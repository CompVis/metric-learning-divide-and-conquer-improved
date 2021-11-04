import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from metriclearning.sampler import pdist
from metriclearning.sampler import topk_mask
from metriclearning.sampler import masked_maximum
from metriclearning.sampler import masked_minimum


class TripletAllLoss(torch.nn.Module):
    """
    Compute all possible triplets in the batch, where loss > 0
    """
    def __init__(self, margin=1.0, **kwargs):
        print(f'Create TripletAllLoss with margin={margin}')
        self.margin = margin
        torch.nn.Module.__init__(self)

    def forward(self, embeddings, labels):
        d = pdist(embeddings)
        pos = torch.eq(
            *[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]
        ).type_as(d) - torch.eye(len(d)).type_as(d)
        # T[a, n, p] = d[a, p]
        T = d.unsqueeze(1).expand(*(len(d),) * 3)
        # M[a, n, p] = pos[a, p] * (1 - pos[a, n])
        # M[a, n, p] == 1 iff (a,p) is positive pair and (a,n) is negative pair.
        M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
        # dist_diff[a, n, p] = max(0, d[a, p] - d[a, n] + margin)
        dist_diff = F.relu(T - T.transpose(1, 2) + self.margin)

        return (M * dist_diff).sum() / M.sum()


class TripletSemihardLoss(torch.nn.Module):
    """
    Computes the triplet loss with semi-hard negative mining.
    For every positive pair find only *one* hardest semihard negative example.

    The loss encourages the positive distances (between a pair of embeddings with
    the same labels) to be smaller than the minimum negative distance - margin.
    The negative distance is selected among the neg pairsi
      which are at least greater than the positive distance (called semi-hard negatives),
      but withing the margin radius from anchor.
     If no such negative exists,
    uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.
    Args:
        labels: 1-D int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        margin: Float, margin term in the loss definition.
    Returns:
        triplet_loss: float32 scalar.
    """
    def __init__(self, margin=1.0, soft=False, **kwargs):
        self.margin = margin
        self.soft = soft
        torch.nn.Module.__init__(self)

    def forward(self, embeddings, labels):
        d = pdist(embeddings)

        pos = torch.eq(
            *[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]
        ).type_as(d) - torch.eye(len(d)).type_as(d)
        # T[a, n, p] = d[a, p]
        T = d.unsqueeze(1).expand(*(len(d),) * 3)
        # M[a, n, p] = pos[a, p] * (1 - pos[a, n])
        # M[a, n, p] == 1 iff (a,p) is positive pair and (a,n) is negative pair.
        M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
        #M = M.byte()
        # dist_diff[a, n, p] = d[a, p] - d[a, n]
        dist_diff = T - T.transpose(1, 2)

        # negatives_outside: smallest D_an where D_an > D_ap.
        mask = M * (dist_diff < 0).type_as(M)
        mask_final = (M * (dist_diff < 0).type_as(M)).sum(dim=1, keepdim=True)
        mask_final = mask_final > 0
        mask_final.squeeze_(dim=1)
        assert len(mask_final.shape) == 2

        # Find the closest semihard negative
        # dist_diff[a, p] = d[a, p] - min_{n}(d[a, n]), where mask > 0
        # shape is [N, N]
        dist_diff_negatives_outside = masked_maximum(dist_diff, mask, dim=1)\
                                        .squeeze_(dim=1)

        # dist_diff[a, p] = d[a, p] - max_{n}(d[a, n]), i.e. select the furthest negative
        # shape is [N, N]
        dist_diff_negatives_inside = masked_minimum(dist_diff, M, dim=1)\
                                        .squeeze_(dim=1)
        # For every positive pair if there is semihard available - use it.
        # If not - means all negatives are closer than positive (lie inside the positive radius),
        # than use the furthest of these.
        dist_diff_semi_hard_negatives = \
            torch.where(mask_final, dist_diff_negatives_outside, dist_diff_negatives_inside)
        if self.soft:
            # soft triplet loss
            # log(1 + exp(d_ap - d_an))
            loss_mat = dist_diff_semi_hard_negatives.exp().log1p()
        else:
            loss_mat = dist_diff_semi_hard_negatives + self.margin
        assert len(loss_mat.shape) == 2
        assert len(pos.shape) == 2


        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        return F.relu(pos * loss_mat).sum() / pos.sum()


class TripletLoss(torch.nn.Module):

    def __init__(self, margin=0.2, soft=False,
                 avg_non_zero_only=True,
                 sampler_args=None, **kwargs):
        self.margin = margin
        self.soft = soft
        self.avg_non_zero_only = avg_non_zero_only
        super(TripletLoss, self).__init__()
        # self.sampler = sampler(margin=margin)
        if sampler_args is not None:
            self.sampler = sampler_args['class'](
                margin=margin,
                **sampler_args['options']
            ).cuda()
        else:
            self.sampler = None

    def forward(self, embeddings, labels):

        anchor_idxs, anchors, positives, negatives = \
            self.sampler(embeddings, labels)

        d_ap = F.pairwise_distance(anchors, positives, p=2)
        d_an = F.pairwise_distance(anchors, negatives, p=2)
        diff = d_ap - d_an
        if self.soft:
            loss = diff.exp().log1p()
        else:
            loss = F.relu(diff + self.margin)
        if self.avg_non_zero_only:
            cnt = (loss > 0).nonzero().size(0)
            return (loss.sum() / cnt) if cnt else loss.sum()
        return loss.mean()
