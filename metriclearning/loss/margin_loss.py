import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from metriclearning.sampler import pdist
from metriclearning.sampler import topk_mask
from metriclearning.sampler import masked_maximum
from metriclearning.sampler import masked_minimum



class MarginLoss(torch.nn.Module):
    """Margin based loss.

    Parameters
    ----------
    nb_classes: int
        Number of classes in the train dataset.
        Used to initialize class-specific boundaries beta.
    margin : float
        Margin between positive and negative pairs.
    nu : float
        Regularization parameter for beta.
    class_specific_beta : bool
        Are class-specific boundaries beind used?
        sampler_args: dict defining the sampler:
            {
             'class': SamplerClass,
             'options': dict of argumets to pass to the class constructor
            }

    Inputs:
        - anchors: sampled anchor embeddings.
        - positives: sampled positive embeddings.
        - negatives: sampled negative embeddings.
        - anchor_classes: labels of anchors. Used to get class-specific beta.

    Outputs:
        Loss value.
    """

    def __init__(self, nb_classes, beta=1.2, margin=0.2, nu=0.0,
 		 class_specific_beta=True, sampler_args=None, **kwargs):
        super(MarginLoss, self).__init__()

        self.nb_classes = nb_classes
        self.class_specific_beta = class_specific_beta
        if class_specific_beta:
            assert nb_classes is not None
            beta = torch.ones(nb_classes, dtype=torch.float32) * beta
        else:
            beta = torch.tensor([beta], dtype=torch.float32)
        self.beta = torch.nn.Parameter(beta)
        self.margin = margin
        self.nu = nu
        if sampler_args is not None:
            self.sampler = sampler_args['class'](**sampler_args['options']).cuda()
        else:
            self.sampler = None

    def forward(self, embeddings, labels):

        anchor_idxs, anchors, positives, negatives = self.sampler(embeddings, labels)
        anchor_classes = labels[anchor_idxs]

        if anchor_classes is not None:
            if self.class_specific_beta:
                # select beta for every sample according to the class label
                beta = self.beta[anchor_classes]
            else:
                beta = self.beta
            beta_regularization_loss = torch.norm(beta, p=1) * self.nu
        else:
            beta = self.beta
            beta_regularization_loss = 0.0
        # TODO: modify to take pairs instead of triplets
        try:
            d_ap = ((positives - anchors)**2).sum(dim=1) + 1e-8
        except Exception as e:
            print(e)
            print(positives.shape, anchors.shape)
            raise e
        d_ap = torch.sqrt(d_ap)
        d_an = ((negatives - anchors)**2).sum(dim=1) + 1e-8
        d_an = torch.sqrt(d_an)

        pos_loss = F.relu(d_ap - beta + self.margin)
        neg_loss = F.relu(beta - d_an + self.margin)

        pair_cnt = torch.sum((pos_loss > 0.0).type(torch.float32) + \
                             (neg_loss > 0.0).type(torch.float32)).type_as(pos_loss)
        loss = torch.sum(pos_loss + neg_loss)
        if pair_cnt > 0.0:
            # Normalize based on the number of pairs.
            loss = (loss + beta_regularization_loss) / pair_cnt
        return loss


class KantorovMarginLoss(nn.Module):
    def __init__(self,
                 alpha=0.2,
                 beta=1.2,
                 distance_threshold=0.5,
                 inf=1e6,
                 eps=1e-6,
                 distance_weighted_sampling=False,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.distance_threshold = distance_threshold
        self.inf = inf
        self.eps = eps
        self.distance_weighted_sampling = distance_threshold
        self.sampler = None

    def forward(self, embeddings, labels):
        assert len(emb) == len(labels)
        return kantorov_margin_loss(
            embeddings,
            labels,
            alpha=self.alpha,
            beta=self.beta,
            distance_threshold=self.distance_threshold,
            inf=self.inf,
            eps=self.eps,
            distance_weighted_sampling=self.distance_weighted_sampling)


def kantorov_margin_loss(
              embeddings,
              labels,
              alpha=0.2,
              beta=1.2,
              distance_threshold=0.5,
              inf=1e6,
              eps=1e-6,
              distance_weighted_sampling=False):
    from metriclearning.sampler import pdist, topk_mask

    d = pdist(embeddings)
    pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d)
                     for dim in [0, 1]]).type_as(d) - torch.autograd.Variable(torch.eye(
                         len(d))).type_as(d)
    num_neg = int(pos.data.sum() / len(pos))
    num_neg = max(1, num_neg)
    assert num_neg > 0

    if distance_weighted_sampling:
        neg = torch.autograd.Variable(
            torch.zeros_like(pos.data).scatter_(
                1,
                torch.multinomial(
                    (d.data.clamp(min=distance_threshold).pow(embeddings.size(-1) - 2) *
                     (1 - d.data.clamp(min=distance_threshold).pow(2) / 4).pow(
                         0.5 * (embeddings.size(-1) - 3))).reciprocal().masked_fill_(
                             pos.data + torch.eye(len(d)).type_as(d.data) > 0, eps),
                    replacement=False,
                    num_samples=num_neg), 1))
    else:
        neg = topk_mask(
            d + inf * ((pos > 0) + (d < distance_threshold)).type_as(d), dim=1, largest=False, K=num_neg)
    L = F.relu(alpha + (pos * 2 - 1) * (d - beta))
    M = ((pos + neg > 0) * (L > 0)).float()
    return (M * L).sum() / M.sum()

