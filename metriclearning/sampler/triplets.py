from __future__ import print_function
from __future__ import division

from collections import Counter
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DistanceWeightedSampler',
           'EasyPeasyPairsSampler',
           'FlexibleTripletSampler',
           'pdist',
           'topk_mask',
           'masked_maximum',
           'masked_minimum']


def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    # TODO: make numerically stable like in TF
    # tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py
    if squared:
        return res
    else:
        return res.clamp(min=eps).sqrt()


def topk_mask(input, dim, K=10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim=dim, **kwargs)[1]
    return torch.zeros_like(input.data).scatter(dim, index, 1.0)


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.
    Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = data.min(dim, keepdim=True)[0]
    masked_maximums = torch.max(
      torch.mul(data - axis_minimums, mask), dim=dim,
        keepdim=True)[0] + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
    Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = data.max(dim, keepdim=True)[0]
    masked_minimums = torch.min(
      torch.mul(data - axis_maximums, mask), dim=dim,
      keepdim=True)[0] + axis_maximums
    return masked_minimums


class DistanceWeightedSampler(nn.Module):
    """Distance weighted sampling.

    Sample negative pairs uniformly according to distances,
        i.e. sampling with weights q(d)**-1.
    See "Sampling matters in deep embedding learning" paper for details.

    Parameters
    ----------
    batch_k : int
        Number of images per class.

    Inputs:
        - **data**: input tensor with shape (batch_size, embed_dim).
        Here we assume the consecutive batch_k examples are of the same class.
        For example, if batch_k = 5, the first 5 examples belong to the same class,
        6th-10th examples belong to another class, etc.

    Outputs:
        - a_indices: indices of anchors.
        - x[a_indices]: sampled anchor embeddings.
        - x[p_indices]: sampled positive embeddings.
        - x[n_indices]: sampled negative embeddings.
        - x: embeddings of the input batch.
    """
    def __init__(self, cutoff=0.5, nonzero_loss_cutoff_dist=1.4,
                 eps=1e-6, **kwargs):
        self.cutoff = cutoff
        self.eps = eps

        # We sample only from negatives that induce a non-zero loss.
        # These are negatives with a distance < nonzero_loss_cutoff.
        # With a margin-based loss, nonzero_loss_cutoff == margin + beta.
        self.nonzero_loss_cutoff_dist = nonzero_loss_cutoff_dist
        logging.info(f'DistanceWeightedSampler(cutoff={self.cutoff}, eps={self.eps},'
                    f' nonzero_loss_cutoff_dist={self.nonzero_loss_cutoff_dist}')
        super(DistanceWeightedSampler, self).__init__()

    @staticmethod
    def get_inverse_pairwise_distance_prob(distances, n_dims, ignore_mask):
        """
        Compute 1 / prob(d) for pairwise distances on the sphere.
        Not normalized
        """
        inv_log_q_d = (2.0 - float(n_dims)) * torch.log(distances) \
                      - (float(n_dims - 3) / 2) * \
                      torch.log(1.0 - 0.25 * (distances ** 2.0))
        inv_log_q_d[ignore_mask] = 0

        # Subtract max(log(distances)) for numerical stability.
        weights = torch.exp(inv_log_q_d - inv_log_q_d.max(dim=1, keepdim=True)[0])
        return weights

    def forward(self, x, labels):
        n = x.shape[0]

        # Cut off to avoid high variance.
        distances = pdist(x).clamp_(min=self.cutoff)

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(distances) for dim in [0, 1]]).type_as(distances) - torch.eye(n).type_as(distances)
        pos_and_self_mask = (pos.data + torch.eye(n).type_as(distances)) > 0
        zero_loss_mask = distances >= self.nonzero_loss_cutoff_dist
        weights = self.get_inverse_pairwise_distance_prob(distances, n_dims=x.shape[1],
                                                          ignore_mask=(pos_and_self_mask | zero_loss_mask))
        weights += 1e-9

        num_neg = int(pos.data.sum() / len(pos) + 0.5)
        num_neg = max(1, num_neg)
        # Make sure that num_neg >= num samples
        # from another class in the batch
        # For ex: class_1: 15 samples, class_2: 5 samples. num_neg = 10 < 5
        class_counter = Counter(labels.data.cpu().numpy())
        #print('Classes count in the batch:', class_counter)
        max_class_size = class_counter.most_common(1)[0][1]
        num_neg = min(num_neg, n - max_class_size)
        assert num_neg <= n - max_class_size
        assert num_neg >= 1

        weights.masked_fill_(pos_and_self_mask, 0.0) \
                .masked_fill_(distances > self.nonzero_loss_cutoff_dist,
                              self.eps)
        assert torch.isfinite(weights).all()

        neg = torch.zeros_like(pos).scatter_(
            1,
            torch.multinomial(weights,
                              num_samples=num_neg,
                              replacement=False),
            1)

        a_indices = []
        p_indices = []
        n_indices = []
        cnt_warnings = 0
        for i in range(n):
            if pos[i].data.sum() < 1:
                #logging.debug(f'No pos for anchor {i} - skip')
                continue

            cur_posits = np.atleast_1d(pos[i].nonzero().squeeze().cpu().numpy())
            cur_negs = np.atleast_1d(neg[i].nonzero().squeeze().cpu().numpy())
            if len(cur_negs) != len(cur_posits):
                if len(cur_posits) < len(cur_negs):
                    if cnt_warnings < 0:
                        logging.debug('Too many negatives: '\
                                       'Anchor idx={}, n_pos={}, n_neg= {}'\
                                        .format(i, len(cur_posits), len(cur_negs)))
                        cnt_warnings += 1
                    # duplicate positives with repetitions
                    cur_posits = np.random.choice(cur_posits, size=len(cur_negs))
                else:
                    if cnt_warnings < 0:
                        logging.debug('Too many negatives: '\
                                       'Anchor idx={}, n_pos={}, n_neg= {}'\
                                        .format(i, len(cur_posits), len(cur_negs)))
                        cnt_warnings += 1
                    # select only subset of positives
                    cur_posits = np.random.choice(cur_posits, size=len(cur_negs),
                                                  replace=False)


            p_indices.extend(cur_posits)
            n_indices.extend(cur_negs)
            a_indices.extend([i] * len(cur_posits))
        assert len(a_indices) == len(p_indices) == len(n_indices), \
                '{}, {}, {}'.format(*map(len, [a_indices, p_indices, n_indices]))
        assert len(a_indices), a_indices
        return a_indices, x[a_indices], x[p_indices], x[n_indices]

    def __repr__(self):
        s = '{name}({batch_k}{cutoff})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


class EasyPeasyPairsSampler(nn.Module):
    """
    Sample for each anchor negative examples
        are K closest points on the distance >= cutoff

    Inputs:
        - **data**: input tensor with shape (batch_size, embed_dim).
    Outputs:
        - a_indices: indices of anchors.
        - x[a_indices]: sampled anchor embeddings.
        - x[p_indices]: sampled positive embeddings.
        - x[n_indices]: sampled negative embeddings.
        - x: embeddings of the input batch.
    """

    def __init__(self, cutoff=0.5, infinity=1e6, eps=1e-6,
                 one_triplet_per_sample=False,
                 **kwargs):
        super(EasyPeasyPairsSampler, self).__init__()
        self.cutoff = cutoff
        self.infinity = infinity
        self.eps = eps
        self.one_triplet_per_sample = one_triplet_per_sample

    def forward(self, x, labels):
        d = pdist(x)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d)
                         for dim in [0, 1]]).type_as(d) - (torch.eye( len(d))).type_as(d)
        num_neg = int(pos.data.sum()) // len(pos)
        num_neg = max(1, num_neg)
        #logging.debug('total num pos pairs = {}; Num_neg={}'.format(int(pos.data.sum()), num_neg))
        neg = topk_mask(d + self.infinity * ((pos > 0) + (d < self.cutoff)).type_as(d),
                        dim=1,
                        largest=False,
                        K=num_neg)

        a_indices = []
        p_indices = []
        n_indices = []
        cnt_warnings = 0
        for i in range(len(d)):
            if self.one_triplet_per_sample:
                a_indices.append(i)
                p_indices.append(np.random.choice(np.where(pos[i].nonzero().squeeze().cpu().numpy())[0]))
                n_indices.append(np.random.choice(np.where(neg[i].nonzero().squeeze().cpu().numpy())[0]))
            else:
                cur_posits = np.atleast_1d(pos[i].nonzero().squeeze().cpu().numpy())
                cur_negs = np.atleast_1d(neg[i].nonzero().squeeze().cpu().numpy())
                if len(cur_negs) != len(cur_posits):
                    if cnt_warnings < 1:
                        logging.debug('Probably too many positives, because of lacking'\
                                        ' classes in the current cluster.'\
                                        ' Anchor idx={}, n_pos={}, n_neg= {}'\
                                        .format(i, len(cur_posits), len(cur_negs)))
                        cnt_warnings += 1
                    min_len = min(map(len, [cur_posits, cur_negs]))
                    cur_posits = cur_posits[:min_len]
                    cur_negs = cur_negs[:min_len]
                p_indices.extend(cur_posits)
                n_indices.extend(cur_negs)
                a_indices.extend([i] * len(cur_posits))

        assert len(a_indices) == len(p_indices) == len(n_indices), \
                '{}, {}, {}'.format(*map(len, [a_indices, p_indices, n_indices]))
        return a_indices, x[a_indices], x[p_indices], x[n_indices]


class FlexibleTripletSampler(nn.Module):
    """
    Flexible Triplet sampler.

    Can even sample EPSHN triplets.
    EPSHN: EasyPositive SemihardNegative.
        Sample 1 triplet per anchor with the easiest positive and
            the closest (hardest) semihard negative
        See "Improved mbeddings with Easy Positive Triplet Mining"

    Parameters
    ----------
    positive : (str) one of ['random', 'easy']
        How to sample positives.
    negative : (str) one of ['random', 'semihard']
        How to sample negatives.

    Inputs:
        - **data**: input tensor with shape (batch_size, embed_dim).

    Outputs:
        - a_indices: indices of anchors.
        - x[a_indices]: sampled anchor embeddings.
        - x[p_indices]: sampled positive embeddings.
        - x[n_indices]: sampled negative embeddings.
        where  x is the embeddings of the input batch.
    """
    def __init__(self, margin=0.2, positive='random', negative='random', **kwargs):
        self.positive = positive
        self.negative = negative
        self.margin = margin

        assert self.positive in ['random', 'easy', 'hard']
        assert self.negative in ['random', 'semihard', 'hard']

        logging.info(f'FlexibleTripletSampler(positive={self.positive}, negative={self.negative},'
                    f' margin={self.margin}')
        super().__init__()

    @staticmethod
    def random_choice_one(values):
        idx = torch.randint(0, values.size(0), size=(1,),
                            dtype=torch.long, device=values.device)
        return values[idx]

    def forward(self, x, labels):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        n = x.shape[0]

        dist_mat = pdist(x.detach())
        max_dist = dist_mat.max()

        a_indices = []
        p_indices = []
        n_indices = []
        for i in range(bs):
            neg = labels != labels[i]
            pos = labels == labels[i]
            pos[i] = 0
            if pos.sum() < 1:
                continue

            d_pos = dist_mat[i, :].clone() * pos

            if not pos.sum() or not neg.sum():
                continue
            if self.positive == 'random':
                idx_pos = np.random.choice(np.where(pos)[0])
            elif self.positive == 'easy':
                # we mask out all negatives
                idx_pos = (dist_mat[i, :] + neg * (max_dist + 0.1)).argmin()
            elif self.positive == 'hard':
                # we mask out all negatives
                idx_pos = (dist_mat[i, :] - neg * (max_dist + 0.1)).argmax()
            else:
                raise ValueError(f'Unknown value for positive mode(={self.positive}')

            good_neg_mask = (dist_mat[i, :] < dist_mat[i, idx_pos] + self.margin)
            if good_neg_mask.sum() < 1:
                continue
            if self.negative == 'random':
                idx_neg = self.random_choice_one(good_neg_mask.nonzero().squeeze())
            elif self.negative == 'semihard':
                good_neg_within_margin_mask = \
                    (dist_mat[i, :] > dist_mat[i, idx_pos]) & good_neg_mask
                idx_neg = self.random_choice_one(
                    good_neg_within_margin_mask.nonzero().squeeze())
            elif self.negative == 'hard':
                # we mask out all positives
                idx_neg = (dist_mat[i, :] + pos * (max_dist + 0.1)).argmin()
            else:
                raise ValueError(f'Unknown value for negative mode(={self.negative}')

            anchors.append(i)
            positives.append(idx_pos)
            negatives.append(idx_neg)

        return anchors, x[anchors], x[positives], x[negatives]


if __name__ == '__main__':
        n_dims = 512
        torch.set_printoptions(precision=16)
        np.set_printoptions(precision=16)
        print("Test inverse distance:")
        test_dists = torch.tensor([np.linspace(0.01, 1.99, 21).tolist()] , dtype=torch.float32).clamp_(min=0.5)
        print('test distances:', test_dists)
        print('weights:', DistanceWeightedSampler.get_inverse_pairwise_distance_prob(test_dists, n_dims=n_dims, pos_mask=torch.zeros_like(test_dists, dtype=torch.uint8)))

        def inverse_sphere_distances(dim, dist, labels, anchor_label ):

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            #log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
#            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse.
            # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

            #q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

        print('----')
        print(inverse_sphere_distances(n_dims, test_dists, np.zeros(len(test_dists)), 1))
