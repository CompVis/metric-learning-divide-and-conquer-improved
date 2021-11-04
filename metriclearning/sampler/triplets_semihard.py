from __future__ import print_function
from __future__ import division

# THIS FILE IS NOT CURRENTLY USED

"""
NOTE: sampling class balanced means that last batch might be dropped
therefore, must create seperate dataloaders without class balance for
sampling semihard triplets; TODO: fix it properly (not like hack now)
"""

"""
NOTE: clean up `semihard`, see `semihard_online`
"""
import torch
from ..utils import predict_batchwise
from ..similarity import pairwise_distance


def semihard_offline(model, dataloader, use_penultimate, margin):
    """
    X, T, I: embeddings, targets, indices of samples
    I, P, N: indices of anchors (all), positives, negatives.

    Vadim's implementation
    """
    import itertools
    semihard_triplets = {}

    X_all, T_all, I_all = predict_batchwise(
        model,
        dataloader,
        use_penultimate=use_penultimate
    )

    # -1 is selected as masking value for next steps
    assert (I_all[I_all < 0]).sum() == 0

    D = pairwise_distance(X_all)

    for t, a in zip(T_all, I_all):

        # positives of anchor
        P = torch.where(
            # if same target label and not same point (index)
            (T_all == t) & (I_all != a),
            I_all,
            torch.ones_like(I_all) * -1)
        P = P.unique()
        P = P[P >= 0] # filter out negative entries

        # negatives of anchor
        N = torch.where(T_all != t, I_all, torch.ones_like(I_all) * -1)
        N = N.unique()
        N = N[N >= 0] # filter out negative entries

        # cartesian product for P/N combinations
        P, N = torch.tensor(list(itertools.product(P, N))).t()

        # actual semi-hard constraint
        mask = (D[a, P] < D[a, N]) & (D[a, N] < D[a, P] + margin)

        # selecting semihard triplets
        A = a.repeat(*P.size())
        valid_indices = torch.masked_select(
            torch.arange(len(A)),
            mask).long()
        all_triplets = torch.stack([A, P, N], dim = 1)
        semihard_triplets[a.item()] = all_triplets[valid_indices]

        # in case that no semihard triplets left, random sampling
        if len(semihard_triplets[a.item()]) == 0:
            print('No semihard triplets left for sample {}.'.format(a))
            print('Choosing 100 triplets without semihard-constraint.')
            semihard_triplets[a.item()] = all_triplets[
                torch.randint(
                    low = 0,
                    high = len(all_triplets),
                    size = (100,)).long()]

    return semihard_triplets


def semihard_online(X, T, margin):
    """
    X, T, I: embeddings, targets, indices of samples
    I, P, N: indices of anchors (all), positives, negatives.

    Vadim's implementation
    """
    import itertools

    triplets = []
    D = pairwise_distance(X)
    D = torch.from_numpy(D)
    I = torch.arange(len(X)).long()

    for a, t in zip(I, T):

        P = I[(T == t) & (I != a)]
        N = I[(T != t)]

        # cartesian product for P/N combinations
        P, N = torch.tensor(list(itertools.product(P, N))).t()

        # actual semi-hard constraint
        mask = (D[a, P] < D[a, N]) & (D[a, N] < D[a, P] + margin)

        # repeat A for selecting all triplets together with mask
        A = a.repeat(len(mask))

        triplets += [*torch.stack([A, P, N], dim = 1)[mask == 1]]

    return torch.stack(triplets)

