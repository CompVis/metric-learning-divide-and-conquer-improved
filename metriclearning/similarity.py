from __future__ import print_function
from __future__ import division

import logging
import random
import torch
import sklearn
import time
import numpy as np

from . import utils
from . import faissext

__all__ = ['pairwise_distance', 'calc_neighbors', 'get_cluster_labels']

TORCH_SKLEARN_BACKEND = 'torch+sklearn'
FAISS_BACKEND = 'faiss'
FAISS_GPU_BACKEND = 'faiss-gpu'
_backends_ = [TORCH_SKLEARN_BACKEND, FAISS_BACKEND, FAISS_GPU_BACKEND]
backends_available = [TORCH_SKLEARN_BACKEND, FAISS_BACKEND, FAISS_GPU_BACKEND]

_DEFAULT_BACKEND_ = FAISS_BACKEND
_DEFAULT_BACKEND_ = FAISS_GPU_BACKEND
#_DEFAULT_BACKEND_ = TORCH_SKLEARN_BACKEND


def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
    feature: 2-D Tensor of size [number of data, feature dimension].
    squared: Boolean, whether or not to square the pairwise distances.
    Returns:
    pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    a = torch.as_tensor(np.atleast_2d(a))
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals).data.cpu().numpy()

    return pairwise_distances


def nearest_neighbors(D, nb_neighbors, labels=None):
    """
    D: pairwise distance matrix for samples.
    returns matrix containing nearest neigbhours for each sample.
    """
    # alt: torch.sort(D, dim = 1)[1][:, 1 : nb_neighbors + 1]
    assert 'tensor' not in str(type(D)).lower(), type(D)
    nns = []
    for d in D:
        nns += [[*np.argsort(d)[: nb_neighbors + 1]]]
    nns = np.asarray(nns)
    for i in range(len(nns)):
        indices = np.nonzero(nns[i, :] != i)[0]
        if len(indices) > nb_neighbors:
            indices = indices[:-1]
        nns[i, :-1] = nns[i, indices]
    nns = nns[:, :-1]

    return nns if labels is None else labels[nns]


def non_nearest_neighbors(D, nb_neighbors, labels=None):
    """
    Like nearest neighbors, but choose set difference with
    equal (like nearest neighbors) number of non nearest neighbors.
    """
    N = []
    for d in D:
        N += [random.sample(
            list(*[torch.sort(d)[1][nb_neighbors + 1:]]), nb_neighbors
        )]
    return N


def random_sample_non_nearest_examples(nns, k, labels=None):
    assert 'tensor' not in str(type(nns)).lower(), type(nns)
    non_nearest = []
    set_all_indices = set(range(len(nns)))
    for i, row in enumerate(nns):
        non_nearest.append(
            np.random.choice(list(set_all_indices - set(row) - {i}), size=k, replace=False)
        )
    return np.array(non_nearest)


def get_cluster_labels(model, data_loader, use_penultimate, nb_clusters,
                       gpu_id=None,
                       backend=_DEFAULT_BACKEND_, with_X = False,
                       ntrials=1,
                       random_state=None):
    """
    Try ntrials times until every cluster has at least 2 GT classes with >= 2 samples
    """
    is_dry_run = (nb_clusters == 1) and not with_X
    if not is_dry_run:
        if not use_penultimate:
            logging.debug('Using the final layer for clustering')
        X_all, T_all, I_all = utils.predict_batchwise(
            model=model,
            dataloader=data_loader,
            use_penultimate=use_penultimate,
            is_dry_run=is_dry_run
        )
        perm = np.argsort(I_all)
        X_all = X_all[perm]
        I_all = I_all[perm]
        T_all = T_all[perm]

        seed = 1234 if random_state is None else random_state.randint(999999)
        for trial in range(ntrials):
            if backend == TORCH_SKLEARN_BACKEND:
                clustering_algorithm = sklearn.cluster.KMeans(
                    n_clusters=nb_clusters, random_state=seed + trial)
                C = clustering_algorithm.fit(X_all).labels_
            else:
                C = faissext.do_clustering(X_all,
                                           num_clusters=nb_clusters,
                                           gpu_ids=None if backend != FAISS_GPU_BACKEND
                                               else torch.cuda.current_device(),
                                           niter=100, nredo=1, verbose=0,
                                           seed=seed + trial)
            is_ok = True
            for c in range(nb_clusters):
                gt_class_labels, counts = np.unique(T_all[C == c], return_counts=True)
                if len(counts[counts >= 2]) < 2:
                    is_ok = False
                    break
            if is_ok:
                # This trial yielded more or less balanced clustering
                break
            else:
                logging.debug(f' - Trial #{trial} of clustering failed')
    else:
        T_all = np.array(data_loader.dataset.ys)
        I_all = np.array(data_loader.dataset.I)
        C = np.zeros(len(T_all), dtype=int)
    if with_X:
        return C, T_all, I_all, np.array(X_all)
    else:
        return C, T_all, I_all


def calc_neighbors(model, dataloader, use_penultimate,
                   nb_neighbors=5, gpu_id=None, backend=_DEFAULT_BACKEND_):
    assert backend in _backends_
    with torch.no_grad():
        X_all, _, I_all = utils.predict_batchwise(
            model,
            dataloader,
            use_penultimate=use_penultimate
        )
    perm = np.argsort(I_all)
    X_all = X_all[perm]
    I_all = I_all[perm]
    if backend == TORCH_SKLEARN_BACKEND:
        D = pairwise_distance(X_all, squared = True)
        nearest_n = nearest_neighbors(
            D, nb_neighbors = nb_neighbors
        )
    else:
        nearest_n, _ = faissext.find_nearest_neighbors(X_all,
                                            k=nb_neighbors, gpu_id=None
                                               if backend != FAISS_GPU_BACKEND
                                               else torch.cuda.current_device())

    nearest_n_not = random_sample_non_nearest_examples(nearest_n, k=nb_neighbors)
    return nearest_n, nearest_n_not


def benchmark_neighbors_search(n=30000, k=11):
    x = torch.rand(n, 512)
    print('Torch:')
    tic = time.time()
    #D = pairwise_distance(x, squared = True)
    #nearest_n = nearest_neighbors(D, nb_neighbors=k)
    print('Elapsed time: {} sec'.format(time.time() - tic))

    print('FAISS:')
    tic = time.time()
    try:
        nns, _ = faissext.find_nearest_neighbors(x.data.cpu().numpy(), k=k, gpu_id=None)
        print(nns.shape)
    except Exception as e:
        print('FAISS failed', e)
    print('Elapsed time: {} sec'.format(time.time() - tic))


def benchmark_clustering(n=100000, k=100):
    clustering_algorithm = sklearn.cluster.KMeans(
        n_clusters=k, verbose=1, n_init=1)
    print('K-means, n={}, k={}'.format(n, k))
    a = torch.rand(n, 512)

    print('FAISS:')
    tic = time.time()
    C = faissext.do_clustering(a.data.cpu().numpy(), num_clusters=k, gpu_ids=None)
    print('Elapsed time: {} sec'.format(time.time() - tic))

    print('Sklearn:')
    tic = time.time()
    C = clustering_algorithm.fit(a.data.cpu().numpy()).labels_
    print('Elapsed time: {} sec'.format(time.time() - tic))


if __name__ == '__main__':

    #a = torch.rand(32, 64)
    #D = pairwise_distance(a)
    #print("pairwise distance matrix: {}".format(D))
    #print("nearest neighbors: {}".format(nearest_neighbors(D, 5)))
    #print("non nearest neighbors: {}".format(non_nearest_neighbors(D, 5)))
    benchmark_clustering()
    benchmark_neighbors_search(10000)
