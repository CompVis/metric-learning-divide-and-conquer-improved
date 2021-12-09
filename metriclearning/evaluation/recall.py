from __future__ import division

import numpy as np
import torch
import sklearn.metrics.pairwise
from .. import faissext


TORCH_SKLEARN_BACKEND = 'torch+sklearn'
FAISS_BACKEND = 'faiss'
FAISS_GPU_BACKEND = 'faiss-gpu'

_DEFAULT_BACKEND_ = FAISS_GPU_BACKEND


def assign_by_euclidian_at_k(X, T, k, gpu_id=None, backend=_DEFAULT_BACKEND_):
    """
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    if backend == TORCH_SKLEARN_BACKEND:
        distances = sklearn.metrics.pairwise.pairwise_distances(X)
        # get nearest points
        nns = np.argsort(distances, axis = 1)[:, :k + 1]
        for i in range(len(nns)):
            indices = np.nonzero(nns[i, :] != i)[0]
            if len(indices) > k:
                indices = indices[:-1]
            nns[i, :-1] = nns[i, indices]
        nns = nns[:, :-1]
        assert nns.shape[1] == k, nns.shape
    else:
        nns, _ = faissext.find_nearest_neighbors(X,
                                                 k=k,
                                                 gpu_id=None if backend != FAISS_GPU_BACKEND
                                                    else torch.cuda.current_device()
        )
    return np.array([[T[i] for i in ii] for ii in nns])


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))


