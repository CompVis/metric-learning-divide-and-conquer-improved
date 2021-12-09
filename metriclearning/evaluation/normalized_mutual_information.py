import sklearn.cluster
import sklearn.metrics.cluster
import torch

TORCH_SKLEARN_BACKEND = 'torch+sklearn'
FAISS_BACKEND = 'faiss'
FAISS_GPU_BACKEND = 'faiss-gpu'

_DEFAULT_BACKEND_ = FAISS_GPU_BACKEND


def cluster_by_kmeans(X, nb_clusters, gpu_id=None, backend=_DEFAULT_BACKEND_):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    if backend == TORCH_SKLEARN_BACKEND:
        C = sklearn.cluster.KMeans(nb_clusters).fit(X).labels_
    else:
        from metriclearning import faissext
        C = faissext.do_clustering(X,
                                   num_clusters=nb_clusters,
                                   gpu_ids=None if backend != FAISS_GPU_BACKEND
                                    else torch.cuda.current_device(),
                                   niter=100, nredo=5, verbose=1)
    return C

def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys, average_method='arithmetic')
