import logging
import numpy as np

from metriclearning import faissext


def log_clustering_stats(C, T):
    pass
    logging.info(' -- Clusters stats:')
    c_labels, c_sizes = np.unique(C, return_counts=True)
    for c, sz in zip(c_labels, c_sizes):
        if c == -1:
            continue
        nb_classes = len(np.unique(T[C == c]))
        logging.info(f' --- {c}: {sz} images ({nb_classes} GT classes]))')

    logging.info(
        ' --- number of unassigned samples: {}'.format(C[C == -1].sum())
    )

def merge_clusters(C, gpu_ids = None):
    C_merge = np.ones_like(C, dtype=np.int32) * -1
    for c in np.unique(C):
        C_merge[C == c] = c // 2
    return C_merge


def divide_one_cluster_with_n_trials(X, T,
                                     gpu_ids=None, ntrials=10,
                                     random_state=None):
    """
    Divide a cluster in 2 smaller.
    Try ntrials times until every split has at least 2 GT classes with >= 2 samples

    random_state: (int) used to sample random seed for faiss runs.
        The seed is hardcoded if random_state is None

        We need to change faiss seed at each k-means run so that the randomly picked
        initialization centroids do not correspond to the same feature ids
        from an epoch to another (what happens if seed is alway the same).

    Returns whatever we have if the last trial reached.
    """
    is_ok = False
    seed = 1234 if random_state is None else random_state.randint(999999)
    for trial in range(ntrials):
        C_bin = faissext.do_clustering(
                X,
                num_clusters=2,
                gpu_ids=gpu_ids,
                nredo=1,
                seed=seed + trial,
            )
        is_ok = True
        # Now check each resultant split that it has enough classes
        for c in range(2):
            gt_class_labels, counts = np.unique(T[C_bin == c], return_counts=True)
            if len(counts[counts >= 2]) < 2:
                is_ok = False
                break

        if is_ok:
            # This trial yielded more or less balanced clustering
            break
        else:
            logging.debug(f' - Trial #{trial} to divide a cluster failed')

    return C_bin



def divide_clusters(X, C, T, ntrials=1, gpu_ids=None, random_state=None):
    """
        Divide each cluster into two clusters using k-means.
        X[i] corresponds to cluster label C[i] and to GT label T[i]

        Note: assumes that C was reassigned (with hungarian) already
    """
    assert len(X) == len(C) == len(T)

    C_div = np.ones_like(C, dtype=np.int32) * -1

    for c in np.unique(C):
        cur_X = X[C == c]
        cur_T = T[C == c]
        cur_indices = np.arange(len(X))[C == c]

        logging.debug(f'Divide cluster {c}:')
        C_bin = divide_one_cluster_with_n_trials(cur_X, cur_T,
                                                 gpu_ids=gpu_ids,
                                                 ntrials=ntrials,
                                                 random_state=random_state)

        # binary: for each c, create two clusters
        for c_bin in range(2):
            C_div[cur_indices[C_bin == c_bin]] = c * 2 + c_bin

    return C_div


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler()
        ]
    )

    gpu_ids = [0]
    nb_clusters_initially = 4
    # 1000 samples with 32 sized embedding
    N = 1000
    X = np.random.randn(N, 32)
    T = np.random.randint(0, 10, size=N, dtype=np.int32)
    C = faissext.do_clustering(
            X,
            nb_clusters_initially
        )


    # divide 4 clusters into 8
    C_div = divide_clusters(X, C, T, ntrials=2, gpu_ids=gpu_ids)

    # merging these 8 clusters will result in same clustering as before
    assert np.array_equal(merge_clusters(C_div), C)

