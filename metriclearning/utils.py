from __future__ import print_function
from __future__ import division

from . import evaluation
from . import similarity
import numpy as np
import torch
import logging
from tqdm import tqdm
from . import faissext

def part_dict(origin_dict, keyword):
    # function to filter out part of the weights of a state_dict
    from collections import OrderedDict
    filtered_dict = OrderedDict()

    for key in origin_dict:
        if keyword not in key:
            filtered_dict[key] = origin_dict[key]

    return filtered_dict


def predict_batchwise(model, dataloader, use_penultimate, is_dry_run=False, learner_id=None):
    # list with N lists, where N = |{image, label, index}|

    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # use tqdm when the dataset is large (SOProducts)
        is_verbose = len(dataloader.dataset) > 0

        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc='predict', disable=not is_verbose):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    if not is_dry_run:
                        # move images to device of model (approximate device)
                        J = J.to(list(model.parameters())[0].device)
                        # predict model output for image
                        J = model(
                            J,
                            use_penultimate = use_penultimate,
                            dset_id=learner_id
                        ).data.cpu().numpy()
                    else:
                        # just a placeholder not to break existing code
                        J = np.array([-1])
                for j in J:
                    A[i].append(np.asarray(j))
        result = [np.stack(A[i]) for i in range(len(A))]
    model.train()
    model.train(model_is_training) # revert to previous training state
    if is_dry_run:
        # do not return features if is_dry_run
        return [None, *result[1:]]
    else:
        return result


def evaluate_in_shop(model, dl_query, dl_gallery, use_penultimate, backend,
        K = [1, 10, 20, 30, 50], with_nmi=True, final_eval=False):

    # calculate embeddings with model and get targets
    X_query, T_query, _ = predict_batchwise(model, dl_query, use_penultimate=use_penultimate)
    X_gallery, T_gallery, _ = predict_batchwise(model, dl_gallery, use_penultimate=use_penultimate)

    nb_classes = dl_query.dataset.nb_classes()
    assert nb_classes == len(set(T_query))

    # calculate full similarity matrix, choose only first `len(X_query)` rows
    # and only last columns corresponding to the column
    T_eval = torch.cat(
        [torch.from_numpy(T_query), torch.from_numpy(T_gallery)])
    X_eval = torch.cat(
        [torch.from_numpy(X_query), torch.from_numpy(X_gallery)])
    D = similarity.pairwise_distance(X_eval)[:len(X_query), len(X_query):]

    D = torch.from_numpy(D)
    # get top k labels with smallest (`largest = False`) distance
    Y = T_gallery[D.topk(k = max(K), dim = 1, largest = False)[1]]

    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T_query, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    if with_nmi:
        # calculate NMI with kmeans clustering
        if final_eval:
            # should only be used during evaluation for trained model
            # nmi will be computed 10 times and we take the mean
            nmi = avg_nmi(backend, nb_classes, X_eval.numpy(), T_eval.numpy())

        else:
            nmi = evaluation.calc_normalized_mutual_information(
                T_eval.numpy(),
                evaluation.cluster_by_kmeans(
                    X_eval.numpy(), nb_classes, backend=backend
                )
            )
            logging.info("NMI: {:.3f}".format(nmi * 100))
    else:
        nmi = None

    if final_eval:
        return nmi, recall, Y
    else:
        return nmi, recall


def evaluate_market(model, dl_query, dl_gallery, K=[1, 5, 10]):
    """
        we will need the camera id information to mask out some sample in gallery
    """

    # calculate embeddings with model and get targets
    X_query, T_query, _ = predict_batchwise(model, dl_query, use_penultimate=False)
    Cid_query = np.asarray(dl_query.dataset.cid)

    X_gallery, T_gallery, _ = predict_batchwise(model, dl_gallery, use_penultimate=False)
    Cid_gallery = np.asarray(dl_gallery.dataset.cid)

    # nns are indices of the nearest neighnpors
    nns, _ = faissext.find_nearest_neighbors(X_gallery, queries=X_query, k=len(X_gallery))

    """quote from original paper
    The search process is performed in a cross-camera mode, i.e., 
    relevant images captured in the same camera as the query are 
    viewed as “junk”. 
    """
    nns_clean = []

    for i in range(len(T_query)):
        # get the current camera id and person id
        cid = Cid_query[i]
        pid = T_query[i]
        # index of the gallery entries with same cid and same pid as current query image
        mask1 = np.where(np.logical_and(Cid_gallery == cid, T_gallery == pid))[0]
        # where these indices appear in nns
        mask2 = np.where(np.in1d(nns[i], mask1))[0]

        nns_clean.append(np.delete(nns[i], mask2))

        # the label of nns
    Y = [[T_gallery[i] for i in ii] for ii in nns_clean]

    recall = []
    for k in K:
        # Y don't have to be ndarray, could be list
        r_at_k = evaluation.calc_recall_at_k(T_query, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    mAP = evaluation.mean_avg_precision(
        T_query=T_query, Y=Y)

    logging.info("mAP: {:.3f}".format(mAP * 100))

    return mAP, recall


def evaluate(model, dataloader, use_penultimate, backend, K = [1, 2, 4, 8],
        with_nmi = False):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader, use_penultimate=use_penultimate)

    if with_nmi:
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(
            T,
            evaluation.cluster_by_kmeans(
                X, nb_classes, backend=backend
            )
        )
    else:
        nmi = 0
    logging.info("NMI: {:.3f}".format(nmi * 100))

    # get predictions by assigning nearest 8 neighbors with euclidian
    Y = evaluation.assign_by_euclidian_at_k(X, T, 8, backend=backend)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return nmi, recall


def eval_model(model, args, dataloader, new_vid=False, with_nmi=True, logging=None):
    """
    eval a trained model loaded from the checkpoint
    new_vid: calculate mean R@k aggregated per class fo VID dataset
    """
    if args['dataset']['selected'] in ['cub', 'cars', 'sop']:
        with torch.no_grad():
            X, T, _ = predict_batchwise(model=model, dataloader=dataloader, use_penultimate=False)

            max_class_size = int(np.max(np.unique(T, return_counts=True)[1]))

            if args['dataset']['selected'] != 'sop':
                # for mAP we need pairwise dist for all samples
                Y = evaluation.assign_by_euclidian_at_k(X, T, len(X), backend=args['backend'])

                assert Y.shape[1] >= max_class_size, 'not enough nns to calculate mARP'

                # There are some samples in eval set have same image but different label/indices
                # i.e. [1668, 2022], [6409, 6442] [6371, 6480]
                # this causing unexpected nns by searching k=8 and k=len(X).
                # To be consistent with the log, we use k=8 to get our R@k

                Y_R = evaluation.assign_by_euclidian_at_k(X, T, 8, backend=args['backend'])
                R_k = []

                for k in [1, 2, 4, 8]:
                    r_at_k = evaluation.calc_recall_at_k(T, Y_R, k)
                    R_k.append(r_at_k)
                    if logging is not None:
                        # TODO apply to other datasets as well
                        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

                mARP = evaluation.mean_avg_R_precision(T, Y, T)

            else:
                # evaluate sop on 1, 10, 100 nns
                Y = evaluation.assign_by_euclidian_at_k(X, T, 100, backend=args['backend'])

                assert Y.shape[1] >= max_class_size, 'not enough nns to calculate mARP'

                R_k = []

                for k in [1, 10, 100]:
                    R_k.append(evaluation.calc_recall_at_k(T, Y, k))

                mARP = evaluation.mean_avg_R_precision(T, Y, T)

            # for a fair nmi we run 10 times clustering with different seed and take the mean as final result
            nb_classes = dataloader.dataset.nb_classes()
            nmi = avg_nmi(args['backend'], nb_classes, X, T) if with_nmi else 0

    elif args['dataset']['selected'] == 'vid':
        with torch.no_grad():
            if len(dataloader) != 1:
                # small, medium and large eval set
                R_k = []
                nmi = []
                mARP = []
                for dl in dataloader:
                    R_k_sub, nmi_sub, mARP_sub = eval_vid(model, backend=args['backend'], dataloader=dl,
                                                          new_vid=new_vid, with_nmi=with_nmi)

                    R_k.append(R_k_sub)
                    nmi.append(nmi_sub)
                    mARP.append(mARP_sub)
            else:
                R_k, nmi, mARP = eval_vid(model, args, dataloader, new_vid)

    elif args['dataset']['selected'] == 'inshop':
        (dl_query, dl_gallery) = dataloader

        nmi, R_k, Y = evaluate_in_shop(model, dl_query, dl_gallery,
                                       use_penultimate=False, backend=args['backend'], final_eval=True)
        T_query = dl_query.dataset.ys
        T_gallery = dl_gallery.dataset.ys

        mARP = evaluation.mean_avg_R_precision(T_query, Y, T_gallery)

    else:
        raise ValueError('Unknown dataset: {}'.format(args['dataset']['selected']))

    return R_k, nmi, mARP


def eval_vid(model, backend, dataloader, new_vid=False, with_nmi=True):
    """
    new_vid: if true - Calculates mean Recall@k aggregated per class (not the same as regular R@k)
    """

    X, T, _ = predict_batchwise(model, dataloader, use_penultimate=False)
    max_class_size = int(np.max(np.unique(T, return_counts=True)[1]))

    # evaluate vid requires 5 nns
    # 120 is the size of the largest class
    Y = evaluation.assign_by_euclidian_at_k(X=X, T=T, k=max_class_size + 1, backend=backend)

    R_k = []

    # the normal evaluation procedure
    for k in [1, 5]:
        R_k.append(evaluation.calc_recall_at_k(T, Y, k))

    if new_vid:
        # in this case the returned R_k has 4 values
        print('!!! Calculate mean R@k aggregated per class')
        for k in [1, 5]:
            R_class = []
            for t in np.unique(T):
                # get recall per class
                R_class.append(np.mean(([1 if t in nns else 0 for nns in Y[T == t][:, :k]])))
            # then average the recall among all class
            R_k.append(np.mean(R_class))

    nb_classes = dataloader.dataset.nb_classes()
    nmi = avg_nmi(backend, nb_classes, X, T) if with_nmi else 0

    mARP = evaluation.mean_avg_R_precision(T, Y, T)

    return R_k, nmi, mARP


def avg_nmi(backend, nb_classes, X, T, runs=10):
    """
    runs: int -- how many runs.

    Compute the nmi multiple times based on different clustering assignments.
    Return the mean and std of these runs.
    """
    Cluster_assignments = []
    for random_seed in range(runs):
        Cluster_assignments.append(faissext.do_clustering(X, num_clusters=nb_classes,
                                                          gpu_ids=None if backend != 'faiss-gpu'
                                                          else torch.cuda.current_device(),
                                                          niter=100, nredo=1, verbose=0,
                                                          seed=random_seed)
                                   )
    nmis = [evaluation.calc_normalized_mutual_information(T, c) for c in Cluster_assignments]

    return np.mean(nmis), np.std(nmis)
