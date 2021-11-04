from __future__ import print_function
from __future__ import division

import collections
import os
import matplotlib
import numpy as np
import logging
import torch
import re
import sys
import time
import json
import random
import shelve
import socket
from tqdm import tqdm

import dataset
import metriclearning.model.inweave
import metriclearning
from metriclearning import faissext
import sklearn.utils
import sklearn.cluster
import math
from collect_env_info import get_pretty_env_info
from eval_model import load_mask_model


os.putenv("OMP_NUM_THREADS", "8")


# __repr__ may contain `\n`, json replaces it by `\\n` + indent
json.dumps_ = lambda **kwargs: json.dumps(
    **kwargs
).replace('\\n', '\n    ')


faiss_memory_holder = None
clustering_random_state = None


def lock_faiss_gpu_memory(args):
    """
    Reserve memory for Faiss if backend is faiss-gpu,

    Usage: wrap make_clustered_dataloaders in
        lock_faiss_gpu_memory() and release_faiss_gpu_memory()
    """
    global faiss_memory_holder

    if args['backend'] == 'faiss-gpu':
        logging.debug('Reserve some memory for FAISS')
        faiss_memory_holder = faissext.reserve_faiss_gpu_memory(gpu_id=0)
    else:
        faiss_memory_holder = None


def release_faiss_gpu_memory():
    global faiss_memory_holder
    if faiss_memory_holder is not None:
        logging.debug('Release memory for FAISS')
        faiss_memory_holder = None


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


class MasksFreezer:
    def __init__(self, opt):
        self.opt = opt
        self.lr = None
        self.is_frozen = False

    def _find_opt_group(self):
        for i, g in enumerate(self.opt.param_groups):
            if g.get('group_name', None) == 'masks':
                return i
        return None

    def freeze(self, epoch=None):
        group_idx = self._find_opt_group()
        if group_idx is None:
            return
        assert self.opt.param_groups[group_idx]['group_name'] == 'masks'
        if self.is_frozen:
            assert self.opt.param_groups[group_idx]['lr'] == 0, 'masks lr={}'.format(self.opt.param_groups[group_idx]['lr'])
            return
        # freeze backbone, just set LR to 0
        if epoch is not None:
            logging.info('Freezing masks at epoch {}.'.format(epoch))
        else:
            logging.info('Freezing masks.')
        self.lr = self.opt.param_groups[group_idx]['lr']
        self.opt.param_groups[group_idx]['lr'] = 0
        self.is_frozen = True

    def unfreeze(self):
        group_idx = self._find_opt_group()
        if group_idx is None:
            return

        if not self.is_frozen:
            return
        logging.info('Unfreezing masks.')
        assert self.opt.param_groups[group_idx]['group_name'] == 'masks'
        if self.lr is None:
            raise ValueError('masks lr must be saved before!')
        self.opt.param_groups[group_idx]['lr'] = self.lr
        self.is_frozen = False


def make_clustered_dataloaders(model, dataloader_init, args,
        reassign = False, I_prev = None, C_prev = None, logging = None,
        e = -1):

    import utils

    def correct_indices(I):
        return torch.sort(torch.LongTensor(I))[1]

    if args['clustering_method']['selected'] == 'kmeans':
        if args['clustering_method']['options']['mode'] == 'adaptive_centroids' and args['nb_clusters'] > 1:
            X, T, I = metriclearning.utils.predict_batchwise(
                model=model,
                dataloader=dataloader_init,
                use_penultimate=args['penultimate_for_clusters'],
                is_dry_run=False
            )

            logging.info('******* CLUSTERING WITH ADAPTIVE CENTROIDS **********')
            if hasattr(dataloader_init, 'kmeans_init'):
                C_new = C_prev[correct_indices(I_prev)]
                logging.info('Adapting centroids with new representation!')
                logging.info(str(dataloader_init.kmeans_init.shape))
                # C_new is Y_1 in notebook
                kmeans_init = dataloader_init.kmeans_init
                X_new = X[correct_indices(I)]
                new_cluster_centers = []
                for c in sorted(list(set(C_new).difference([-1]))):
                    print(c)
                    new_cluster_centers.append(
                        X_new[C_new == c].mean(axis = 0)
                    )

                logging.info('distances between old kmeans init' + \
                        ' and new adpated centroids')
                logging.info(str(
                    metriclearning.similarity.pairwise_distance(
                    torch.cat([
                        torch.FloatTensor(kmeans_init),
                        torch.FloatTensor(new_cluster_centers)
                    ])
                )[:len(kmeans_init), len(kmeans_init):]))

                kmeans_init = np.array(new_cluster_centers)

            else:
                logging.info('FIRST time init with k-means++!')
                kmeans_init = 'k-means++'

            clustering_algorithm = sklearn.cluster.KMeans(
                n_clusters=args['nb_clusters'], init = kmeans_init)
            C = clustering_algorithm.fit(X).labels_
            dataloader_init.kmeans_init = clustering_algorithm.cluster_centers_

        else:

            C, T, I, X = metriclearning.similarity.get_cluster_labels(
                model,
                dataloader_init,
                use_penultimate = args['penultimate_for_clusters'],
                nb_clusters = args['nb_clusters'],
                backend = args['backend'],
                with_X = True,
                ntrials=30,
                random_state=clustering_random_state
            )

    elif args['nb_clusters'] == 1:
        num_items_total = len(dataloader_init.dataset)
        assert len(dataloader_init.dataset.I) == num_items_total
        T = np.array(dataloader_init.dataset.ys, dtype=int)
        I = np.array(dataloader_init.dataset.I, dtype=int)

        I = np.hstack([I for c in range(args['nb_clusters'])])
        T = np.hstack([T for c in range(args['nb_clusters'])])
        C = np.hstack([[c] * num_items_total for c in range(args['nb_clusters'])])
        assert len(I) == len(T) == len(C) == args['nb_clusters'] * num_items_total

    if reassign == True:
        # get correct indices for samples by sorting them and return arg sort
        perm = correct_indices(I)
        I = I[perm]
        T = T[perm]
        C = C[perm]
        if args['hierarchy_method'] in ['top_bot', 'bot_top']:
            X = X[perm]

        # also use the same indices of sorted samples for previous data
        perm = correct_indices(I_prev)
        I_prev = I_prev[perm]
        C_prev = C_prev[perm]
        assert np.array_equal(I, I_prev), (I, I_prev)

        logging.info('Reassigning clusters...')
        logging.info('Calculating NMI for consecutive cluster assignments...')
        logging.info('NMI(prev, cur) = {}'.format(
            metriclearning.evaluation.calc_normalized_mutual_information(
            C[I],
            C_prev[I_prev]
        )))
        if args['reassign_random'] == True:
            # don't use reassignment with cost matrix; i.e. reassign randomly
            logging.info('not reassigning!')
        else:
            # assign s.t. least costs w.r.t. L1 norm
            C, costs = dataset.loader.reassign_clusters(C_prev = C_prev,
                    C_curr = C, I_prev = I_prev, I_curr = I)
            logging.info(f'Costs before reassignment = {costs.diagonal().sum()}')
            logging.info('\n' + str(costs))
            _, costs = dataset.loader.reassign_clusters(C_prev = C_prev,
                    C_curr = C, I_prev = I_prev, I_curr = I)
            # after printing out the costs now, the trace of matrix should
            # have lower numbers than other entries in matrix
            logging.info(f'Costs after reassignment = {costs.diagonal().sum()}')
            logging.info('\n' + str(costs))

    utils.log_clustering_stats(C, T)

    if args['hierarchy_method'] in ['top_bot'] and e > 0:

        if args['hierarchy_method'] == 'top_bot':
            if args['nb_clusters'] < args['nb_clusters_final']:
                args['nb_clusters'] = int(args['nb_clusters'] * 2)
                logging.info('Setting nb_clusters = {}'.format(args['nb_clusters']))
                logging.info('Divide each cluster in 2')
                C_undivided = C

                assert np.array_equal(I, np.arange(len(I))), I
                C = utils.divide_clusters(
                    X = X,
                    C = C,
                    T = T,
                    ntrials=30,
                    gpu_ids = args['cuda_device'] if args['backend'] != 'faiss' else None,
                    random_state=clustering_random_state
                )
                assert np.array_equal(
                    utils.merge_clusters(
                        C,
                        gpu_ids=args['cuda_device'] if args['backend'] != 'faiss' else None,
                    ),
                    C_undivided
                )

                utils.log_clustering_stats(C, T)

        else:
            if args['nb_clusters'] > args['nb_clusters_final']:
                args['nb_clusters'] = int(args['nb_clusters'] / 2)
                logging.info('Setting nb_clusters = {}'.format(args['nb_clusters']))


    #  remove labels s.t. minimum 2 samples per class per cluster
    if args['supervised'] == True:
        for c in range(args['nb_clusters']):
            cnt_removed = 0
            for t in np.unique(T[C == c]):
                if (T[C == c] == t).sum().item() == 1:
                    # assign to cluster -1 if only one sample from class
                    C[(T == t) & (C == c)] = -1
                    cnt_removed += 1
            if cnt_removed:
                logging.debug(f' --- Removed {cnt_removed} images (w/o pos pair)  from cluster {c}')

    dls = dataset.loader.make_trainloaders_from_clusters(
        C = C, I = I, model = model, args = args
    )

    return dls, X, C, T, I


def evaluate(model, dataloaders, logging, layers = ['final', 'penultimate'],
        loader_types = ['eval', 'init'], backend='faiss', args = None):
    model.eval()
    scores = {}
    for ltype in loader_types:
        scores[ltype] = {}
        logging.info("--- Data Loader: {} ---".format(ltype))
        for layer in layers:
            logging.info("-- Layer: {} --".format(layer))
            if args is not None and args['dataset']['selected'] == 'inshop':
                logging.info("Using dataset `InShop`")
                dl_query = dataset.loader.make_loader(args, model,
                    'eval', inshop_type = 'query')
                dl_gallery = dataset.loader.make_loader(args, model,
                    'eval', inshop_type = 'gallery')

                scores[ltype][layer] = metriclearning.utils.evaluate_in_shop(
                    model,
                    dl_query = dl_query,
                    dl_gallery = dl_gallery,
                    use_penultimate = True if layer == 'penultimate' else False,
                    backend = backend)

            elif args is not None and args['dataset']['selected'] == 'market':
                logging.info("Using dataset `Market1501`")
                # we could use the param 'inshop_type' to do similar work for market for now
                dl_query = dataset.loader.make_loader(args, model,
                                                      'eval', inshop_type='query')
                dl_gallery = dataset.loader.make_loader(args, model,
                                                        'eval', inshop_type='gallery')

                scores[ltype][layer] = metriclearning.utils.evaluate_market(
                    model,
                    dl_query=dl_query,
                    dl_gallery=dl_gallery)
            else:
                scores[ltype][layer] = metriclearning.utils.evaluate(
                    model,
                    dataloaders[ltype],
                    use_penultimate=True if layer == 'penultimate' else False,
                    backend=backend
                )
    return scores



def train_batch(model, criterion, opt, args, batch, dset, first_run=True):
    X = batch[0].cuda(non_blocking=True)
    T = batch[1].cuda(non_blocking=True) # class labels
    I = batch[2] # image ids

    opt.zero_grad()

    # if force full embedding, call forward pass with dset_id=None
    # to ignore masking
    if args['force_full_embedding']:
        #normalize the whole feature layer output
        M = model(X, dset_id=None)
    else:
        #normalize according to the clusters
        M = model(X, dset_id=dset.id)

    M = torch.nn.functional.normalize(M, p=2, dim=1)

    l_emb = criterion(M, T)
    loss = l_emb

    mask = model.masks[dset.id]
    mask_norm = mask.norm(1)

    def calc_orthogonality(vectors):
      from torch.nn.functional import relu
      sim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
      m = torch.zeros(len(vectors), len(vectors))
      s = 0
      for i in range(len(vectors)):
        for j in range(len(vectors)):
          if i != j:
              sim_ij = sim(relu(vectors[i]), (relu(vectors[j])))
              # make correlation be 0, remove negative one as well
              m[i][j] = sim_ij.abs()
              s +=  sim_ij
      if len(vectors) > 1:
        s /= (len(vectors) * (len(vectors) - 1))
      return m, s

    m, loss_mask = calc_orthogonality(model.masks)

    loss_mask = loss_mask * args['masking_lambda']

    # if using full embedding, ignore masking
    if args['force_full_embedding']:
        if model.masks[0].requires_grad:
            for p in model.masks:
                logging.info('Stopping to optimize masks.')
                p.requires_grad = False
            assert len(model.opt.param_groups) in [3, 4]
            assert model.opt.param_groups[-1]['group_name'] == 'masks'
            del model.opt.param_groups[-1]
    else:
        loss = loss + loss_mask

    if args['is_debug'] and first_run:
        logging.info(
            '{}(loss emb) + {}(loss mask) = {}'.format(
            l_emb.item(),
            loss_mask,
            loss.item()
            )
        )
        logging.info('l1 norm of mask {} w/o rescaling: {}'.format(dset.id, mask_norm))
        logging.info('cosine similarity:')
        logging.info(m.detach().cpu().numpy())

    loss.backward()
    opt.step()

    if isinstance(loss_mask, torch.Tensor):
        loss_mask = loss_mask.item()
    return l_emb.item(), loss_mask


def get_criterion(args):
    name = args['criterion']['selected']
    loss_class = metriclearning.loss.__dict__[name]
    dataset_name = args['dataset']['selected']
    num_classes = len(args['dataset']['types'][dataset_name]['classes']['train'])
    logging.debug('Create {} loss. Num classes={}'.format(name, num_classes))
    # use the same margin loss for every cluster
    criterion = \
            loss_class(nb_classes=num_classes,
                       sampler_args=args['criterion']['sampler'],
                       **args['criterion']['types'][name]).cuda()
    return criterion


def get_optimizer(args, model, criterion):

    class OptimizerGroup(object):
        """
        Group several optimizers in one object
        """
        def __init__(self, optimizers):
            self.optimizers = optimizers
            # optimizer which will use scheduler
            self.optimizer_for_scheduler = optimizers[0]

        def zero_grad(self):
            for opt in self.optimizers:
                opt.zero_grad()

        def step(self):
            for opt in self.optimizers:
                opt.step()

    extra_opt_params = []
    if args['criterion']['selected'] == 'MarginLoss' \
       and args['criterion']['types']['MarginLoss']['lr_beta'] > 0:
        # we assume we have the same loss isntance for all clusters
        assert not isinstance(criterion, collections.Iterable)
        extra_opt_params = [{
                             'group_name': 'loss_params',
                             'params': criterion.parameters(),
                             'lr': args['criterion']['types']['MarginLoss']['lr_beta'],
                             'weight_decay': 0.0
                            }]
    elif args['criterion']['selected'] == 'NPairsLoss' \
       and args['criterion']['types']['NPairsLoss']['lr_alpha'] > 0:
        extra_opt_params = [{
                             'group_name': 'loss_params',
                             'params': criterion.parameters(),
                             'lr': args['criterion']['types']['NPairsLoss']['lr_alpha'],
                             'weight_decay': 0.0
                            }]

    opt = getattr(torch.optim, args['opt']['selected'])(
        [
            # DON'T CHANGE POSITION, because used for setting LR,
            # when freezing
            {
                'params': model.parameters_dict['backbone'],
                **args['opt']['features_w']
            },
            {
                'params': model.parameters_dict['embedding'],
                **args['opt']['embedding_w']
            }
        ] + \
        extra_opt_params + \
        [
            {
                'group_name': 'masks',
                'params': model.masks,
                **args['opt']['mask']
            }
        ],
        **args['opt']['base']
    )

    optimizers = [opt]

    # make sure that model is on first place in optimizers[0]
    assert len(optimizers[0].param_groups[0]['params']) > 2

    # Currently LR scheduler is completely disabled
    assert args['opt']['features_w']['lr'] == args['opt']['embedding_w']['lr']

    return OptimizerGroup(optimizers)


def read_num_epoch_trained(db_path):
    """
    Read information about the the number of epochs trained from db
    """
    try:
        f = shelve.open(db_path, flag='r')
    except Exception as e:
        logging.debug(e)
        raise IOError('Db file {} not found!'.format(db_path))
    if 'metrics' in f:
        try:
            m = f['metrics']
            max_epoch = np.max(list(m.keys()))
        except EOFError:
            max_epoch = -1
    else:
        max_epoch = -1
    f.close()
    return max_epoch + 1


def read_ckpt_info_from_df(db_path):
    """
    Read information about the best checkpoint from the db file
    """
    try:
        f = shelve.open(db_path, flag='r')
    except Exception as e:
        logging.debug(e)
        raise IOError('Db file {} not found!'.format(db_path))
    m = f['metrics']
    epochs = np.arange(0, np.max(list(m.keys())))
    best_epoch_idx = np.argmax([m[e]['score']['eval']['final'][1][0] for e in epochs])
    best_epoch = epochs[best_epoch_idx]
    best_recall = m[best_epoch]['score']['eval']['final'][1][0]
    f.close()
    return best_epoch, best_recall


def wandb_log_metrics(metrics, e):

    metric_names = [
        *['R@{}'.format(i) for i in [1,]],
    ]
    metric_values = [*metrics[e]['score']['eval']['final'][1],
                    ]
    # w&b doesn't allow negative step number, use 0 again in that case
    step = e if e >= 0 else 0
    wnb.log({k: v for k, v in zip(metric_names, metric_values)}, step=step)


def log_extra_info_after_epoch(args, criterion):
    if args['criterion']['selected'] == 'MarginLoss' \
       and args['criterion']['types']['MarginLoss']['lr_beta'] > 0:
        beta = criterion.beta.detach().cpu().numpy()
        k = min(len(beta) // 2, 10)
        logging.info(' - Margin loss beta: [{} ... {}]'.format(
                ' '.join(str(round(i.item(), 3)) for i in beta[:k]),
                ' '.join(str(round(i.item(), 3)) for i in beta[-k:])
            )
        )


def start(args, metrics):

    """
    Import `plt` after setting `matplotlib` backend to `agg`, because `tkinter`
    missing. If `agg` set, when this module is imported, then plots can not
    be displayed in jupyter notebook, because backend can be set only once.
    """
    db_path = os.path.join(args['log']['path'], args['log']['name'])
    if os.path.exists(db_path + '.dat'):
        print(f'Found db file: {db_path}.dat')
        num_epochs_trained = read_num_epoch_trained(db_path)
        if num_epochs_trained > args['nb_epochs'] - 10:
            print(f'The model was already trained for '
                         f'{num_epochs_trained}/{args["nb_epochs"]}\n'
                         f'Aborting.')
            return
        elif num_epochs_trained > 0:
            print(f'The model was already trained only for '
                         f'{num_epochs_trained}/{args["nb_epochs"]}\n'
                         f'Retrain from scratch.')

    import matplotlib.pyplot as plt

    # create logging directory
    os.makedirs(args['log']['path'], exist_ok = True)

    # warn if log file exists already and wait for user interaction
    import warnings
    _fpath = os.path.join(args['log']['path'], args['log']['name'])
    if os.path.exists(_fpath):
        warnings.warn('Log file exists already: {}'.format(_fpath))
        print('Appending underscore to log file and database')
        args['log']['name'] += '_'

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(
                "{0}/{1}.log".format(args['log']['path'], args['log']['name'])
            ),
            logging.StreamHandler()
        ]
    )
    env_cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    logging.info('--\nThe script was run with the following command:\n' + \
                 '==================================================\n' + \
                 (
                     f'CUDA_VISIBLE_DEVICES={env_cuda_visible_devices} '
                     if env_cuda_visible_devices else ''
                 ) + \
                 'python ' + ' '.join(sys.argv) + '\n' +
                 '==================================================\n')
    logging.info(f'Hostname {socket.gethostname()}')
    logging.info('\n' + get_pretty_env_info())

    # print summary of args
    logging.info(
        json.dumps_(obj = args, indent=4, cls = JSONEncoder, sort_keys = True)
    )

    torch.cuda.set_device(args['cuda_device'])

    if not os.path.isdir(args['log']['path']):
        os.mkdir(args['log']['path'])

    seed = args['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # set random seed for all gpus
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global clustering_random_state
    if args['clustering_random_state'] is not None:
        clustering_random_state = np.random.RandomState(args['clustering_random_state'])

    # print out GPU info, since different GPU architectures may act differently during training
    logging.info('Current GPU information: {}'.format(torch.cuda.get_device_properties(torch.cuda.current_device())))

    lock_faiss_gpu_memory(args)

    model = metriclearning.model.make(args).cuda()
    #wnb.watch(model)



    if args['checkpoint'] is None:
        ckpt_paths = [os.path.join(args['log']['path'], args['log']['name'] + \
                                  '-before-finetune.pt'),
                      os.path.join(args['log']['path'], args['log']['name'] + \
                                  '.pt')]
        for p in ckpt_paths:
            if os.path.exists(p):
                print('### Not loading the checkpoint. Retrain again')
                break
                args['checkpoint'] = p

    if args['checkpoint'] is not None:
        logging.info('Loading checkpoint from {}'.format(args['checkpoint']))
        if not os.path.exists(args['checkpoint']):
            logging.error('Checkpoint {} not found!'.format(args['checkpoint']))
            raise IOError(args['checkpoint'])

        db_path = os.path.splitext(args['checkpoint'])[0]
        if '-full-emb-' in db_path:
            db_path = db_path.rsplit('before-finetune', 1)[0]
            db_path = db_path.rsplit('-full-emb-', 1)[0]
            db_path = db_path.rsplit('_ep', 1)[0]
        else:
            db_path = db_path.rsplit('-before-finetune', 1)[0]
            db_path = db_path.rsplit('_ep', 1)[0]

        best_epoch, best_recall = read_ckpt_info_from_df(db_path)
        m = re.search(r'ep(\d+)\.pt', args['checkpoint'])
        if m is not None:
            start_epoch = int(m.groups()[0]) + 1
        else:
            start_epoch = best_epoch + 1

        # TODO: alpha of the Npair loss are not stored,
        #  checkpoint of a model trained with Npair-loss won't give you exact same result
        args, model = load_mask_model(args, args['checkpoint'])

        logging.info('Loaded model at epoch {}; best_epoch: {}, R@1={}'.format(start_epoch - 1, best_epoch, best_recall))
    else:

        start_epoch = 0
        best_epoch = -1
        best_recall = 0

    dataloaders = {}
    for dl_type in ['init', 'eval']:
        if args['dataset']['selected'] in ['inshop', 'market']:
            # query and gallery initialized in `make_clustered_dataloaders`
            if dl_type == 'init':
                dataloaders[dl_type] = dataset.loader.make_loader(args, model,
                    dl_type, inshop_type = 'train')
        else:
            dataloaders[dl_type] = dataset.loader.make_loader(args, model,
                dl_type)

    criterion = get_criterion(args)
    opt = get_optimizer(args, model, criterion)
    model.opt = opt.optimizers[0]

    if args['hierarchy_method'] in ['top_bot', 'bot_top']:
        if args['hierarchy_method'] == 'top_bot':
            #NOTE: args['nb_clusters_final'] is number of cluster to be reached at the end
            #      args['nb_clusters'] becomes the current clusters num
            args['nb_clusters_final'] = args['nb_clusters']
            args['nb_clusters'] = 1
            if start_epoch > 0:
                assert args['checkpoint'] is not None
                if start_epoch // args['recluster']['mod_epoch'] > math.log(args['nb_clusters_final'], 2) * args['recluster']['mod_epoch']:
                    args['nb_clusters'] = args['nb_clusters_final']
                else:
                    args['nb_clusters'] = int(2**(start_epoch // args['recluster']['mod_epoch']))
            logging.info('Start from {} clusters'.format(args['nb_clusters']))

        elif args['hierarchy_method'] == 'bot_top':
            args['nb_clusters_final'] = 1
            args['nb_clusters'] = args['nb_clusters']
        logging.info('From {} to {} clusters.'.format(
            args['nb_clusters'], args['nb_clusters_final'])
        )
    elif args['hierarchy_method'] == 'none':
        pass
    else:
        logging.error('--- hierarchy method not known, may be typo ---')
        raise SystemExit

    # we need faiss to evaluate
    release_faiss_gpu_memory()
    logging.info("Evaluating initial model...")
    metrics[-1] = {
        'score': evaluate(model, dataloaders, logging,
                        ['final'] + \
                        (['penultimate'] if args['penultimate_at_eval'] else []),
                        ['eval'],
                        backend=args['backend'],
                        args = args)}
    wandb_log_metrics(metrics, -1)

    dataloaders['train'], X, C, T, I = make_clustered_dataloaders(model,
            dataloaders['init'], args, reassign = False, logging = logging)
    lock_faiss_gpu_memory(args)

    if args['save_masks']:
        metrics[-1].update({'C': C, 'T': T, 'I': I, 'masks': [model_mask.detach().cpu().numpy().astype(np.half) for model_mask in model.masks]})
    else:
        metrics[-1].update({'C': C, 'T': T, 'I': I})

    logging.debug('Printing only first 200 classes (because of SOProducts)')
    for c in range(args['nb_clusters']):
        if len(dataloaders['train'][c].dataset.ys):
            logging.debug(str(np.bincount(dataloaders['train'][c].dataset.ys)[:200]))
            plt.hist(dataloaders['train'][c].dataset.ys, bins = 100)
            plt.show()
        else:
            logging.debug('Empty cluster {}'.format(c))

    logging.info("Training for {} epochs.".format(args['nb_epochs']))
    t1 = time.time()


    def freeze():
        nonlocal opt
        # freeze backbone, just set LR to 0
        logging.info('Freezing backbone at epoch {}.'.format(e))
        opt.optimizers[0].param_groups[0]['lr'] = 0

    def unfreeze():
        nonlocal opt
        logging.info('Unfreezing backbone at epoch {}.'.format(e))
        opt.optimizers[0].param_groups[0]['lr'] = opt.optimizers[0]\
                .param_groups[1]['lr']

    masks_freezer = MasksFreezer(model.opt)

    for e in range(start_epoch, args['nb_epochs']):
        is_best = False
        model.eval()

        metrics[e] = {}
        time_per_epoch_1 = time.time()
        losses_per_epoch = collections.defaultdict(list)

        # initially set mod epoch freeze and unfreeze
        assert args['mod_epoch_freeze'] < args['recluster']['mod_epoch']

        if args['stop_recluster']:
            if (args['hierarchy_method'] == 'top_bot' and \
                    args['nb_clusters'] == args['nb_clusters_final']) or \
                    (args['hierarchy_method'] == 'none' and e >= args['mod_epoch_freeze']):
                logging.info('Stopping to recluster.')
                logging.info(
                    'Setting mod epoch to 1000, mod epoch freeze to 0.')
                args['recluster']['mod_epoch'] = 1000
                args['mod_epoch_freeze'] = 0
                logging.info(
                    'Setting LR of features to LR of embedding.')
                opt.optimizers[0].param_groups[0]['lr'] = opt.optimizers[
                    0
                ].param_groups[1]['lr']
                args['stop_recluster'] = False

        if e >= args['force_full_embedding_epoch']:
            args['force_full_embedding'] = True
            if e == args['force_full_embedding_epoch']:
                #unfreeze()
                logging.info(
                    'Starting to use the entire embedding every iter...')


        if args['recluster']['enabled'] and args['nb_clusters'] > 0:

            if e % args['recluster']['mod_epoch'] == 0 or \
                    e % args['recluster']['mod_epoch'] < \
                    args['mod_epoch_freeze']:
                if args['mod_epoch_freeze'] > 0:
                    freeze()
            elif e % args['recluster']['mod_epoch'] == \
                    args['mod_epoch_freeze'] and \
                    args['mod_epoch_freeze'] > 0:
                unfreeze()

            if e % args['recluster']['mod_epoch'] == 0:
                logging.info("Reclustering dataloaders...")
                if args['recluster']['method']['selected'] == 'reassign':
                    release_faiss_gpu_memory()
                    dataloaders['train'], X, C, T, I =  \
                            make_clustered_dataloaders(
                        model, dataloaders['init'], args, reassign = True,
                        C_prev = C, I_prev = I, logging = logging, e = e)
                    lock_faiss_gpu_memory(args)
                    for c in range(args['nb_clusters']):
                        ys = dataloaders['train'][c].dataset.ys
                        if len(ys):
                            logging.debug('Cluster {}: num GT classes = {} ({} samples)'\
                                  .format(c, len(np.unique(ys)), len(ys)))
                        else:
                            logging.debug('Cluster {}: Empty!'.format(c))
                else:
                    release_faiss_gpu_memory()
                    dataloaders['train'], X, C, T, I = \
                            make_clustered_dataloaders(
                        model, dataloaders['init'], args, logging = logging)
                    lock_faiss_gpu_memory(args)
                if args['save_masks']:
                    metrics[e].update({'C': C, 'T': T, 'I': I, 'masks': [model_mask.detach().cpu().numpy().astype(np.half) for model_mask in model.masks]})
                else:
                    metrics[e].update({'C': C, 'T': T, 'I': I})

        if args['masks_freeze_one_cluster']:
            if args['nb_clusters'] == 1:
                masks_freezer.freeze(epoch=e)
            else:
                masks_freezer.unfreeze()

        mdl = dataset.loader.merge_dataloaders(
            dataloaders['train'], **args['dataloader']['merged']
        )

        logging.info(f'Optimizer: {opt.optimizers[0]}')
        logging.info('LR of backbone: {}'.format(
            opt.optimizers[0].param_groups[0]['lr']
        ))

        model.train()

        num_batches_approx = max(
            [len(dl) for dl in dataloaders['train']]
        ) * len(dataloaders['train'])

        first_batch_run = True
        for batch, dset in tqdm(mdl,
                                total=num_batches_approx,
                                disable=num_batches_approx < 100,
                                desc='Train epoch {}'.format(e)):
            loss, loss_masks = train_batch(model, criterion, opt, args, batch, dset, first_run=first_batch_run)
            # now disable first_batch_run, to disable printing of log
            first_batch_run = False

            losses_per_epoch['loss_metric'].append(loss)
            losses_per_epoch['loss_masks'].append(loss_masks)

        logging.info(model.summarize_masks())

        time_per_epoch_2 = time.time()
        mean_loss_metric = np.mean(losses_per_epoch['loss_metric'])
        mean_loss_masks = np.mean(losses_per_epoch['loss_masks'])
        logging.info(
            "Epoch: {}, loss: {} + {}, time (seconds): {:.2f}.".format(
                e,
                mean_loss_metric,
                mean_loss_masks,
                time_per_epoch_2 - time_per_epoch_1
            )
        )
        log_extra_info_after_epoch(args, criterion)

        release_faiss_gpu_memory()
        tic = time.time()
        metrics[e].update({
            'score': evaluate(model, dataloaders, logging,
                        ['final'] + \
                        (['penultimate'] if args['penultimate_at_eval'] else []),
                        ['eval'],
                        backend=args['backend'],
                        args = args),
            'loss': {
                'train': mean_loss_metric,
                'loss_masks': mean_loss_masks,
            },

        })
        logging.debug('Evaluation total elapsed time: {:.2f} s'.format(time.time() - tic))
        wandb_log_metrics(metrics, e)
        wnb.log({
            'loss_metric': mean_loss_metric,
            'loss_masks': mean_loss_masks
        }, step=e)

        lock_faiss_gpu_memory(args)

        recall_curr = metrics[e]['score']['eval']['final'][1][0]
        if recall_curr > best_recall:
            best_recall = recall_curr
            best_epoch = e
            is_best = True
            logging.info('Best epoch!')
            wnb.log({'R@1_best': best_recall}, step=best_epoch)

        model.current_epoch = e
        with shelve.open(
            os.path.join(
                args['log']['path'], args['log']['name']),
            writeback = True
            ) as _f:
            if 'args' not in _f:
                _f['args'] = args
            if 'metrics' not in _f:
                _f['metrics'] = {}
                # if initial model evaluated, append metrics
                if -1 in metrics:
                    _f['metrics'][-1] = metrics[-1]
            _f['metrics'][e] = metrics[e]
            if args['save_model'] and is_best:
                if e < args['finetune_epoch']:
                    if not args['force_full_embedding']:
                        save_suff = '-before-finetune.pt'
                    else:
                        save_suff = '-full-emb-before-finetune.pt'
                else:
                    save_suff = '.pt'
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args['log']['path'], args['log']['name'] + save_suff
                    )
                )
            if args['save_model'] and not isinstance(args['save_model'], bool) and e % args['save_model'] == 1:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args['log']['path'], args['log']['name'] + '_ep{}.pt'.format(e)
                    )
                )


    t2 = time.time()
    logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
    logging.info("Best R@1 = {} at epoch {}.".format(best_recall, best_epoch))


def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required = True, type = str)
    parser_args = parser.parse_args(sys.argv[1:3])

    from importlib import import_module
    args = import_module('experiment.' + parser_args.experiment).make_args()

    matplotlib.use('agg')

    global wnb
    if not args['wandb_enabled']:

        class Blank:
            def log(*args, **kwargs):
                pass
        wnb = Blank()
    else:
        import wandb
        wnb = wandb

    metrics = {}
    start(args, metrics)


if __name__ == '__main__':
    wnb = None

    main()
