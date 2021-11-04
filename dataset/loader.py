from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import logging
import random
from .bytebox.cub import CUBirds
from .bytebox.cars import Cars
from .bytebox.sop import SOProducts
from .bytebox.inshop import InShop
from .bytebox.vid import VehicleID
from .bytebox.utils import make_transform
from .bytebox.utils import make_albumentation



datasets = {
    'cub': CUBirds,
    'cars': Cars,
    'sop': SOProducts,
    'inshop': InShop,
    'vid': VehicleID,
}


def make_loader(args, model, type, I = None, inshop_type = None):
    """
    I :     indices for selecting subset of dataset
    type:   'init', 'eval' or 'train'.
    """
    selected_dataset = args['dataset']['selected']
    # inshop_types: train, query, gallery; basically instead of labels/classes
    if selected_dataset == 'inshop':
        if args['album'] and args['album'] != 'ColorHeavy':
            ds = datasets[selected_dataset](
                root=args['dataset']['types'][selected_dataset]['root'],
                dset_type=inshop_type,
                transform=None,
                albumentation=make_albumentation(mode=args['album'],
                                                 **args['img_transform_parameters'],
                                                 is_train=args['dataset']['augmentation'][type]
                                                 )
            )
        else:
            if args['album'] == 'ColorHeavy':
                hinton = True
                logging.debug('Using ColorHeavy augmentations')
            else:
                logging.debug('Using standard augmentation')
                hinton = False

            ds = datasets[selected_dataset](
                root=args['dataset']['types'][selected_dataset]['root'],
                dset_type=inshop_type,
                transform=make_transform(
                    **model.img_normalization_parameters,
                    **args['img_transform_parameters'],
                    is_train = args['dataset']['augmentation'][type],
                    using_hinton=hinton
                )
            )

    else:
        # for the dataset other than In-Shop
        if args['album'] and args['album'] != 'ColorHeavy':
            ds = datasets[selected_dataset](
                root=args['dataset']['types'][selected_dataset]['root'],
                classes=args['dataset']['types'][selected_dataset]['classes'][type],
                transform=None,
                albumentation=make_albumentation(mode=args['album'],
                                                   **args['img_transform_parameters'],
                                                   is_train = args['dataset']['augmentation'][type]
                )
            )
        else:
            if args['album'] == 'ColorHeavy':
                hinton = True
                logging.debug('Using ColorHeavy augmentations')
            else:
                logging.debug('Using torchvision augmentation')
                hinton = False

            ds = datasets[selected_dataset](
                root = args['dataset']['types'][selected_dataset]['root'],
                classes = args['dataset']['types'][selected_dataset]['classes'][type],
                transform = make_transform(
                    **model.img_normalization_parameters,
                    **args['img_transform_parameters'],
                    is_train = args['dataset']['augmentation'][type],
                    using_hinton=hinton
                )
            )
    if type == 'train':
        ds.set_subset(I)
        if args['dataloader'][type]['_batch_sampler'] is None \
                and args['dataloader'][type]['_sampler'] is None:
            dl_options = args['dataloader'][type].copy()
            dl_options.pop('_batch_sampler')
            dl_options.pop('_sampler')
            dl = torch.utils.data.DataLoader(ds, **dl_options)
        elif args['dataloader'][type]['_sampler'] is not None:
            dl_options = args['dataloader'][type].copy()
            sampler = dl_options['_sampler']['class']\
                    (ds, **dl_options['_sampler']['options'])
            del dl_options['_sampler']
            del dl_options['_batch_sampler']
            dl = torch.utils.data.DataLoader(ds,
                                         sampler=sampler, **dl_options)
        else:
            dl_options = args['dataloader'][type].copy()
            bsampler = dl_options['_batch_sampler']['class']\
                    (ds, **dl_options['_batch_sampler']['options'])
            del dl_options['_sampler']
            del dl_options['_batch_sampler']
            dl = torch.utils.data.DataLoader(ds,
                                         batch_sampler=bsampler, **dl_options)

    else:
        dl = torch.utils.data.DataLoader(ds, **args['dataloader'][type])
    return dl


def make_trainloaders_from_clusters(C, I, model, args):
    dataloaders = [[None] for c in range(args['nb_clusters'])]
    for c in range(args['nb_clusters']):
        dataloaders[c] = make_loader(
            args = args, model = model, type = 'train', I = I[C == c],
            inshop_type = 'train')
        dataloaders[c].dataset.id = c
    return dataloaders


# for plotting assigned clusters compared with clusters from previous epoch(s)
def plot_histograms(C_prev, C_curr, T_prev, T_curr):
    import matplotlib.pyplot as plt
    nb_clusters = max(C_curr).item()
    classes_range_lo = min(T_curr).item()
    classes_range_hi = max(T_curr).item()

    def plot_histogram(c, C, T):
        plt.hist(
            T[C == int(c)].numpy(),
            alpha = 0.5,
            range = (classes_range_lo, classes_range_hi + 1)
        )

    for c in range(nb_clusters + 1):
        plot_histogram(c, C_prev, T_prev)
        plot_histogram(c, C_curr, T_curr)
        plt.show()

def reassign_clusters(C_prev, C_curr, I_prev, I_curr):
    from scipy.optimize import linear_sum_assignment
    nb_clusters = max(C_prev).item() + 1 # cluster ids start from 0
    assert set(
        i.item() for i in np.unique(I_prev)
    ) == set(i.item() for i in np.unique(I_curr))
    I_max = max(I_curr).item() + 1
    I_all = {
        'prev': torch.zeros(nb_clusters, I_max),
        'curr': torch.zeros(nb_clusters, I_max)
    }
    I = {'prev': I_prev, 'curr': I_curr}
    C = {'prev': C_prev, 'curr': C_curr}

    for e in ['prev', 'curr']:
        for c in range(nb_clusters):
            _C = C[e]
            _I = I[e]
            I_all[e][c, _I[_C == c]] = 1

    costs = torch.zeros(nb_clusters, nb_clusters)
    for i in range(nb_clusters):
        for j in range(nb_clusters):
            costs[i, j] = torch.norm(
                I_all['curr'][i] - I_all['prev'][j],
                p = 1
            )

    reassign_prev, reassign_curr = linear_sum_assignment(costs)

    C_reassigned = C['curr'].copy()

    for a_prev, a_curr in zip(reassign_prev, reassign_curr):
        C_reassigned[C['curr'] == int(a_curr)] = int(a_prev)

    return C_reassigned, costs


def merge_dataloaders(dls_non_iter, mode, sampling_mode):
    """
    mode 1: dl1: b1, b2,... dM, dl2: b1, b2, ... bM, ...
    mode 2: dl1: b1, dl2: b1, dlN: b1, dl1: b2, dl2: b2, ... dlN: b2
    mode 3: like mode 2, but permute order of data loaders (N, 2, 5, ...)
    """
    nb_batches_per_dl = [len(dl) for dl in dls_non_iter]
    if sampling_mode == 'under':
        nb_batches = min(nb_batches_per_dl)
    elif sampling_mode == 'over':
        nb_batches = max(nb_batches_per_dl)
    I = range(len(dls_non_iter))

    length = len(dls_non_iter)

    dls = [iter(dl) for dl in dls_non_iter]

    if mode == 1:
        k = 0
        for i in I:
            for j in range(nb_batches):
                k += 1
                b = next(dls[i], None)
                if b == None:
                    # initialize new dataloader in case no batches left
                    dls[i] = iter(dls_non_iter[i])
                    b = next(dls[i])
                yield b, dls_non_iter[i].dataset
    elif mode == 2 or mode == 3:
        for j in range(nb_batches):
            if mode == 3:
                # shuffle dataloaders
                I = random.sample(I, len(I))
            for i in I:
                b = next(dls[i], None)
                if b == None:
                    # initialize new dataloader in case no batches left
                    dls[i] = iter(dls_non_iter[i])
                    b = next(dls[i])
                yield b, dls_non_iter[i].dataset
