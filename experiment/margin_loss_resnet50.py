from __future__ import print_function

import math
import torch

from . import base
import copy
from dataset.sampler import ClassBalancedSampler
from dataset.sampler import AdaptiveClassBalancedSampler
from dataset.sampler import RandomBatchSampler
from dataset.npairs import NPairs
from metriclearning.sampler import *
import metriclearning.similarity
import os

batch_samplers = None

DEFAULT_BASE_LR = 1e-5
optimizers = {
        'newlr2': {
            'selected': 'Adam',
            'base': {
                'lr': DEFAULT_BASE_LR,
                'weight_decay': 1e-3,
            },
            'features_w': {
                'lr': DEFAULT_BASE_LR,
                'weight_decay': 1e-3,
            },
            'features_b': {
                'lr': DEFAULT_BASE_LR,
                'weight_decay': 1e-3,
            },
            'embedding_w': {
                'lr': DEFAULT_BASE_LR,
                'weight_decay': 1e-3,
            },
            'embedding_b': {
                'lr': DEFAULT_BASE_LR,
                'weight_decay': 1e-3,
            }
        },
}

img_transform_parameters = {
    'sz_resize': 256,
    'sz_crop': 224,
}


def empty_if_default(value, default_value):
    if value != default_value:
        return '-' + str(value)
    else:
        return ''


def _args(batch_size, sz_resize, sz_crop, nb_clusters, num_workers=None,
          base_lr=1e-5, emb_lr_mult=1.0, weight_decay=1e-3,
          margin=0.2, soft_loss=False,
          mask_lr_mult=1.0, mask_wd_mult=0.0,
          features_pooling='avg',
          fixed_mask=False):

    global batch_samplers
    batch_samplers = {
                        'balanced_old': {
                            'class': ClassBalancedSampler,
                            'options': {
                                'num_samples_per_class': 4,
                                'batch_size': batch_size,
                            },
                        },
                        'adaptbalanced': {
                            'class': AdaptiveClassBalancedSampler,
                            'options': {
                                'num_samples_per_class': 4,
                                'num_replicas': 1,
                                'batch_size': batch_size,
                                'small_class_action': 'sample_other'
                            },
                        },
                        'npairs': {
                            'class': NPairs,
                            'options': {
                                'num_samples_per_class': 4,
                                'batch_size': batch_size,
                            }
                        },
                        'randombatch': {
                            'class': RandomBatchSampler,
                            'options': {
                                'min_num_classes_per_batch': 2,
                                'batch_size': batch_size,
                                'num_replicas': 1,
                            }
                        }
                    }

    if num_workers is None:
        if nb_clusters > 1:
            num_workers = 2
        else:
            num_workers = 4
    d = {
        'wandb_enabled': False,
        'reassign_random': False,
        'random_seed': 1,
        'save_model': True,
        'resnet': {
            'bn_learnable': False
        },
        'supervised': True,
        'dataset': {
            'selected': 'cub',
            'types': {
                'cub': {
                    'classes': {
                        'train': range(0, 100),
                        'init': range(0, 100),
                        'eval': range(100, 200)
                    }
                }
            }
        },
        'criterion': {
            'selected': 'MarginLoss',
            'sampler': {
                'class': EasyPeasyPairsSampler,
                'options': {
                    'cutoff': 0.5,
                    # distance weighted sampling options
                    # Fix to something higher if uoy use learnable beta or triplet loss
                    'nonzero_loss_cutoff_dist': 1.2 + margin
                }
            },
            'types': {
                'MarginLoss': {
                    'margin': margin,
                    'beta': 1.2,
                    'class_specific_beta': False,
                    'lr_beta': 0.0,
                    'nu': 0.0,
                },
                'TripletAllLoss': {
                    'margin': margin,
                },
                'TripletSemihardLoss': {
                    'margin': margin,
                    'soft': soft_loss
                },
                'TripletLoss': {
                    'margin': margin,
                    'soft': soft_loss
                },
                'NPairsLoss': {
                    'alpha': 1.0,
                    'lr_alpha': 1.0
                },
            }
        },
        'img_transform_parameters': {
            'sz_resize': sz_resize,
            'sz_crop': sz_crop,
         },
        'checkpoint': None,
        'penultimate_at_eval': False,
        'features_dropout_prob': 0.01,
        'nb_epochs': 100,
        'force_full_embedding_epoch': float('inf'),
        'force_full_embedding': False,
        'opt': {
            'selected': 'Adam',
            'base': {
                    'lr': base_lr,
                    'weight_decay': weight_decay,
            },
            'features_w': {
                'lr': base_lr,
                'weight_decay': weight_decay,
            },
            'features_b': {
                'lr': base_lr,
                'weight_decay': weight_decay,
            },
            'embedding_w': {
                'lr': base_lr * emb_lr_mult,
                'weight_decay': weight_decay,
            },
            'embedding_b': {
                'lr': base_lr * emb_lr_mult,
                'weight_decay': weight_decay,
            },
            'mask': {
                # in case of fixed mask, masks will no longer be learnable (lr=0)
                'lr': base_lr * mask_lr_mult if not fixed_mask else 0,
                'weight_decay': mask_wd_mult * weight_decay if not fixed_mask else 0
            },
        },
        'lr_scheduler': {
            'class': torch.optim.lr_scheduler.StepLR,
            'params': {
                'step_size': float('inf'),
                'gamma': 0.5
            }
        },
        'dataloader': {
            'train': {
                'num_workers': num_workers,
                'drop_last': True,
                'shuffle': None,
                'batch_size': batch_size,
                '_batch_sampler': batch_samplers['balanced_old'],
                '_sampler': batch_samplers['npairs'],
            },
            'init': {
                'drop_last': False,
                'shuffle': False,
                'batch_size': batch_size,
                'num_workers': num_workers
            },
            'eval': {
                'drop_last': False,
                'shuffle': False,
                'batch_size': batch_size,
                'num_workers': 4
            },
            'merged': {
                'mode': 2,
                'sampling_mode': 'over'
            }
        },
        'recluster': {
            'enabled': True,
            'mod_epoch': 2,
        },
        'clustering_method': {
            'selected': 'kmeans',
            'options': dict()
        },
        'penultimate_for_clusters': True,
        'model': {
            'pretrained': True,
            'embedding': {
                'init_splitted': False,
                # other values: 'pt_default', all in torch.nn.init.<init>
                'init_type': 'pt_default',
                'init_fn_kwargs': dict()
            },
            # values: max for maxpooling and avg for averagepooling
            'features_pooling': features_pooling,
            'fixed_mask': fixed_mask
        },
    }

    if d['criterion']['selected'] == 'kantorov_margin':
        del d['criterion']['sampler']

    return d


def parse_clustering_method(val):
    if val is None:
        # default value
        val = 'kmeans'
    val = val.strip().lower()
    if val == 'kmeans':
        res = {
            'selected': 'kmeans',
            'options': dict(mode='')
        }
    elif val == 'kmeans_adaptive_centroids':
        res = {
            'selected': 'kmeans',
            'options': dict(mode='adaptive_centroids')
        }
    else:
        raise ValueError('unknown clustering-method={}'.format(val))
    return res


def parse_bool(val):

    """
    Parse bool value from cli.
    Must be represented by integer number 0 or 1.

    Return: (bool) True/False
    """
    try:
        val = int(val)
        assert val == 0 or val == 1, 'Wrong value: {}'.format(val)
    except:
        raise ValueError('Cannot parse flag {}. It must an integer 0 or 1'.format(val))
    return bool(val)


def make_args():
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default = 'resnet50', type = str)
    parser.add_argument('--criterion', default = 'MarginLoss', choices=['MarginLoss', 'KantorovMarginLoss',
                                                                       'TripletSemihardLoss',
                                                                       'TripletAllLoss',
                                                                       'TripletLoss',
                                                                       'NPairsLoss'])

    parser.add_argument('--optimizer', default = 'Adam', choices=['Adam', 'AdamW'])
    parser.add_argument('--nb-clusters', default = 1, type = int)
    parser.add_argument('--nb-epochs', type = int, default=100)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save-model', type=int, default=True)
    parser.add_argument('--save-masks', type=int, default=False)
    parser.add_argument('--mod-epoch', type = int, default=2)
    parser.add_argument('--num-samples-per-class', type=int, default=4)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--soft-loss', action='store_true')
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sz-resize', type=int,
                        default=img_transform_parameters['sz_resize'])
    parser.add_argument('--sz-crop', type=int,
                        default=img_transform_parameters['sz_crop'])
    parser.add_argument('--sz-embedding', default=512, type=int)
    parser.add_argument('--cuda-device', default = 0, type = int)
    parser.add_argument('--penultimate-for-clusters', default=1, type=parse_bool)
    parser.add_argument('-i', '--experiment-id', default='2', type=str)
    parser.add_argument('--backend', default='faiss',
                        choices=metriclearning.similarity.backends_available)
    parser.add_argument('-seed', '--random-seed', default = 1, type = int)
    parser.add_argument('-crs', '--clustering-random-state', default=None, type=int,
                       help='if None the same seed will be used for every clustering run,'\
                             'if not None will sample seed randomly for every clustering run')

    parser.add_argument('--dataset', default='cub', choices=['cub', 'cars', 'sop', 'inshop', 'vid'])
    parser.add_argument('--clustering-method',
                        default=parse_clustering_method(None),
                        type=parse_clustering_method)
    parser.add_argument('--batch-sampler', default='balanced_old', choices=['adaptbalanced',
                                                                            'balanced_old',
                                                                            'npairs',
                                                                            'randombatch'])
    parser.add_argument('--small-class-action', default='sample_other', choices=['sample_other', 'duplicate',
                                                                                 'smaller_batch'])
    parser.add_argument('--sampler', default='easypeasy', choices=['distanceweighted',
                                                                   'easypeasy',
                                                                   'HPHN',
                                                                   'EPHN',
                                                                   'EPSHN',
                                                                   'RPSHN',
                                                                   'RPHN'
                                                                  ])

    parser.add_argument('--base-lr', default=DEFAULT_BASE_LR, type=float,
                        help="the basic learning rate. Other lr will be multiplied by some factor on top")
    parser.add_argument('--emb-lr-mult', default=1.0, type=float,
                        help="the lr multiplier of the embedding layer")
    parser.add_argument('--lr-beta', default=0.0, type=float)
    parser.add_argument('--lr-alpha', default=1.0, type=float)
    parser.add_argument('--lr-step', default=float('inf'), type=float,
                        help="the lr scheduler step. Currently disabled.")
    parser.add_argument('--hierarchy-method', default='top_bot', choices=['none', 'top_bot', 'bot_top'],
                        help="All the results on paper obtained with top_bot. bot_top is redundant.")
    parser.add_argument('--mask-lr-mult', default=1.0, type=float,
                        help="lr mutiplier for masks")
    parser.add_argument('--mask-wd-mult', default=0.0, type=float,
                        help="weight decay multiplier for masks")
    parser.add_argument('--mask-relu', default=1, type=parse_bool)
    parser.add_argument('--weight-decay', default=1e-3, type=float)
    # only used for NPairs exp
    parser.add_argument('--alpha', default=1.0, type=float, help='initial value for alpha')
    parser.add_argument('--init-method', default=None, type=str)
    parser.add_argument('--lr-gamma', default=0.5, type=float, help="learning rate decay gamma")
    parser.add_argument('--force-full-embedding-epoch', type=int, default=float('inf'),
                        help='after this epoch update the full embedding using all clusters')
    parser.add_argument('--with-l1', action = 'store_true')
    parser.add_argument('--is-debug', action = 'store_true')
    parser.add_argument('--ratio-classes', type = float, default = 1.0,
        help = 'Ratio of classes used for training and evaluation.'
    )
    parser.add_argument('--mod-epoch-freeze', type = int, default=0)
    parser.add_argument('--stop-recluster', action = 'store_true')
    parser.add_argument('--reset-learners-fix', default = 1,
            type = int, choices = [0, 1])
    parser.add_argument('--verbose', action = 'store_true')
    parser.add_argument('--log-dir', type = str,
        default = 'margin_loss_resnet50_masks')
    parser.add_argument('--wandb-enabled', action='store_true')
    parser.add_argument('--album', default=None, choices=['Geom', 'GeomColor','ColorHeavy'],
        help='Check tab.5 in the paper to choose the suitable augmentation. None means standard way used in torchvision'
    )
    parser.add_argument('--init-clusters', default = 1, type=int, choices=[1, 2])
    parser.add_argument('--num-replicas',  default=1, type=int,
                        help='number of replicas of each sample. The batch size must be divisible by it.')

    parser.add_argument('--dataset-dir', default=os.getcwd()+'/DataPath', type=str, help='Path to training data.')


############################## params for mask ##################################
    parser.add_argument(
        '--masking-init',
        default='ones',
        choices=['normal', 'ones']
    )
    parser.add_argument(
        '--masking-lambda',
        default=1,
        type=float
    )
    parser.add_argument(
        '--masks-freeze-one-cluster',
        default=False,
        type=parse_bool
    )
    parser.add_argument(
        '--masks-sum-for-full-emb',
        default=False,
        type=parse_bool
    )
    parser.add_argument(
        '--masks-normalize-before-sum',
        default=None,
        choices=['l1'],
        type=str,
        help='Normalization to apply before summing up the masks. No normalization by default'
    )
    parser.add_argument(
        '--features-pooling',
        default='avg',
        choices=['avg', 'max']
    )

    parser.add_argument(
        '--fixed-mask',
        default=False,
        type=parse_bool
    )

    # ignore first arguments, 0: file_name, 1: --experiment 3: "experiment_name"
    cli_args = vars(parser.parse_args(sys.argv[3:]))

    if cli_args['masks_normalize_before_sum'] and \
       not cli_args['masks_sum_for_full_emb']:
        raise ValueError('masks_normalize_before_sum can be True '
                         'only if masks_sum_for_full_emb is True')

    cli_args_orig = copy.deepcopy(cli_args)

    # how your dataset structure should look like
    dataset_roots = {
            'cub': cli_args['dataset_dir'] + '/cub-200-2011/images',
            'cars': cli_args['dataset_dir'] + '/cars196',
            'sop': cli_args['dataset_dir'] + '/sop',
            'inshop': cli_args['dataset_dir'] + '/in-shop',
            'vid': cli_args['dataset_dir'] + '/vehicle-id/VehicleID_V1.0',
    }

############### the following part is mainly for logging and debug, usually no need to change them ###################

    args_exp = _args(
        cli_args.pop('batch_size'),
        cli_args.pop('sz_resize'),
        cli_args.pop('sz_crop'),
        cli_args['nb_clusters'],
        cli_args.pop('num_workers'),
        base_lr=cli_args.pop('base_lr'),
        emb_lr_mult=cli_args['emb_lr_mult'],
        weight_decay=cli_args['weight_decay'],
        margin=cli_args['margin'],
        soft_loss=cli_args['soft_loss'],
        mask_lr_mult=cli_args['mask_lr_mult'],
        mask_wd_mult=cli_args['mask_wd_mult'],
        features_pooling=cli_args['features_pooling'],
        fixed_mask=cli_args['fixed_mask']
    )

    args_exp['stop_recluster'] = cli_args.pop('stop_recluster')

    args_exp['reset_learners_fix'] = cli_args.pop('reset_learners_fix')

    args_exp['verbose'] = cli_args.pop('verbose')

    args_exp['mod_epoch_freeze'] = cli_args.pop('mod_epoch_freeze')
    args_exp['ratio_classes'] = cli_args.pop('ratio_classes')

    args_exp['dataset']['selected'] = cli_args.pop('dataset')
    args_exp['criterion']['selected'] = cli_args.pop('criterion')
    args_exp['opt']['selected'] = cli_args.pop('optimizer')
    args_exp['lr_scheduler']['params']['step_size'] = cli_args.pop('lr_step')
    args_exp['lr_scheduler']['params']['gamma'] = cli_args.pop('lr_gamma')
    args_exp['recluster']['mod_epoch'] = cli_args.pop('mod_epoch')
    args_exp['album'] = cli_args.pop('album')

    assert args_exp['mod_epoch_freeze'] < args_exp['recluster']['mod_epoch'], \
        'Attention! While num_clsuters > 1 Backbone will always be frozen!'

    if cli_args['batch_sampler'] == 'npairs':
        args_exp['dataloader']['train']['_batch_sampler'] = None
        args_exp['dataloader']['train']['drop_last'] = True
        args_exp['dataloader']['train']['_sampler'] = batch_samplers['npairs']
    else:
        args_exp['dataloader']['train']['_sampler'] = None
        args_exp['dataloader']['train']['batch_size'] = None
        args_exp['dataloader']['train']['drop_last'] = None
        args_exp['dataloader']['train']['_batch_sampler'] = batch_samplers[cli_args['batch_sampler']]
        if cli_args['batch_sampler'] == 'adaptbalanced':
            args_exp['dataloader']['train']['_batch_sampler']\
                    ['options']['small_class_action'] = cli_args.pop('small_class_action')
        if cli_args['batch_sampler'] in ['adaptbalanced', 'randombatch']\
                and cli_args['num_replicas'] > 1:
            assert args_exp['album'] is not None, 'Using replica without extra augmentations is redundant'
            args_exp['dataloader']['train']['_batch_sampler'] \
                ['options']['num_replicas'] = cli_args.pop('num_replicas')


    if cli_args['lr_beta'] > 0:
        args_exp['criterion']['types']['MarginLoss']['lr_beta'] = cli_args.pop('lr_beta')
        args_exp['criterion']['types']['MarginLoss']['class_specific_beta'] = True
    args_exp['criterion']['types']['NPairsLoss']['alpha'] = cli_args.pop('alpha')
    args_exp['criterion']['types']['NPairsLoss']['lr_alpha'] = cli_args.pop('lr_alpha')
    if cli_args['init_method'] is None:
        pass
    elif cli_args['init_method'] == 'pt_splitted':
        args_exp['model']['embedding']['init_splitted'] = True
    elif cli_args['init_method'].startswith('xavier_uniform_'):
        args_exp['model']['embedding']['init_splitted'] = True
        gain = float(cli_args['init_method'].rsplit('_', 1)[1])
        args_exp['model']['embedding']['init_type'] = 'xavier_uniform_'
        args_exp['model']['embedding']['init_fn_kwargs'] = dict(gain=gain)
    elif cli_args['init_method'].startswith('xavier_normal_'):
        args_exp['model']['embedding']['init_splitted'] = True
        gain = float(cli_args['init_method'].rsplit('_', 1)[1])
        args_exp['model']['embedding']['init_type'] = 'xavier_normal_'
        args_exp['model']['embedding']['init_fn_kwargs'] = dict(gain=gain)
    elif cli_args['init_method'].startswith('kaiming_normal_'):
        assert cli_args['init_method'] in ['kaiming_normal_fan_in', 'kaiming_normal_fan_out']
        mode = cli_args['init_method'].split('kaiming_normal_', 1)[1]
        args_exp['model']['embedding']['init_splitted'] = True
        args_exp['model']['embedding']['init_type'] = 'kaiming_normal_'
        args_exp['model']['embedding']['init_fn_kwargs'] = dict(mode=mode, nonlinearity='linear')
    elif cli_args['init_method'] == 'trunc_normal':
        args_exp['model']['embedding']['init_splitted'] = False
        args_exp['model']['embedding']['init_type'] = cli_args['init_method']
    else:
        raise ValueError('Unknown init method: {}'.format(cli_args['init_method']))


    if cli_args['nb_clusters'] == 1:
        args_exp['recluster'] = {'enabled': False}
    if args_exp['dataloader']['train']['_batch_sampler'] is not None:
        args_exp['dataloader']['train']['batch_size'] = 1
        if 'num_samples_per_class' in \
           args_exp['dataloader']['train']['_batch_sampler']['options']:
            args_exp['dataloader']['train']['_batch_sampler']\
                    ['options']['num_samples_per_class'] = cli_args.pop('num_samples_per_class')
        else:
            cli_args.pop('num_samples_per_class')

    if args_exp['dataloader']['train']['_sampler'] is not None:
        args_exp['dataloader']['train']['_sampler']\
                ['options']['num_samples_per_class'] = cli_args.pop('num_samples_per_class')

    elif cli_args['sampler'] == 'distanceweighted':
        args_exp['criterion']['sampler']['class'] = DistanceWeightedSampler
    elif cli_args['sampler'] == 'easypeasy':
        args_exp['criterion']['sampler']['class'] = EasyPeasyPairsSampler
    elif cli_args['sampler'] == 'HPHN':
        # hard positive, hard negative
        args_exp['criterion']['sampler']['class'] = FlexibleTripletSampler
        args_exp['criterion']['sampler']['options']['positive'] = 'hard'
        args_exp['criterion']['sampler']['options']['negative'] = 'hard'
    elif cli_args['sampler'] == 'EPHN':
        # easy positive, hard negative
        args_exp['criterion']['sampler']['class'] = FlexibleTripletSampler
        args_exp['criterion']['sampler']['options']['positive'] = 'easy'
        args_exp['criterion']['sampler']['options']['negative'] = 'hard'
    elif cli_args['sampler'] == 'EPSHN':
        # easy positive, semihard negative
        args_exp['criterion']['sampler']['class'] = FlexibleTripletSampler
        args_exp['criterion']['sampler']['options']['positive'] = 'easy'
        args_exp['criterion']['sampler']['options']['negative'] = 'semihard'
    elif cli_args['sampler'] == 'HPSHN':
        # hard positive, semihard negative
        args_exp['criterion']['sampler']['class'] = FlexibleTripletSampler
        args_exp['criterion']['sampler']['options']['positive'] = 'hard'
        args_exp['criterion']['sampler']['options']['negative'] = 'semihard'
    elif cli_args['sampler'] == 'RPSHN':
        # random positive, semihard negative
        args_exp['criterion']['sampler']['class'] = FlexibleTripletSampler
        args_exp['criterion']['sampler']['options']['positive'] = 'random'
        args_exp['criterion']['sampler']['options']['negative'] = 'semihard'
    elif cli_args['sampler'] == 'RPHN':
        # random positive, hard negative
        args_exp['criterion']['sampler']['class'] = FlexibleTripletSampler
        args_exp['criterion']['sampler']['options']['positive'] = 'random'
        args_exp['criterion']['sampler']['options']['negative'] = 'hard'
    else:
        raise ValueError('Unknown sampler: {}'.format(cli_args['sampler']))


    for k in cli_args:
        args_exp[k] = cli_args[k]

    if args_exp['hierarchy_method'] not in [None, 'none']:
        hierarchy_method_str = '-hrc-{}'.format(args_exp['hierarchy_method'])
    else:
        hierarchy_method_str = ''

    if args_exp['clustering_random_state'] is None:
        clustering_random_state_str = ''
    else:
        clustering_random_state_str = '-crs{}'.format(args_exp['clustering_random_state'])

    if math.isfinite(args_exp['lr_scheduler']['params']['step_size']):
        step_suffix = '-step{}'.format(args_exp['lr_scheduler']['params']['step_size'])
    else:
        step_suffix = ''

    force_full_emb_suff = ''
    if args_exp['force_full_embedding_epoch'] <= 0:
        force_full_emb_suff = '_force-full-emb0'
    elif math.isfinite(args_exp['force_full_embedding_epoch']):
        force_full_emb_suff = '_force-full-emb{}'.format(args_exp['force_full_embedding_epoch'])

    if cli_args['batch_sampler'] == 'adaptbalanced':
        batch_sampler_suff = 'adaptbalanced' + ('-' + args_exp['dataloader']['train']['_batch_sampler']\
                                                    ['options']['small_class_action']
                                               if args_exp['dataloader']['train']['_batch_sampler']\
                                                    ['options']['small_class_action'] != 'sample_other'
                                                else '')
        batch_sampler_suff += ('-rep' + str(args_exp['dataloader']['train']['_batch_sampler']\
                                                    ['options']['num_replicas'])
                               if args_exp['dataloader']['train']['_batch_sampler']\
                                                    ['options']['num_replicas'] > 1
                               else '')
    elif cli_args['batch_sampler'] == 'randombatch':
        batch_sampler_suff = cli_args['batch_sampler']

        batch_sampler_suff += ('-rep' + str(args_exp['dataloader']['train']['_batch_sampler'] \
                                                ['options']['num_replicas'])
                               if args_exp['dataloader']['train']['_batch_sampler'] \
                                      ['options']['num_replicas'] > 1
                               else '')
    else:
        batch_sampler_suff = cli_args['batch_sampler']

    beta_suffix = ''
    lr_beta = args_exp['criterion']['types']['MarginLoss']['lr_beta']
    if lr_beta > 0:
        beta_suffix = f'lrbeta{lr_beta}'

############################## final file name for the model and .log ##############################

    # DIVR3 means redo division into 2 clusrters 3 times (in faiss)
    # NREDO5 means redo clustering 5 times every reclsutering, including splitting clusters in 2 (in faiss)
    args_exp['log'] = {
        'name': '{}-exp-{}-e-{}{sz_crop}-c-{}{hierarchy_method_str}{clustering_random_state_str}-NREDO1{mod_epoch_freeze}{features_pool}{}{}{}{}{}{}{force_full_emb}-newlr2_{lr_suff}{beta_suffix}-{sampler}-{album}{mask_relu}{fixed_mask}'.format(
            args_exp['dataset']['selected'],
            args_exp['experiment_id'],
            args_exp['sz_embedding'],
            args_exp['nb_clusters'],
            '-{}'.format(cli_args['init_method']) if cli_args['init_method'] is not None else '',
            step_suffix,
            '-margin{}'.format(cli_args['margin']) if cli_args['margin'] != 0.2 else '',
            '-kantorov_margin' if args_exp['criterion']['selected'] == 'kantorov_margin' else '',
            '-' + batch_sampler_suff + \
            ('-{}'.format(args_exp['recluster']['mod_epoch']) \
                if args_exp['recluster']['enabled'] \
                    and args_exp['recluster']['mod_epoch'] != 2 else '') + \
            ('-{}_{}'.format(args_exp['clustering_method']['selected'],
                             args_exp['clustering_method']['options']['mode'][:3]) \
                if args_exp['clustering_method']['selected'] != 'kmeans' else ''),
            '-clust_final' if not cli_args['penultimate_for_clusters'] else '',
            force_full_emb=force_full_emb_suff,
            lr_suff=
            empty_if_default(args_exp['opt']['selected'], optimizers['newlr2']['selected']) + \
            (str(args_exp['opt']['base']['lr']) if args_exp['opt']['base']['lr'] != DEFAULT_BASE_LR else '') + \
            ('-emb_lrm{}'.format(cli_args['emb_lr_mult']) if cli_args['emb_lr_mult'] != 1.0 else '') + \
            ('-wd{}'.format(cli_args['weight_decay']) \
                if cli_args['weight_decay'] != optimizers['newlr2']['base']['weight_decay'] \
                else ''),
            beta_suffix=beta_suffix,
            hierarchy_method_str=hierarchy_method_str,
            clustering_random_state_str=clustering_random_state_str,
            mod_epoch_freeze=f'-frz{args_exp["mod_epoch_freeze"]}' if args_exp['mod_epoch_freeze'] else '',
            features_pool='-{}pool'.format(args_exp['features_pooling']) if args_exp['features_pooling'] != 'avg' else '',
            sampler=cli_args['sampler'],
            album=args_exp['album'] if args_exp['album'] else '',
            fixed_mask='FIXEDM' if args_exp['fixed_mask'] else '',
            mask_relu='-mnorelu' if not args_exp['mask_relu'] else '',
            sz_crop='-sz{}'.format(args_exp['img_transform_parameters']['sz_crop']) if args_exp['img_transform_parameters']['sz_crop'] != img_transform_parameters['sz_crop'] else '',
        ),
        'path': 'log/{}'.format(cli_args.pop('log_dir'))
    }


    final_args = base.make_args(dataset_roots)
    del final_args['opt']

    # reduce classes for testing code or hyperparameter search
    def reduce_classes(crange, ratio):
        # crange is class range, i.e. for CUB init/eval: [0,100]/[100,200]
        return range(
            min(crange),
            min(crange) + round((max(crange) + 1 - min(crange)) * ratio)
        )

    base.recursive_dict_update(final_args, args_exp)
    if args_exp['ratio_classes'] != 1:
        for x in ['init', 'train', 'eval']:
          final_args['dataset']['types'][
                args_exp['dataset']['selected']
            ]['classes'][x] = reduce_classes(
                crange = final_args['dataset']['types'][
                        args_exp['dataset']['selected']
                    ]['classes'][x],
                ratio = args_exp['ratio_classes']
            )

    # wandb setting
    if final_args['wandb_enabled']:
        import wandb
        wandb.init(project="DivideAndConquer",
                   name=args_exp['log']['name'],
                   notes='notes',
                   tags=['default'],
                   config=cli_args_orig)

    return final_args

