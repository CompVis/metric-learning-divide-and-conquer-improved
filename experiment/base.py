import torch

def update(a, b, a_prev = None, b_prev = None, k = None):
    """
    Recursively update dictionary `a` with dictionary `b`.

    If some keys are missing in in `a` but present in `b`
        then they will be created.
    If key `k` exist in both `a` and `b` than
        a[k] will be owerwritten with b[k]
    If key `k` exists only in `a` it will be left unchanged.

    NOTE: Existing values might be updated and new items added.
        But nothing will be removed from dict `a`.
    """
    if a is None or type(b) != dict:
        assert a_prev is not None
        a_prev[k] = b
    else:
        for k in b:
            if k not in a:
                a[k] = b[k]
            else:
                update(a[k], b[k], a, b, k)
            # print(k, a[k])


recursive_dict_update = update


def recursive_apply_fn(a, fn, a_prev=None):
    """
    Recursively apply a function fn to every key of dictionary `a`

    NOTE: Existing values might be updated
        But nothing will be removed from dict `a`.
    """
    if isinstance(a, dict):
        for k in a:
            a[k] = fn(a[k])
            update(a[k], fn, a)
            # print(k, a[k])


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default = 'alexnet', type = str)
    parser.add_argument('--save-model', action = 'store_true')
    parser.add_argument('--cuda-device', default = 0, type = int)
    parser.add_argument('--nb-epochs', default = 25, type = int)
    parser.add_argument('--experiment-id', default = 0, type = int)
    return parser


def make_args(dataset_roots):
    args = {
        'is_debug': False,
        'reassign_random': False,
        'random_seed': 0,
        'save_model': False,
        'resnet': {
            'bn_learnable': False
        },
        'supervised': True,
        'experiment_id': 0,
        'penultimate_for_triplets': True,
        'with_l1': False,
        'criterion': {
            'selected': 'TripletAllLoss',
            'types': {
                'TripletAllLoss': {
                    'margin': 0.2,
                },
                'MarginLoss': {
                    'margin': 0.2,
                    'beta': 1.2,
                    'class_specific_beta': True,
                    'lr_beta': 0.1,
                    'nu': 0.0,
                },
                'KantorovMarginLoss': {
                    'margin': 0.2 # currently is hardcoded
                }
            }
        },
        'dataset': {
            'selected': 'cub',
            'types': {
                'vid': {
                    'root': dataset_roots['vid'],
                    'classes': {
                        'train': range(0, 13164),
                        'init': range(0, 13164),
                        # small: 800, mid: +1600, large: +2400
                        'eval': range(13164, 13164 + 2400)
                    }
                },
                'inshop': {
                    'root': dataset_roots['inshop'],
                    'classes': {
                        'train': range(0, 3997),
                        'init': range(0, 3997),
                        'eval': range(0, 3985)
                    }
                },
                'cub': {
                    'root': dataset_roots['cub'],
                    'classes': {
                        'train': range(0, 100),
                        'init': range(0, 100),
                        'eval': range(100, 200)
                    }
                },
                'cars': {
                    'root': dataset_roots['cars'],
                    'classes': {
                        'train': range(0, 98),
                        'init': range(0, 98),
                        'eval': range(98, 196)
                    }
                },
                'sop': {
                    'root': dataset_roots['sop'],
                    'classes': {
                        'train': range(0, 11318),
                        'init': range(0, 11318),
                        'eval': range(11318, 22634)
                    }
                }
            },
            'augmentation': {
                'train': True,
                'eval': False,
                'init': False
            }
        },
        'recluster': {
            'enabled': True,
            'mod_epoch': 2,
            'method': {
                'types': ['reset', 'reassign', 'const'],
                'selected': 'reassign'
            }
        },
        'clustering_method': {
            'selected': 'kmeans', # or labels
            'options': dict()
        },
        'model': {
            'pretrained': True,
            'embedding': {
                'init_splitted': False,
                # other values: 'pt_default', all in torch.nn.init.<init>
                'init_type': 'pt_default',
                'init_fn_kwargs': dict()
            },
            # values: max for maxpooling and avg for averagepooling
            'features_pooling': 'avg'
        },
        'arch': 'vgg11bn',
        'backend': 'faiss',
        'checkpoint': None,
        'penultimate_at_eval': True,
        'penultimate_for_clusters': True,
        'penultimate_for_neighbors': True,
        'penultimate_at_train': False,
        'normalize_output': True,
        'nb_clusters': 3,
        'sz_embedding': 64,
        'features_dropout_prob': 0.01,
        'nb_epochs': 10,
        'finetune_epoch': float('inf'), # keep it here, otherwise will raise error during training.
        'kantorov_bgr': False,
        'force_full_embedding': False,
        'opt': {
            'base': {
                    'lr': 1e-3,
                    'weight_decay': 1e-4
            },
            'features': {
                'lr': 1e-3 * 1e-2,
                'weight_decay': 1e-4
            },
            'embedding': {
                'lr': 1e-3 * 1e-1,
                'weight_decay': 1e-4
            }
        },
        'lr_scheduler': {
            'class': torch.optim.lr_scheduler.StepLR,
            'params': {
                'step_size': float('inf'),
                'gamma': 1.0
            }
        },
        'log': {
            'path': 'log',
            'name': 'test'
        },
        'dataloader': {
            'train': {
                'drop_last': False,
                'shuffle': True,
                'batch_size': 42,
                'pin_memory': True,
                'num_workers': 4,
                '_batch_sampler': None,
                '_sampler': None
            },
            'init': {
                'drop_last': False,
                'shuffle': False,
                'batch_size': 42,
                'pin_memory': True,
                'num_workers': 4
            },
            'eval': {
                'drop_last': False,
                'shuffle': False,
                'batch_size': 42,
                'pin_memory': True,
                'num_workers': 4
            },
            'merged': {
                'mode': 2,
                'sampling_mode': 'over'
            }
        },
        'cuda_device': 0
    }
    return args

