
from . import base

def _args():
    return {
        'arch': 'bn_inception',
        'supervised': True,
        'dataset': {
            'types': {
                'cub': {
                    'classes': {
                        'train': range(0, 10),
                        'init': range(0, 10),
                        'eval': range(100, 110)
                    }
                }
            }
        },
        'nb_epochs': 25,
    }

def make_args():
    import sys
    import argparse

    args = base.make_args()
    args_exp = _args()

    parser = base.make_parser()
    parser.add_argument('--nb-clusters', required = True, type = int)
    parser.add_argument('--sz-embedding', default = 64, type = int)
    parser.add_argument('--is-unsupervised', dest = 'supervised', 
        action = 'store_false')

    # ignore first arguments, 1: file name, 2: --experiment 3: experiment name
    parser_args = parser.parse_args(sys.argv[3:])
    if parser_args.nb_clusters == 1:
        args_exp['recluster'] = {'enabled': False}

    for arg in parser_args.__dict__:
        args_exp[arg] = parser_args.__dict__[arg]

    args_exp['log'] = {
        'name': 'exp-{}-e-{}-c-{}'.format(
            args_exp['experiment_id'],
            args_exp['sz_embedding'],
            args_exp['nb_clusters']
        ),
        'path': 'log/nb-clusters/{}'.format(
            'supervised' if parser_args.supervised == True else 'unsupervised'
        )
    }
    base.update(args, args_exp)
    return args
