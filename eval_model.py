import torch
import metriclearning
import dataset
import argparse
import numpy as np
import glob
import shelve
from metriclearning.utils import eval_model, evaluate, part_dict, predict_batchwise
import experiment
from experiment.base import recursive_dict_update


def load_mask_model(args_restored, model_path):
    args = experiment.margin_loss_resnet50.make_args()

    # To use default values if the value was not stored beforehand
    recursive_dict_update(args, args_restored)

    model = metriclearning.model.make(args).cuda()
    while len(model.masks) < args['nb_clusters']:
        model.reset_learners_indices() # to duplicate proper number of masks

    state_dict = torch.load(model_path,
                 map_location={'cuda:{}'.format(i): 'cuda:{}'.format(args['cuda_device']) for i in range(10)})
    model.load_state_dict(state_dict)

    return args, model

def fetch_recall(metrics, max_e=400):
    return [metrics[m]['score']['eval']['final'][1][0] for m in metrics if m < max_e and m >= 0]

def eval_procedure(shelves, log_path):

    metrics = []
    args = []

    for path in shelves:
        # retrieve the log and metric info
        with shelve.open(path) as f:
            metrics.append(f['metrics'])
            args.append(f['args'])
        print(path)

    for i in range(len(metrics)):

        best_epoch = np.argmax(fetch_recall(metrics[i], max_e=400))

        path = log_path + args[i]['log']['name']

        if 'nb_clusters_final' in args[i].keys():
            args[i]['nb_clusters'] = args[i]['nb_clusters_final']

        # choose the correct model checkpoint based on best epoch
        if len(glob.glob(path + '*.pt')) == 1:
            model_path = (glob.glob(path + '*.pt')[0])
        else:
            for name in glob.glob(path + '*.pt'):
                if '-full-emb-before-' in name:
                    model_path = name
                    args[i]['force_full_embedding'] = True

        if 'img_transform_parameters' not in args[i].keys():
            # extra check for transformation. Some models may not have this key
            args[i]['img_transform_parameters'] = {
                                        'sz_resize': 256,
                                        'sz_crop': 224,
                                    }

        # for mask model
        args[i], model = load_mask_model(args[i], model_path)

       # print(args[i]['masks_sum_for_full_emb'])
       # print(args[i]['force_full_embedding'])

        # prepare the evaluation set for each model
        selected = args[i]['dataset']['selected']

        if args[i]['dataset'] ['selected'] == 'inshop':
            # evaluation dataloaders for inshop differs from others
            dl_query = dataset.loader.make_loader(args[i], model,
                                'eval', inshop_type = 'query')
            dl_gallery = dataset.loader.make_loader(args[i], model,
                                'eval', inshop_type = 'gallery')
            dl_ev = (dl_query, dl_gallery)


        elif args[i]['dataset'] ['selected'] == 'vid':
            # For vid we need to evaluatate on different subsets
            large = args[i]['dataset']['types']['vid']['classes']['eval']
            medium = range(large.start, large.start + 1600)
            small = range(large.start, large.start + 800)

            # the sizes of the largest class in three sub sets are 79, 80, 118 respectively
            dl_ev = []
            for size in [small, medium, large]:
                args[i]['dataset']['types']['vid']['classes']['eval'] = size
                dl_ev.append(dataset.loader.make_loader(args[i], model, 'eval'))

        else:
            # For other datasets
            dl_ev = dataset.loader.make_loader(args[i], model, 'eval')

        print('  evaluating model: \n --{}--\n'.format(args[i]['log']['name']))
        print('\n\nBest epoch: {}'.format(best_epoch))
        print('nb clusters: {}'.format(args[i]['nb_clusters']))

        R_k, nmi, mARP = eval_model(model, args[i], dl_ev, new_vid=False)

        print('nmi: \n{};\n R@k: \n{};\n mARP: \n{}'.format(100*np.around(nmi,4), 100*np.around(R_k,4), 100*np.around(mARP,4)))

        del model

        print('---------------NEXT MODEL---------------\n')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type = str)
    args = parser.parse_args()
    log_path = args.log_path
    logs = glob.glob(log_path+'*.log')

    shelves = [log.split('.log')[0] for log in logs]
    eval_procedure(shelves, log_path)

if __name__ == '__main__':
    main()
