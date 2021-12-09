
"""
Abstract function for embedding models. Currently supports AlexNet and VGG
since both have similar forwarding pass and features/classifier structure.
"""

from math import ceil
import logging
import torch
from .init import init_weights, init_bias
import numpy as np
import torchvision
import torch
from ..bn_inception import BNInception, bn_inception
from torch.nn import Linear, Dropout, AvgPool2d, MaxPool2d
from torch.nn import AdaptiveMaxPool2d, AdaptiveAvgPool2d
import torch.nn.functional as F

# currently supported model architectures
model_architectures = {
    'alexnet': torchvision.models.alexnet,
    'vgg11': torchvision.models.vgg11,
    'vgg11bn': torchvision.models.vgg11_bn,
    'bn_inception': bn_inception
}

_sz_features = {
    # ResNet hardcoded in `resnet` module
    torchvision.models.AlexNet: 256,
    torchvision.models.VGG: 512,
    BNInception: 1024
}


def log_verbose(message, args, with_print=False):
    if args['verbose']:
        logging.debug(message)
    if with_print:
        print(message)


def init_splitted(layer, weight_init, nb_clusters, sz_embedding, **kwargs):
    from math import ceil
    if weight_init == None:
        # default pytorch init
        # since default initializer is :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
        # :math:`k = \frac{1}{\text{in\_features}}`.
        # It doe snot depend on number of output features
        _layer = torch.nn.Linear(layer.weight.shape[1], layer.weight.shape[0])
        layer.weight.data = _layer.weight.data
        layer.bias.data = _layer.bias.data
    else:
        for c in range(nb_clusters):
            i = torch.arange(
                c * ceil(sz_embedding / nb_clusters),
                # cut off remaining indices, e.g. if > embedding size
                min(
                    (c + 1) * ceil(
                        sz_embedding / nb_clusters
                    ),
                    sz_embedding
                )
            ).long()
            _layer = torch.nn.Linear(layer.weight.shape[1], len(i))
            layer.weight.data[i] = weight_init(_layer.weight.data, **kwargs)
            layer.bias.data[i] = _layer.bias.data


def make_parameters_dict(model, filter_module_names):
    """
    Separates model parameters into 'backbone' and other modules whose names
    are given in as list in `filter_module_names`, e.g. ['embedding_layer'].
    """

    # init parameters dict
    D = {k: [] for k in ['backbone', *filter_module_names]}
    for name, param in model.named_parameters():
        name = name.split('.')[0]
        if name not in filter_module_names:
            D['backbone'] += [param]
        else:
            D[name] += [param]

    # verify that D contains same number of parameters as in model
    nb_total = len(list(model.parameters()))
    nb_dict_params = sum([len(D[d]) for d in D])
    assert nb_total == nb_dict_params
    return D


def embed_model(model, args, sz_embedding, normalize_output=True):
    if type(model) in [
        torchvision.models.AlexNet, torchvision.models.VGG, BNInception,
        torchvision.models.ResNet
        ]:

        model.opt = None

        def reset_embedding():
            if args['model']['embedding']['init_type'] == 'trunc_normal':
                log_verbose('Initializing with truncated normal.', args)
                init_weights(model.embedding)
                init_bias(model.embedding)

            else:
                # embedding init
                if args['model']['embedding']['init_type'] == 'pt_default':
                    if args['model']['embedding']['init_splitted']:
                        init_splitted(model.embedding, None,
                            args['nb_clusters'], args['sz_embedding'])
                    else:
                        # leave embedding as it was initialized by pytorch
                        pass
                else:
                    logging.info('Init splitted embedding using {}(...)'.format(
                                 args['model']['embedding']['init_type']))
                    _weight_init = getattr(
                        torch.nn.init,
                        args['model']['embedding']['init_type']
                    )
                    init_splitted(model.embedding, _weight_init,
                        args['nb_clusters'], args['sz_embedding'],
                        **args['model']['embedding']['init_fn_kwargs'])

        assert not hasattr(model, 'features_pooling')
        assert not hasattr(model, 'features_dropout')

        if args['model']['features_pooling'] == 'avg':
            if args['dataset']['selected'] == 'market':
                # the market has rectangle input, so the pooling layer differs from others
                # default input size (384, 128)

                # similar to PCB model we remove the downsampling at layer4 of ResNet50
                model.layer4[0].downsample[0].stride = (1,1)
                model.layer4[0].conv2.stride = (1,1)
                model.features_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
            else:
                model.features_pooling = AdaptiveAvgPool2d(1)
        elif args['model']['features_pooling'] == 'max':
            model.features_pooling = AdaptiveMaxPool2d(1)
        # TODO: Make dropout as optional parameter!
        model.features_dropout = Dropout(p = args['features_dropout_prob'])

        # choose arbitrary parameter for selecting GPU/CPU
        dev = list(model.parameters())[0].device
        if type(model) != torchvision.models.ResNet:
            model.sz_features_output = _sz_features[type(model)]
        torch.random.manual_seed(args['random_seed'] + 1)
        model.embedding = Linear(model.sz_features_output, sz_embedding).to(dev)
        model.reset_embedding = reset_embedding

        # for fair comparison between different cluster sizes
        torch.random.manual_seed(args['random_seed'] + 1)
        np.random.seed(args['random_seed'] + 1)
        model.reset_embedding()

        if type(model) == BNInception:
            features_parameters = (
                p for n, p in model.named_parameters() if n not in [
                    'embedding.weight', 'embedding.bias']
            )
        else:
            features_parameters = model.features.parameters()


        model.parameters_dict = make_parameters_dict(
            model = model,
            filter_module_names = ['embedding']
        )

        if not normalize_output:
            raise NotImplementedError('We always normalize embedding output')

        def summarize_masks(masks):
            # to better track mask value in log
            s = 'mask values:'
            for j, m in enumerate(masks):
                mask = masks[j].clone().detach()
                mask /= mask.max()
                l1norm = mask.norm(p=1)
                s += '\n{}: ({})  {} ... {}'.format(
                    j,
                    l1norm,
                    ' '.join(str(round(i.item(), 3)) for i in mask[:20]),
                    ' '.join(str(round(i.item(), 3)) for i in mask[-20:])
                )
            s += '\n'
            return s

        # model.log_masks = lambda: log_verbose(
        #     summarize_masks(model.masks), args, with_print=True
        # )
        model.summarize_masks = lambda: summarize_masks(model.masks)

        def fixed_mask_split(pre_mask):
            """
            :param array: the previous mask (tensor) containing consecutive 0 and 1s.
            :yield: the 2 children masks

            e.g. for input like [1,1,1,1,1,1,1,1], it returns 2 arrays [1,1,1,1,0,0,0,0,]
            and [0,0,0,0,1,1,1,1]
            """
            array = pre_mask.cpu().detach().numpy()
            mask_len = np.sum(array) // 2
            children = [np.copy(array), np.copy(array)]
            # where the first nonzero element is
            idx_st = np.nonzero(array)[0][0]
            # from the idx to next mask_len elements should keep old value (1s)
            keep = np.arange(idx_st, idx_st + mask_len)
            # the rest will be set to 0
            children[0][~np.in1d(np.arange(len(array)), keep)] = 0
            # complement to the one above
            children[1] = array - children[0]

            for c in children:
                yield c

        def reset_learners_indices():
            # important to keep args global, otherwise not updated here
            nonlocal args
            nb_clusters = args['nb_clusters']
            sz_embedding = args['sz_embedding']

            # HERE
            # only create new masks if number of clusters changed
            while len(model.masks) < nb_clusters:
                if not args['fixed_mask']:
                    # children mask will first copy parent's. Since they are learnable
                    # they will diverge later because of different samples.
                    masks = torch.nn.ParameterList(
                        # NOTE:
                        # use the following, as written in PyTorch doc!
                        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.new_tensor
                        torch.nn.Parameter(m.clone().detach().requires_grad_(True))
                        # select one mask
                        for m in model.masks
                            # duplicate twice (because of split)
                            for i in range(2)
                    )
                    model.masks = masks
                    log_verbose('Initialized masks...', args)
                    log_verbose(
                        summarize_masks(model.masks), args, with_print=False
                    )
                    if model.opt is not None:
                        # remove previous masks from optimizer and add new ones
                        assert len(model.opt.param_groups) in [3, 4]
                        assert model.opt.param_groups[-1]['group_name'] == 'masks'
                        del model.opt.param_groups[-1]
                        model.opt.add_param_group({
                            'group_name': 'masks',
                            'params': model.masks,
                            **args['opt']['mask']
                        })
                        log_verbose('Reset optimizer params for masks...\n' +\
                                    str(model.opt.param_groups[-1]), args)
                else:
                    # to use fixed mask: ones for selected idx and zeros for the rest
                    # NOTE: still requires grad, but remember to exclude them from optimizer
                    assert args['masking_init'] == 'ones', 'Fixed masks should use ones as initial values'

                    masks = torch.nn.ParameterList(
                        torch.nn.Parameter(data=torch.from_numpy(child).cuda())
                        # split the mask into 2 sub-masks while remaining same dimension
                        for m in model.masks
                            for child in fixed_mask_split(m)
                    )
                    model.masks = masks
                    log_verbose('Initialized masks...', args)
                    log_verbose(
                        summarize_masks(model.masks), args, with_print=False
                    )
                    if model.opt is not None:
                        # No opt is actually applied as the lr is set to be zero in this case.
                        # But to keep consistent with the masked case to avoid raising some error.
                        assert len(model.opt.param_groups) in [3, 4]
                        assert model.opt.param_groups[-1]['group_name'] == 'masks'
                        del model.opt.param_groups[-1]
                        model.opt.add_param_group({
                            'group_name': 'masks',
                            'params': model.masks,
                            **args['opt']['mask']
                        })

        # at the very beginning the mask are all ones
        model.masks = torch.nn.ParameterList(
            [torch.nn.Parameter(
                data=torch.ones(sz_embedding)
            )]
        )
        model.reset_learners_indices = reset_learners_indices

        if args['masking_init'] == 'normal':
            assert not args['fixed_mask'], 'Fixed mask cannot be used with the normal distr mask init'
            print('Initializing masks with normal distribution.')
            model.masks[0].data.normal_(0.9, 0.7)

        log_verbose(summarize_masks(model.masks), args, with_print=False)


        def forward(x,
                    use_penultimate=False,
                    dset_id=None):
            """
            x: input batch
            use_penultimate: return features from penultimate layer
                             instead of the final layer
            dset_id: learner id, used to apply appropriate mask.
                     No mask applied if None
            """

            if args['reset_learners_fix']:
                if not args['force_full_embedding']:
                    # During each forward the function will check
                    # the current nb_clusters and generate new masks if needed
                    reset_learners_indices()
                    assert len(model.masks) == args['nb_clusters'], \
                        f"{len(model.masks)} != {args['nb_clusters']}"
            x = model.features(x)
            x = model.features_pooling(x)
            x = model.features_dropout(x)
            x = x.view(x.size(0), -1)


            # Use either one of them
            if not use_penultimate:
                x = model.embedding(x)
                if dset_id is not None:
                    x *= (F.relu(model.masks[dset_id]) if args['mask_relu'] else model.masks[dset_id])
                elif args['masks_sum_for_full_emb']:
                    masks = [(F.relu(m) if args['mask_relu'] else m) for m in model.masks]
                    if args['masks_normalize_before_sum'] == 'l1':
                        masks = [F.normalize(m, p=1, dim=0) for m in masks]
                    else:
                        assert args['masks_normalize_before_sum'] is None
                    x *= sum([m for m in masks])
            else:
                # use penultimate
                pass
            # at eval not normalized, so normalize here
            x = F.normalize(x, p=2, dim=1)

            return x

        model.forward = forward

    else:
        raise ValueError(
            'Model type `{}` currently not supported.'.format(
                type(model)))
