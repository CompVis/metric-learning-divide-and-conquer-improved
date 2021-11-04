
from . import torchvision_models as _torchvision_models
from .bn_inception import bn_inception as _bn_inception
from .inweave import embed_model
from . import resnet as _resnet

def set_path_to_models(path):
    import os
    os.environ['TORCH_MODEL_ZOO'] = path


# supported model architectures
_arch = {
    'alexnet': _torchvision_models.alexnet,
    'vgg11': _torchvision_models.vgg11,
    'vgg11bn': _torchvision_models.vgg11_bn,
    'bn_inception': _bn_inception,
    'resnet18': _resnet.resnet18,
    'resnet34': _resnet.resnet34,
    'resnet50': _resnet.resnet50,
    'resnet101': _resnet.resnet101,
    'resnet152': _resnet.resnet152
}

def factory():
    return list(_arch.keys())

def make(args, embedded = True):
    if args['arch'][:6] == 'resnet':
        model = _arch[args['arch']](
            pretrained = args['model']['pretrained'],
            bn_learnable = args['resnet']['bn_learnable']
        )
    else:
        model = _arch[args['arch']](pretrained = args['model']['pretrained'])
    if embedded:
        embed_model(
            model = model,
            args = args,
            sz_embedding = args['sz_embedding'],
            normalize_output = args['normalize_output']
        )
    return model

