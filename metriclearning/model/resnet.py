
import torchvision
import torch


def resnet(model_type, bn_learnable, pretrained = True):
    model = getattr(torchvision.models, model_type)(pretrained)
    model.features = torch.nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4
    )
    if model_type in ['resnet18', 'resnet34']:
        model.sz_features_output = 512
    elif model_type in ['resnet50', 'resnet101', 'resnet152']:
        model.sz_features_output = 2048
    # deactivate learnable parameters for batchnorm
    if bn_learnable == False:
        for module in filter(
            lambda m: type(m) == torch.nn.BatchNorm2d, model.modules()
        ):
            module.eval()
            module.train = lambda _: None
        
    return model


def resnet18(pretrained = True, bn_learnable = True):
    return resnet('resnet18', pretrained = pretrained, 
            bn_learnable = bn_learnable)


def resnet34(pretrained = True, bn_learnable = True):
    return resnet('resnet34', pretrained = pretrained, 
            bn_learnable = bn_learnable)


def resnet50(pretrained = True, bn_learnable = True):
    return resnet('resnet50', pretrained = pretrained, 
            bn_learnable = bn_learnable)


def resnet101(pretrained = True, bn_learnable = True):
    return resnet('resnet101', pretrained = pretrained, 
            bn_learnable = bn_learnable)


def resnet152(pretrained = True, bn_learnable = True):
    return resnet('resnet152', pretrained = pretrained, 
            bn_learnable = bn_learnable)


