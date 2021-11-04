
import torchvision
from torchvision.models import *


img_normalization_parameters = {
    'rgb_to_bgr': False,
    'intensity_scale': [[0, 1], [0, 1]],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}


for model_type in [AlexNet, DenseNet, Inception3, ResNet, SqueezeNet, VGG]:
    #used only for if pytoirch built-in augmentations enabled
    model_type.img_normalization_parameters = img_normalization_parameters

