
# package works only with `model` package, thus init `model`, then use this
from ..bn_inception import BNInception, bn_inception
from .models import embed_model
import torchvision

# supported model architectures
model_architectures = {
    'alexnet': torchvision.models.alexnet,
    'vgg11': torchvision.models.vgg11,
    'vgg11bn': torchvision.models.vgg11_bn,
    'bn_inception': bn_inception
}
