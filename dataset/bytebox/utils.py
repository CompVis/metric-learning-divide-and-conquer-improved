from __future__ import print_function
from __future__ import division
from torchvision import transforms
import PIL.Image
import torch
import cv2
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

def std_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).std(dim = 1)


def mean_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).mean(dim = 1)


class Identity():
    # used for skipping transforms
    def __call__(self, im):
        return im


class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor

class ResizeV2(A.Resize):
    """
        torchvision.transforms version of resize reimplemented in Albumentation
    """
    def __init__(self, height, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(ResizeV2, self).__init__(always_apply, p)
        self.height = height
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):

        h, w = img.shape[:2]

        if (w <= h and w == self.height) or (h <= w and h == self.height):
            return img

        if w < h:
            ow = self.height
            oh = int(self.height * h / w)
            return F.resize(img, oh, ow, interpolation=interpolation)
        else:
            oh = self.height
            ow = int(self.height * w / h)
            return F.resize(img, oh, ow, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        pass

    def get_transform_init_args_names(self):
        pass


def make_transform(sz_resize = 256, sz_crop = 224, mean = [104, 117, 128],
        std = [1, 1, 1], rgb_to_bgr = True, is_train = True,
        intensity_scale = None, using_hinton=False):
    if not using_hinton:
        return transforms.Compose([
            RGBToBGR() if rgb_to_bgr else Identity(),
            transforms.RandomResizedCrop(sz_crop) if is_train else Identity(),
            transforms.Resize(sz_resize) if not is_train else Identity(),
            transforms.CenterCrop(sz_crop) if not is_train else Identity(),
            transforms.RandomHorizontalFlip() if is_train else Identity(),
            transforms.ToTensor(),
            ScaleIntensities(
                *intensity_scale) if intensity_scale is not None else Identity(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ])

    else:
        # ColorHeavy in the paper, adapted from one of Hinton's paper
        # in case you wonder where the param name comes from
        return transforms.Compose([
            transforms.RandomResizedCrop(sz_crop) if is_train else Identity(),
            transforms.Resize(sz_resize) if not is_train else Identity(),
            transforms.CenterCrop(sz_crop) if not is_train else Identity(),
            transforms.RandomHorizontalFlip(p=0.5) if is_train else Identity(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8) if is_train else Identity(),
            transforms.RandomGrayscale(p=0.2) if is_train else Identity(),
            transforms.ToTensor(),
            ScaleIntensities(
                *intensity_scale) if intensity_scale is not None else Identity(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ])


def make_albumentation(mode='light', sz_resize=256, sz_crop=224, is_train=True):

    if mode == 'light':
        # a replica of the standard augs used in torchvision
        return A.Compose([
            A.RandomResizedCrop(height=sz_crop, width=sz_crop) if is_train else A.NoOp(), # A.NoOp() eq to Identity() in torchvision
            ResizeV2(sz_resize) if not is_train else A.NoOp(),
            A.CenterCrop(sz_crop, sz_crop) if not is_train else A.NoOp(),
            A.HorizontalFlip(p=0.5) if is_train else A.NoOp(),
            A.Normalize(),  # a combination of ScaleIntensities and Normalization
            ToTensorV2(),
        ])

    elif mode == 'GeomColor':
        # "Geom+Color" in the paper
        # increase the rotation limit and the size of the cropped holes. Also extra brightness/contrast adjust
        # also change the sequence of the augmentations
        return A.Compose([
            A.RandomResizedCrop(height=sz_crop, width=sz_crop) if is_train else A.NoOp(),  # A.NoOp() eq to Identity()
            ResizeV2(sz_resize) if not is_train else A.NoOp(),
            A.CenterCrop(sz_crop, sz_crop) if not is_train else A.NoOp(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5) if is_train else A.NoOp(),
            A.HorizontalFlip(p=0.5) if is_train else A.NoOp(),
            A.ShiftScaleRotate(scale_limit=0, rotate_limit=45, p=0.5) if is_train else A.NoOp(),
            A.CoarseDropout(max_holes=5, min_holes=1, max_height=40, max_width=40, p=0.5) if is_train else A.NoOp(),
            A.Blur(p=0.5) if is_train else A.NoOp(), #using a random-sized (3-7) average kernel.
            A.Normalize(), # a combination of ScaleIntensities and Normalization
            ToTensorV2(),
        ])

    elif mode == 'Geom':
        # "Geom" in the paper
        # change the order. First shift and rotate, then crop, which makes it less likely results in some artifact
        # region. No color related augs are applied
        return A.Compose([
            A.ShiftScaleRotate(scale_limit=0, rotate_limit=45, p=0.5) if is_train else A.NoOp(),
            A.RandomResizedCrop(height=sz_crop, width=sz_crop) if is_train else A.NoOp(),  # A.NoOp() eq to Identity()
            ResizeV2(sz_resize) if not is_train else A.NoOp(),
            A.CenterCrop(sz_crop, sz_crop) if not is_train else A.NoOp(),
            A.HorizontalFlip(p=0.5) if is_train else A.NoOp(),
            A.CoarseDropout(max_holes=5, min_holes=1, max_height=40, max_width=40, p=0.5) if is_train else A.NoOp(),
            A.Blur(p=0.5) if is_train else A.NoOp(), #using a random-sized (3-7) average kernel.
            A.Normalize(), # a combination of ScaleIntensities and Normalization
            ToTensorV2(),
        ])

    else:
        raise ValueError("No legal albumentation is chosen.")
