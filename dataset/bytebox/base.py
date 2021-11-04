from __future__ import print_function
from __future__ import division

import torch
import torchvision.transforms.functional as F
import cv2
import os
import torchvision
import numpy as np
#from .utils import ScaleNormalize


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform=None, albumentation=None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.albumentation = albumentation
        if self.transform is not None and self.albumentation is not None:
            raise ValueError('Either transform or albumentation can ' +
                             'be enabled, but not both.')

        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)


    def __getitem__(self, index):
        # Read an image with OpenCV, otherwise albumentation won't work
        im = cv2.imread(self.im_paths[index])
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        assert im is not None, f'Failed to read: {self.im_paths[index]}'
        assert im.size, f'Error! Empty image: {self.im_paths[index]}'
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # some transforms are performed on PIL image
            im = F.to_pil_image(im)
            im = self.transform(im)
        if self.albumentation is not None:
            im = self.albumentation(image=im)['image']
            # im is tensor after albumentation. To keep the pipeline consistent, perform

            # ScaleIntensities and Normalization were moved to dataloader
            # im = ScaleNormalize(im)

        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
