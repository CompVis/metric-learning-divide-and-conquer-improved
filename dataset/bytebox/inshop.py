import torch
import os

import torchvision.transforms.functional as F
import cv2

class InShop(torch.utils.data.Dataset):
    """
    For the In-Shop Clothes Retrieval dataset, we use the predefined
    25, 882 training images of 3,997 classes for training. The test
    set is partitioned into a query set (14,218 images of 3,985 classes)
    and a gallery set (12, 612 images of 3, 985 classes)
    """
    def __init__(self, root, classes = None, dset_type = 'train',
            transform = None, albumentation = None):
        with open(
            os.path.join(
                root, 'Eval/list_eval_partition.txt'
            ), 'r'
        ) as f:
            lines = f.readlines()

        self.transform = transform
        self.albumentation = albumentation

        if self.transform is not None and self.albumentation is not None:
            raise ValueError('Either transform or albumentation can ' +
                             'be enabled, but not both.')
        # store for using later '__getitem__'
        self.dset_type = dset_type

        nb_samples = int(lines[0].strip('\n'))
        assert nb_samples == 52712

        torch.utils.data.Dataset.__init__(self)
        self.im_paths = {'train': [], 'query': [], 'gallery': []}
        self.ys = {'train': [], 'query': [], 'gallery': []}
        I_ = {'train': 0, 'query': 0, 'gallery': 0}
        self.I = {'train': [], 'query': [], 'gallery': []}
        # start from second line, since 0th and 1st contain meta-data
        for line in lines[2:]:
            im_path, im_id, eval_type = [
                l for l in line.split(' ') if l != '' and l != '\n']
            y = int(im_id.split('_')[1])
            self.im_paths[eval_type] += [os.path.join(root, im_path)]
            self.ys[eval_type] += [y]
            self.I[eval_type] += [I_[eval_type]]
            I_[eval_type] += 1

        nb_samples_counted = len(self.im_paths['train']) + \
                len(self.im_paths['gallery']) + len(self.im_paths['query'])
        assert nb_samples_counted == nb_samples

        # verify that labels are sorted for next step
        self.ys['query'] == sorted(self.ys['query'])
        self.ys['gallery'] == sorted(self.ys['gallery'])

        assert len(self.ys['train']) == 25882
        assert len(self.ys['query']) == 14218
        assert len(self.ys['gallery']) == 12612

        # verify that query and gallery have same labels
        assert set(self.ys['query']) == set(self.ys['gallery'])

        # labels of query and gallery are like [1, 1, 7, 7, 8, 11, ...]
        # condense them such that ordered without spaces,
        # i.e. 1 -> 1, 7 -> 2, ...
        idx_to_class = {idx: i for i, idx in enumerate(
            sorted(set(self.ys['query']))
        )}
        for _type in ['query', 'gallery']:
            self.ys[_type] = list(
                map(lambda x: idx_to_class[x], self.ys[_type]))

        # same thing for train labels
        idx_to_class = {idx: i for i, idx in enumerate(
            sorted(set(self.ys['train']))
        )}
        self.ys['train'] = list(
            map(lambda x: idx_to_class[x], self.ys['train']))

        # should be 3997 classes for training, 3985 for query/gallery
        assert len(set(self.ys['train'])) == 3997
        assert len(set(self.ys['query'])) == 3985
        assert len(set(self.ys['gallery'])) == 3985

        self.im_paths = self.im_paths[dset_type]
        self.ys = self.ys[dset_type]
        self.I = self.I[dset_type]

    def __len__(self):
        return len(self.ys)

    def nb_classes(self):
        return len(set(self.ys))

    def __getitem__(self, index):
        # Read an image with OpenCV, otherwise albumentation won't work
        im = cv2.imread(self.im_paths[index])
        # By default OpenCV uses BGR color space for color images,
        # need to convert the image to RGB color space.
        assert im is not None, f'Failed to read: {self.im_paths[index]}'
        assert im.size, f'Error! Empty image: {self.im_paths[index]}'
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            im = F.to_pil_image(im)
            im = self.transform(im)
        if self.albumentation is not None:
            im = self.albumentation(image=im)['image']
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [
            self.im_paths[i] for i in I]

