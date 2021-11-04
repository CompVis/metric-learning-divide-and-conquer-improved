from __future__ import print_function
from __future__ import division

import logging
import numpy as np


class ClassBalancedSampler(object):
    """
    Sampler that generates class balanced indices with classes chosen randomly.
    For example, choosing batch_size = 32 and nun_samples_per_class = 8
    will result in
    32 indices, which point to 8 samples from 32/8=4 randomly picked classes.

    If number of classes is not enough - some classes would be sampled several times.
    """

    def __init__(self, dataset, batch_size=70, num_samples_per_class=5):
        logging.debug('->>>Create ClassBalancedSampler batch_size={}, s_per_class={}'\
                      .format(batch_size, num_samples_per_class))

        assert batch_size % num_samples_per_class == 0, \
            "batch size must be divisable by num_samples_per_class"
        self.targets = np.array(dataset.ys)
        self.C = list(set(self.targets))
        self.C_index = {
            c: np.where(self.targets == c)[0] for c in self.C}
        for c in self.C:
            np.random.shuffle(self.C_index[c])
        self.C_count = {c: 0 for c in self.C}
        self.count = 0
        self.num_classes = batch_size // num_samples_per_class
        self.num_samples_per_class = num_samples_per_class
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        is_not_enough_classes = len(self.C) < self.num_classes
        if is_not_enough_classes:
            logging.warn(('Not enough classes to sample batches: have={},'
                         'required={}').format(len(self.C), self.num_classes))
        while self.count + self.batch_size < len(self.dataset):
            C = np.random.choice(self.C, self.num_classes, replace=is_not_enough_classes)
            indices = []
            for class_ in C:
                if self.C_count[class_] + self.num_samples_per_class\
                   > len( self.C_index[class_]):
                    indices.extend(
                        np.random.choice(self.C_index[class_],
                                         self.num_samples_per_class,
                                         replace=len(self.C_index[class_]) < self.num_samples_per_class))
                else:
                    indices.extend(
                        self.C_index[class_][
                            self.C_count[class_]:self.C_count[
                                class_] + self.num_samples_per_class])
                self.C_count[class_] += self.num_samples_per_class
                if self.C_count[class_] >= len( self.C_index[class_]):
                    np.random.shuffle(self.C_index[class_])
                    self.C_count[class_] = 0
            assert self.count % self.batch_size == 0, self.count
            assert len(indices) == self.batch_size
            yield indices
            self.count += self.num_classes * self.num_samples_per_class

    def __len__(self):
        return len(self.dataset) // self.batch_size


class AdaptiveClassBalancedSampler(object):
    """
    Sampler that generates class balanced indices with classes chosen randomly.
    For example, choosing batch_size = 32 and nun_samples_per_class = 8
    will result in
    32 indices, which point to 8 samples from 32/8=4 randomly picked classes.

    If number of classes is not enough - it will sample more images per every class

    Args:
        allow_smaller_batches: if True:
                                    if the class size is small allow to use < num_samples_per_class.
                                    This avoids repetition of the samples in the batch.
                               if False (default):
                                    sample from small classes with repetitions
    """

    def __init__(self, dataset, batch_size=70,
                 num_samples_per_class=5,
                 num_replicas=1,
                 small_class_action="duplicate"):
        logging.debug('->>>Create ClassBalancedSampler(small_class_action={}) batch_size={}, s_per_class={}, num_replicas={}'\
                      .format(small_class_action, batch_size, num_samples_per_class, num_replicas))
        assert batch_size % num_replicas == 0, \
            "batch size must be divisable by num_replicas"

        batch_size_wo_replicas = batch_size // num_replicas
        assert batch_size_wo_replicas % num_samples_per_class == 0, \
            "batch size must be divisable by num_samples_per_class"
        self.targets = np.array(dataset.ys)
        self.C = list(set(self.targets))
        self.C_index = {
            c: np.where(self.targets == c)[0] for c in self.C}
        for c in self.C:
            np.random.shuffle(self.C_index[c])
        self.C_count = {c: 0 for c in self.C}
        self.count = 0
        self.num_classes = None
        self.min_num_samples_per_class = num_samples_per_class
        self.num_samples_per_class = self.min_num_samples_per_class
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.batch_size_wo_replicas = batch_size_wo_replicas
        self.batch_size = batch_size
        self.small_class_action = small_class_action
        if small_class_action not in ['duplicate', 'sample_other', 'smaller_batch']:
            raise ValueError(f'Unknown small_class_action: {small_class_action}')
        self.small_class_action = small_class_action


    def __iter__(self):
        self.num_samples_per_class = self.min_num_samples_per_class
        self.num_classes = self.batch_size_wo_replicas // self.num_samples_per_class

        self.count = 0
        is_not_enough_classes = len(self.C) < self.num_classes
        if is_not_enough_classes:
            self.num_samples_per_class = self.batch_size_wo_replicas // len(self.C)
            logging.warn(('Not enough classes to sample batches: have={},'
                         'required={}. Will sample >= {} sampels per class').format(len(self.C), self.num_classes, self.num_samples_per_class))
            self.num_classes = len(self.C)
        assert self.num_classes > 1, 'Cannot sample batches with {} classes!'.format(self.num_classes)
        # TODO: implement allow_smaller_batches when class is too small
        while self.count + self.batch_size_wo_replicas <= len(self.dataset):
            # Select which classes would be sampled twise in this batch.
            # Set slightly_larger_classes will be non-empty only when batch_size_wo_replicas is not divisible by the number of classes
            num_larger_classes = self.batch_size_wo_replicas - self.num_samples_per_class * self.num_classes
            assert num_larger_classes < self.num_classes, '{} < {}'.format(num_larger_classes, self.num_classes)
            slightly_larger_classes = set(np.random.choice(self.C, num_larger_classes, replace=False))

            C = np.random.choice(self.C, self.num_classes, replace=False)
            indices = []
            for class_ in C:
                cur_class_num_samples_to_take = self.num_samples_per_class + int(class_ in slightly_larger_classes)
                if self.small_class_action in ['smaller_batch', 'sample_other']:
                    cur_class_num_samples_to_take = min(cur_class_num_samples_to_take,
                                                        len(self.C_index[class_]))
                else:
                    assert self.small_class_action == 'duplicate'
                if self.C_count[class_] + cur_class_num_samples_to_take\
                   > len(self.C_index[class_]):
                    indices.extend(
                        np.random.choice(self.C_index[class_],
                                         cur_class_num_samples_to_take,
                                         replace=len(self.C_index[class_]) < cur_class_num_samples_to_take))
                else:
                    indices.extend(
                        self.C_index[class_][
                            self.C_count[class_]:self.C_count[class_] + cur_class_num_samples_to_take
                        ])
                self.C_count[class_] += cur_class_num_samples_to_take
                if self.C_count[class_] >= len(self.C_index[class_]):
                    np.random.shuffle(self.C_index[class_])
                    self.C_count[class_] = 0
                self.count += cur_class_num_samples_to_take
            if self.small_class_action == 'sample_other' and len(indices) < self.batch_size_wo_replicas:
                # randomly pick from the images which are not in the batch yet
                num_extra_to_take = self.batch_size_wo_replicas - len(indices)
                allowed_to_select = set(np.arange(len(self.targets))) - set(indices)
                allowed_to_select = list(allowed_to_select)
                indices.extend(
                    np.random.choice(allowed_to_select, num_extra_to_take,
                                     replace=len(allowed_to_select) < num_extra_to_take)
                )
                self.count += num_extra_to_take

            assert self.small_class_action == 'smaller_batch' or (self.count % self.batch_size_wo_replicas == 0), self.count
            assert self.small_class_action == 'smaller_batch' or len(indices) == self.batch_size_wo_replicas
            if len(indices) < self.batch_size_wo_replicas:
                logging.debug('Current batch is small ({} < {})'.format(len(indices), self.batch_size_wo_replicas))
            # Repeat the batch num_replicas times
            yield np.tile(np.array(indices), reps=self.num_replicas)

    def __len__(self):
        return len(self.dataset) // self.batch_size


class RandomBatchSampler(object):
    """
    Sampler that generates random batches with at least 2 classes
    """

    def __init__(self, dataset, batch_size=70, min_num_classes_per_batch=2, num_replicas=1):
        logging.debug('->>>Create RandombatchSampler batch_size={}, num_replicas={}'\
                      .format(batch_size, num_replicas))

        assert batch_size % num_replicas == 0, \
            "batch size must be divisible by num_replicas"

        batch_size_wo_replicas = batch_size // num_replicas

        self.targets = np.array(dataset.ys)
        assert min_num_classes_per_batch > 1
        if len(np.unique(self.targets)) < min_num_classes_per_batch:
            logging.error(f'Dataset must contain > {min_num_classes_per_batch} classes. [dataset size={len(self.targets)}]')
            raise ValueError()
        self.permutation = np.random.permutation(len(self.targets))
        self.count = 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.batch_size_wo_replicas = batch_size_wo_replicas
        self.batch_size = batch_size
        self.min_num_classes_per_batch = min_num_classes_per_batch


    def __iter__(self):
        self.count = 0
        self.permutation = np.random.permutation(len(self.targets))
        num_batches = 0

        while num_batches < self.__len__():
            if self.count + self.batch_size_wo_replicas >= len(self.dataset):
                self.permutation = np.random.permutation(len(self.targets))
                self.count = 0

            indices = self.permutation[self.count:self.count + self.batch_size_wo_replicas]
            self.count += len(indices)
            if len(np.unique(self.targets[indices])) < self.min_num_classes_per_batch:
                continue

            assert self.count % self.batch_size_wo_replicas == 0, self.count
            assert len(indices) == self.batch_size_wo_replicas
            # Repeat the batch num_replicas times
            yield np.tile(np.array(indices), reps=self.num_replicas)
            num_batches += 1

    def __len__(self):
        return len(self.dataset) // self.batch_size
