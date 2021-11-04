from .base import *
import numpy as np

class SOProducts(BaseDataset):
    nb_train_all = 59551
    nb_test_all = 60502
    def __init__(self, root, classes, transform=None, albumentation=None):
        BaseDataset.__init__(self, root, classes, transform, albumentation)

        classes_train = range(0, 11318)
        classes_test = range(11318, 22634)
        self.super_class = []
        
        if classes.start in classes_train:
            if classes.stop - 1 in classes_train:
                train = True

        if classes.start in classes_test:
            if classes.stop - 1 in classes_test:
                train = False

        with open(
            os.path.join(
            root,
            'Ebay_{}.txt'.format('train' if train else 'test')
            )
        ) as f:

            f.readline()
            index = 0
            nb_images = 0

            for (image_id, class_id, _, path) in map(str.split, f):
                nb_images += 1
                if int(class_id) - 1 in classes:
                    self.im_paths.append(os.path.join(root, path))
                    self.super_class.append(self.im_paths[-1].split('sop/')[1].split('/')[0])
                    self.ys.append(int(class_id) - 1)
                    self.I += [index]
                    index += 1
            # create a hash map corresponding super class name to an int        
            name_to_int = {name: value for (value, name) in enumerate(np.unique(self.super_class))}
            self.super_class = [name_to_int[x] for x in self.super_class]
            
            if train:
                assert nb_images == type(self).nb_train_all
            else:
                assert nb_images == type(self).nb_test_all
