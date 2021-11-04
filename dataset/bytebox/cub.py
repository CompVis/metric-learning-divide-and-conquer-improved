from .base import *


class CUBirds(BaseDataset):
    def __init__(self, root, classes, transform=None, albumentation=None):
        BaseDataset.__init__(self, root, classes, transform, albumentation)

        nb_images = 0
        index = 0
        image_folder = torchvision.datasets.ImageFolder(root=root)
        self.class_to_idx = image_folder.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        # ImageFolder.imgs: List of (image path, class_index) tuples
        for i in image_folder.imgs:
            # i[1]: label, i[0]: image_path
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if fn[:2] != '._':
                nb_images += 1
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(root, i[0]))
                index += 1

        assert np.array_equal(np.asarray(self.I), np.arange(len(self.ys)))

        if '2011' in root:
            assert nb_images == 11788
        elif '2010' in root:
            assert nb_images == 6033

