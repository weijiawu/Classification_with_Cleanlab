import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import random
from  datasets.base_dataset import BaseDataset

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

class Dateloader(BaseDataset):

    def __init__(self, root,mode,noise_mode="asym",rate=0.6,target_size=256,noise_file='',right_file='', viz=False, debug=False,dataset = "ImageNet"):
        super(Dateloader, self).__init__(target_size, viz, debug)
        self.mode = mode
        self.rate = rate  # noise ratio
        self.root = root
        self.dataset = dataset
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise
        self.IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
        self.images = []
        self.gt = []

        if self.dataset == "ImageNet":
            if self.mode == "train":
                self.root = os.path.join(self.root,"train")
            else:
                self.root = os.path.join(self.root, "val")
            self.classes, self.class_to_idx = self.find_classes(self.root)
            self.images, self.gt = self.make_dataset(self.root, self.class_to_idx, extensions=self.IMG_EXTENSIONS)

        elif self.dataset == "cifar10":
            if self.mode == "train":
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (self.root, n)
                    data_dic = unpickle(dpath)
                    self.images.append(data_dic['data'])
                    self.gt = self.gt + data_dic['labels']
                self.images = np.concatenate(self.images)

                self.images = self.images.reshape((50000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))

                if noise_mode!=None:
                    if os.path.exists(noise_file):
                        noise_label = json.load(open(noise_file, "r"))
                    else:  # inject noise
                        noise_label = []
                        idx = list(range(50000))
                        random.shuffle(idx)
                        num_noise = int(self.rate * 50000)
                        noise_idx = idx[:num_noise]
                        for i in range(50000):
                            if i in noise_idx:
                                if noise_mode == 'sym':
                                    if dataset == 'cifar10':
                                        noiselabel = random.randint(0, 9)
                                    elif dataset == 'cifar100':
                                        noiselabel = random.randint(0, 99)
                                    noise_label.append(noiselabel)
                                elif noise_mode == 'asym':
                                    noiselabel = self.transition[self.gt[i]]
                                    noise_label.append(noiselabel)
                            else:
                                noise_label.append(self.gt[i])
                        print("save noisy labels to %s ..." % noise_file)

                        json.dump(noise_label, open(noise_file, "w"))
                        json.dump(self.gt, open(right_file, "w"))
                        # json.dump(self.gt
                        #         , open("/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/right.json", "w"))

                    self.gt = noise_label

            else:
                test_dic = unpickle('%s/test_batch' % self.root)
                self.images = test_dic['data']
                self.images = self.images.reshape((10000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.gt = test_dic['labels']
        else:
            print("improper dataset")

    def make_dataset(self,root, class_to_idx, extensions):
        """Make dataset by walking all images under a root.

        Args:
            root (string): root directory of folders
            class_to_idx (dict): the map from class name to class idx
            extensions (tuple): allowed extensions

        Returns:
            images (list): a list of tuple where each element is (image, label)
        """
        images = []
        gt = []
        root = os.path.expanduser(root)
        for class_name in sorted(os.listdir(root)):
            _dir = os.path.join(root, class_name)
            if not os.path.isdir(_dir):
                continue

            for _, _, fns in sorted(os.walk(_dir)):
                num = 0
                for fn in sorted(fns):
                    num = num+1
                    if has_file_allowed_extension(fn, extensions):
                        path = os.path.join(root,class_name, fn)
                        images.append(path)
                        gt.append(class_to_idx[class_name])
                    if num>300:
                        break
                break
        return images,gt

    def find_classes(self, root):
        """Find classes by folders under a root.

        Args:
            root (string): root directory of folders

        Returns:
            classes (list): a list of class names
            class_to_idx (dict): the map from class name to class idx
        """
        classes = [
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images)

    # def get_imagename(self, index):
    #     return self.image[index][0]

    def load_image_gt(self, index):
        '''
        根据索引加载ground truth
        '''
        image = self.images[index]
        gt = self.gt[index]

        return image, gt

if __name__ == "__main__":
    import torch
    from torch.utils import data
    from torch import nn
    from torch.optim import lr_scheduler
    import os
    import time
    import numpy as np
    dataset = Dateloader("/data/glusterfs_cv_04/public_data/imagenet/CLS-LOC/train", mode="train", dataset="ImageNet")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        pin_memory=True)

    for i, (image,gt) in enumerate(data_loader):
        print(image)