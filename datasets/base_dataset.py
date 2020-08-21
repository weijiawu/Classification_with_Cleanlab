import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from datasets.augment import Augmentation
import os
def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w == tw and h == th:
        return imgs


    return imgs


class BaseDataset(data.Dataset):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None
    def __init__(self, root,mode,target_size=256, viz=False, debug=False,dataset = "ImageNet"):
        super(BaseDataset, self).__init__()
        self.mode = mode
        self.target_size = target_size
        self.viz = viz
        self.debug = debug
        self.dataset = dataset


        self.transform_train , self.transform_test = Augmentation(dataset)

    def load_image_gt(self, index):
        '''
        根据索引值返回图像、label
        :param index:
        :return:
        '''
        return None, None


    def pull_item(self, index):

        image_path, gt = self.load_image_gt(index)

        if self.dataset == "ImageNet":
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.fromarray(image_path)

        if self.mode == "train":
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)

        return image,gt

