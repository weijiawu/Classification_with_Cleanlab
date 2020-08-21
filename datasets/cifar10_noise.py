
from torchvision import datasets, transforms
import os
import random
import json
import numpy as np
transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise
def get_datasets(traindir,rate,noise_mode="sym",dataset = 'cifar10',noise_file="",right_file="",mode="train"):
    valdir = os.path.join(traindir, 'test')
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]),
    )
    if mode=="test":
        return val_dataset

    traindir = os.path.join(traindir, "train")
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]),
        )

    right_labels = [label for img, label in train_dataset]

    if os.path.exists(noise_file):
        noise_label = np.array(json.load(open(noise_file, "r")))
    else:
        noise_label = []
        idx = list(range(50000))
        random.shuffle(idx)
        num_noise = int(rate * 50000)
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
                    noiselabel = transition[right_labels[i]]
                    noise_label.append(noiselabel)
            else:
                noise_label.append(right_labels[i])
        # if training labels are provided use those instead of dataset labels
        json.dump(noise_label, open(noise_file, "w"))
        json.dump(right_labels, open(right_file, "w"))

    if noise_label is not None:
        # with open(args.train_labels, 'r') as rf:
        #     train_labels_dict = json.load(rf)
        # print(train_dataset.imgs[0])
        train_dataset.imgs = [(fn, noise_label[i]) for i,(fn, _) in
                              enumerate(train_dataset.imgs)]
        train_dataset.samples = train_dataset.imgs
        # print("load noise label")
    return train_dataset