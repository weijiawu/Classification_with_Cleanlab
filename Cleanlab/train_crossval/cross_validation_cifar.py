# -*- coding: utf-8 -*-
# @Time    : 13/8/20 11:03 AM
# @Author  : weijiawu
from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement)
from network.efficientnet import EfficientNet
from network.models import create_model
import logging
import logging.config
import copy
from sklearn.model_selection import StratifiedKFold
from cleanlab.util import (
    value_counts, clip_values, clip_noise_rates, round_preserving_row_totals,
    assert_inputs_are_valid,
)
from sklearn.linear_model import LogisticRegression as LogReg
# Make sure python version is compatible with pyTorch
from cleanlab.util import VersionWarning
import os
python_version = VersionWarning(
    warning_str="pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
    list_of_compatible_versions=[2.7, 3.5, 3.6, 3.7],
)
if python_version.is_compatible():  # pragma: no cover
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    from torch.utils.data.sampler import SubsetRandomSampler
    import numpy as np
    import random
    import json
from sklearn.base import BaseEstimator
class cross_val(BaseEstimator):  # Inherits sklearn classifier
    '''Wraps a PyTorch CNN for the MNIST dataset within an sklearn template by defining
    .fit(), .predict(), and .predict_proba() functions. This template enables the PyTorch
    CNN to flexibly be used within the sklearn architecture -- meaning it can be passed into
    functions like cross_val_predict as if it were an sklearn model. The cleanlab library
    requires that all models adhere to this basic sklearn template and thus, this class allows
    a PyTorch CNN to be used in for learning with noisy labels among other things.'''

    def __init__(
            self,
            data_root = "",
            batch_size=128,
            epochs=6,
            log_interval=50,  # Set to None to not print
            lr=0.01,
            momentum=0.5,
            no_cuda=False,
            seed=1,
            test_batch_size=100,
            loader=None,
            model_name = "efficientnet-b0",
            num_classes = 10,
            logger = None,
            MNIST_TRAIN_SIZE=50000
    ):
        self.root = data_root
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.model_name = model_name
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        self.logger = logger
        self.MNIST_TRAIN_SIZE = MNIST_TRAIN_SIZE
        torch.manual_seed(self.seed)
        if self.cuda:  # pragma: no cover
            torch.cuda.manual_seed(self.seed)

        # Instantiate PyTorch model
        self.logger.info('| Building net...')
        self.model = create_model(model_name, num_classes)
        self.model = torch.nn.DataParallel(self.model)

        # self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10).cuda()
        if self.cuda:  # pragma: no cover
            self.model.cuda()

        self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.loader = loader

    def fit(self, train_idx, train_labels=None, sample_weight=None):
        '''This function adheres to sklearn's "fit(X, y)" format for compatibility with scikit-learn.
        ** All inputs should be numpy arrays, not pyTorch Tensors
        train_idx is not X, but instead a list of indices for X (and y if train_labels is None).
        This function is a member of the cnn class which will handle creation of X, y from
        the train_idx via the train_loader.'''

        if train_labels is not None and len(train_idx) != len(train_labels):
            raise ValueError("Check that train_idx and train_labels are the same length.")

        if sample_weight is not None:  # pragma: no cover
            if len(sample_weight) != len(train_labels):
                raise ValueError("Check that train_labels and sample_weight are the same length.")
            class_weight = sample_weight[np.unique(train_labels, return_index=True)[1]]
            class_weight = torch.from_numpy(class_weight).float()
            if self.cuda:
                class_weight = class_weight.cuda()
        else:
            class_weight = None

        # Data loading code
        traindir = os.path.join(self.root, 'train')

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


        # Use provided labels if not None, o.w. use MNIST dataset training labels
        if train_labels is not None:
            # Create sparse tensor of train_labels with (-1)s for labels not in train_idx.
            # We avoid train_data[idx] because train_data may very large, i.e. image_net
            sparse_labels = np.zeros(self.MNIST_TRAIN_SIZE, dtype=int) - 1
            sparse_labels[train_idx] = train_labels
            train_dataset.targets = sparse_labels


        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            #             sampler=SubsetRandomSampler(train_idx if train_idx is not None else range(MNIST_TRAIN_SIZE)),
            sampler=SubsetRandomSampler(train_idx),
            batch_size=self.batch_size,
            **self.loader_kwargs
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        CEloss = nn.CrossEntropyLoss()
        # Train for self.epochs epochs
        for epoch in range(1, self.epochs + 1):

            # Enable dropout and batch norm layers
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:  # pragma: no cover
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.model(data)
                loss = CEloss(output, target)
                loss.backward()
                optimizer.step()
                if self.log_interval is not None and batch_idx % self.log_interval == 0:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_idx),
                               100. * batch_idx / len(train_loader), loss.item()))



    def predict(self, idx=None):
        # get the index of the max probability
        probs = self.predict_proba(idx)
        return probs.argmax(axis=1)


    def predict_proba(self, idx=None):

        valdir = os.path.join(self.root, 'train')
        holdout_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]),
        )

        holdout_dataset.imgs = [holdout_dataset.imgs[i] for i in idx]
        holdout_dataset.samples = holdout_dataset.imgs



        loader = torch.utils.data.DataLoader(
            dataset=holdout_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            **self.loader_kwargs
        )

        # sets model.train(False) inactivating dropout and batch-norm layers
        self.model.eval()

        # Run forward pass on model to compute outputs
        outputs = []
        correct = 0
        total = 0
        self.logger.info("test sample: "+str(len(loader)))
        for data, targets in loader:
            if self.cuda:  # pragma: no cover
                data = data.cuda()
                targets = targets.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = self.model(data)
                _,pred = torch.max(output,1)

                total += targets.size(0)
                correct += pred.eq(targets).cpu().sum().item()
            outputs.append(output)
        acc = 100. * correct / total
        self.logger.info("\n| Validation\t Net  Acc: %.2f%%" % acc)
        # Outputs are log_softmax (log probabilities)
        # outputs = torch.cat(outputs, dim=0)
        # Convert to probabilities and return the numpy array of shape N x K
        # out = outputs.cpu().numpy() if self.cuda else outputs.numpy()
        # pred = np.exp(out)

        probs = np.concatenate([
            torch.nn.functional.softmax(z, dim=1).cpu().numpy() for z in outputs
        ])


        return probs



def cross_val_pred_proba(
    X,
    s,
    clf=LogReg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    thresholds=None,
    seed=None,
    calibrate=True,
):

    assert_inputs_are_valid(X, s)
    # Number of classes
    K = len(np.unique(s))

    # Ensure labels are of type np.array()
    s = np.asarray(s)

    # Create cross-validation object for out-of-sample predicted probabilities.
    # CV folds preserve the fraction of noisy positive and
    # noisy negative examples in each class.
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=seed)

    # Intialize psx array
    psx = np.zeros((len(s), K))

    # Split X and s into "cv_n_folds" stratified folds.
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X, s)):

        clf_copy = copy.deepcopy(clf)

        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv = X[cv_train_idx], X[cv_holdout_idx]
        s_train_cv, s_holdout_cv = s[cv_train_idx], s[cv_holdout_idx]

        # Fit the clf classifier to the training set and
        # predict on the holdout set and update psx.
        clf_copy.fit(X_train_cv, s_train_cv)
        psx_cv = clf_copy.predict_proba(X_holdout_cv)  # P(s = k|x) # [:,1]
        psx[cv_holdout_idx] = psx_cv[:, :K]

    return psx



transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

def generate_noise(data,rate, noise_mode,noise_file="",right_file=""):
    right_labels = [label for img, label in
              datasets.ImageFolder(os.path.join(data, "train/")).imgs]


    noise_label = []
    idx = list(range(50000))
    random.shuffle(idx)
    num_noise = int(rate * 50000)
    noise_idx = idx[:num_noise]

    for i in range(50000):
        if i in noise_idx:
            if noise_mode == 'sym':
                noiselabel = random.randint(0, 9)
                noise_label.append(noiselabel)
            elif noise_mode == 'asym':
                noiselabel = transition[right_labels[i]]
                noise_label.append(noiselabel)
        else:
            noise_label.append(right_labels[i])

    json.dump(noise_label, open(noise_file, "w"))
    json.dump(right_labels, open(right_file, "w"))



