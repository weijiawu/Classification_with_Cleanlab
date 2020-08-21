# coding: utf-8

# ## A cleanlab compatible PyTorch CNN classifier.
#
# ## Note to use this model you'll need to have pytorch installed
# See: https://pytorch.org/get-started/locally/

# In[ ]:


# Python 2 and 3 compatibility
from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement)
from network.efficientnet import EfficientNet
from network.models import create_model
import logging
import logging.config
# In[ ]:


# Make sure python version is compatible with pyTorch
from cleanlab.util import VersionWarning
import os
python_version = VersionWarning(
    warning_str="pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
    list_of_compatible_versions=[2.7, 3.5, 3.6, 3.7],
)

# In[ ]:


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

# In[ ]:


MNIST_TRAIN_SIZE = 50000
MNIST_TEST_SIZE = 10000

from sklearn.base import BaseEstimator


class CNN(BaseEstimator):  # Inherits sklearn classifier
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
            test_batch_size=MNIST_TEST_SIZE,
            # Set to 'test' to force fit() and predict_proba() on test_set
            # Be careful setting this, it will override every other loader
            # If you set this to 'test', but call .predict(loader = 'train')
            # then .predict() will still predict on test!
            loader=None,
            model_name = "efficientnet-b0",
            num_classes = 10,
            logger = None,
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

    def fit(self, train_idx, train_labels=None, sample_weight=None, loader='train'):
        '''This function adheres to sklearn's "fit(X, y)" format for compatibility with scikit-learn.
        ** All inputs should be numpy arrays, not pyTorch Tensors
        train_idx is not X, but instead a list of indices for X (and y if train_labels is None).
        This function is a member of the cnn class which will handle creation of X, y from
        the train_idx via the train_loader.'''
        if self.loader is not None:
            loader = self.loader
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
            sparse_labels = np.zeros(MNIST_TRAIN_SIZE if loader == 'train' else MNIST_TEST_SIZE, dtype=int) - 1
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



    def predict(self, idx=None, loader=None):
        # get the index of the max probability
        probs = self.predict_proba(idx, loader)
        return probs.argmax(axis=1)

    def predict_proba(self, idx=None, loader=None):
        if self.loader is not None:
            loader = self.loader
        if loader is None:
            is_test_idx = idx is not None and len(idx) == MNIST_TEST_SIZE and np.all(
                np.array(idx) == np.arange(MNIST_TEST_SIZE))
            loader = 'test' if is_test_idx else 'train'
        if loader == 'train':
            valdir = os.path.join(self.root, 'train')
            dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]),
            )

        else:
            self.logger.info("real validation")
            valdir = os.path.join(self.root, 'test')
            dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]),
            )
        # Filter by idx
        # if idx is not None:
        #     if (loader == 'train' and len(idx) != MNIST_TRAIN_SIZE) or (
        #             loader == 'test' and len(idx) != MNIST_TEST_SIZE):
        #         # dataset.imgs = dataset.imgs[idx]
        #         # dataset.class_to_idx = dataset.class_to_idx[idx]
        #         # dataset.classes = dataset.classes[idx]
        #
        #         dataset.imgs = [dataset.imgs[i] for i in idx]
        #         dataset.class_to_idx = [dataset.class_to_idx[i] for i in idx]
        #         dataset.classes = [dataset.classes[i] for i in idx]
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if loader == 'train' else self.test_batch_size,
            **self.loader_kwargs
        )

        # sets model.train(False) inactivating dropout and batch-norm layers
        self.model.eval()

        # Run forward pass on model to compute outputs
        outputs = []
        correct = 0
        total = 0
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
        outputs = torch.cat(outputs, dim=0)
        # Convert to probabilities and return the numpy array of shape N x K
        out = outputs.cpu().numpy() if self.cuda else outputs.numpy()
        pred = np.exp(out)

        return pred

if __name__ =="__main__":
    import os
    # Now you can use your model with `cleanlab`. Here's one example:
    from cleanlab.classification import LearningWithNoisyLabels
    lnl = LearningWithNoisyLabels(clf=CNN("/data/glusterfs_cv_04/11121171/data/CIFAR/CIFAR10-noise"),n_jobs=1,cv_n_folds=5,prune_method="prune_by_class")
    train_data = np.array(list(range(50000)))
    train_labels = np.array([label for img, label in
              datasets.ImageFolder(os.path.join("/data/glusterfs_cv_04/11121171/data/CIFAR/CIFAR10-noise", "train")).imgs])
    # train_labels = decode_idx1_ubyte("/data/glusterfs_cv_04/11121171/data/MINIST/train-labels.idx1-ubyte")
    model = lnl.fit(train_data,train_labels)
    lnl.predict_proba(idx=np.array(list(range(10000))),loader="test")