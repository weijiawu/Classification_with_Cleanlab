import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from cleanlab.util import VersionWarning
from lib.utils import AverageMeter,accuracy,adjust_learning_rate
import time
python_version = VersionWarning(
    warning_str="pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
    list_of_compatible_versions=[2.7, 3.5, 3.6, 3.7],
)
import numpy as np
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
import argparse
import os
import random
import torch.backends.cudnn as cudnn
from datasets.dataloader import Dateloader

import torch.optim as optim
import logging
import logging.config
import os
import json
from datasets.cifar10_noise import get_datasets
from Cleanlab.cifar10 import CNN
# Now you can use your model with `cleanlab`. Here's one example:
from cleanlab.classification import LearningWithNoisyLabels
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='classification')

# Model path
parser.add_argument('--exp_name', help='Where to store logs and models')
parser.add_argument('--resume', default="/data/glusterfs_cv_04/11121171/AAAI_EAST/Baseline/EAST_v1/model_save/model_epoch_826.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--data_path', default="/data/glusterfs_cv_04/11121171/data/CIFAR/CIFAR10-noise", type=str,
                    help='the test image of target domain ')
parser.add_argument('--workspace', default="/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/", type=str,
                    help='save model')
parser.add_argument('--Backbone', type=str, default="efficientnet-b1", help='FeatureExtraction stage. '
                                                                     'ResNet18|ResNet34|ResNet50'
                                                                     'MobileNet_v1|MobileNet_v2|Mobilenetv3'
                                                                     'vgg11|vgg16|vgg19'
                                                                     'efficientnet-b0|efficientnet-b1'
                                                                     'shufflenet_v2_x0_5|shufflenet_v2_x1_0|shufflenet_v2_x1_5'
                                                                      "inception_v3"
                                                                      "mnasnet0_5|"
                                                                      "densenet121"
                                                                      "ResNeXt29_32x4d|ResNeXt29_2x64d"
                                                                        )
parser.add_argument('--Datasets', type=str, default="cifar10", help=' ImageNet|Clothing|CIFAR10|CIFAR100')
parser.add_argument('--num_classes', type=int, default=10, help=' classification')
parser.add_argument('--rate', type=float, default=0.4, help='noise ratio')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')


# Training strategy
parser.add_argument('--epoch_iter', default=250, type = int,
                    help='the max epoch iter')
parser.add_argument('--batch_size', default=256, type = int,
                    help='batch size of training')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=10, type=int,
                    help='Number of workers used in dataloading')

opt = parser.parse_args()


def train(opt,logger):
    model = CNN(data_root="/data/glusterfs_cv_04/11121171/data/CIFAR/CIFAR10-noise",
                batch_size=opt.batch_size,epochs=opt.epoch_iter,model_name=opt.Backbone,logger=logger)
    # prune_by_class

    # prune_by_noise_rate
    lnl = LearningWithNoisyLabels(clf=model,n_jobs=1,cv_n_folds=5,prune_method="both")

    train_data = np.array(list(range(50000)))
    # train_labels = np.array([label for img, label in
    #           datasets.ImageFolder(os.path.join("/data/glusterfs_cv_04/11121171/data/CIFAR/CIFAR10-noise", "train")).imgs])

    # noise_file = '%s/%s/%.1f_%s.json' % (opt.workspace,f'workspace/{opt.exp_name}', opt.rate, opt.noise_mode)
    noise_file = "/data/glusterfs_cv_04/11121171/AAAI_NL/cleanlab/examples/cifar10/workspace/noise_label.json"
    noise_label = json.load(open(noise_file, "r"))


    lnl1_1 = lnl.fit(train_data,np.array(noise_label))

    lnl.predict_proba(idx=np.array(list(range(10000))),loader="test")

    dataset_val = get_datasets(opt.data_path, opt.rate, noise_mode="sym", dataset='cifar10', mode="test")
    # dataset_val = Dateloader(opt.data_path,mode="test",noise_mode=opt.noise_mode, rate=opt.rate,dataset=opt.Datasets)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)

    validate(data_loader_val,lnl1_1.model,logger)



def validate(val_loader, model,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    logger.info("    ---------------------------------------------------------------")
    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg
if __name__ == '__main__':


    if not opt.exp_name:
        opt.exp_name = f'{opt.Backbone}-{opt.Datasets}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/{opt.exp_name}', exist_ok=True)

    # 通过下面的方式进行简单配置输出方式与日志级别
    logging.basicConfig(
        filename=os.path.join(f'/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/{opt.exp_name}',"logger_noise.log"),
        level=logging.INFO,filemode='w')

    logging.debug('debug message')
    logging.info('info message')
    logging.error('error message')
    logging.critical('critical message')
    logger = logging.getLogger('project')

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()


    if opt.num_gpu > 1:
        logger.info('------ Use multi-GPU setting ------')
        logger.info('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.num_workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

    train(opt,logger)


