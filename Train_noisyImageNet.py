import torch
from cleanlab.util import VersionWarning
from lib.utils import AverageMeter,accuracy
import time
from network.models import create_model
from torch import nn
python_version = VersionWarning(
    warning_str="pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
    list_of_compatible_versions=[2.7, 3.5, 3.6, 3.7],
)
import numpy as np
if python_version.is_compatible():  # pragma: no cover
    import argparse
    import torch
    import numpy as np

from cleanlab.latent_estimation import (
    estimate_py_noise_matrices_and_cv_pred_proba,
)
from cleanlab.pruning import get_noise_indices
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import argparse
import random
import torch.optim as optim
import torch.backends.cudnn as cudnn

import logging.config
import os
from datasets.cifar10_noise import get_datasets
import json
from Cleanlab.train_crossval.cross_validation_cifar import cross_val,generate_noise,cross_val_pred_proba
from Cleanlab.eval_noise_change import get_rate_change
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
parser.add_argument('--Backbone', type=str, default="efficientnet-b0", help='FeatureExtraction stage. '
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
parser.add_argument('--Datasets', type=str, default="ImageNet", help=' ImageNet|Clothing|CIFAR10|CIFAR100|OpenImage')
parser.add_argument('--num_classes', type=int, default=10, help=' classification')
parser.add_argument('--rate', type=float, default=0.4, help='noise ratio')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')


# Training strategy
parser.add_argument('--epoch_iter', default=500, type = int,
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


# Cross validation
parser.add_argument('--is_crossval', default=False, type=bool,
                    help=' whether using cross validation')
parser.add_argument('--cv_n_folds', default=5, type=int,
                    help=' number of cross validation')
parser.add_argument('--cross_batch_size', default=512, type = int,
                    help='batch size of cross validation ')
parser.add_argument('--cross_epoch_iter', default=300, type = int,
                    help='the max epoch iter')
parser.add_argument('--n_jobs', default=1, type = int,
                    help='int (Windows users may see a speed-up with n_jobs = 1)')
parser.add_argument('--frac_noise', default=0, type = int,
                    help='Value in range (0, 1] that determines the fraction of noisy example')
parser.add_argument('--prune_method', type=str, default="both" , help='prune_method. '
                                                            "both|prune_by_noise_rate|prune_by_class"
                                                                        )
parser.add_argument('--cross_Backbone', type=str, default="efficientnet-b0", help='FeatureExtraction in cross validation. '
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
opt = parser.parse_args()


def main(opt,logging):
    noise_file = '%s/%s/%.1f_%s.json' % (opt.workspace, f'workspace/{opt.exp_name}', opt.rate, opt.noise_mode)
    right_file = '%s/%s/right.json' % (opt.workspace, f'workspace/{opt.exp_name}')

    if not os.path.exists(right_file) or not os.path.exists(noise_file):
        logging.info("generate noise label")
        generate_noise(opt.data_path,opt.rate, opt.noise_mode,noise_file=noise_file,right_file=right_file)
    ## 加载 label
    right_label = np.array(json.load(open(right_file, "r")))
    noise_label = np.array(json.load(open(noise_file, "r")))


    # 是否使用交叉验证得到预测概率
    if opt.is_crossval:
        #初始化交叉验证流程
        #batch size： 交叉验证的batch size
        cross_val_model = cross_val(data_root=opt.data_path,
                                           batch_size=opt.cross_batch_size,epochs=opt.cross_epoch_iter,model_name=opt.cross_Backbone,logger=logger)

        # 加载noise label计算交叉验证结果
        train_data = np.array(list(range(50000)))

        psx = cross_val_pred_proba(X=train_data,s=noise_label,clf=cross_val_model,cv_n_folds=opt.cv_n_folds)

        # 保存psx
        wfn = '%s/%s/%.1f_%s_psx' % (opt.workspace, f'workspace/{opt.exp_name}', opt.rate, opt.noise_mode)
        np.save(wfn, psx)

    wfn = '%s/%s/%.1f_%s_psx.npy' % (opt.workspace, f'workspace/{opt.exp_name}', opt.rate, opt.noise_mode)
    probs = np.load(wfn)

    get_rate_change(right_label,noise_label,probs,frac_noise=opt.frac_noise,prune_method=opt.prune_method,logger=logging)

    # Get the indices of the examples we wish to prune
    noise_mask = get_noise_indices(
        noise_label,
        probs,
        frac_noise=opt.frac_noise,
        prune_method=opt.prune_method,
        n_jobs=opt.n_jobs,
    )

    train_data = np.array(list(range(50000)))
    x_mask = ~noise_mask
    x_pruned = train_data[x_mask]
    s_pruned = noise_label[x_mask]

    train(opt,x_pruned,s_pruned)

def train(opt,train_idx,train_labels):
    """ dataset preparation """
    print("dataset preparation ...")

    # Data loading code
    traindir = os.path.join(opt.data_path, 'train')

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
        sparse_labels = np.zeros(50000, dtype=int) - 1
        sparse_labels[train_idx] = train_labels
        train_dataset.targets = sparse_labels

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        #             sampler=SubsetRandomSampler(train_idx if train_idx is not None else range(MNIST_TRAIN_SIZE)),
        sampler=SubsetRandomSampler(train_idx),
        batch_size=opt.batch_size,
    )


    # validation dataloader
    dataset_val = get_datasets(opt.data_path, opt.rate, noise_mode="sym", dataset='cifar10', mode="test")
    # dataset_val = Dateloader(opt.data_path,mode="test",noise_mode=opt.noise_mode, rate=opt.rate,dataset=opt.Datasets)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)


    logging.info('| Building net...')
    model = create_model(opt.Backbone,opt.num_classes)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones =[80, 170, 250, 330, 450], gamma=0.2)
    CEloss = nn.CrossEntropyLoss()

    best_acc_top1 = 40
    for epoch in range(opt.epoch_iter):
        model.train()
        lr_scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (image,gt) in enumerate(train_loader):

            start_time = time.time()
            inputs, labels = image.cuda(), gt.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CEloss(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, opt.epoch_iter, i + 1, int(len(train_loader)), time.time() - start_time, loss.item()))
            logging.info('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, opt.epoch_iter, i + 1, int(len(train_loader)), time.time() - start_time, loss.item()))
        if epoch>1:
            best_acc_top1,best_acc_top5 = validate(data_loader_val, model,logging,best_acc_top1)
            model.train()
        logging.info("----------------------------------------------------------")
        logging.info("                 best_acc_top1: {:.3f}".format(float(best_acc_top1)))
        logging.info("                    lr: {:.3f}".format(float(optimizer.param_groups[0]['lr'])))
        logging.info("----------------------------------------------------------")
        logging.info('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(len(train_loader)),
                                                                         time.time() - epoch_time))
        logging.info(time.asctime(time.localtime(time.time())))



def validate(val_loader, model,logging,best_acc_top1):
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
    logging.info("    ---------------------------------------------------------------")
    logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    if top1.avg > best_acc_top1:
        best_acc_top1 = top1.avg
        logging.info("best top1 acc:  %.2f%%"% best_acc_top1)
        logging.info('| Saving Best Net ...')
        # torch.save(model.state_dict(), save_point)
        torch.save(model.state_dict(), os.path.join('%s/%s/' % (opt.workspace, f'workspace/{opt.exp_name}'), f'{opt.Backbone}-{opt.Datasets}'+'.pth'))

    return best_acc_top1, top5.avg



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

    main(opt,logger)


