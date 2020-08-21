import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
import os
import time
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import argparse
import os
from datasets.cifar10_noise import get_datasets
import random
import torch.backends.cudnn as cudnn
from datasets.dataloader import Dateloader
from network.models import create_model
import torch.optim as optim
import logging
import logging.config
from lib.utils import AverageMeter,accuracy,adjust_learning_rate
from sklearn.mixture import GaussianMixture
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='classification')

# Model path
parser.add_argument('--exp_name', help='Where to store logs and models')
parser.add_argument('--resume', default="/data/glusterfs_cv_04/11121171/AAAI_EAST/Baseline/EAST_v1/model_save/model_epoch_826.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--data_path', default='/data/glusterfs_cv_04/11121171/data/CIFAR/CIFAR10-noise', type=str,
                    help='the test image of target domain ')
parser.add_argument('--workspace', default="/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise", type=str,
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
parser.add_argument('--Datasets', type=str, default="cifar10", help=' ImageNet|Clothing|CIFAR10|CIFAR100')
parser.add_argument('--num_classes', type=int, default=10, help=' classification')
parser.add_argument('--rate', type=float, default=0.4, help='noise ratio')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--manualSeed', type=int, default=222, help='for random seed setting')


# Training strategy
parser.add_argument('--epoch_iter', default=300, type = int,
                    help='the max epoch iter')
parser.add_argument('--batch_size', default=256, type = int,
                    help='batch size of training')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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


def train(opt):
    """ dataset preparation """
    print("dataset preparation ...")
    logging.info("dataset preparation ...")
    # dataset = Dateloader(opt.data_path,mode="train",noise_mode=opt.noise_mode, rate=opt.rate,dataset = opt.Datasets,
    #                      noise_file='%s/%.1f_%s.json' % (f'./workspace/{opt.exp_name}', opt.rate, opt.noise_mode),
    #                      right_file= '%s/right.json' % (f'./workspace/{opt.exp_name}'))

    dataset = get_datasets(opt.data_path,opt.rate,noise_mode="sym",dataset = 'cifar10',
                           noise_file='%s/%.1f_%s.json' % (f'./workspace/{opt.exp_name}', opt.rate, opt.noise_mode),
                         right_file= '%s/right.json' % (f'./workspace/{opt.exp_name}'),mode="train")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        drop_last=True,
        pin_memory=True)

    dataset_val = get_datasets(opt.data_path,opt.rate,noise_mode="sym",dataset='cifar10',mode="test")

    # dataset_val = Dateloader(opt.data_path,mode="test",noise_mode=opt.noise_mode, rate=opt.rate,dataset=opt.Datasets)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)

    print('| Building net...')
    logging.info('| Building net...')
    model = create_model(opt.Backbone,opt.num_classes)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones =[80, 170, 250, 330, 450], gamma=0.2)
    CEloss = nn.CrossEntropyLoss()
    # conf_penalty = NegEntropy()

    best_acc = 40
    for epoch in range(opt.epoch_iter):
        model.train()
        lr_scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (image,gt) in enumerate(data_loader):

            start_time = time.time()
            inputs, labels = image.cuda(), gt.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CEloss(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, opt.epoch_iter, i + 1, int(len(data_loader)), time.time() - start_time, loss.item()))
            logging.info('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, opt.epoch_iter, i + 1, int(len(data_loader)), time.time() - start_time, loss.item()))
        if epoch>1:
            validate(data_loader_val, model, CEloss)
            best_acc = test(epoch,model,data_loader_val,best_acc)
            model.train()
        logging.info("----------------------------------------------------------")
        logging.info("                    best_acc: {:.3f}".format(float(best_acc)))
        logging.info("                    lr: {:.3f}".format(float(optimizer.param_groups[0]['lr'])))
        logging.info("----------------------------------------------------------")
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(len(data_loader)),time.time() - epoch_time))
        logging.info('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(len(data_loader)),
                                                                         time.time() - epoch_time))
        logging.info(time.asctime(time.localtime(time.time())))

def test(epoch,  model,val_loader,best_acc):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    logging.info("\n| Validation\t Net  Acc: %.2f%%" % acc)
    if acc > best_acc:
        best_acc = acc
        logging.info("best acc:  %.2f%%"% best_acc)
        logging.info('| Saving Best Net ...')
        # torch.save(model.state_dict(), save_point)
        torch.save(model.state_dict(), os.path.join('%s/%s/' % (opt.workspace, f'workspace/{opt.exp_name}'),
                                                    f'{opt.Backbone}-{opt.Datasets}' + '.pth'))
    return best_acc

def validate(val_loader, model, criterion):
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
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logging.info("    ---------------------------------------------------------------")
    logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg

if __name__ == '__main__':


    if not opt.exp_name:
        opt.exp_name = f'{opt.Backbone}-{opt.Datasets}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./workspace/{opt.exp_name}', exist_ok=True)

    # 通过下面的方式进行简单配置输出方式与日志级别
    logging.basicConfig(
        filename=os.path.join(f'./workspace/{opt.exp_name}',"logger.log"),
        level=logging.INFO,filemode='w')

    logging.debug('debug message')
    logging.info('info message')
    logging.error('error message')
    logging.critical('critical message')


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
        logging.info('------ Use multi-GPU setting ------')
        logging.info('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.num_workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

    train(opt)


