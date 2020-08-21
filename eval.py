import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import argparse
import json
import os
import random
import torch.backends.cudnn as cudnn
from datasets.dataloader import Dateloader
from network.models import create_model
import torch.optim as optim
import logging
import logging.config
from thop import profile
from thop import clever_format
from sklearn.mixture import GaussianMixture


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='classification')

# Model path
parser.add_argument('--exp_name', help='Where to store logs and models')
parser.add_argument('--resume', default="/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/model_save/MobileNet_v2-cifar10.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--val_path', default="/data/glusterfs_cv_04/11121171/data/CIFAR/cifar10", type=str,
                    help='val dataset path')
parser.add_argument('--Backbone', type=str, default="MobileNet_v2", help='FeatureExtraction stage. '
                                                                     'ResNet18|ResNet34|ResNet50'
                                                                     'MobileNet_v1|MobileNet_v2'
                                                                     'VGG11|VGG16|VGG19'
                                                                     'Efficient_v1')
parser.add_argument('--Datasets', type=str, default="CIFAR10", help=' ImageNet|Clothing|CIFAR10|CIFAR100')
parser.add_argument('--num_classes', type=str, default=10, help=' classification')
parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')


# Training strategy
parser.add_argument('--epoch_iter', default=8000, type = int,
                    help='the max epoch iter')
parser.add_argument('--batch_size', default=100, type = int,
                    help='batch size of training')
parser.add_argument('--num_workers', default=10, type=int,
                    help='Number of workers used in dataloading')

opt = parser.parse_args()

def get_parameter(model):
    params = list(model.parameters())  # 所有参数放在params里
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j  # 每层的参数存入l，这里也可以print 每层的参数
        k = k + l  # 各层参数相加
    print("all params:" + str(k))  # 输出总的参数

    # 可替换为自己的模型及输入
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("flops:",flops,"params:",params)

def test(opt):
    """ dataset preparation """
    print("dataset preparation ...")


    dataset_val = Dateloader(opt.val_path,mode="train",\
                             noise_file="/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/MobileNet_v2-cifar10-Seed1111/0.4_sym.json",
                             dataset=opt.Datasets)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)

    print('| Building net...')
    model = create_model(opt.Backbone,opt.num_classes)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    model.load_state_dict(torch.load(opt.resume))

    # get_parameter(model)

    model.eval()
    correct = 0
    total = 0
    CE = nn.CrossEntropyLoss(reduction='none')
    losses = torch.zeros(50000)
    index = 0

    righ_label = json.load(
        open("/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/right.json",
             "r"))
    noise_label = json.load(open(
        "/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/MobileNet_v2-cifar10-Seed1111/0.4_sym.json",
        "r"))
    # rate = np.array((righ_label==noise_label))
    riggt = [1 if righ_label[i] == noise_label[i] else 0 for i in range(len(righ_label))]
    rate = np.array(riggt).sum() / len(righ_label)
    print("sample number: 50000", rate)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader_val)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index] = loss[b]
                index+=1

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    # acc = 100. * correct / total
    # print("\n| Validation\t Net  Acc: %.2f%%" % acc)
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    input_loss = losses.reshape(-1, 1)
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    # prob = [1 if i<0.5 else 0 for i in prob]
    # json.dump(prob
    #           , open(
    #         "/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/prob.json",
    #         "w"))

    for threshold_one in range(0,11):
        threshold_one = threshold_one*0.1
        truee = (prob > threshold_one).nonzero()[0]
        righ_label_1 = []
        noise_label_1 = []
        for i in range(len(righ_label)):
            if i in truee:
                righ_label_1.append(righ_label[i])
                noise_label_1.append(noise_label[i])
        # righ_label_1 = righ_label * (prob < threshold_one)
        # noise_label_1 = noise_label * (prob < threshold_one)
        riggt = [1 if righ_label_1[i] != noise_label_1[i] else 0 for i in range(len(righ_label_1))]
        rate = np.array(riggt).sum() / len(righ_label_1)
        print("sample number:",np.array(prob > threshold_one).sum(),"after ",threshold_one," threshold:",rate)

    # print(np.array(prob < 0.5).sum())
    # print(prob)
    # print(len(prob))





if __name__ == '__main__':

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
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.num_workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

    test(opt)


