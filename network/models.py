import torch
import torch.nn as nn
from network.resnet import resnet18,resnet34,resnet50
from network.mobilenet import mobilenet_v2
from network.shufflenetv2 import shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5
from thop import profile
from thop import clever_format
from network.efficientnet import EfficientNet
from network.inception import inception_v3
from network.mnasnet import mnasnet0_5,mnasnet0_75,mnasnet1_0,mnasnet1_3
from network.vgg import vgg11,vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn,vgg13,vgg16,vgg19
from network.densenet import densenet121
import os
from network.mobilenetv3 import mobilenetv3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def create_model(Model,num_classes):
    if Model == "ResNet18":
        model = resnet18(pretrained=False)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
        model = model.cuda()

    elif Model == "ResNet34":
        model = resnet34(pretrained=False).cuda()
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
        model = model.cuda()

    elif Model == "ResNet50":
        model = resnet50(pretrained=False).cuda()
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
        model = model.cuda()

    elif Model == "MobileNet_v2":
        model = mobilenet_v2(num_classes = num_classes,pretrained=False).cuda()

    elif Model == "Mobilenetv3":
        model = mobilenetv3(n_class = num_classes,pretrained=False).cuda()

    elif Model == "shufflenet_v2_x0_5":
        model = shufflenet_v2_x0_5(pretrained=False,num_classes = num_classes).cuda()

    elif Model == "shufflenet_v2_x1_0":
        model = shufflenet_v2_x1_0(pretrained=False,num_classes = num_classes).cuda()

    elif Model == "shufflenet_v2_x1_5":
        model = shufflenet_v2_x1_5(pretrained=False,num_classes = num_classes).cuda()

    elif "efficientnet" in Model :
        model = EfficientNet.from_pretrained(Model,num_classes=num_classes,pretrained=False).cuda()

    elif Model == "inception_v3":
        model = inception_v3(pretrained=False,num_classes=num_classes).cuda()


    elif Model == "mnasnet0_5":
        model = mnasnet0_5(pretrained=False, num_classes=num_classes).cuda()

    elif Model == "vgg11":
        model = vgg11(pretrained=False, num_classes=num_classes).cuda()

    elif Model == "vgg11_bn":
        model = vgg11_bn(pretrained=False, num_classes=num_classes).cuda()

    elif Model == "vgg19":
        model = vgg19(pretrained=False, num_classes=num_classes).cuda()

    elif Model == "densenet121":
        model = densenet121(pretrained=False).cuda()

    else:
        print("model error")

    # input = torch.randn(1, 3, 32, 32).cuda()
    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("------------------------------------------------------------------------")
    # print("                              ",Model)
    # print( "                    flops:", flops, "    params:", params)
    # print("------------------------------------------------------------------------")

    return model
#
# model = create_model("ResNeXt29_2x64d",10)
# input = torch.randn(1, 3, 32, 32).cuda()
#
# flops, params = profile(model, inputs=(input,))
# flops, params = clever_format([flops, params], "%.3f")
# # print(output.shape)
# print("------------------------------------------------------------------------")
# print( "                     flops:", flops, "params:", params)
# print("------------------------------------------------------------------------")


