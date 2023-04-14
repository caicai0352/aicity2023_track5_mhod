import timm
import torch
import logging
import math

import torch
from torch import nn

import logging
import math

import torch
from torch import nn
from torchvision import models

from fastreid.layers import SplAtConv2d, get_norm, DropBlock2D
from fastreid.utils.checkpoint import get_unexpected_parameters_message, get_missing_parameters_message


NUM_FINETUNE_CLASSES=39
swsl_pretrain_path='/data1/yuwq/semi_weakly_supervised_resnet50-16a12f1b.pth'
# se_pretrain_path='/data1/yuwq/seresnext50_32x4d_racm-a304a460.pth'
# se_pretrain_path='/data1/yuwq/seresnext101_32x8d_ah-e6bc4c0a.pth'
# se_pretrain_path='/data1/yuwq/seresnet50_ra_224-8efdb4bb.pth'
se_pretrain_path='/mnt/home/seresnet152d_ra2-04464dd2.pth'
resnet152_pretrain_path='/mnt/home/resnet152-394f9c45.pth'


def swsl_resnet50():
    model = timm.create_model('swsl_resnet50', pretrained=True)
    classifier = nn.Sequential()
    model.fc=classifier
    print(model)
    # model_dict = model.state_dict()

    # for k,v  in model_dict.items():
    #     print(k)
    #print(pretrained_dict)

    # state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
    state_dict = torch.load(swsl_pretrain_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model_dict = model.state_dict()

    for k in model_dict:
        print(k)


    # model.load_state_dict(torch.load('/data1/yuwq/semi_weakly_supervised_resnet50-16a12f1b.pth'))
    return model

def seresnet152d():
    # model = timm.create_model('seresnext50_32x4d', pretrained=False)
    # model = timm.create_model('seresnext101_32x8d', pretrained=False)
    model = timm.create_model('seresnet152d', pretrained=False)
    classifier = nn.Sequential()
    model.fc=classifier
    print(model)
    # state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
    state_dict = torch.load(se_pretrain_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model_dict = model.state_dict()

    for k in model_dict:
        print(k)


    # model.load_state_dict(torch.load('/data1/yuwq/semi_weakly_supervised_resnet50-16a12f1b.pth'))
    return model

def seresnet50():
    model = timm.create_model('seresnet50', pretrained=False)
    # model = timm.create_model('seresnext101_32x8d', pretrained=False)
    classifier = nn.Sequential()
    model.fc=classifier
    print(model)
    # state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
    state_dict = torch.load(se_pretrain_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model_dict = model.state_dict()

    for k in model_dict:
        print(k)


    # model.load_state_dict(torch.load('/data1/yuwq/semi_weakly_supervised_resnet50-16a12f1b.pth'))
    return model

def resnet152():
    model = models.resnet152(pretrained=False)
    classifier = nn.Sequential()
    model.fc = classifier
    print(model)
    # state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
    state_dict = torch.load(resnet152_pretrain_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model_dict = model.state_dict()

    for k in model_dict:
        print(k)
    return model