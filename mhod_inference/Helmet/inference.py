# -*- encoding: utf-8 -*-
"""
@FileName ：predict_net_rap.py
@Time ： 2021/10/14 15:30
@Auth ： Ying
"""

import logging
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.build import _root, build_reid_train_loader, build_reid_test_loader
from fastreid.data.transforms import build_transforms
from fastreid.utils import comm

from fastreid.data.data_utils import read_image
from train_net import AttrTrainer
from fastattr import *
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import os
import pandas as pd
import numpy as np
# from torchsummary import summary
import pdb
import os.path as osp

label_class_map = {0:6,1:7}

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    import time
    global classes

    model_path = "../../weights/tou_18_256x12_no_padding.pth"
    img_folder = "../../data/crop_test_frame/images"
    p2_txt = "../../data/p2.txt"
    save_path = "../../data/p2_cls.txt"

    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.HEADS.NUM_CLASSES = 2

    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = AttrTrainer.build_model(cfg)
    model.eval()

    # transform
    #print(model.backbone.state_dict().keys())
    
    size_test = [256, 192]
    size_test = [256, 192]
    cfg.INPUT.SIZE_TEST = size_test

    # dataloader
    tfn = build_transforms(cfg, is_train=False)

    p2_label = np.loadtxt(p2_txt,delimiter=",")
    #print(p2_label)
    select_out = []

    with torch.no_grad():
        Checkpointer(model).load(model_path)  # load trained model

        for p2 in p2_label:
            video_id,frame_id,bb_left,bb_top,bb_width,bb_height,_,confidence = p2
            if int(video_id) < 10:
                image_name = '00'+str(int(video_id))+'_'+ str(int(frame_id))+'.jpg'
            elif int(video_id) != 100:
                image_name = '0'+str(int(video_id))+'_'+ str(int(frame_id))+'.jpg'
            else:
                image_name = str(int(video_id))+'_'+ str(int(frame_id))+'.jpg'
            image = os.path.join(img_folder,image_name)

            img = read_image(image)
            img = img.crop((int(bb_left), int(bb_top), int(bb_left+bb_width), int(bb_top+bb_height)))
            # img.save('1.png')
            img_trans = tfn(img).unsqueeze(0)
            img_trans = img_trans.to("cuda")

            img_ = {
                "images": img_trans,
                "targets": torch.as_tensor([0]),
                "img_paths": image,
            }

            outputs = model(img_)

            result = outputs.cpu().numpy()[0]
            result = label_class_map[np.argmax(result)]
            select_out.append([video_id,frame_id,bb_left,bb_top,bb_width,bb_height,result,confidence]) 

    np.savetxt(save_path,select_out,fmt="%d,%d,%d,%d,%d,%d,%d,%lf",delimiter=",")    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(args)
    main(args)
    print('finish')


    
    



