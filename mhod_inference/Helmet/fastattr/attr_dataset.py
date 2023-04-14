# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset

from fastreid.data.data_utils import read_image
import random
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image, ImageOps


class AttrDataset(Dataset):
    """Image Person Attribute Dataset"""

    def __init__(self, cfg, img_items, transform, attr_dict,):
        self.img_items = img_items
        self.transform = transform
        self.attr_dict = attr_dict
        self.size_train = cfg.INPUT.SIZE_TRAIN
        

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, labels = self.img_items[index]
        img = read_image(img_path)
        # image = np.asarray(img)
        # img = self.img_resize(image)
        
        # img = Image.fromarray(img)
        if self.transform is not None: img = self.transform(img)
        labels = torch.as_tensor(labels)
        

        
    
        return {
            "images": img,
            "targets": labels,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.attr_dict)

    @property
    def sample_weights(self):
        sample_weights = torch.zeros(self.num_classes, dtype=torch.float32)
        for _, attr in self.img_items:
            sample_weights += torch.as_tensor(attr)
        sample_weights /= len(self)
        # sample_weights[[8,18,21,28,48,58,72,93]]=0.99
        return sample_weights
    
    def img_resize(self,img):
        h,w = img.shape[0],img.shape[1]
        obj_h = self.size_train[0]
        obj_w = self.size_train[1]
        if h > obj_h or w > obj_w:
            scale = max(h / float(obj_h), w / float(obj_w))
            new_h,new_w=int(h/scale),int(w/scale)

        else:
            scale= min(float(obj_h) / h,  float(obj_w)/w)
            new_h,new_w= int(h*scale),int(w*scale)

        resize_img = cv2.resize(img, (new_w, new_h))

        if(obj_w - new_w) % 2 != 0 and (obj_h - new_h) %2 == 0:
            top, bottom, left, right = (obj_h - new_h) // 2, (obj_h - new_h) // 2, (obj_w - new_w) // 2 + 1, (obj_w - new_w) // 2
        elif (obj_w - new_w) % 2 == 0 and (obj_h - new_h) %2 != 0:
            top, bottom, left, right = (obj_h - new_h) // 2+1, (obj_h - new_h) // 2, (obj_w - new_w) // 2 + 1, (obj_w - new_w) // 2
        elif (obj_w - new_w) % 2 == 0 and (obj_h - new_h) %2 == 0:
            top, bottom, left, right = (obj_h - new_h) // 2, (obj_h - new_h) // 2, (obj_w - new_w) // 2 , (obj_w - new_w) // 2
        else:
            top, bottom, left, right = (obj_h - new_h) // 2+1, (obj_h - new_h) // 2, (obj_w - new_w) // 2 + 1, (obj_w - new_w) // 2
        
        pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return pad_img

            



