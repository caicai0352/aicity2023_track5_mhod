# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import json
import os.path as osp
import os
from re import I
import shutil

import numpy as np
from scipy.io import loadmat

from fastreid.data.datasets import DATASET_REGISTRY

from .bases import Dataset
from pycocotools.coco import COCO
import random


@DATASET_REGISTRY.register()
class COCOBAIDU(Dataset):
    """Pedestrian quality dataset.
    *k training images + *k test images.
    The folder structure should be:
        root/
            train/ # images
            test/
            val/
                'Resource/super\:'+path   #images
                'Annotation/coco.json'   #labels
    """

    def __init__(self, root='', **kwargs):

        self.class_map = ['helmet']
        self.label_class_map = {
        'helmet': ['with','no'],
        }

        train, val, test, attr_dict = self.extract_data()
        super(COCOBAIDU, self).__init__(train, val, test, attr_dict=attr_dict, **kwargs)

    
    def extract_data(self):
      
        def extend_label(name):
            attrs=[]
            paths=[]
            data=[]
            for path,attr in name:
                attrs.append(attr)
                paths.append(path)
            attrs=np.array(attrs)
            extend_attrs = np.zeros((attrs.shape[0],2))
            index_begin = 0
            for i, key in enumerate(self.label_class_map):
                att_num = len(self.label_class_map[key])
                index_end = index_begin + att_num
                label_onehot = np.eye(att_num)[attrs[:, i].astype(np.int)]
                extend_attrs[:,index_begin:index_end] = label_onehot
                index_begin = index_end
            for i in range(len(paths)):
                data.append((paths[i],extend_attrs[i,:]))

            return data
                 
        train_path= '../../data/crop_images'
        all_data = []
        
        for path in os.listdir(train_path):
            full_path = os.path.join(train_path,path)
            class_name = int(path.split('_')[-1].strip('.jpg'))
            # if class_name == 2:
            #     continue
            label=[]
            label.append(class_name)
            # print(path,label)
            all_data.append((full_path,label))
        
        train = all_data[:int(0.9*len(all_data))]
        test = all_data[int(0.9*len(all_data)):]

        val=test
        test= extend_label(test)
        train= extend_label(train)
        
        attrs=[]
        for k in self.label_class_map:
            attrs.extend(self.label_class_map[k])
        attr_dict = {i: str(attr) for i, attr in enumerate(attrs)}
        print(attr_dict)

        return train, val, test, attr_dict
