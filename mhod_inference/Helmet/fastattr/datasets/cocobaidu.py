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
        

        def extract_name(path):
            if osp.exists(path):
                names=[]
                f= open((path),'r')
                for line in f.readlines():
                    names.append(line.strip('\n'))
                f.close()
            return names

        def extract_label(path):
            if osp.exists(path):
                atts=[]
                atts = np.loadtxt(path).astype(int)
                return atts
            

        #统计每个属性的个数
        def cal_num(datas):
            label=[]
            for data in datas:
                label.append(data[1])
            label = np.array(label)
            for i,k in enumerate(self.label_class_map):
                label_i = label[:,i]
                for iv,v in enumerate(self.label_class_map[k]):
                    num = sum(label_i==iv)
                    print(k,v,num)
        
        
        def cal_number(datas,name):
            result_dict = dict()
            for data in datas:
                label = data[1]
                for i,k in enumerate(self.label_class_map):
                    label_i = label[i]
                    for iv,v in enumerate(self.label_class_map[k]):
                        if k not in result_dict:
                            result_dict[k] = dict()
                        if v not in result_dict[k]:
                            result_dict[k][v] = 0
                        if label_i == iv:
                            result_dict[k][v] += 1
            os.makedirs('./runs_drvpassger/',exist_ok=True)
            with open('./runs_drvpassger/'+name+".json", "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
            with open('./runs_drvpassger/'+name+".txt",'w',encoding='utf-8') as f:
                for key in result_dict:
                    for i in result_dict[key]:
                        f.write(str(result_dict[key][i]))
                        f.write('\n')

        
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
            '''np.savetxt('/mnt/home/baidu_prc/2.5wtestdata_labels_paths.txt',attrs)
            
            with open('/mnt/home/baidu_prc/2.5wtestdata_img_paths.txt', "w", encoding='utf-8') as f:
                    for lines in range(len(paths)):
                        f.write(paths[lines]+"\n")'''
            return data
            
        
        def save_gt(name):
            paths=[]
            attrs=[]
            for path,attr in name:
                paths.append(path)
                attrs.append(attr)
            with open('/mnt/home/aicity_track5/crop_images/val_test_paths.txt', "w", encoding='utf-8') as f:
                for lines in range(len(paths)):
                    f.write(paths[lines]+"\n")
            # np.savetxt('/mnt/home/person_att/quality_data/quality_10_18_2000_label.txt',attrs)
        
        def extract_test(path):
            img_list=os.listdir(path)
            data = []
            for img in img_list:
                img_path=osp.join(path,img)
                attrs = np.array([0] * 4, np.float32)
                data.append((img_path,attrs))
            return data
        

            
        train_path= '/mnt/home/aicity_track5/crop_images/train_tou/images'
        test_path= '/mnt/home/aicity_track5/p2_crop_kuobian'
        train=[]
        test=[]
        
        for path in os.listdir(train_path):
            full_path = os.path.join(train_path,path)
            class_name = int(path.split('_')[-1].strip('.jpg'))
            # if class_name == 2:
            #     continue
            label=[]
            label.append(class_name)
            # print(path,label)
            train.append((full_path,label))
        for path in os.listdir(test_path):
            full_path = os.path.join(test_path,path)
            # class_name = int(path.split('_')[-1].strip('.jpg'))
            # if class_name == 2:
            #     continue
            label=[]
            label.append(random.randint(0,1))
            # label.append(class_name)
            test.append((full_path,label))

        val=test
        save_gt(test)
        cal_number(train,'train')
        cal_number(test,'test')
        test= extend_label(test)
        train= extend_label(train)
        print(train[1])
        # attrs = self.class_map
        # attr_dict = {i: str(attr) for i, attr in enumerate(attrs)}
        attrs=[]
        for k in self.label_class_map:
            attrs.extend(self.label_class_map[k])
        attr_dict = {i: str(attr) for i, attr in enumerate(attrs)}
        print(attr_dict)

        return train, val, test, attr_dict
