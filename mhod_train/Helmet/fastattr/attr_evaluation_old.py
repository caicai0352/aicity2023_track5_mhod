# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import json
import os.path as osp
import os

import copy
import logging
from collections import OrderedDict
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
from matplotlib import font_manager



import torch

from fastreid.evaluation.evaluator import DatasetEvaluator
from fastreid.utils import comm
import numpy as np

logger = logging.getLogger("fastreid.attr_evaluation")


class AttrEvaluator(DatasetEvaluator):
    def __init__(self, cfg, attr_dict, thres=0.5, output_dir=None):
        self.cfg = cfg
        self.attr_dict = attr_dict
        self.thres = thres
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.pred_logits = []
        self.gt_labels = []


    def reset(self):
        self.pred_logits = []
        self.gt_labels = []

    def process(self, inputs, outputs):
        self.gt_labels.extend(inputs["targets"].to(self._cpu_device))
        self.pred_logits.extend(outputs.to(self._cpu_device, torch.float32))

    @staticmethod
    def get_attr_metrics(gt_labels, pred_logits, thres):
        # print(pred_logits.shape)

        label_class_map = {
        'helmet': ['with','no'],
        }

       
    
        index_list=np.zeros((gt_labels.shape[0],1))
        
        eps = 1e-20
         #对预测结果，各类别计算softmax找最大值，置1
        pred_labels = np.zeros(pred_logits.shape)
        index_begin = 0
        for i, key in enumerate(label_class_map):
            att_num = len(label_class_map[key])
            index_end = index_begin + att_num
            index = range(index_begin,index_end)
            c = np.argmax(pred_logits[:,index],axis=1)
            index_list[:,i]=c
            pred_onehot = np.eye(att_num)[c]
            pred_labels[:, index_begin:index_end] = pred_onehot
            index_begin = index_end
        #np.savetxt('/mnt/home/aicity_track5/crop_images/0308_pre.txt',index_list)
        
        #计算混淆矩阵
        AttrEvaluator.confMax(index_list,gt_labels,label_class_map)    

        # Compute label-based metric
        overlaps = pred_labels * gt_labels
        correct_pos = overlaps.sum(axis=0)
        real_pos = gt_labels.sum(axis=0)
        pre_pos= pred_labels.sum(axis=0)
        inv_overlaps = (1 - pred_labels) * (1 - gt_labels)
        correct_neg = inv_overlaps.sum(axis=0)
        real_neg = (1 - gt_labels).sum(axis=0)


        precision=correct_pos/(pre_pos+eps)
        term1 = correct_pos / (real_pos + eps)
        term2 = correct_neg / (real_neg + eps)
        label_mA_verbose = (term1 + term2) * 0.5
        acc= correct_pos.sum()/(pred_labels.shape[0])
        # #保存各个属性的precision,recall及ma
        # np.savetxt('./projects/prc/precision.txt',precision)
        # np.savetxt('./projects/prc/recall.txt',term1)
        # np.savetxt('./projects/prc/ma.txt',label_mA_verbose)
    
      
        label_mA = label_mA_verbose.mean()
        label_mRecall= term1.mean()
        label_mPrec = precision.mean()
        

        results = OrderedDict()
        results["Accu"] = acc * 100
        results["mA"] = label_mA * 100
        results["metric"] = label_mA * 100
        results["Recall"] = label_mRecall*100
        results["Prec"] = label_mPrec*100
        return results

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            pred_logits = comm.gather(self.pred_logits)
            pred_logits = sum(pred_logits, [])

            gt_labels = comm.gather(self.gt_labels)
            gt_labels = sum(gt_labels, [])

            if not comm.is_main_process():
                return {}
        else:
            pred_logits = self.pred_logits
            gt_labels = self.gt_labels

        pred_logits = torch.stack(pred_logits, dim=0).numpy()
        gt_labels = torch.stack(gt_labels, dim=0).numpy()

        # Pedestrian attribute metrics
        thres = self.cfg.TEST.THRES
        self._results = self.get_attr_metrics(gt_labels, pred_logits, thres=thres)

        return copy.deepcopy(self._results)

    def confMax(index_list,gt_labels,label_class_map):
        index_begin=0
        for i,key in enumerate(label_class_map):
            labels=label_class_map[key]
            prelab=index_list[:,i]
            print(labels)
            att_num = len(label_class_map[key])
            index_end = index_begin + att_num
            index = range(index_begin,index_end)
            gtlab= np.argmax(gt_labels[:,index],axis=1)
            # gtlab=gt_labels[:,i]
            # print(f"hhhh{gtlab}")
            confusion_mat=confusion_matrix(gtlab,prelab)
            print("confusion_mat.shape : {}".format(confusion_mat.shape))
            print("confusion_mat : {}".format(confusion_mat))

            index_begin = index_end

    

