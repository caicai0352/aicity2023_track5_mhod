# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.modeling.heads import EmbeddingHead
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from fastreid.layers.weight_init import weights_init_kaiming


@REID_HEADS_REGISTRY.register()
class AttrHead(EmbeddingHead):
    def __init__(self, cfg):
        super().__init__(cfg)
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.bnneck = nn.BatchNorm1d(num_classes)
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(neck_feat.size(0), -1)

        logits = F.linear(neck_feat, self.weight)
        logits = self.bnneck(logits)

        # Evaluation
        if not self.training:
            
            label_class_map = {
        'helmet': ['with','no'],
        }

            index_begin = 0
            for i, key in enumerate(label_class_map):
                att_num = len(label_class_map[key])
                index_end = index_begin + att_num
                index = range(index_begin,index_end)
                logits[:, index_begin:index_end] = F.softmax(logits[:, index_begin:index_end], dim=1)
                index_begin = index_end
        
            return logits

        return {
            "cls_outputs": logits,
        }
