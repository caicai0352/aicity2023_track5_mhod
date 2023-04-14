# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F


def ratio2weight(targets, ratio):
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    weights[targets > 1] = 0.0
    return weights


def cross_entropy_sigmoid_loss(pred_class_logits, gt_classes, sample_weight=None):
    
    loss = F.binary_cross_entropy_with_logits(pred_class_logits, gt_classes, reduction='none')

    if sample_weight is not None:
        targets_mask = torch.where(gt_classes.detach() > 0.5,
                                   torch.ones(1, device="cuda"), torch.zeros(1, device="cuda"))  # dtype float32
        weight = ratio2weight(targets_mask, sample_weight)
        loss = loss * weight

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt
    return loss

def corss_entropy_softmax_loss(pred_class_logits,gt_classes,sample_weight=None):
    attrdict_index = {
        'helmet': ['with','no'],
        }

    total_loss=0
    num_attr=len(attrdict_index)
    index_begin = 0
    for i,key in enumerate(attrdict_index):
        att_num = len(attrdict_index[key])
        index_end = index_begin + att_num
        index = range(index_begin,index_end)
        target=gt_classes[:,index].argmax(dim=1)
        # print(sample_weight)
        if sample_weight is not None:
             weight=torch.exp(1-sample_weight[index])
        # print(weight)

        loss = F.cross_entropy(pred_class_logits[:,index],target, weight=weight,reduction='none')

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
        loss = loss.sum() / non_zero_cnt
        total_loss += loss
        index_begin = index_end
        # print(f"啦啦啦{total_loss}")
    return total_loss/num_attr

def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2):
    num_classes = pred_class_outputs.size(1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt

    return loss