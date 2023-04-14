import json
import os
import numpy as np
import cv2
from tqdm import tqdm

def get_detectron2_labels(path):
    results = []
    if not os.path.exists(path):
        labels = np.zeros((0,5))
        return labels
    #print(path)
    f = open(path,'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        if int(float(line[0])) != 0:
            continue
        if float(line[5]) <0.9:
            continue
        line = np.array([float(m) for m in line])
        line[[1,3]] = line[[1,3]]*1920
        line[[2,4]] = line[[2,4]]*1080
        # if len(line) == 8:
        #     li = np.zeros((9))
        #     li[:8] = line
        #     line = li
        results.append(line)
    return results

def get_yolo(path):
    results = []
    if not os.path.exists(path):
        labels = np.zeros((0,5))
        return labels
    f = open(path,'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        if float(line[5]) <0.3:
            continue
        if int(float(line[0])) != 0:
            continue
        line = np.array([float(m) for m in line])
        results.append(line)
    return results

def get_p2(bboxes):
    dx = np.mean(bboxes[:,6])
    dy = np.mean(bboxes[:,7])
    x_list = list(bboxes[:,2])
    #print(bboxes[:,6:])
    if dx > 0:
        box = [bboxes[x_list.index(min(x_list)),:]]
    else:
        box = [bboxes[x_list.index(max(x_list)),:]]
    return box


def xywh2xyxy(the_boxes):
    box = np.array(the_boxes)
    center_x = the_boxes[:, 1]
    center_y = the_boxes[:, 2]
    width = the_boxes[:, 3]
    height = the_boxes[:, 4]

    box[:, 1] = center_x - width / 2
    box[:, 2] = center_y - height / 2
    box[:, 3] = center_x + width / 2
    box[:, 4] = center_y + height / 2
    return box

def xyxy2xywh(the_boxes):
    box = np.zeros_like(the_boxes)
    x1 = the_boxes[:, 0]
    y1 = the_boxes[:, 1]
    x2 = the_boxes[:, 2]
    y2 = the_boxes[:, 3]

    box[:, 0] = (x2 + x1) / 2
    box[:, 1] = (y2 + y1) / 2
    box[:, 2] = x2 - x1
    box[:, 3] = y2 - y1
    return box

colormap = [(0, 255, 0), (132, 112, 255), (0, 191, 255)]  # 色盘，可根据类别添加新颜色

def show_xyxy(x, w1, h1, img,color_i):
    label, top_left_x, top_left_y, bottom_right_x, bottom_right_y,conf = x[:6]
    # if conf<0.3:
    #     return img

    # 绘制矩形框
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[color_i], 2)
    cv2.putText(img, str(round(float(conf),2)), (int(top_left_x), int(top_left_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colormap[1], 2)
    
    return img

def get_iou(bbox1,bbox2):
    x1,y1,x2,y2 = bbox1[1:5]
    areas = (y2 - y1) * (x2 - x1)
    bbox2 = np.array(bbox2)
    areas2 = (bbox2[:,4] - bbox2[:,2]) * (bbox2[:,3] - bbox2[:,1])

    x11 = np.maximum(x1, bbox2[:,1])
    y11 = np.maximum(y1, bbox2[:,2])
    x22 = np.minimum(x2, bbox2[:,3])
    y22 = np.minimum(y2, bbox2[:,4])


    # 如果边框相交, x22 - x11 > 0, 如果边框不相交, w(h)设为0
    w = np.maximum(0, x22 - x11)
    h = np.maximum(0, y22 - y11)

    overlaps = w * h

    ious = overlaps / (areas + areas2 - overlaps + 1e-7)

    return ious


def nms(dets, thresh):
    """
    :param dets: numpy矩阵
    :param thresh: iou阈值
    :return:
    """

    result = []

    # 3类
    class_num = 4
    for each in range(class_num):
        the_boxes = dets[np.where(dets[:, 0] == each)]

        center_x = the_boxes[:, 2]
        center_y = the_boxes[:, 3]
        width = the_boxes[:, 4]
        height = the_boxes[:, 5]
        confidence = the_boxes[:, 1]

        index = confidence.argsort()[::-1]

        keep = []

        while index.size > 0:
            best = index[0]
            keep.append(np.expand_dims(the_boxes[best, :], axis=0))

            ious = get_iou(index, best, center_x, center_y, width, height)

            idx = np.where(ious <= thresh)[0]

            index = index[idx + 1]

        result.append(np.concatenate(keep, axis=0))

    return np.concatenate(result, axis=0)

images = []

detectron2_dir = '../data/detectron2_result_sort'
deta_dir = '../data/deta_result/merge/txt'

save_path = '../data/p2.txt'
select_out = []
for files in tqdm(os.listdir(detectron2_dir)):
    Draw = False
    detectron2_path = os.path.join(detectron2_dir,files)
    deta_path = os.path.join(deta_dir,files)

    detectron2_labels = get_detectron2_labels(detectron2_path)
    deta_labels = get_yolo(deta_path)
    # print(detectron2_labels)
    # print(deta_labels)
    i = []
    if len(detectron2_labels)>=3:
        detectron2_labels = xywh2xyxy(np.array(detectron2_labels))
        for bbox1 in deta_labels:
            p2 = np.where(get_iou(bbox1,detectron2_labels)>0.175)
            # print(len(p2[0]))
            if len(p2[0])>=3:
                i.extend(p2[0])
                Draw= True
                #print(files)
    if Draw:
        select_index = np.array(detectron2_labels[i])
        box = get_p2(select_index)
        video_id = int(files.split('_')[0])
        frame_id = int(files.split('_')[1].strip('.txt'))
        class_name,bb_left,bb_top,bb_right,bb_bottom,confidence = box[0][:-2]
        bb_width = bb_right-bb_left
        bb_height = bb_bottom-bb_top
        select_out.append([video_id,frame_id,bb_left,bb_top,bb_width,bb_height,0,confidence])

np.savetxt(save_path,select_out,fmt="%d,%d,%d,%d,%d,%d,%d,%lf",delimiter=",")

        









