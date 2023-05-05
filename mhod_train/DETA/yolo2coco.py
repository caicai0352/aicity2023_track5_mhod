"""
YOLO 格式的数据集转化为 COCO 格式的数据集
--root_dir 输入根路径
--save_path 保存文件的名字(没有random_split时使用)
--random_split 有则会随机划分数据集，然后再分别保存为3个文件。
--split_by_file 按照 ./train.txt ./val.txt ./test.txt 来对数据集进行划分。
"""

import os
import cv2
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/mnt/home/datasets/aicity2023/aicity5/crop_yolo_data/train_all/',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_path', type=str,default='./data/aicity5/', help="if not split the dataset, give a path to a json file")
parser.add_argument('--save_name', type=str,default='train_all.json', help="if not split the dataset, give a path to a json file")

arg = parser.parse_args()

def yolo2coco(arg):
    root_path = arg.root_dir
    print("Loading data from ",root_path)

    assert os.path.exists(root_path)
    originLabelsDir = os.path.join(root_path, 'labels',)                                        
    originImagesDir = os.path.join(root_path, 'images',)
    with open('classes_7.txt') as f:
        classes = f.read().strip().split()
    # images dir name
    indexes = os.listdir(originImagesDir)

    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        txtFile = index.replace('.jpg','.txt')
        # 读取图像的宽和高
        im = cv2.imread(os.path.join(originImagesDir, index))
        # print(os.path.join(originImagesDir, index))
        height, width, _ = im.shape
        # 添加图像的信息
        dataset['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            continue
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(float(label[0]))   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    os.makedirs(arg.save_path,exist_ok=True)
    json_name = os.path.join(arg.save_path,arg.save_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(json_name))

if __name__ == "__main__":
    yolo2coco(arg)