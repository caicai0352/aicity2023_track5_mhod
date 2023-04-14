import cv2
import os
import numpy as np
from tqdm import tqdm

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        l = np.array([x.split() for x in f.read().splitlines()],np.float64)
    return l

label_dir = '../data/deta_result/merge/txt'
p2_txt = '../data/p2_cls.txt'

out_labels = []
save_path = '../data/submit.txt'
for label_name in tqdm(os.listdir(label_dir)):
    video_id = int(label_name.split('_')[0])
    # print(video_id)
    frame_id = int(label_name.split('_')[1].strip('.txt'))
    label_path = os.path.join(label_dir,label_name)
    #print(label_path)
    sub = read_txt(label_path)
    for sub_labels in sub: 
        confidence = sub_labels[5]
        # if confidence<0.65:
        #     continue
        class_name = int(sub_labels[0] +1)
        bb_left = max(0,round(sub_labels[1]))
        bb_top = max(0,round(sub_labels[2]))
        bb_width = round(sub_labels[3]-sub_labels[1])
        if bb_left+bb_width>1920:
            bb_width = 1920-bb_left
        bb_height = round(sub_labels[4]-sub_labels[2])
        if bb_top+bb_height>1080:
            bb_height = 1080-bb_top
        if class_name == 5:
            if confidence<0.65 or bb_width<=150 or bb_height<=150:
                continue
        elif class_name == 4:
            if confidence < 0.1 or bb_width<=80 or bb_height<=80:
                continue
        else:
            if confidence < 0.7 or bb_width<=145 or bb_height<=145:
                continue
        out_labels.append([video_id,frame_id,bb_left,bb_top,bb_width,bb_height,class_name,confidence])
        if bb_left+bb_width>1920 or bb_top+bb_height>1080:
            print(bb_left,bb_top,bb_left+bb_width,bb_top+bb_height,label_name)

p2_label = np.loadtxt(p2_txt,delimiter=",")
for p2 in p2_label:
    out_labels.append(p2)
print(len(out_labels))
np.savetxt(save_path,out_labels,fmt="%d,%d,%d,%d,%d,%d,%d,%lf",delimiter=",")

    

        