import os
import cv2
import numpy as np
import pdb
def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        l = np.array([x.split() for x in f.read().splitlines()],np.float64)
    return l

image_dir = '../data/crop_yolo_data/train/images'
lable_dir = '../data/crop_yolo_data/train/labels'
new_image_savepath = '../data/crop_images/'

os.makedirs(new_image_savepath,exist_ok=True)
new_labels = []

for image_path in os.listdir(image_dir):
    label_path = image_path.replace('jpg', 'txt')
    full_image_path = os.path.join(image_dir,image_path)
    frame=cv2.imread(full_image_path)
    height, width = frame.shape[:2]
    
    full_label_path = os.path.join(lable_dir,label_path)
    labels = read_txt(full_label_path)

    for label in labels:
        new_label = []
        class_before = int(label[0])
        if class_before == 0:
            continue
        if class_before ==1 or class_before ==3 or class_before ==5:
            class_after = 0
        else:
            class_after = 1
            
        new_label.append(class_after)
        new_labels.append(new_label)
        det = label[1:5]
        #进行扩边
        box = det.copy()
        # print(box)
        # pdb.set_trace()
        
        det[[0,2]] *= width
        det[[1,3]] *= height

        det[0] -= det[2]/2
        det[1] -= det[3]/2
        det[2] += det[0]
        det[3] += det[1]
        x0, y0, x1, y1  = list(map(round, det))

        x0=max(0,x0)
        y0=max(0,y0)
        x1=min(width,x1)
        y1=min(height,y1)
        save_frame = frame[y0:y1, x0:x1, :]
        save_name = image_path.split('.jpg')[0]+f"_{box[0]}_{box[1]}_{box[2]}_{box[3]}_{class_after}"
        save_image_savepath = os.path.join(new_image_savepath,save_name+'.jpg')
        cv2.imwrite(save_image_savepath,save_frame)        
        


    
