# coding:utf-8
import os
from tqdm import tqdm

def get_yolo_merge(path1,path2):
    results = []
    if not os.path.exists(path1):
        labels1 = np.zeros((0,5))
    if not os.path.exists(path2):
        labels2 = np.zeros((0,5))      
    f = open(path1,'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        if int(float(line[0])) == 3:
            continue
        line = [float(m) for m in line]
        results.append(line)
    f.close()

    f = open(path2,'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        if int(float(line[0])) != 3:
            continue
        line = [float(m) for m in line]
        results.append(line)  
    f.close() 
    return results

# 图片文件夹，后面的/不能省
input_1 = '../../data/deta_result/model1/txt/'
input_2 = '../../data/deta_result/model2/txt/'

# xml存放的文件夹，后面的/不能省
txt_save = '../../data/deta_result/merge/txt/'
os.makedirs(txt_save,exist_ok=True)

input_1_paths = os.listdir(input_1)

for txt_name in tqdm(input_1_paths): 
    txt_path1 = os.path.join(input_1,txt_name)
    txt_path2 = os.path.join(input_2,txt_name)   
    label = get_yolo_merge(txt_path1,txt_path2)
    txt_file = open(os.path.join(txt_save,txt_name), 'w') 
    for la in label:
       txt_file.write(" ".join([str(a) for a in la])+"\n")
