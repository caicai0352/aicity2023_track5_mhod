import os.path
import os.path
import numpy as np
import cv2
import argparse
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--gt_txt", help="path to gt txt", default="../data/aicity2023_track5/gt.txt")
ap.add_argument("-v", "--input_video_dir", help="path to input video dir", default="../data/aicity2023_track5/videos")
ap.add_argument("-o", "--output_video_dir", help="path to output video dir", default="../data/crop_yolo_data")
ap.add_argument("-p", "--part_gt_dir", help="path to part gt dir", default="../data/crop_yolo_data/part_gt")

args = vars(ap.parse_args())
print(args)

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
label=["motorbike","DHelmet","DNoHelmet","P1Helmet","P1NoHelmet","P2Helmet","P2NoHelmet"]

def sort_output(txt_path):
    with open(txt_path, 'r') as f:
        list = []
        for line in f:
            list.append(line.strip())

    with open(txt_path, "w") as f:
        for item in sorted(list, key=lambda x: int(str(x).split(',')[0])):
            f.writelines(item)
            f.writelines('\n')
        f.close()

def convert(line,imgWidth,imgHeight):
    left, top, width, height, classes = np.array(line[2:],np.float)
    x = (left + width / 2.0) / imgWidth
    y = (top + height / 2.0) / imgHeight
    w = width / imgWidth
    h = height / imgHeight
    classes = classes-1
    return np.array([classes, '%.6f'%x, '%.6f'%y, '%.6f'%w, '%.6f'%h],np.float) # 保留6位小数


def mot2txt(source_file, bboxs, frame,frame_id, save_dir):
    source_name = os.path.splitext(os.path.basename(source_file.name))[0]
    im_h,im_w = frame.shape[0],frame.shape[1]
    labels = []
    for i in range(bboxs):
        line = source_file.readline().strip().split(',')
        if int(source_name.split('_')[-1])==100 and int(line[0])>50 and int(line[1]) == 14:
            print(line)
            continue
        labels.append(convert(line,im_w,im_h))

    image_name = source_name + '_' + '%06d'%frame_id + '.jpg' # MOT17-02-FRCNN-000001.jpg
    newimgpath = os.path.join(save_dir,'train', 'images', image_name) # images/train/MOT17-02-FRCNN-000001.jpg
    cv2.imwrite(newimgpath,frame)

    label_name = source_name + '_' + '%06d'%frame_id + '.txt'
    labelpath = os.path.join(save_dir,'train', 'labels', label_name)
    print(labelpath)
    np.savetxt(labelpath,np.array(labels))

    return labels


def main():
    gt_txt = args["gt_txt"]
    path_video_dir = args["input_video_dir"]
    paths_video = os.listdir(path_video_dir)
    output_video_dir = args['output_video_dir']
    os.makedirs(output_video_dir,exist_ok=True)
    part_gt_dir = args['part_gt_dir']
    os.makedirs(part_gt_dir,exist_ok=True)
    
    os.makedirs(os.path.join(output_video_dir,'train', 'images'),exist_ok=True)
    os.makedirs(os.path.join(output_video_dir,'train', 'labels'),exist_ok=True)

    sort_output(gt_txt)  # 这是一个对txt文本结果排序的代码，key=video id，根据video编号排序
    for i in tqdm(range(len(paths_video))):
        path_video=os.path.join(path_video_dir,paths_video[i])
        save_video_path=os.path.join(output_video_dir,paths_video[i])
        gt=np.loadtxt(gt_txt,delimiter=',').astype(int)
        gt_part=[]
        for j in range(gt.shape[0]):
            if gt[j][0] == i+1:
                gt_part.append(gt[j,1:])
        save_partgt_path=os.path.join(part_gt_dir,f'gt_part_{i+1}.txt')
        np.savetxt(save_partgt_path,np.array(gt_part),fmt='%i',delimiter=',')

        sort_output(save_partgt_path)
        draw_txt = save_partgt_path
        cap = cv2.VideoCapture(path_video)

        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率<每秒中展示多少张图片>
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
        print(fps, width, height)
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height), True)

        source_file = open(draw_txt)
        # 把frame存入列表img_names
        img_names = []
        for line in source_file:
            staff = line.split(',')
            img_name = staff[0]
            img_names.append(img_name)

        # 将每个frame的bbox数目存入字典
        name_dict = {}
        for i in img_names:
            if img_names.count(i):
                name_dict[i] = img_names.count(i)
        #print(name_dict)
        source_file.close()

        source_file = open(draw_txt)
        # draw_mot
        if cap.isOpened():
            i = 0
            read_frame_index = 1
            while True:
                frame_index = [idx for idx in name_dict]
                (flag, frame) = cap.read()  # 读取每一张 flag<读取是否成功> frame<内容>
                if not flag:
                    print("haha")
                    break  # 当获取完最后一帧就结束
                if i >= len(frame_index):
                    # out.write(frame)
                    continue
        
                if read_frame_index != int(frame_index[i]):
                    read_frame_index+=1
                    # out.write(frame)
                    continue
                try:
                    frame_out = mot2txt(source_file, name_dict[str(frame_index[i])], frame,int(frame_index[i]),output_video_dir)
                    i+=1
                except IndexError:
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                read_frame_index+=1
        else:
            print('视频打开失败！')


if __name__ == '__main__':
    main()
