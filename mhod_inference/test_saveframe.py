import os.path
import os.path
import numpy as np
import cv2
import argparse
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--input_video_dir", help="path to input video dir", default="../data/aicity2023_track5_test/videos/")
ap.add_argument("-o", "--output_video_dir", help="path to output video dir", default="../data/crop_test_frame")

args = vars(ap.parse_args())

def main():
    path_video_dir = args["input_video_dir"]
    paths_video = os.listdir(path_video_dir)
    output_video_dir = os.path.join(args['output_video_dir'] , 'images') 
    os.makedirs(output_video_dir,exist_ok=True)
    
    for i in tqdm(range(len(paths_video))):
        path_video=os.path.join(path_video_dir,paths_video[i])
        cap = cv2.VideoCapture(path_video)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率<每秒中展示多少张图片>
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
        #print(fps, width, height)

        if cap.isOpened():
            read_frame_index = 1
            while True:
                (flag, frame) = cap.read()  # 读取每一张 flag<读取是否成功> frame<内容>
                if not flag:
                    #print("haha")
                    break  # 当获取完最后一帧就结束
                frame_out = cv2.imwrite(os.path.join(output_video_dir,paths_video[i][:-4]+'_'+str(read_frame_index)+'.jpg'),frame)   
                read_frame_index+=1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print('视频打开失败！')
   
    os.makedirs(output_video_dir.replace('images','labels'),exist_ok=True)  
    # generate label
    for file in tqdm(os.listdir(output_video_dir)):
        f = open(os.path.join(output_video_dir.replace('images','labels'), file.replace('.jpg', '.txt')), 'w')
        f.write('0 0.11126041666666667 0.3880277777777778 0.0028593750000000047 0.005240740740740712\n')
        f.close()
    print('finish generate label')  


if __name__ == '__main__':
    main()
