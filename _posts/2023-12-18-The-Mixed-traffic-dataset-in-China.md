---
layout:     post
title:      Trajectory dataset of Mixed traffic flow in China
subtitle:   Dataset
date:       2023-12-18
author:      Zhangcun Yan
header-img: img-post/interaction-behavior-modeling/summary/traj_track.png
catalog: true
tags:
    - Technology Notebook
---


**The Mixed traffic dataset in China** 

this dataset was shared on kaggle, You can get it by this url{https://www.kaggle.com/datasets/zcyan2/mixed-traffic-trajectory-dataset-in-from-shanghai}.

we extracted images from video by frame, and labeled the objective of traffic user of each image.

To extract images from video, we design this code for automatic manner. 

```python
import cv2
import os
import re

def extract_frames_with_gpu(video_path, output_path,frame_interval):
    # open the video
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置帧的计数器
    frame_count = 0
    # 获取CUDA设备信息
    cv2.setUseOptimized(True)
    cv2.setNumThreads(64)
    cv2.ocl.setUseOpenCL(True)
    while True:
        # 读取一帧
        ret, frame = cap.read()
        # 检查是否成功读取帧
        if not ret:
            break
        # 每隔frame_interval帧保存一张图片
        if frame_count % frame_interval == 0:
            output_frame_path = f"{output_path}/frame_{frame_count}.jpg"
            cv2.imwrite(output_frame_path, frame)
        frame_count += 1
    # 关闭视频文件
    cap.release()
    print(f"Frames extracted at {output_path} with {frame_interval} frame interval.")

def Extract_file_name(file_name):
    match = re.search(r'C(\d+)', file_name)
    if match:
        extracted_text = match.group(0)
        print(extracted_text)  # 这会打印出 C0004
    return extracted_text

def create_subfolder(parent_folder, new_folder_name):
    # 拼接路径，创建新的子文件夹路径
    new_folder_path = os.path.join(parent_folder, new_folder_name)
    # 检查文件夹是否已经存在
    if not os.path.exists(new_folder_path):
        # 如果不存在，则创建新的子文件夹
        os.makedirs(new_folder_path)
        print(f"子文件夹 '{new_folder_name}' 已创建成功.")
    else:
        print(f"子文件夹 '{new_folder_name}' 已经存在.")
    return new_folder_path

# 定义自动提取和保存的路径
def File_procession(Input_file_path,Output_file_path,frame_interval):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    for i in range(len(files1)):
        Input_file_path_2 = Input_file_path+"/"+ files1[i]
        # 创建子文件夹
        first_subfile = create_subfolder(Output_file_path, files1[i])
        files2 = os.listdir(Input_file_path_2)
        for j in range(len(files2)):
            work_file = Input_file_path_2+"/"+files2[j]
            video_name = Extract_file_name(files2[j])
            output_file_path_2 = Output_file_path+files1[i]
            # 创建子文件夹
            creat_second_file = create_subfolder(first_subfile, video_name)
            save_path = Output_file_path + files1[i]+"/"+video_name+"/"
            Extract_image = extract_frames_with_gpu(work_file,save_path,frame_interval)
    return Extract_image

# path of the input and output dataset
Input_file_path = r'D:\dataset\video'
Output_file_path = r'D:\dataset\imagelabeling\intersection/'
frame_interval = 100
File_procession(Input_file_path,Output_file_path,frame_interval)
```

Then, we need put all of the images into a new folder and rename each images.

```python
import os

def Rename_image(image_file_path,image_file_save_path):
    "open each vidoe file and rename the image by the name and id"
    for  root, dirs, files in os.walk(image_file_path):
        count = 1
        for dir_name in dirs:
            # Create the subfile
            subfolder_path = os.path.join(root, dir_name)
            # get the file list in subfile
            files_in_subfolder = [f for f in os.listdir(subfolder_path) ]
            for file_name in files_in_subfolder:
                subsubfolder_path = os.path.join(subfolder_path, file_name)
                images_in_subfolder = [m for m in os.listdir(subsubfolder_path)]
                i = 0
                for image_id in images_in_subfolder:
                    new_name = f"roadside_{count:06d}.jpg"  # 例如：image_001.jpg
                    # creat the whole path of the oldpath 
                    old_path = os.path.join(subsubfolder_path, image_id)
                    new_path = os.path.join(image_file_save_path, new_name)
                    os.rename(old_path, new_path)
                    count += 1

image_file_path = r"D:\dataset\imagelabeling\intersection"
image_file_save_path = r"D:\dataset\imagelabeling\Imagedataset"
Rename_image(image_file_path,image_file_save_path)
```

We have finished the images dataset, after operated by above codes. Then, we will label the objective in each images.  For the roadside view, there are 6819 images,  and 2460 images from drone. 

 The object of two view are different, roadside view contain 8 classes:  Car; Bus(travel bus); Truck; Large-truck; Bike; Motorbike;E-bike;President. Drone view contains : Car,Bus,Truck,Large-truck,motorbike.  Now, we should select a method suited to label the objective of image effectively. and the type of these data can be used to train Yolov7 and Mask-RCNN. Here, we summary the methods of labelling image. 

**Label the image and train the Yolov8 model.**  (the code of exacting trajectory from video based on the Yolov8 and Deep-stort)

Validation of the parameters of pixies coordinators and world coordinators. Here, recommend the George to validate the coordinators.

Use **ffmpeg** split the video into different pieces. with the fellow codes:

```python
ffmpeg -i testfile.mp4 -c copy -f segment -segment_time 1200 testfile_piece_%02d.mp4
```

Use **ffmpeg** convert .mp4 format into .avi format.

```python
ffmpeg -i test.wmf test.mp4
```

In order to make sure the coordinator from  the world system and pixies is matching, we selected the software **George** to validation the error between the word and pixies.

| ID   | coordinators   | Conclude coordinators | errors        |
| :--- | :------------- | :-------------------- | :------------ |
| 0    | (29.63, 39.20) | (28.17,39.36)         | (-1.46,0.16)  |
| 1    | (0.88, 39.20)  | (0.64,39.01)          | (-0.24,-0.19) |
| 2    | (2.13,7.38)    | (1.81,7.47)           | (-0.32,0.09)  |
| 3    | (39.39,0.26)   | (39.70,0.17)          | (0.31,-0.09)  |
| 4    | (15.26,39.20)  | (16.96,39.23)         | (1.70,0.03)   |
