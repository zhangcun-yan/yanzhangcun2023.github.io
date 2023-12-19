---
layout:     post
title:      Micro-driving behavior modeling
subtitle:   summary
date:       2023-12-18
author:     Zhangcun Yan
header-img: img-post/interaction/causal_discovery.png
catalog: true
tags:
    - Research Notebook
---

## Micro-driving behavior modeling and analyzing based on the High-resolution traffic video in the Intersection

The implementation of computer vision algorithms has being bombed by the breakthroughs in computing power especially in the intelligent transportation systems. But there some ramparts between computer vision algorithm and traditionally traffic theory. Here, I want to show whole procedures for Modeling the interaction behavior between motorized and Non-Motorized  vehicles based the High-resolution traffic video from roadside view. To summarize those procedures and provide cleaning tutorials for the ordinal student, Some of the details background information was omitted and just saved the main steps. 

### Record video
**First**, we should prepare the basic information of object scenarios that contains the unobstructed videos, the Geometric size of the object intersection, and the signal plans. These information will help you to convert picture information to real world. For instance, the fellow figures show **the real world coordination system** and the some details about basic information.  You can establish the real world coordinator system basic on geometric information from the high-resolution measures experiment. On the other way, you can get less five marked points of the intersections by the GPS location system app on mobile phones. 
<br>
| <img src="img-post/interaction/changjidong-moyu.jpg" alt="changjidong-moyu" style="zoom:13%;" /> | <img src="img-post/interaction/Figure_sharing space in intersection.jpg" alt="Figure_sharing space in intersection" style="zoom:13%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                       a). Moyu-changji                       |                      Longchang-jiyanglu                      |
| <img src="img-post/interaction/Jianhe-xianxia road.png" alt="Jianhe-xianxia road" style="zoom:13%;" /> | <img src="img-post/interaction/WN_intersection.png" alt="WN_intersection" style="zoom:13%;" /> |
|                        jianhe-xianxia                        |                        Ningwu-Hejian                         |

​	**Figure.1. The methodology of estimating the real world coordinator**

**Second**, Make the datasets and train the detection model, you should label the object which you chosen based on your research topic(such as: Car, Bus, truck, Fright, president, electrical bicycles and bicycles so on!) in you video. Some software can help you finish this task effectively. such as, [ImageLabel](https://create.roblox.com/docs/reference/engine/classes/ImageLabel). And then, chose suite computer vision algorithm as the detection model, Here, I introduce **Yolov8** algorithm.
<br>
<img src="E:\Academic\project\Drivingbehaviormoding\study procedure\Figure\trajectory tracker.jpg" alt="trajectory tracker " style="zoom:70%;" />

​                             **Figure.2. The framework of object detection and trajectory tracking**

**Third**, Detection the object and Tracking the Trajectories. Now, we can connect the detection model with tracking model. The better tracking model is the evolution of Deepsort algorithm which is employed in our framework. 
<br>
| <img src="img-post/interaction/Object_detection.jpg" alt="Object_detection" style="zoom:10%;" /> | <img src="img-post/interaction/轨迹追踪.png" alt="轨迹追踪" style="zoom:40%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                       Object detection                       |                     trajectory tracking                      |

​                                        **Figure 3. The processor of detecting objects and tracking trajectories ** 
<br>
**Fourth**, Reconstruction of the orginal trajectories. Due to the orginal trajectory is saving in different formats, I should be reorganize before next analysis step. There are four procedures should be implemented: reformation, filter and reconstruction. The details of this procedure shown as fellow.
<br>
| <img src="img-post/interaction/image-20231212164828225.png" alt="image-20231212164828225" style="zoom:48%;" /> | <img src="img-post/interaction/image-20231212165137586.png" alt="image-20231212165137586" style="zoom:48%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| JH-XX                                                        | MY-CJ                                                        |
| <img src="img-post/interaction/image-20231212165413946.png" alt="image-20231212165413946" style="zoom:48%;" /> | <img src="img-post/interaction/image-20231212165623939.png" alt="image-20231212165623939" style="zoom:48%;" /> |
| LC-JY                                                        | NW-HJ                                                        |

<br>
*The code of reformation:*
<br>
```python
# Basic packages 
import math
import pandas as pd
import numpy as np
import os

# define new format of saving data.
# The basic information: vehicle_ID ，vehicle_type,  x , y, speed, tan_acc, lat_acc, time

```python
def Data_format_switch(data_path):
    ult_res = {}
    data1 = pd.read_csv(data_path) 
    veh_ID_list = []
    veh_type_list = []
    veh_x_list = []
    veh_y_list = []
    veh_speed_list = []
    veh_tan_acc_list = []
    veh_lat_acc_list = []
    veh_time_list = []
    vehicle_angle_list = []
    for i in range(len(data1)):
        str_res = data1.iloc[i][0] 
        ls = str_res.split(";") 
        vehicle_ID = ls[0]
        vehicle_type = ls[1]
        vehicle_trajectory = ls[10:]
        vehicle_trajectory = [vehicle_trajectory[j:j+7] for j in range(0,len(vehicle_trajectory)-1,7)] 
        vehicle_trajectory = np.array(vehicle_trajectory) 
        vehicle_trajectory_T = vehicle_trajectory.T 
        ls_x = vehicle_trajectory_T[0]
        ls_x = [np.float64(m) for m in ls_x] 
        ls_y = vehicle_trajectory_T[1]
        ls_y = [np.float64(m) for m in ls_y]
        ls_speed = vehicle_trajectory_T[2]
        ls_speed = [np.float64(m) for m in ls_speed]
        ls_tan_acc = vehicle_trajectory_T[3]
        ls_tan_acc = [np.float64(m) for m in ls_tan_acc]
        ls_lat_acc = vehicle_trajectory_T[4]
        ls_lat_acc = [np.float64(m) for m in ls_lat_acc]
        ls_time = vehicle_trajectory_T[5]
        ls_time = [np.float64(m) for m in ls_time]
        vehicle_angle = vehicle_trajectory_T[6]
        vehicle_angle = [np.float64(m) for m in vehicle_angle]
        vehicle_ID_ls = [vehicle_ID] * len(ls_time)
        vehicle_type_ls = [vehicle_type]*len(ls_time)
        veh_ID_list= veh_ID_list + vehicle_ID_ls
        veh_type_list = veh_type_list + vehicle_type_ls
        veh_x_list = veh_x_list + ls_x
        veh_y_list = veh_y_list + ls_y
        veh_speed_list = veh_speed_list + ls_speed
        veh_tan_acc_list = veh_tan_acc_list + ls_tan_acc
        veh_lat_acc_list = veh_lat_acc_list + ls_lat_acc
        veh_time_list = veh_time_list + ls_time
        vehicle_angle_list = vehicle_angle_list + vehicle_angle
    ult_res['vehicle_id'] = veh_ID_list
    ult_res['vehicle_type'] = veh_type_list
    ult_res['frame_time'] = veh_time_list
    ult_res['world_x'] = veh_x_list
    ult_res['world_y'] = veh_y_list
    ult_res['vehicle_speed'] = veh_speed_list
    ult_res['vehicle_tan_acc'] = veh_tan_acc_list
    ult_res['vehicle_lat_acc'] = veh_lat_acc_list
    ult_res['Angle'] = vehicle_angle_list
    ult_res = pd.DataFrame(ult_res)
    return ult_res

def File_procession(Input_file_path,Output_file_path):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    for i in range(len(files1)):
        work_file = Input_file_path +files1[i]
        print(work_file)
        save_path = Output_file_path +'/'+ files1[i]
        Trajectory_denoise = Data_format_switch(work_file)
        Trajectory_denoise.to_csv(save_path, index=False, header=True)
    return Trajectory_denoise

# The pathfile of the data 
input_path = r'E:/CodeResource/000_Traffic_conflict_risk_analysis/'
output_path = r'E:/CodeResource/000_Traffic_conflict_risk_analysis/Data_clearning'
Trajectory_denoise = File_procession(input_path,output_path)
```
*The code of denoise, in the first step, we should calculate the variable of the vehicle motion, then denoise the trajectory.*

```python
# calculate the kinetic parameter
def XY(groundtraj,caompartraj):
    g_World_x = np.array(groundtraj['world_x'].astype(float))
    g_World_y = np.array(groundtraj['world_y'].astype(float))
    com_World_x = np.array(caompartraj['world_x'].astype(float))
    com_World_y = np.array(caompartraj['world_y'].astype(float))
    return g_World_x,g_World_y,com_World_x,com_World_y

def Velocity(trajdata):
    """recalculate the velocity, the velocity contain the X_velocity and Y-velocity and the speed"""
    """定义初速度为0"""
    len_x = trajdata.shape[0]
    wordld_x = np.array(trajdata.world_x)
    wordld_y = np.array(trajdata.world_y)
    velocity_x = (wordld_x[1:len_x]-wordld_x[0:len_x-1])/0.04
    velocity_y = (wordld_y[1:len_x]-wordld_y[0:len_x-1])/0.04
    velocity_x = np.insert(velocity_x,0,0)
    velocity_y = np.insert(velocity_y,0,0)
    return velocity_x,velocity_y

def Accelection(trajdata):
    """定结束时刻的加速度为0"""
#     print(trajdata)
    len_x = trajdata.shape[0]
    velocity_x = np.array(trajdata.speed_x)
    velocity_y = np.array(trajdata.speed_y)
    accelection_x = (velocity_x[1:len_x]-velocity_x[0:len_x-1])/0.04
    accelection_y = (velocity_y[1:len_x]-velocity_y[0:len_x-1])/0.04
    accelection_x = np.insert(accelection_x,0,0)
    accelection_y = np.insert(accelection_y,0,0)
#     print(accelection_x)
    accelection_x[1] = 0
    accelection_y[1] = 0
    return accelection_x,accelection_y

def Aclculate_Jerk(trajdata):
    """计算急动度"""
    len_x = trajdata.acc_x.shape[0]
    acc_x = np.array(trajdata.acc_x)
    acc_y = np.array(trajdata.acc_y)
    JJerk_x = (acc_x[1:len_x] - acc_x[0:(len_x-1)])/0.04
    JJerk_y = (acc_y[1:len_x] - acc_y[0:(len_x-1)])/0.04
    JJerk_x = np.insert(JJerk_x,0,0)
    JJerk_y = np.insert(JJerk_y,0,0)
    JJerk_x[2] = 0
    JJerk_y[2] = 0
    return JJerk_x,JJerk_y

def Angle(trajectorydata):
    "calculate the Angle of vehilce"
    Angle_veh = np.array(trajectorydata.Angle)
    return Angle_veh
```


```python
# wavelet algorithm for denoising
import numpy as np
import pywt
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt

# Denoise the trajectory
def wavelet_reduce_noise(input_data_path,output_data_path):
    'if you use this code you should format the trajectory at first.'
    Traje_clearn = pd.read_csv(input_data_path)
    traj_data_df = pd.DataFrame(Traje_clearn)
    df = traj_data_df
    vehid = pd.unique(df.vehicle_id)
    Wavelet_traj = df[['vehicle_id','frame_time','vehicle_type']]
    Wt = pd.DataFrame(Wavelet_traj)
    Wt['world_x']=''
    Wt['world_y']=''
    Wt['speed_x']=''
    Wt['speed_y']=''
    Wt['acc_x']=''
    Wt['acc_y']=''
    Wt['Jerk_x']=''
    Wt['Jerk_y']=''
    Wt['Angle'] =''
    for id in range(0,len(vehid),1):
        veh_id = vehid[id]
        B =[]
        V =[]
        ACC = []
        Jerk=[]
        Angle_veh = []
        traj_data = df[df.vehicle_id==veh_id]
        if len(traj_data)>=20:
            Frame_id = pd.unique(traj_data.frame_time)
            TRAJ_world_x = traj_data['world_x']
            TRAJ_world_y = traj_data['world_y']
            min_row = traj_data.loc[traj_data['frame_time']== min(Frame_id),].index[0]
            max_row = traj_data.loc[traj_data['frame_time']== max(Frame_id),].index[0]
            AA = traj_data.iloc[min_row:max_row+1,[3,4]]
            A = df.iloc[min_row:max_row+1,[3,4]]
            # x_denoise = denoise_wavelet(TRAJ_world_x, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='sym8',
            #                             rescale_sigma='True')
            W_X = pd.array(denoise_wavelet(TRAJ_world_x, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            B.append(W_X)
            W_Y = pd.array(denoise_wavelet(TRAJ_world_y, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            B.append(W_Y)
            BB = pd.DataFrame(B)
            BBB = np.transpose(BB)
            if A.shape[0]<BBB.shape[0]:
                CB = BBB.iloc[0:A.shape[0],[0,1]]
            else:
                CB = BBB.iloc[0:len(BBB),[0,1]]
            Wt.iloc[min_row:max_row+1,[3,4]] = CB
            'Calculate the speed with wordx and word y'
            WTraj = Wt.iloc[min_row:max_row+1,0:5]
            speed_x,speed_y = Velocity(WTraj)
            "Denoise the speed of the vehicle"
            speed_x = pd.Series(speed_x)
            speed_x = speed_x.astype(np.float64)
            W_speed_x = pd.array(denoise_wavelet(speed_x, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            V.append(W_speed_x)
            speed_y = pd.Series(speed_y)
            speed_y = speed_y.astype(np.float64)
            W_speed_Y = pd.array(denoise_wavelet(speed_y, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            V.append(W_speed_Y)
            VV = pd.DataFrame(V)
            VVV = np.transpose(VV)
            if A.shape[0]<VVV.shape[0]:
                CBB = VVV.iloc[0:A.shape[0],[0,1]]
            else:
                CBB = VVV.iloc[0:len(VVV),[0,1]]
            Wt.iloc[min_row:max_row+1,[5,6]] = CBB
            # 计算加速度
            WWTraj = Wt.iloc[min_row:max_row+1,0:7]
            acc_x,acc_y = Accelection(WWTraj)
            acc_x = pd.Series(acc_x)
            acc_x = acc_x.astype(np.float64)
            W_acc_x = pd.array(denoise_wavelet(acc_x,method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            ACC.append(W_acc_x)
            acc_y = pd.Series(acc_y)
            acc_y = acc_y.astype(np.float64)
            W_acc_y = pd.array(denoise_wavelet(acc_y,method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            ACC.append(W_acc_y)
            ACCC = pd.DataFrame(ACC)
            AACCC = np.transpose(ACCC)
            if A.shape[0]<AACCC.shape[0]:
                CBBB = AACCC.iloc[0:A.shape[0],[0,1]]
            else:
                CBBB = AACCC.iloc[0:len(AACCC),[0,1]]
            Wt.iloc[min_row:max_row+1,[7,8]] = CBBB
            'calculate the jerk value'
            WWWTTraj = Wt.iloc[min_row:max_row+1,0:9]
            WWWTTTraj = pd.DataFrame(WWWTTraj)
            jerxx,jeryy = Aclculate_Jerk(WWWTTTraj)
            jerxx = pd.Series(jerxx)
            jerxx = jerxx.astype(np.float64)
            W_jerk_x = pd.array(denoise_wavelet(jerxx,method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            Jerk.append(W_jerk_x)
            jeryy = pd.Series(jeryy)
            jeryy = jeryy.astype(np.float64)
            W_jerk_y = pd.array(denoise_wavelet(jeryy,method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            Jerk.append(W_jerk_y)
            Jerkk = pd.DataFrame(Jerk)
            JJerkk = np.transpose(Jerkk)
            if A.shape[0]<VVV.shape[0]:
                CBBBB = JJerkk.iloc[0:A.shape[0],[0,1]]
            else:
                CBBBB = JJerkk.iloc[0:len(JJerkk),[0,1]]
            Wt.iloc[min_row:max_row+1,[9,10]] = CBBBB
            "Add the Angle of the vehicle"
            Angle_vehicle_traj = Angle(traj_data)
            Angle_veh.append(Angle_vehicle_traj)
            Angle_veh = pd.DataFrame(Angle_veh)
            Anngle_veh = np.transpose(Angle_veh)
            Wt.iloc[min_row:max_row+1,[11]] = Anngle_veh
        #     print(Wt.iloc[min_row:max_row+1,[9,10]])
    Wt.to_csv(output_data_path,index=False, header=True)
    return Wt

```python
# processing the file one by one
def File_procession(Input_file_path,Output_file_path):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    for i in range(len(files1)):
        work_file = Input_file_path +files1[i]
        print(work_file)
        save_path = Output_file_path +'/'+ files1[i]
        Trajectory_denoise = wavelet_reduce_noise(work_file,save_path)
    return Trajectory_denoise

# the dataset files about input path and the output data path
Input_file_path = r'D:/dataset/Changji-moyu/output_Trajectory/Reorginazation/step1/'
Output_file_path = r'D:/dataset/Changji-moyu/output_Trajectory/Reorginazation/step2'
Trajectory_denoise = File_procession(Input_file_path,Output_file_path)
```

**Show the result of trajectories, compare the variable related to the speed and the acceleration **
<br>
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def SpeedHistmap(data1,data2):
  plt.hist(data1, bins=20, density=True, alpha=0.4, label='Speed_x')
  plt.hist(data2, bins=20, density=True, alpha=0.4, label='Speed_y')
  plt.xlabel('speed(km/h)',fontsize=12)
  plt.xticks(fontsize=12,rotation=0)
  plt.yticks(fontsize=12)
  plt.ylabel('Frequency',fontsize=12)
  plt.legend(loc="upper right",fontsize=12)   #设置图例字体大小
  plt.tight_layout()
  plt.grid()
  plt.show()
    
def AccHistmap(data1,data2):
  plt.hist(data1, bins=20,color='red',density=True, alpha=0.4, label='acc_x')
  plt.hist(data2, bins=20,color='blue', density=True, alpha=0.4, label='acc_y')
  plt.xlabel('acc(m/s^2)',fontsize=12)
  plt.xticks(fontsize=12,rotation=0)
  plt.yticks(fontsize=12)
  plt.ylabel('Frequency',fontsize=12)
  plt.legend(loc="upper right",fontsize=12)   #设置图例字体大小
  plt.tight_layout()
  plt.grid()
  plt.show()
```

```python
def Compare_montion(Trajectory_No_filter,Trajectory_filter,veh_id):
    'compare the denoise effect'
    Traj_no_veh = pd.read_csv(Trajectory_No_filter)
    Traj_no_veh = pd.DataFrame(Traj_no_veh)
    Traj_filter_veh = pd.read_csv(Trajectory_filter)
    Traj_filter_veh = pd.DataFrame(Traj_filter_veh)
    Traj_nf_vehid = Traj_no_veh[Traj_no_veh['vehicle_id']==veh_id]
    Traj_f_vehid = Traj_filter_veh[Traj_filter_veh['vehicle_id']==veh_id]
    # plt.scatter(Traj_nf_vehid['frame_time'],Traj_nf_vehid['speed_x'])
    plt.scatter(Traj_nf_vehid['frame_time'], Traj_nf_vehid['speed_y'])
    # plt.scatter(Traj_f_vehid['frame_time'], Traj_f_vehid['speed_x'])
    plt.scatter(Traj_f_vehid['frame_time'], Traj_f_vehid['speed_y'])
    plt.show()
```

Before calculate the conflict risk indictors we split the trajectory into different time pieces with the signal control time for accelerating the speed of calculation.  
<br>

```python
# @ time
import os
import pandas as pd
import re

def splittrajectoryfile(traj_file_path,output_path,split_time_file):
    """read the file of trajectory and the file of splittime"""
    traj_data = pd.read_csv(traj_file_path)
    split_time_data = pd.read_csv(split_time_file,header=None)
    for time_point_id in range(0,len(split_time_data),1):
        time_point = split_time_data.iloc[time_point_id,0]
        if time_point_id >=1:
            time_point_last = split_time_data.iloc[time_point_id-1, 0]
            split_traj_data = traj_data[(float(time_point_last)*0.04<traj_data['frame_time']) & (traj_data['frame_time']<= float(time_point) * 0.04)]
        else:
            split_traj_data = traj_data[traj_data['frame_time'] <= float(time_point) * 0.04]
        "create a file to save the split traj data"
        save_file_split_traj_path = output_path +'/'+str(time_point)+".csv"
        split_traj_data.to_csv(save_file_split_traj_path, index=False, header=True)
    return

def Extract_file_name(file_name):
    match = re.search(r'C(\d+)', file_name)
    if match:
        extracted_text = match.group(0)
        print(extracted_text) 
    return extracted_text

def create_subfolder(parent_folder, new_folder_name):
    new_folder_path = os.path.join(parent_folder, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"subfile '{new_folder_name}' created successful.")
    else:
        print(f"subfile '{new_folder_name}' have been excited.")
    return new_folder_path
```

# define the input data path and the output data path

```python
def File_procession(Traj_file_path,Output_file_path,split_time_path):
    "This function will process the csv in the file path"
    files1 = os.listdir(Traj_file_path)
    for i in range(len(files1)):
        Traj_file_path_2 = Traj_file_path+"/"+ files1[i]
        # create the subfile
        first_subfile = create_subfolder(Output_file_path, files1[i])
        files2 = os.listdir(Traj_file_path_2)
        for j in range(len(files2)):
            traj_work_file = Traj_file_path_2+"/"+files2[j]
            split_time_file = split_time_path+"/" +files1[i]+ "/"+files2[j]
            Traj_video_name = Extract_file_name(files2[j])
            # output_file_path_2 = Output_file_path+files1[i]
            # # create the subfile
            output_split_traj_path = create_subfolder(first_subfile, Traj_video_name)
            # save_path = Output_file_path + files1[i]+"/"+Traj_video_name+"/"
            splittrajectoryfile(traj_work_file,output_split_traj_path,split_time_file)
    return

Traj_file_path = r"D:\dataset\Intersection\Data_processing\Denoise"
split_time_path = r"D:\dataset\Intersection\Video_cycle_split"
Output_file_path = r"D:\dataset\Intersection\Data_processing\Split_with_cycle_time/"
File_procession(Traj_file_path,Output_file_path,split_time_path)
```


```PYTHON
# @ time
import os
import pandas as pd
import re


def splittrajectoryfile(traj_file_path, output_path, split_time_file):
    """read the file of trajectory and the file of splittime"""
    traj_data = pd.read_csv(traj_file_path)
    split_time_data = pd.read_csv(split_time_file, header=0)
    split_time_data = pd.DataFrame(split_time_data)
    video_id = str(output_path.split('\\')[-1])
    split_time = split_time_data[split_time_data['video_id']==video_id]
    for time_point_id in range(1, len(split_time), 1):
        time_point = split_time.iloc[time_point_id, 4]
        if time_point_id <= 0:
            split_traj_data = traj_data[traj_data['frame_time'] <= float(time_point)]
        else:
            time_point_last = split_time.iloc[time_point_id - 1, 4]
            split_traj_data = traj_data[(float(time_point_last)  < traj_data['frame_time']) & (
                        traj_data['frame_time'] <= float(time_point) )]
        "create a file to save the split traj data"
        save_file_split_traj_path = output_path + '/' + str(time_point) + ".csv"
        split_traj_data.to_csv(save_file_split_traj_path, index=False, header=True)
    return


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


# define the input data path and the output data path
def File_procession(Traj_file_path, Output_file_path, split_time_path):
    "This function will process the csv in the file path"
    files1 = os.listdir(Traj_file_path)
    for i in range(len(files1)):
        file_name = files1[i].split('.')[0]
        Traj_file_path_2 = Traj_file_path + "/" + files1[i]
        # create the subfile
        first_subfile = create_subfolder(Output_file_path, file_name)
        files2 = os.listdir(Traj_file_path_2)
        for j in range(len(files2)):
            traj_work_file = Traj_file_path_2 + "/" + files2[j]
            # split_time_file = split_time_path + "/" + files1[i] + "/" + files2[j]
            Traj_video_name = files2[j].split('.')[0]
            # # create the subfile
            output_split_traj_path = create_subfolder(first_subfile, Traj_video_name)
            # save_path = Output_file_path + files1[i]+"/"+Traj_video_name+"/"
            splittrajectoryfile(traj_work_file, output_split_traj_path, split_time_path)
    return


Traj_file_path = r"D:\dataset\Intersection\Data_processing\denoise"
split_time_path = r"D:\dataset\Intersection\signal_time_in_video\Video_cycle_split/NW/NW_SIGNAL.csv"
Output_file_path = r"D:\dataset\Intersection\Data_processing\Split_with_cycle_time/"
File_procession(Traj_file_path, Output_file_path, split_time_path)
```
<br>
More inductor were employed in this research, such as TTC, PET, and  delta V MTTC and so on! The logistic of calculation are two kinds, first one is for by the conflict pairs, other one is by frame.  
<br>
**Firth, extracting the conflict event chain! **The next step is calculate the indictors of traffic conflicts, such as the TTC, PET, Delta-V and the risk field in the real time data. Here, summary the key step of processing. 
<br>
The idea of calculating the inductor traffic conflict between different vehicles. 
<br>


<img src="E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\conflict_calculate processing.png" alt="conflict_calculate processing" style="zoom:25%;" />

​							**Figure. 4. Calculate the TTC of two vehicles**



Here, we designs a framework which can calculate conflict risk inductors by different modal. The dynamic inductor  update frame by frame,  the static inductor calculated with whole trajectories. 
$$
Dynamic-inductors: (TTC,MTTC,\Delta{V})\\
Static-inductors:(PET)
$$

```python
# @ time
# calculate the TTC for each pair of trajectory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from itertools import combinations as comb
import os
import time
import re

def Speed_Caluca(Traj_OD,time_gap):
    Delta = (Traj_OD.values[1] - Traj_OD.values[0])
    squre = [num * num for num in Delta]
    Sum = sum(squre)
    Distance = math.sqrt(Sum)
    Speed = Distance / time_gap
    return Speed

def Montion_Equations(Traj_OD):
    Line_parmeter = pd.DataFrame(columns=['A', 'B', 'C'],index=[1]) # if you want create a new datafram you should add index=[1], it mean row
    Line_parmeter['A'] = Traj_OD['world_y'].values[1]-Traj_OD['world_y'][0]
    Line_parmeter['B'] = Traj_OD['world_x'].values[0] - Traj_OD['world_x'][1]
    Line_parmeter['C']= Traj_OD['world_x'].values[1]*Traj_OD['world_y'].values[0]-Traj_OD['world_x'].values[0]*Traj_OD['world_y'].values[1]
    return Line_parmeter

def Point_of_cross(Line_parmeter_1,Line_parmeter_2):
    Point_cross = pd.DataFrame()
    D = (Line_parmeter_1['A'].values)*(Line_parmeter_2['B'].values)-(Line_parmeter_2['A'].values)*(Line_parmeter_1['B'].values)
    # what is the D?
    if D !=0:
        point_x = ((Line_parmeter_1['B'].values)*(Line_parmeter_2['C'].values)-(Line_parmeter_2['B'].values)*(Line_parmeter_1['C'].values))/D
        Point_cross['world_x'] = point_x
        point_y = ((Line_parmeter_2['A'].values)*(Line_parmeter_1['C'].values)-(Line_parmeter_1['A'].values)*(Line_parmeter_2['C'].values))/D
        Point_cross['world_y'] = point_y
    # else:
    #     print("no conflict point")
    return Point_cross

def TTC(Point_cross,speed_A,speed_B,Traj_OD_A,Traj_OD_B,Delta):
  Delta_dis = Traj_OD_A.values[1]-Traj_OD_B.values[1]
  squre_dis = [num*num for num in Delta_dis]
  sum_dis = sum(squre_dis)
  Dis = math.sqrt(sum_dis)
  if len(Point_cross)>=1:
    Delta_a = Point_cross.values[0]-Traj_OD_A.values[1]
    squre_a = [num*num for num in Delta_a]
    sum_dis_a = sum(squre_a)
    Dis_cross_a = math.sqrt(sum_dis_a)
    if speed_A!=0:
      time_point_a = Dis_cross_a/speed_A
      # Calculate the time for car B to reach the conflict point
      Delta_b = Point_cross.values[0]-Traj_OD_B.values[1]
      squre_b = [num*num for num in Delta_b]
      sum_dis_b = sum(squre_b)
      Dis_cross_b = math.sqrt(sum_dis_b)
      time_point_b = Dis_cross_b/speed_B
      ttc_time_aver = abs(time_point_a-time_point_b)
      if ttc_time_aver < Delta:
        TTC = max(time_point_a,time_point_b)
      else:
        TTC = 10000
    else:
      TTC = 22222
  else:
    TTC = 9999999
  return TTC

def Veh_motion_state(combinations1,traj_all_data,vehicle_pair_id,Time_frame):
    'extracte the motion state of vehicle_id'
    M_veh_id = combinations1.iloc[vehicle_pair_id][0]
    NM_veh_id = combinations1.iloc[vehicle_pair_id][1]    
    Traj_mveh_id = traj_all_data[traj_all_data['vehicle_id'] == M_veh_id]
    Traj_nmveh_id = traj_all_data[traj_all_data['vehicle_id'] == NM_veh_id]    
    Time_frame = np.around(Time_frame, decimals=2)
    Traj_mveh_id_time_O = Traj_mveh_id[Traj_mveh_id['frame_time'] == Time_frame]
    Traj_nmveh_id_time_O = Traj_nmveh_id[Traj_nmveh_id['frame_time'] == Time_frame]
    Next_frame = Time_frame+0.04
    Next_frame = np.around(Next_frame, decimals=2)
    Traj_mveh_id_time_D = Traj_mveh_id[Traj_mveh_id['frame_time'] == Next_frame]
    Traj_nmveh_id_time_D = Traj_nmveh_id[Traj_nmveh_id['frame_time'] == Next_frame]
    Traj_mveh_id_OD = pd.concat([Traj_mveh_id_time_O, Traj_mveh_id_time_D], ignore_index=True)
    Traj_nmveh_id_OD = pd.concat([Traj_nmveh_id_time_O, Traj_nmveh_id_time_D], ignore_index=True)
    MVeh_a_point = Traj_mveh_id_OD[['world_x', 'world_y']]
    NMVeh_a_point = Traj_nmveh_id_OD[['world_x', 'world_y']]
    return MVeh_a_point,NMVeh_a_point

def Creat_trajectory_pair(trajectory_data):
    "This function create the trajectory pair form the trajectory data"
    Objectory_ids = pd.unique(trajectory_data['vehicle_id'])
    # the pair of the vehicle id
    combinations1 = list(comb(Objectory_ids, 2))
    combinations1 = pd.DataFrame(combinations1)
    return combinations1

def Extract_same_time(first_vehicle_inf,second_vehicle_inf):
    "extract the same time frame time"
    first_vehicle_all_frame = first_vehicle_inf['frame_time']
    second_vehicle_all_frame = second_vehicle_inf['frame_time']
    same_time_frame = pd.Series(list(set(first_vehicle_all_frame) & set(second_vehicle_all_frame)))
    return same_time_frame

def Extract_traj_by_frame_time(trajectory,same_time_frame):
    "extract the trajectory by the frame time"
    "the input is the first_vehicle_inf or the second_vehicle_inf"
    same_time_inf = trajectory[trajectory['frame_time'].isin(same_time_frame)]
    return same_time_inf

def Extract_same_frame_time(combinations1,trajectory_data,id):
    M_veh_id = combinations1.iloc[id][0]
    NM_veh_id = combinations1.iloc[id][1]
    first_vehicle_inf = trajectory_data[trajectory_data['vehicle_id'] == M_veh_id]
    second_vehicle_inf = trajectory_data[trajectory_data['vehicle_id'] == NM_veh_id]
    "Search the same frame time of the two vehicles"
    second_vehicle_inf = second_vehicle_inf.reset_index(drop=True)
    F_vehicle_frame_time = first_vehicle_inf['frame_time']
    S_vehicle_frame_time = second_vehicle_inf['frame_time']
    same_time_frame = np.intersect1d(F_vehicle_frame_time, S_vehicle_frame_time)
    same_time_frame = np.sort(same_time_frame)
    return same_time_frame

def Extract_inf_of_each_vehicle_by_same_time(combinations1,trajectory_data,id,same_time_frame):
    M_veh_id = combinations1.iloc[id][0]
    NM_veh_id = combinations1.iloc[id][1]
    Mvehicle_inf = trajectory_data[trajectory_data['vehicle_id'] == M_veh_id]
    NMvehicle_inf = trajectory_data[trajectory_data['vehicle_id'] == NM_veh_id]
    MV_trajectory_same_time = Extract_traj_by_frame_time(Mvehicle_inf, same_time_frame)
    NMV_trajectory_same_time = Extract_traj_by_frame_time(NMvehicle_inf, same_time_frame)
    return MV_trajectory_same_time,NMV_trajectory_same_time

def Ture_veh_pairs(combinations1,trajectory_data):
    "this function make sure the same time of the two vehicles"
    delete_id = []
    for id in range(0,len(combinations1),1):
        # print(str(id) + '/' + str(len(combinations1)))
        same_time_frame = Extract_same_frame_time(combinations1,trajectory_data,id)
        if np.size(same_time_frame)<=0:
            # print(same_time_frame)
            delete_id.append(id)
    combinations1 = combinations1.drop(delete_id)
    combinations1 = combinations1.reset_index(drop=True)
    return combinations1

def Extract_the_MV_NMV_pairs(trajectory,combinations1):
    """"extract all mv-nmv interaction pairs"""
    delete_no_id = []
    for id in range(0, len(combinations1), 1):
        F_veh_id = combinations1.iloc[id][0]
        S_veh_id = combinations1.iloc[id][1]
        F_veh_type_value = trajectory.loc[trajectory['vehicle_id']==F_veh_id,'vehicle_type']
        F_veh_type = pd.unique(F_veh_type_value)
        S_veh_type_value = trajectory.loc[trajectory['vehicle_id']==S_veh_id,'vehicle_type']
        S_veh_type = pd.unique(S_veh_type_value)
        if (F_veh_type == S_veh_type):
            delete_no_id.append(id)
        elif F_veh_type != " Car" and S_veh_type != " Car":
            delete_no_id.append(id)
    combinations_MV_NMV = combinations1.drop(delete_no_id)
    combinations_MV_NMV = combinations_MV_NMV.reset_index(drop=True)
    return combinations_MV_NMV

def Extract_the_conflict_pairs_by_dis(trajectory,combinations_MNM,id,same_time_frame):
    # extract the vehicle pair with the dis less than 20mi
    F_veh_id = combinations_MNM.iloc[id][0]
    S_veh_id = combinations_MNM.iloc[id][1]
    F_veh_traj = trajectory[trajectory['vehicle_id']==F_veh_id]
    S_veh_traj = trajectory[trajectory['vehicle_id'] == S_veh_id]
    F_veh_type_value = F_veh_traj[F_veh_traj['frame_time'].isin(same_time_frame)]
    F_X_Y = F_veh_type_value[['world_x','world_y']]
    S_veh_type_value = S_veh_traj[S_veh_traj['frame_time'].isin(same_time_frame)]
    S_X_Y = S_veh_type_value[['world_x', 'world_y']]
    A = F_X_Y.values
    B = S_X_Y.values
    distances = [np.linalg.norm(np.array(a) - np.array(b)) for a, b in zip(A, B)]
    min_dis = min(distances)
    return distances,min_dis

def Calculate_TTC_value(trajectory_path,Columns_name,time_gap,Delta):
    "this function main to check in the same time of the trajectory"
    start_time = time.time()
    trajectory_data_orginal = pd.read_csv(trajectory_path)
    # We only focus on the interaction between MV and NMV
    trajectory_data_orginal = pd.DataFrame(trajectory_data_orginal)
    trajectory_data = trajectory_data_orginal[trajectory_data_orginal["vehicle_type"].isin([' Car',' Bicycle',' Motorcycle'])]
    combinations2 = Creat_trajectory_pair(trajectory_data)
    combinations1 = Ture_veh_pairs(combinations2, trajectory_data)
    combinations_MNM = Extract_the_MV_NMV_pairs(trajectory_data, combinations1)
    # saving the conflict information
    df_all_veh_pairs = pd.DataFrame(columns=Columns_name)
    for id in range(0,len(combinations_MNM),1):
        print(str(id)+'/'+str(len(combinations_MNM)))
        same_time_frame = Extract_same_frame_time(combinations_MNM, trajectory_data, id)
        "extract the same time series"
        df_one_veh_pair = pd.DataFrame(columns=Columns_name)
        MV_trajectory_same_time, NMV_trajectory_same_time = Extract_inf_of_each_vehicle_by_same_time(combinations_MNM,trajectory_data,id,same_time_frame)
        distances,min_dis = Extract_the_conflict_pairs_by_dis(trajectory_data, combinations_MNM, id, same_time_frame)
        if min_dis<=20:
            for frame_time_id in same_time_frame:
                df_one_frame = pd.DataFrame(columns=Columns_name)
                first_trajectory_rame_time_id = MV_trajectory_same_time[MV_trajectory_same_time['frame_time']==frame_time_id]
                second_trajectory_rame_time_id = NMV_trajectory_same_time[NMV_trajectory_same_time['frame_time'] == frame_time_id]
                MVeh_a_point, NMVeh_a_point = Veh_motion_state(combinations_MNM,trajectory_data,id,frame_time_id)
                if (len(MVeh_a_point) >= 2) & (len(NMVeh_a_point) >= 2):
                    F_veh_speed = Speed_Caluca(MVeh_a_point, time_gap)
                    S_veh_speed = Speed_Caluca(NMVeh_a_point, time_gap)
                    F_Line_parmeter = Montion_Equations(MVeh_a_point)
                    S_Line_parmeter = Montion_Equations(NMVeh_a_point)
                    'Calculate the cross point'
                    cross_point = Point_of_cross(F_Line_parmeter, S_Line_parmeter)
                    # if there are no cross points, we should stop the next step
                    if len(cross_point) != 0:
                        'Calculate the TTC'
                        conflict_TTC = TTC(cross_point, F_veh_speed, S_veh_speed, MVeh_a_point, NMVeh_a_point, Delta)
                        # creat a list which contain the vehicle information and the PET at every fram
                        df_one_frame[['F_vehicle_id', 'F_frame_time','F_vehicle_type','F_world_x', 'F_world_y', 'F_speed_x','F_speed_y','F_acc_x','F_acc_y','F_Jerk_x','F_Jerk_y','F_Angle']] = first_trajectory_rame_time_id
                        df_one_frame['S_vehicle_id'] = second_trajectory_rame_time_id['vehicle_id'].values
                        df_one_frame['S_vehicle_type'] = second_trajectory_rame_time_id['vehicle_type'].values
                        df_one_frame['S_world_x'] = second_trajectory_rame_time_id['world_x'].values
                        df_one_frame['S_world_y'] = second_trajectory_rame_time_id['world_y'].values
                        df_one_frame['S_speed_x'] = second_trajectory_rame_time_id['speed_x'].values
                        df_one_frame['S_speed_y'] = second_trajectory_rame_time_id['speed_y'].values
                        df_one_frame['S_acc_x'] = second_trajectory_rame_time_id['acc_x'].values
                        df_one_frame['S_acc_y'] = second_trajectory_rame_time_id['acc_y'].values
                        df_one_frame['S_Jerk_x'] = second_trajectory_rame_time_id['Jerk_x'].values
                        df_one_frame['S_Jerk_y'] = second_trajectory_rame_time_id['Jerk_y'].values
                        df_one_frame['S_Angle'] = second_trajectory_rame_time_id['Angle'].values
                        df_one_frame['cross_point_x'] = cross_point.values[0][0]
                        df_one_frame['cross_point_y'] = cross_point.values[0][1]
                        df_one_frame['TTC'] = conflict_TTC
                        df_one_veh_pair_add = [df_one_veh_pair,df_one_frame]
                        df_one_veh_pair = pd.concat(df_one_veh_pair_add)
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            print(f"Elapsed Time: {elapsed_time} seconds")
            df_all_veh_pairs_add = [df_all_veh_pairs,df_one_veh_pair]
            df_all_veh_pairs = pd.concat(df_all_veh_pairs_add)
    return df_all_veh_pairs

def create_subfolder(parent_folder, new_folder_name):
    new_folder_path = os.path.join(parent_folder, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"subfile '{new_folder_name}' created successful.")
    else:
        print(f"subfile '{new_folder_name}' have been excited.")
    return new_folder_path

def File_procession(Input_file_path,Output_file_path,time_gap,Delta,Columns_name):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    for i in range(len(files1)):
        Input_file_path_2 = Input_file_path+"/"+ files1[i]
        # create the subfile
        first_subfile = create_subfolder(Output_file_path, files1[i])
        files2 = os.listdir(Input_file_path_2)
        for j in range(len(files2)):
            work_file = Input_file_path_2+"/"+files2[j]
            # Traj_split_time_name = Extract_file_name(files2[j])
            output_split_traj_path = first_subfile + "/" + files2[j]
            Trajectory_conflict = Calculate_TTC_value(work_file,Columns_name,time_gap,Delta)
            Trajectory_conflict.to_csv(output_split_traj_path, index=False, header=True)
    return Trajectory_conflict

Columns_name = ['F_vehicle_id','F_frame_time','F_vehicle_type','F_world_x','F_world_y','F_speed_x','F_speed_y','F_acc_x', 'F_acc_y', 'F_Jerk_x', 'F_Jerk_y','F_Angle','S_vehicle_id','S_vehicle_type','S_world_x','S_world_y','S_speed_x','S_speed_y','S_acc_x','S_acc_y','S_Jerk_x','S_Jerk_y','S_Angle','cross_point_x','cross_point_y','TTC']
Delta = 0.5
time_gap = 0.04
data_path =  r'D:\dataset\Intersection\Data_processing\Split_with_cycle_time\JH/'
output_path = r'D:\dataset\Intersection\Data_processing\Riskinductor\TTC'
Trajectory_conflict = File_procession(data_path,output_path,time_gap,Delta,Columns_name)
```

The flowchart of modeling interaction behavior between Motorized vehicle and Non-Motorized vehicles.

![flowchart](img-post/interaction/flowchart.png)

The distribution of the conflict risk shows as fellow, the data from the Longchang-Ningwu intersections.

​                    **Figure. 5. The result of the risk inductors calculated from the trajectories**
<br>
**sixth, Modeling the evolving procession of interaction behavior between the motorized and Non-motorized vehicles.** 
<br>
a) **The first step is extract the event chain about the course of interacting.**

```python
import numpy as np
import pandas as pd
import os

# 对冲突类型进行分类，这边保留碰撞对象中涉及机动车的所有事故
def Extract_conflict_relate_car(conflcit_envent_data_file):
    df = pd.read_csv(conflcit_envent_data_file)
    df = pd.DataFrame(df)
    #  提取其中一辆车是机动车的交互事件
    interfere_motorveh = df[(df['F_vehicle_type'] == " Car") | (df['S_vehicle_type'] == " Car")]
    return interfere_motorveh
# @ time

# 对数据进行重定义，筛选dataframe列表中第一辆车的类型不是car的所有行对应的列表，再调换新列表中第一辆车信息与第二辆车信息。
def Restructure_data_TTC(interfere_motor_vehicle,Columns_name):
    df = pd.DataFrame(interfere_motor_vehicle)
    df_first_car = df[df['F_vehicle_type']==" Car"]
    df_first_no_car = df[df['F_vehicle_type']!=" Car"]
    new_df_first_no_car = df_first_no_car.reset_index(drop=True)
    columnss = new_df_first_no_car.columns.tolist()
    # 这边必须是：间隔
    new_columns = columnss[12:13] + columnss[1:2]+ columnss[13:23]+columnss[0:1]+columnss[2:12] + columnss[23:26]
    df_new_no_car_first = new_df_first_no_car[new_columns]
    df_new_no_car_first.columns.name = None
#     df_new_no_car_first_T = df_new_no_car_first.rename(columns=dict(zip(df.columns, Columns_name)), inplace=False)
    df_new_no_car_first.columns = Columns_name
    New_df = pd.concat([df_first_car,df_new_no_car_first])
    return New_df

def calculate_the_area_blow_2(event_data):
    "calculate the area blow the 2 seconds"
    df = pd.DataFrame(event_data)
    df_ttc_blow_2 = df[df['TTC']<=2]
    area_df_ttc_below_2 = 0.12*df_ttc_blow_2['TTC']
    total_area = sum(area_df_ttc_below_2)
    return total_area

# extract the conflict event chain
def Extract_event_by_ttc(envent_data,colmun_name):
    # create a empty matrix , save the conflict event chain
    ttc_event_chain = pd.DataFrame(columns=colmun_name)
    # acquire the veh pair
    veh_pairs = envent_data[['F_vehicle_id','S_vehicle_id']]
    # delete the repeat row
    veh_pairs_unique = veh_pairs.drop_duplicates()
    # extract the veh pairs
    pairs = list(zip(veh_pairs_unique['F_vehicle_id'], veh_pairs_unique['S_vehicle_id']))
    for id in pairs:
        conflict_env = envent_data[(envent_data['F_vehicle_id'] == id[0]) & (envent_data['S_vehicle_id'] == id[1])]
        # delete the no TTC value <10
        # has_not_nan = conflict_env['TTC'].notna().any()
        conflict_env['TTC'].fillna(10, inplace=True)
        conflict_env.loc[conflict_env['TTC'] >= 10, 'TTC'] = 10
        conflict_env = conflict_env.reset_index(drop=True)
        min_ttc = min(conflict_env['TTC'])
        N_ttc_low_5 = len(conflict_env[conflict_env["TTC"] <= 5])
        if (min_ttc<=2) and (N_ttc_low_5>=45):
            frame_min_ttc = conflict_env.loc[conflict_env['TTC'] == min_ttc,'F_frame_time'].iloc[0]
            befor_frame = round(frame_min_ttc - 5, 2)
            after_frame = round(frame_min_ttc + 5, 2)
            Event_chain = conflict_env[
                (conflict_env['F_frame_time'] >= befor_frame) & (conflict_env['F_frame_time'] <= after_frame)]
            total_area = calculate_the_area_blow_2(Event_chain)
            Event_chain['area_below_2'] = total_area
            ttc_event_chain = pd.concat([ttc_event_chain, Event_chain])
    return ttc_event_chain

def create_subfolder(parent_folder, new_folder_name):
    new_folder_path = os.path.join(parent_folder, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"subfile '{new_folder_name}' created successful.")
    else:
        print(f"subfile '{new_folder_name}' have been excited.")
    return new_folder_path

def File_procession(Input_file_path, Output_file_path,Columns_name,total_dataset_path):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    Total_event_chain = pd.DataFrame(columns=Columns_name)
    for i in range(len(files1)):
        Input_file_path_2 = Input_file_path + "/" + files1[i]
        # create the subfile
        first_subfile = create_subfolder(Output_file_path, files1[i])
        files2 = os.listdir(Input_file_path_2)
        for j in range(len(files2)):
            work_file = Input_file_path_2 + "/" + files2[j]
            output_event_chain_path = first_subfile + "/" + files2[j]
            interfere_motorveh = Extract_conflict_relate_car(work_file)
            New_df = Restructure_data_TTC(interfere_motorveh, Columns_name)
            Event_veh_pairs_chain = Extract_event_by_ttc(New_df,Columns_name)
            Event_veh_pairs_chain['video_id'] = str(files1[i])
            Event_veh_pairs_chain.to_csv(output_event_chain_path, index=False, header=True)
            total_chain_path = total_dataset_path
            Total_event_chain = pd.concat([Total_event_chain,Event_veh_pairs_chain])
    Total_event_chain.to_csv(total_chain_path,index=False, header=True)
    return
```

```python
# extract conflict event chain from conflict risk data
Columns_name = ['F_vehicle_id','F_frame_time','F_vehicle_type','F_world_x','F_world_y','F_speed_x','F_speed_y','F_acc_x', 'F_acc_y', 'F_Jerk_x', 'F_Jerk_y','F_Angle','S_vehicle_id','S_vehicle_type','S_world_x','S_world_y','S_speed_x','S_speed_y','S_acc_x','S_acc_y','S_Jerk_x','S_Jerk_y','S_Angle','cross_point_x','cross_point_y','TTC']
total_dataset_path = 'D:/dataset/Intersection/Data_processing/Conflict_event/TTC/Total_dataset/MY_conflict.csv'
ttc_data_path = r"D:/dataset/Intersection/Data_processing/Riskinductor/TTC/MY"
conflict_event_save = r'D:/dataset/Intersection/Data_processing/Conflict_event/TTC/MY'
File_procession(ttc_data_path, conflict_event_save,Columns_name,total_dataset_path)
```

**b) The second step define the variables related to the severity of conflict risk.**

This step, we connect interaction procession with the environment impact factor, Signal control, intersection design factor and other dataset. 

```PYTHON
#  Signal control, 
import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import norm
import pandas as pd
import math
import numpy as np

def signial_time_variable(conflict_event, signal_data_path, colum_name3):
    "Merage the signal variable with the other variables"
    conflict_event_data = pd.DataFrame(conflict_event)
    signal_data = pd.read_csv(signal_data_path)
    df_signal_data = pd.DataFrame(signal_data)
    # we need select the same time frame in time list
    ttc_event_chain_3 = pd.DataFrame(columns=colum_name3)
    vid_nums = pd.unique(conflict_event_data['Video_id'])
    for vid in vid_nums:
    # for vid in ['C00040']:
        print(vid)
        vid_lc_data = conflict_event_data[conflict_event_data['Video_id'] == vid]
        lc_data_signal_vid = df_signal_data[df_signal_data['Video_id'] == vid]
        pairs_nums = vid_lc_data[['F_vehicle_id','S_vehicle_id']]
        pairs_nums = pairs_nums.drop_duplicates(subset=['F_vehicle_id', 'S_vehicle_id'])
        pairs_nums = pairs_nums.values.tolist()
        vid_lc_data_time = vid_lc_data.drop_duplicates()
        # vid_lc_data_time = vid_lc_data_time.values.tolist()
        for pair_id in pairs_nums:
            # print(pair_id)
            pair_id_event = vid_lc_data_time[(vid_lc_data_time['F_vehicle_id'] == pair_id[0]) & (vid_lc_data_time['S_vehicle_id'] == pair_id[1])]
            # extract the traffic information in the same time
            Frame_time = pd.unique(pair_id_event['F_frame_time'])
            for time_id in Frame_time:
                # search the inf with same time_frame
                print(time_id)
                conflcit_event_data_time_id = pair_id_event[pair_id_event['F_frame_time'] == time_id]
                # here i need the choice the time variable with the signal plan
                frame_time_value = float(time_id)
                signal_inf = lc_data_signal_vid[(lc_data_signal_vid['G_B_F'] <= frame_time_value) & (lc_data_signal_vid['R_B_F'] >= frame_time_value)]
                # here, divid into detial function level
                print(signal_inf['Y_B_F'].values)
                if frame_time_value<=float(signal_inf['Y_B_F'].values):
                    light_Color = "green"
                    light_time = frame_time_value-float(signal_inf['G_B_F'].values)
                elif frame_time_value<=float(signal_inf['R_B_F'].values):
                    light_Color = "yellow"
                    light_time = frame_time_value - float(signal_inf['Y_B_F'].values)
                    if light_time>3:
                        light_Color = "red"
                        light_time = light_time-3
                print(light_time,light_Color)
                signal_inf = signal_inf.reset_index(drop=True)
                conflcit_event_data_time_id['cycle_id'] = signal_inf.loc[0, 'cycle_id']
                conflcit_event_data_time_id['phase'] = signal_inf.loc[0, 'phase']
                conflcit_event_data_time_id['light_color'] = str(light_Color)
                conflcit_event_data_time_id['light_time'] = float(light_time)
                # rowid += 1
                ttc_event_chain_3 = pd.concat([ttc_event_chain_3, conflcit_event_data_time_id])
                    # print(ttc_event_chain_3)
    return ttc_event_chain_3
# @ time
```

```PYTHON
colum_name3 = ["Video_id",'F_vehicle_id', 'F_frame_time', 'F_vehicle_type', 'F_world_x', 'F_world_y', 'F_speed_x', 'F_speed_y', 'F_acc_x', 'F_acc_y', 'F_Jerk_x', 'F_Jerk_y', 'F_Angle','S_vehicle_id', 'S_vehicle_type', 'S_world_x', 'S_world_y', 'S_speed_x', 'S_speed_y', 'S_acc_x','S_acc_y', 'S_Jerk_x', 'S_Jerk_y', 'S_Angle', 'cross_point_x', 'cross_point_y', 'TTC']
conflict_event_path = r"D:\dataset\Intersection\Data_processing\Conflict_event\TTC\Total_dataset/MY_conflict.csv"
conflict_event = pd.read_csv(conflict_event_path)
signal_data_path = r"D:\dataset\Intersection\signal_time_in_video\Total_signal/MY_SIGNAL.csv"
all_variable_add_signal = signial_time_variable(conflict_event,signal_data_path,colum_name3)
save_path3 = r"D:\dataset\Intersection\Data_processing\Conflict_event\TTC\Add_signal_with_event/event_signal_MY.csv"
all_variable_add_signal.to_csv(save_path3,index=False, header=True)
```

Define the variable of driving_purpose. In this step,  we need match label each trajectory by the turn direction classic. the classic of all the direction in this intersection is divided into 12 kinds, as show in the fellow figure.

|                     Direction of Turning                     |                       MY intersection                        |                       JH intersection                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="img-post/interaction/Direction_of_intersection.jpg" alt="Direction_of_intersection" style="zoom:18%;" /> | <img src="img-post/interaction/MY.png" alt="MY" style="zoom:40%;" /> | <img src="img-post/interaction/myplot.png" alt="myplot" style="zoom:40%;" /> |
<br>
The logistics of this code is that 
<br>
First, we should integrate the trajectory by the intersection name.
<br>


```python
# intergate the trajectory together. 
def File_procession(Input_file_path,Output_file_path):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    total_data = pd.DataFrame()
    for i in range(len(files1)):
        work_file = Input_file_path +files1[i]
        Traj = pd.read_csv(work_file)
        video_id = files1[i][:-4]
        Traj['video_id'] = video_id
        total_data = pd.concat([total_data,Traj])
    total_data.to_csv(Output_file_path, index=False, header=True)
    return total_data

Input_file_path = r"D:\dataset\Intersection\Data_processing\Mixed_traffic_flow_in_intersection_of_china\MY/"
Output_file_path = r'D:\dataset\Intersection\Data_processing\Mixed_traffic_flow_in_intersection_of_china\Total_data_set/MY_total.csv'
total_data = File_procession(Input_file_path,Output_file_path)
```

<br>

```python
Org_traj = pd.read_csv(orj_traj_data_path)
Org_traj = pd.DataFrame(Org_traj)
# ,' Motorcycle',' Bicycle'
Vehicle_type = [' Car']
Org_traj = Org_traj[Org_traj['vehicle_type'].isin(Vehicle_type)]
Org_traj = pd.DataFrame(Org_traj)
Org_traj =  Org_traj[Org_traj['world_x']<=50]
Org_traj =  Org_traj[Org_traj['world_x']>=0]
Org_traj =  Org_traj[Org_traj['world_y']>=-10]
Org_traj =  Org_traj[Org_traj['world_y']<=50]
# Org_traj = Org_traj[Org_traj['world_x']<=50 & Org_traj['world_y']<=50 & Org_traj['world_x']>=0 & Org_traj['world_y']>=-10]
plt.figure(figsize=(18,18))
plt.scatter(Org_traj['world_x'],Org_traj['world_y'],s=0.05)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid()
plt.show()
```

Define_direction.

```python
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import os

def Devided_driving_direction(trajectory_data,direction_devied_path):
    dir_plan = pd.read_csv(direction_devied_path)
    # make sure the plane of the trajectory
    Orj_point = trajectory_data[['world_x','world_y']].head(10)
    End_point = trajectory_data[['world_x','world_y']].tail(10)
    Orj_point_x = Orj_point['world_x'].mean()
    Orj_point_y = Orj_point['world_y'].mean()
    End_point_x = End_point['world_x'].mean()
    End_point_y = End_point['world_y'].mean()
    print(dir_plan.iloc[0,0])
    print(dir_plan.iloc[0,1])
    if (Orj_point_x <dir_plan.iloc[0,0] ) and (Orj_point_y > dir_plan.iloc[0,1]):
        Orginal_site = 'N_in'
        if (End_point_x >dir_plan.iloc[0,2]) and (End_point_y >dir_plan.iloc[0,3]):
            End_site = 'N_out'
            direction = 'NN'
        elif (End_point_x <dir_plan.iloc[1,2]) and (End_point_y >dir_plan.iloc[1,3]):
            End_site = 'W_out'
            direction = 'NW'
        elif (End_point_x <dir_plan.iloc[2,2]) and (End_point_y <dir_plan.iloc[2,3]):
            End_site = 'S_out'
            direction = "NS"
        elif (End_point_x >dir_plan.iloc[3,2]) and (End_point_y <dir_plan.iloc[3,3]):
            End_site = 'E_out'
            direction = "NE"
        else:
            End_site = 'nan'
            direction = "nan"
    elif (Orj_point_x < dir_plan.iloc[1,0]) and (Orj_point_y < dir_plan.iloc[1,1]):
        Orginal_site = 'W_in'
        if (End_point_x >dir_plan.iloc[0,2]) and (End_point_y >dir_plan.iloc[0,3]):
            End_site = 'N_out'
            direction = 'WN'
        elif (End_point_x <dir_plan.iloc[1,2]) and (End_point_y >dir_plan.iloc[1,3]):
            End_site = 'W_out'
            direction = 'WW'
        elif (End_point_x <dir_plan.iloc[2,2]) and (End_point_y <dir_plan.iloc[2,3]):
            End_site = 'S_out'
            direction = "WS"
        elif (End_point_x >dir_plan.iloc[3,2]) and (End_point_y <dir_plan.iloc[3,3]):
            End_site = 'E_out'
            direction = "WE"
        else:
            End_site = 'nan'
            direction = "nan"
    elif (Orj_point_x > dir_plan.iloc[2,0]) and (Orj_point_y < dir_plan.iloc[2,1]):
        Orginal_site = 'S_in'
        if (End_point_x >dir_plan.iloc[0,2]) and (End_point_y >dir_plan.iloc[0,3]):
            End_site = 'N_out'
            direction = 'SN'
        elif (End_point_x <dir_plan.iloc[1,2]) and (End_point_y >dir_plan.iloc[1,3]):
            End_site = 'W_out'
            direction = 'SW'
        elif (End_point_x <dir_plan.iloc[2,2]) and (End_point_y <dir_plan.iloc[2,3]):
            End_site = 'S_out'
            direction = "SS"
        elif (End_point_x >dir_plan.iloc[3,2]) and (End_point_y <dir_plan.iloc[3,3]):
            End_site = 'E_out'
            direction = "SE"
        else:
            End_site = 'nan'
            direction = "nan"
    elif (Orj_point_x > dir_plan.iloc[3,0]) and (Orj_point_y > dir_plan.iloc[3,1]):
        Orginal_site = 'E_in'
        if (End_point_x >dir_plan.iloc[0,2]) and (End_point_y >dir_plan.iloc[0,3]):
            End_site = 'N_out'
            direction = 'EN'
        elif (End_point_x <dir_plan.iloc[1,2]) and (End_point_y >dir_plan.iloc[1,3]):
            End_site = 'W_out'
            direction = 'EW'
        elif (End_point_x <dir_plan.iloc[2,2]) and (End_point_y <dir_plan.iloc[2,3]):
            End_site = 'S_out'
            direction = "ES"
        elif (End_point_x >dir_plan.iloc[3,2]) and (End_point_y <dir_plan.iloc[3,3]):
            End_site = 'E_out'
            direction = "EE"
        else:
            End_site = 'nan'
            direction = "nan"
    else:
        Orginal_site = 'nan'
        End_site = 'nan'
        direction = 'nan'
    return Orginal_site,End_site,direction

def Define_direction(conf_event_data_path, orj_traj_data_path, direction_path,Output_data_path):
    con_event_dir = pd.DataFrame()
    con_event = pd.read_csv(conf_event_data_path)
    con_event = pd.DataFrame(con_event)
    Org_traj = pd.read_csv(orj_traj_data_path)
    Org_traj = pd.DataFrame(Org_traj)
    con_veh_pairs = con_event[['F_vehicle_id', 'S_vehicle_id', 'Video_id']]
    df = pd.DataFrame(con_veh_pairs)
    con_veh_pairs = df.drop_duplicates(subset=['F_vehicle_id', 'S_vehicle_id', 'Video_id'])
    veh_pairs = con_veh_pairs.values
    for veh_pair in veh_pairs:
        Org_traj_conf_pair = Org_traj[Org_traj['video_id'] == veh_pair[2]]
        M_org_traj = Org_traj_conf_pair[Org_traj_conf_pair['vehicle_id'] == veh_pair[0]]
        N_org_traj = Org_traj_conf_pair[Org_traj_conf_pair['vehicle_id'] == veh_pair[1]]
        con_event_veh_pair = con_event[con_event["Video_id"]==veh_pair[2]]
        con_event_veh_pair_trj = con_event_veh_pair[con_event_veh_pair["F_vehicle_id"]==veh_pair[0]]
        # divided the driving propose into 12 class
        M_Orginal_site, M_End_site, M_direction = Devided_driving_direction(M_org_traj,direction_path)
        N_Orginal_site, N_End_site, N_direction = Devided_driving_direction(N_org_traj, direction_path)
        con_event_veh_pair_trj['F_O'] = M_Orginal_site
        con_event_veh_pair_trj['F_D'] = M_End_site
        con_event_veh_pair_trj['F_Dir'] = M_direction
        con_event_veh_pair_trj['S_O'] = N_Orginal_site
        con_event_veh_pair_trj['S_D'] = N_End_site
        con_event_veh_pair_trj['S_Dir'] = N_direction
        con_event_dir = pd.concat([con_event_dir,con_event_veh_pair_trj])
    con_event_dir.to_csv(Output_data_path,index=False, header=True)
    return con_event_dir

conf_event_data_path = r"D:\dataset\Intersection\Data_processing\Conflict_event\TTC\Add_signal_with_event/event_signal_JH.csv"
orj_traj_data_path = r"D:\dataset\Intersection\Data_processing\Mixed_traffic_flow_in_intersection_of_china\Total_data_set/JH_total.csv"
direction_devied_path = r"D:\dataset\Intersection\Driving_direction/JHD.csv"
out_put_path = r"D:\dataset\Intersection\Data_processing\Conflict_event\TTC\Add_dir_with_event/JH_add_dir.csv"
Define_direction(conf_event_data_path,orj_traj_data_path,direction_devied_path,out_put_path)
```

Calculate the all variables and define. At first, we can connect the dir_signal file with ordinal data.
<br>
```python
import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import norm
import pandas as pd
import math
import numpy as np



def signial_time_variable(conflict_event, signal_data_path):
    "Merage the signal variable with the other variables"
    conflict_event_data = pd.DataFrame(conflict_event)
    signal_data = pd.read_csv(signal_data_path)
    df_signal_data = pd.DataFrame(signal_data)
    # we need select the same time frame in time list
    ttc_event_chain_3 = pd.DataFrame()
    vid_nums = pd.unique(conflict_event_data['Video_id'])
    for vid in vid_nums:
    # for vid in ['C00040']:
        vid_lc_data = conflict_event_data[conflict_event_data['Video_id'] == vid]
        lc_data_signal_vid = df_signal_data[df_signal_data['video_id'] == vid]
        pairs_nums = vid_lc_data[['F_vehicle_id','S_vehicle_id']]
        pairs_nums = pairs_nums.drop_duplicates(subset=['F_vehicle_id', 'S_vehicle_id'])
        pairs_nums = pairs_nums.values.tolist()
        vid_lc_data_time = vid_lc_data.drop_duplicates()
        # vid_lc_data_time = vid_lc_data_time.values.tolist()
        for pair_id in pairs_nums:
            # print(pair_id)
            pair_id_event = vid_lc_data_time[(vid_lc_data_time['F_vehicle_id'] == pair_id[0]) & (vid_lc_data_time['S_vehicle_id'] == pair_id[1])]
            # extract the traffic information in the same time
            Frame_time = pd.unique(pair_id_event['F_frame_time'])
            for time_id in Frame_time:
                # search the inf with same time_frame
                # print(time_id,vid)
                conflcit_event_data_time_id = pair_id_event[pair_id_event['F_frame_time'] == time_id]
                # here i need the choice the time variable with the signal plan
                frame_time_value = float(time_id)
                signal_inf = lc_data_signal_vid[(lc_data_signal_vid['G_B_F'] <= frame_time_value) & (lc_data_signal_vid['R_B_F'] >= frame_time_value)]
                # here, divid into detial function level
                # print(signal_inf['Y_B_F'].values)
                if frame_time_value<=float(signal_inf['Y_B_F'].values):
                    light_Color = "green"
                    light_time = frame_time_value-float(signal_inf['G_B_F'].values)
                elif frame_time_value<=float(signal_inf['R_B_F'].values):
                    light_Color = "yellow"
                    light_time = frame_time_value - float(signal_inf['Y_B_F'].values)
                    if light_time>3:
                        light_Color = "red"
                        light_time = light_time-3
                # print(light_time,light_Color)
                signal_inf = signal_inf.reset_index(drop=True)
                conflcit_event_data_time_id['cycle_id'] = signal_inf.loc[0, 'cycle_id']
                conflcit_event_data_time_id['phase'] = signal_inf.loc[0, 'phase']
                conflcit_event_data_time_id['light_color'] = str(light_Color)
                conflcit_event_data_time_id['light_time'] = float(light_time)
                ttc_event_chain_3 = pd.concat([ttc_event_chain_3,conflcit_event_data_time_id])
                # print(ttc_event_chain_3)
    return ttc_event_chain_3
# @ timeE


# colum_name3 = ["Video_id","frame_time","F_vehicle_id","F_type","F_world_x","F_world_y","F_speed_x","F_speed_y","F_speed","F_acc_x","F_acc_y","F_acc","F_dir","S_vehicle_id","S_type","S_dir","S_world_x","S_world_y","S_speed","S_acc","S_jerk","S_Angle","F_to_c","S_to_c","F_to_S","V_F_S","ACC_F_S","Ang_F_S","conflict_x","conflict_y","TTC","intersection_name","area_below_2","F_state_n","F_state_m","S_state_n","S_state_m",'cycle_id','phase','begin_time','end_time','begin_frame_time','end_frame_time','phase_length','over_phase_time']
# colum_name3 = ["Video_id",'F_vehicle_id', 'F_frame_time', 'F_vehicle_type', 'F_world_x', 'F_world_y', 'F_speed_x', 'F_speed_y', 'F_acc_x', 'F_acc_y', 'F_Jerk_x', 'F_Jerk_y', 'F_Angle','S_vehicle_id', 'S_vehicle_type', 'S_world_x', 'S_world_y', 'S_speed_x', 'S_speed_y', 'S_acc_x','S_acc_y', 'S_Jerk_x', 'S_Jerk_y', 'S_Angle', 'cross_point_x', 'cross_point_y', 'TTC','area_below_2']
conflict_event_path = r"D:\dataset\Intersection\Data_processing\Conflict_event\Conflict_event_chain_by_ttc\Total_dataset/LC_conflict.csv"
conflict_event = pd.read_csv(conflict_event_path)
signal_data_path = r"D:\dataset\Intersection\signal_time_in_video/LC_SIGNAL.csv"
all_variable_add_signal = signial_time_variable(conflict_event,signal_data_path)
save_path3 = r"D:\dataset\Intersection\Data_processing\Conflict_event\Add_signal_with_event/event_signal_LC.csv"
all_variable_add_signal.to_csv(save_path3,index=False, header=True)
```

 contact with the ordinal information! 

```python
def Add_ord_information(conf_event_data_path, orj_traj_data_path,Output_data_path):
    # add the ordinal information about the vehicle kitmate
    con_event_all_inf_dir = pd.DataFrame()
    con_event = pd.read_csv(conf_event_data_path)
    con_event = pd.DataFrame(con_event)
    df = pd.DataFrame(con_event)
    veh_pairs = df[['F_vehicle_id','S_vehicle_id','Video_id']].drop_duplicates()
    Org_traj = pd.read_csv(orj_traj_data_path)
    Org_traj = pd.DataFrame(Org_traj)
    veh_pairs = veh_pairs.values
    for veh_pair in veh_pairs:
        print(veh_pair)
        # search each pair
        Org_traj_conf_pair = Org_traj[Org_traj['video_id'] == veh_pair[2]]
        M_org_traj = Org_traj_conf_pair[Org_traj_conf_pair['vehicle_id'] == veh_pair[0]]
        N_org_traj = Org_traj_conf_pair[Org_traj_conf_pair['vehicle_id'] == veh_pair[1]]
        con_video = df[df['Video_id']== veh_pair[2]]
        M_event = con_video[con_video['F_vehicle_id']==veh_pair[0]]
        MN_event = M_event[M_event['S_vehicle_id']==veh_pair[1]]
        MN_event = MN_event.rename(columns={'F_frame_time': 'frame_time'})
        MN_df = pd.DataFrame(MN_event)
        df2 = pd.DataFrame(M_org_traj)
        MN_df_m = pd.merge(MN_df,df2[['frame_time','vehicle_speed','vehicle_tan_acc','vehicle_lat_acc','Angle']],on ='frame_time')
        MN_df_m = MN_df_m.rename(columns={'vehicle_speed': 'M_speed', 'vehicle_tan_acc': 'M_tan_acc', 'vehicle_lat_acc': 'M_lat_acc','Angle':'M_Angle'})
        df3 = pd.DataFrame(N_org_traj)
        MN_df_mn = pd.merge(MN_df_m, df3[
            ['frame_time','vehicle_speed', 'vehicle_tan_acc', 'vehicle_lat_acc', 'Angle']], on='frame_time')
        MN_df_mn = MN_df_mn.rename(columns={'vehicle_speed': 'N_speed', 'vehicle_tan_acc': 'N_tan_acc',
                     'vehicle_lat_acc': 'N_lat_acc', 'Angle': 'N_Angle'})
        con_event_all_inf_dir = pd.concat([MN_df_mn,con_event_all_inf_dir])
    con_event_all_inf_dir.to_csv(Output_data_path,index=False, header=True)
    return con_event_all_inf_dir

def Map_show(orj_traj_data_path):
    Org_traj = pd.read_csv(orj_traj_data_path)
    Org_traj = pd.DataFrame(Org_traj)
    # ,' Motorcycle',' Bicycle',' Car',
    Vehicle_type = [' Motorcycle',' Bicycle',' Car']
    Org_traj = Org_traj[Org_traj['vehicle_type'].isin(Vehicle_type)]
    Org_traj = pd.DataFrame(Org_traj)
    Org_traj = Org_traj[Org_traj['world_x'] <= 30]
    Org_traj = Org_traj[Org_traj['world_x'] >= -12]
    Org_traj = Org_traj[Org_traj['world_y'] >= -15]
    Org_traj = Org_traj[Org_traj['world_y'] <= 35]
     # class the vehicle type
    Car = Org_traj[Org_traj['vehicle_type']==' Car']
    Motorcycle = Org_traj[Org_traj['vehicle_type'] == ' Motorcycle']
    Bicycle = Org_traj[Org_traj['vehicle_type'] == ' Bicycle']
    plt.figure(figsize=(10, 10))
    plt.scatter(Car['world_x'], Car['world_y'], s=0.5,color='b',alpha=0.05,label ="Car")
    # plt.colorbar()
    plt.scatter(Motorcycle['world_x'],Motorcycle['world_y'], s=0.5,color="g",alpha=0.1, label='Motorcycle')
    # plt.colorbar()
    plt.scatter(Bicycle['world_x'],Bicycle['world_y'], s=0.5,color="y", alpha=0.1,label='Bicycle')
    # plt.colorbar()
    plt.xlabel("X (m)",fontsize=14)
    plt.ylabel("Y (m)",fontsize=14)
    # plt.legend("G:Car")
    plt.legend(fontsize=14)
    plt.grid()
    plt.show()
    return
```



Define variabel

```python
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
# independent variable contain driving purpose,Kinematic status,Surrounding enviroment,
# Interaction behavior,vehicle type, Relative status, Signal control.

# first variable about driving purpose
"here we just devid into foure kinds: turning, left,stragit,right"
def Driving_purpose(con_event_data):
    DIR = pd.DataFrame()
    df = pd.DataFrame(con_event_data)
    # M_dir_type = con_event['M_Dir'].unique()
    conditions_1 = [
        (df['M_Dir']=='NN')|(df['M_Dir']=='WW')|(df['M_Dir']=='SS')|(df['M_Dir']=='EE'),
        (df['M_Dir']=='NE')|(df['M_Dir']=='WN')|(df['M_Dir']=='SW')|(df['M_Dir']=='ES'),
        (df['M_Dir']=='NS')|(df['M_Dir']=='WE')|(df['M_Dir']=='SN')|(df['M_Dir']=='EW'),
        (df['M_Dir']=='NW')|(df['M_Dir']=='WS')|(df['M_Dir']=='SE')|(df['M_Dir']=='EN'),
        (df['M_Dir'].isnull())]
    conditions_2 = [
        (df['N_Dir'] == 'NN') | (df['N_Dir'] == 'WW') | (df['N_Dir'] == 'SS') | (df['N_Dir'] == 'EE'),
        (df['N_Dir'] == 'NE') | (df['N_Dir'] == 'WN') | (df['N_Dir'] == 'SW') | (df['N_Dir'] == 'ES'),
        (df['N_Dir'] == 'NS') | (df['N_Dir'] == 'WE') | (df['N_Dir'] == 'SN') | (df['N_Dir'] == 'EW'),
        (df['N_Dir'] == 'NW') | (df['N_Dir'] == 'WS') | (df['N_Dir'] == 'SE') | (df['N_Dir'] == 'EN'),
        (df['M_Dir'].isnull())]
    choices = ['T','L','S','R',np.nan]
    df['MDIR'] = np.select(conditions_1, choices, default=0)
    df['NDIR'] = np.select(conditions_2, choices, default=0)
    return df

# second Kinematic status
def euclidean_distance(point1, point2):
    "Calculate the distance between two points"
    x1, y1 = point1[0]
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_distances(one_point, other_points):
    "Calculate the distance between the two points"
    other_points = [(row['world_x'], row['world_y']) for _, row in other_points.iterrows()]
    distances = []
    for point in other_points:
        distance = euclidean_distance(one_point, point)
        distances.append(distance)
    return distances


def Dis_to_conflitc(veh_conf_data):
    df = pd.DataFrame(veh_conf_data)
    df['MV'] = (((df['F_speed_x']) ** 2 + (df['F_speed_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['MACC'] = (((df['F_acc_x']) ** 2 + (df['F_acc_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['NV'] = (((df['S_speed_x']) ** 2 + (df['S_speed_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['NACC'] = (((df['S_acc_x']) ** 2 + (df['S_acc_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['MOA'] = (((df['M_tan_acc']) ** 2 + (df['M_lat_acc']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['NOA'] = (((df['N_tan_acc']) ** 2 + (df['N_lat_acc']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['NTC'] = (((df['F_world_x'] - df['cross_point_x']) ** 2 + (df['F_world_y'] - df['cross_point_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['NTC'] = (((df['S_world_x'] - df['cross_point_x']) ** 2 + (df['S_world_y'] - df['cross_point_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['MTN'] = (((df['F_world_x'] - df['S_world_x']) ** 2 + (df['F_world_y'] - df['S_world_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['VMN'] = (((df['F_speed_x'] - df['S_speed_x']) ** 2 + (df['F_speed_y'] - df['S_speed_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    df['AMN'] = (((df['F_acc_x'] - df['S_acc_x']) ** 2+(df['F_acc_y'] - df['S_acc_y']) ** 2).abs()).apply(lambda x: x ** 0.5)
    return df

# third Surrounding enviroment bupdate with frame by frame
def Select_the_same_frame_vehicel(orginal_data_with_veh_pair_id,con_event_pair):
    "The function of this code is to calculate the influence from surrounding vehicle."
    Sub_veh = pd.DataFrame()
    Org_data = pd.DataFrame(orginal_data_with_veh_pair_id)
    Frame_time = pd.unique(con_event_pair['frame_time'])
    M_veh_id = pd.unique(con_event_pair['F_vehicle_id'])
    N_veh_id = pd.unique(con_event_pair['S_vehicle_id'])
    for time_id in Frame_time:
        # select all vehicle in sametime
        Sub_veh_time = pd.DataFrame()
        veh_in_inters = Org_data[Org_data['frame_time']==time_id]
        Obj_inf = con_event_pair[con_event_pair['frame_time']==time_id]
        # calaulate the distance between veh to object
        veh_in_inters = veh_in_inters.drop(veh_in_inters[(veh_in_inters['vehicle_id'] == M_veh_id[0]) | (veh_in_inters['vehicle_id'] == N_veh_id[0])].index)
        M_position = Obj_inf[['F_world_x', 'F_world_y']]
        N_position = Obj_inf[['S_world_x', 'S_world_y']]
        def calculate_distance(row):
            return ((row['world_x'] - M_position['F_world_x'].iloc[0]) ** 2 + (row['world_y'] - M_position['F_world_y'].iloc[0]) ** 2) ** 0.5
        veh_in_inters['M_dis'] = veh_in_inters.apply(calculate_distance, axis=1)

        def calculate_distance(row):
            return ((row['world_x'] - N_position['S_world_x'].iloc[0]) ** 2 + (
                        row['world_y'] - N_position['S_world_y'].iloc[0]) ** 2) ** 0.5
        veh_in_inters['N_dis'] = veh_in_inters.apply(calculate_distance, axis=1)
        # stage 2: calculate the number of vehicle,twe point: vehicle type and distance

        " here we setting the thesoud distance value 20 for vehicle and 10 for non-motorized vehicle"
        M_sub_veh = veh_in_inters[veh_in_inters['M_dis']<=20]
        N_sub_veh = veh_in_inters[veh_in_inters['N_dis']<=10]

        # calculate the vehicle number
        M_sub_N = len(M_sub_veh[(M_sub_veh['vehicle_type']==' Bicycle')|(M_sub_veh['vehicle_type']==' Motorcycle') |(M_sub_veh['vehicle_type']==' Tuk-Tuk')])
        M_sub_M_big = len(M_sub_veh[(M_sub_veh['vehicle_type'] == ' Bus')|(M_sub_veh['vehicle_type'] == ' Heavy Vehicle')])
        M_sub_M_Med = len(M_sub_veh[(M_sub_veh['vehicle_type'] == ' Light Truck')|(M_sub_veh['vehicle_type'] == ' Van')])
        M_sub_M_sam = len(M_sub_veh[(M_sub_veh['vehicle_type']==' Car')])
        M_sub_P = len(M_sub_veh[(M_sub_veh['vehicle_type']==' Pedestrian')])
        N_sub_N = len(N_sub_veh[(N_sub_veh['vehicle_type']==' Bicycle') | (N_sub_veh['vehicle_type'] == ' Motorcycle') | (N_sub_veh['vehicle_type'] == ' Tuk-Tuk')])
        N_sub_M_big = len(N_sub_veh[(N_sub_veh['vehicle_type'] == ' Bus')|(N_sub_veh['vehicle_type'] == ' Heavy Vehicle')])
        N_sub_M_Med = len(N_sub_veh[(N_sub_veh['vehicle_type'] == ' Light Truck')|(N_sub_veh['vehicle_type'] == ' Van')])
        N_sub_M_sam = len(N_sub_veh[(N_sub_veh['vehicle_type']==' Car')])
        N_sub_P = len(N_sub_veh[(N_sub_veh['vehicle_type'] == ' Pedestrian')])
        # add the variable of vehicle number
        Obj_inf = pd.DataFrame(Obj_inf)
        Obj_inf['M_sub_M'] = int(3*M_sub_M_big + 1.5*M_sub_M_Med+M_sub_M_sam)
        Obj_inf['M_sub_N'] = M_sub_N
        Obj_inf['M_sub_P'] = M_sub_P
        Obj_inf['N_sub_M'] = int(3*N_sub_M_big + 1.5*N_sub_M_Med+N_sub_M_sam)
        Obj_inf['N_sub_N'] = N_sub_N
        Obj_inf['N_sub_P'] = N_sub_P
        Sub_veh = pd.concat([Obj_inf,Sub_veh])
    return Sub_veh

# fourth Interaction behavior
def Brake_behavior(con_event_pair_data):
    df = pd.DataFrame(con_event_pair_data)
    # calculate the mean of first second
    window_size = 25
    # m
    overall_mean = df['MOA'].mean()
    rolling_means = df['MOA'].rolling(window=window_size, min_periods=1).mean()
    df['MOAM'] = rolling_means.fillna(overall_mean)
    # n
    overall_mean = df['NOA'].mean()
    rolling_means = df['NOA'].rolling(window=window_size, min_periods=1).mean()
    df['NOAM'] = rolling_means.fillna(overall_mean)

    df.loc[df['MOA'] - df['MOAM'] >=0.5, 'M_bra_MOA'] = 'acc'
    df.loc[abs(df['MOA'] - df['MOAM']) < 0.5, 'M_bra_MOA'] = 'same'
    df.loc[df['MOA'] - df['MOAM'] <= -0.5, 'M_bra_MOA'] = 'brake'

    df.loc[df['NOA']- df['NOAM']>=0.5,'N_bra_NOA']='acc'
    df.loc[abs(df['NOA']-df['NOAM'])< 0.5,'N_bra_NOA']='same'
    df.loc[df['NOA']-df['NOAM']<=-0.5,'N_bra_NOA']='brake'

    # second_acc
    overall_mean = df['MACC'].mean()
    rolling_means = df['MACC'].rolling(window=window_size, min_periods=1).mean()
    df['MACCM'] = rolling_means.fillna(overall_mean)
    # n
    overall_mean = df['NACC'].mean()
    rolling_means = df['NACC'].rolling(window=window_size, min_periods=1).mean()
    df['NACCM'] = rolling_means.fillna(overall_mean)

    df.loc[df['MACC'] - df['MACCM'] >= 0.5, 'M_bra_MACC'] = 'acc'
    df.loc[abs(df['MACC'] - df['MACCM']) < 0.5, 'M_bra_MACC'] = 'same'
    df.loc[df['MACC'] - df['MACCM'] <= -0.5, 'M_bra_MACC'] = 'brake'

    df.loc[df['NACC'] - df['NACCM'] >= 0.5, 'N_bra_NACC'] = 'acc'
    df.loc[abs(df['NACC'] - df['NACCM']) < 0.5, 'N_bra_NACC'] = 'same'
    df.loc[df['NACC'] - df['NACCM'] <= -0.5, 'N_bra_NACC'] = 'brake'
    return df

# direction
def Escape_behavior(con_event_pair_data):
    Escape_beh = pd.DataFrame()
    df = pd.DataFrame(con_event_pair_data)
    # calculate the mean of first second

    window_size = 25
    # m
    overall_mean = df['M_Angle'].mean()
    rolling_means = df['M_Angle'].rolling(window=window_size, min_periods=1).mean()
    df['M_Angle_M'] = rolling_means.fillna(overall_mean)

    overall_variance = df['M_Angle'].var()
    rolling_variances = df['M_Angle'].rolling(window=window_size, min_periods=1).var()
    df['M_Angle_V'] = rolling_variances.fillna(overall_variance)
    #
    overall_mean = df['N_Angle'].mean()
    rolling_means = df['N_Angle'].rolling(window=window_size, min_periods=1).mean()
    df['N_Angle_M'] = rolling_means.fillna(overall_mean)

    overall_variance = df['N_Angle'].var()
    rolling_variances = df['N_Angle'].rolling(window=window_size, min_periods=1).var()
    df['N_Angle_V'] = rolling_variances.fillna(overall_variance)

    df.loc[df['M_Angle'] - df['M_Angle_M'] >=0.523, 'M_esc_beh'] = 'A_esc'
    df.loc[abs(df['M_Angle'] - df['M_Angle_M']) < 0.523, 'M_esc_beh'] = 'keep'
    df.loc[df['M_Angle'] - df['M_Angle_M'] <= -0.523, 'M_esc_beh'] = 'N_esc'

    df.loc[df['N_Angle']- df['N_Angle_M']>=0.523,'N_esc_beh']='A_esc'
    df.loc[abs(df['N_Angle']-df['N_Angle_M'])< 0.523,'N_esc_beh']='keep'
    df.loc[df['N_Angle']-df['N_Angle_M']<=-0.523,'N_esc_beh']='N_esc'
    return df

#five vehicle type
def MN_Angle(con_event_chain_data):
    df = pd.DataFrame(con_event_chain_data)
    M_position = df[['F_world_x',"F_world_y"]].values
    N_position = df[['S_world_x',"S_world_y"]].values
    dot_product = np.sum(M_position * N_position, axis=1)
    norm_vec1 = np.linalg.norm(M_position, axis=1)
    norm_vec2 = np.linalg.norm(N_position, axis=1)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    angles_radians = np.arccos(cosine_similarity)
    df['MN_angle'] = np.degrees(angles_radians)
    return df

def Define_variabel(conf_event_data_path, orj_traj_data_path,Output_data_path):
    con_event_dir = pd.DataFrame()
    con_event = pd.read_csv(conf_event_data_path)
    con_event_a = pd.DataFrame(con_event)
    # step1: calculate the sqrt variable
    con_event_b = Dis_to_conflitc(con_event_a)
    # step2: calculate the direction
    con_event_c = Driving_purpose(con_event_b)
    Org_traj = pd.read_csv(orj_traj_data_path)
    Org_traj = pd.DataFrame(Org_traj)
    con_veh_pairs = con_event_c[['F_vehicle_id', 'S_vehicle_id', 'Video_id']]
    df = pd.DataFrame(con_veh_pairs)
    con_veh_pairs = df.drop_duplicates(subset=['F_vehicle_id', 'S_vehicle_id', 'Video_id'])
    veh_pairs = con_veh_pairs.values
    for veh_pair in veh_pairs:
        print(veh_pair)
        Org_traj_conf_pair = Org_traj[Org_traj['video_id'] == veh_pair[2]]
        cevent_veh_pair = con_event_c[con_event_c["Video_id"]==veh_pair[2]]
        cevent_veh_pair_trj = cevent_veh_pair[cevent_veh_pair["F_vehicle_id"]==veh_pair[0]]
        chain_veh_pair = cevent_veh_pair_trj[cevent_veh_pair_trj['S_vehicle_id']==veh_pair[1]]
        # calculate the variavle
        # step2: calculate the direction
        Sur_event = Select_the_same_frame_vehicel(Org_traj_conf_pair, chain_veh_pair)
        Sur = pd.DataFrame(Sur_event)
        chain_veh_pair_1 = pd.merge(chain_veh_pair, Sur[['frame_time','M_sub_M','M_sub_N','M_sub_P','N_sub_M','N_sub_N','N_sub_P']],on='frame_time')
        # step3: calculate the ACC behaviour
        Brake_b = Brake_behavior(chain_veh_pair_1)
        chain_veh_pair_2 = pd.merge(chain_veh_pair_1, Brake_b[
            ['frame_time', 'M_bra_MOA','N_bra_NOA','M_bra_MACC','N_bra_NACC']],on='frame_time')
        # step4: calculate the escape behaviour
        Escape_b = Escape_behavior(chain_veh_pair_2)
        chain_veh_pair_3 = pd.merge(chain_veh_pair_2, Escape_b[
            ['frame_time','M_Angle_M','M_Angle_V','M_esc_beh','N_Angle_M','N_Angle_V','N_esc_beh']],on='frame_time')
        # step5: calculate the ACC behaviour
        MNA = MN_Angle(chain_veh_pair_3)
        chain_veh_pair_4 = pd.merge(chain_veh_pair_3, MNA[['frame_time','MN_angle']],on='frame_time')
        #, suffixes=('_chain_veh_pair_3','_MNA')
        con_event_dir = pd.concat([con_event_dir,chain_veh_pair_4])
    con_event_dir.to_csv(Output_data_path,index=False, header=True)
    return con_event_dir


con_event_chain_path = r"D:\dataset\Intersection\Data_processing\Conflict_event\Add_ordinal_information/JH_org_dir.csv"
org_traj_data_path = r"D:\dataset\Intersection\Data_processing\Convert_format\total_data/JH_total.csv"
save_data_path = r"D:\dataset\Intersection\Data_processing\Conflict_event\variable\ordinal_variable/JH_var.csv"

# con_event_chain_path = r"Dataset/Add_ordinal_information/NW_add_org_inf.csv"
# org_traj_data_path = r"Dataset/total_data/LC_total.csv"
# save_data_path = r"Dataset/ordinal_variable/LC_var.csv"
Define_variabel(con_event_chain_path, org_traj_data_path,save_data_path)
```
<br>


**c) The  third step model the interaction behavior based on the ordinal logit model and the causality inference model.**
<br>








**d) The forth step model the interaction behavior based on the dynamic theory, dynamic Bayesian model and dynamic causality discovery theory.**

<br>





**e) The finally step build the model deflection the anomaly behavior based the dynamic theory.**
<br>
