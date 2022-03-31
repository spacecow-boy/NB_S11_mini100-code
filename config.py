# -*- coding: utf-8 -*-
# @Time : 2020/4/7 19:46 
# @Author : Zhao HL
# @File : config.py
import os
import torch
import numpy as np
from collections import namedtuple
from torchvision import transforms
'''
配置文件
初次训练时，需要修改Root_path
其他参数配置，建议理解代码后再尝试修改
'''

# region parameters
# region paths
Root_path = r'E:\temp\unet训练arcgis数据教程\NB_S11_mini100数据集及训练代码'            # 数据集总路径
Data_path = os.path.join(Root_path, "data")                                 # 数据路径
Imgs_path = os.path.join(Data_path, "img")                                  # 数据集-影像文件夹
GTs_path = os.path.join(Data_path, "gt")                                    # 数据集-标注文件夹
GTs_color_path = os.path.join(Data_path, "gt_color")                        # 数据集-标注可视化文件夹
Pres_path = os.path.join(Data_path, "pre")                                  # 测试时，模型预测结果文件夹
Pres_color_path = os.path.join(Data_path, "pre_color")                     # 测试时，模型预测结果可视化文件夹

Infer_path = os.path.join(Data_path, "infer")                               # 推理数据存放文件夹
Infer_image_path = os.path.join(Infer_path, 'img.tif')                      # 需要推理的大幅遥感影像路径
Infer_predict_path = os.path.join(Infer_path, 'pre.tif')                    # 大幅遥感影像的最终推理结果路径
Infer_predict_color_path = os.path.join(Infer_path, 'pre_color.tif')        # 大幅遥感影像的最终推理结果可视化路径
Infer_imgs_path = os.path.join(Infer_path, 'infer_img')                     # 大幅遥感影像，裁剪为模型需要尺寸的小影像的存放路径
Infer_pres_path = os.path.join(Infer_path, 'infer_pre')                     # 小影像的预测结果的存放路径；用于合成最终推理结果
InferSize_csv_path = os.path.join(Infer_path, 'size.csv')                   # 推理影像size存放路径

Result_path = os.path.join(Root_path, "result")                             # 各结果存放路径
Split_csv_path = os.path.join(Result_path, "split.csv")                     # 数据集划分结果
Model_file_torch = os.path.join(Result_path, "unet_torch.pth")              # 模型参数存放路径
Record_csv_path = os.path.join(Result_path, 'unet_record.csv')              # 训练过程loss、acc字典存放路径
Test_cfm_path = os.path.join(Result_path, 'cfm.csv')                        # 测试时，混淆矩阵存放路径
Test_mIoU_path = os.path.join(Result_path, 'mIoU.csv')                      # 测试时，mIoU存放路径

Dir_structure = [                                                           # 文件夹结构
    Data_path,
        Imgs_path,
        GTs_path,
        Pres_path,
        GTs_color_path,
        Pres_color_path,
        Infer_path,
            Infer_imgs_path,
            Infer_pres_path,
    Result_path,
]
# endregion

# region image parameter
Img_size = 200                                                              # 图像尺寸
Img_chs = 3                                                                 # 图像通道数
Cls = namedtuple('cls', ['name', 'id', 'color'])                            # 标注信息，包括：类别名、类别id、可视化RGB颜色
Clss = [
    Cls('None', 0, (0, 0, 0)),
    Cls('road', 1, (128, 0, 0)),
    Cls('build_resident', 2,  (75, 0, 130)),
    Cls('build_industrial', 3,(255, 215, 0)),
    Cls('build_service', 4, (34, 139, 34)),
    Cls('build_village', 5, (255, 0, 0)),
    Cls('greenland', 6, (0, 0, 128)),
    Cls('forest', 7,  (0, 128, 128)),
    Cls('bareland', 8, (128, 128, 0)),
    Cls('farmland', 9,  (160, 82, 45)),
    Cls('waterbody', 10, (72, 209, 204)),
    Cls('other', 11, (255, 0, 255)),
]
Labels_nums = len(Clss)                                                     # 标注类别数
# endregion

# region unet parameter
Layer1_chs = 32
Layer2_chs = 64
Layer3_chs = 128
Layer4_chs = 256
Layer5_chs = 512
# endregion

# region hpyerparameter
Learning_rate = 1e-2                                                        # 学习率
Batch_size = 1                                                              # 输入批次数
Epochs = 10                                                                 # 迭代论述
ReLoad = False                                                              # 是否加载已有参数
# endregion

# region other
Net_show = False                                                            # 是否打印网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # 运行设备：cuda表示在GPU上运行，cpu则在CPU上运行
img_transform = transforms.Compose([                                        # 影像变换
    transforms.ToTensor(),
])
gt_transform = None                                                         # 标注变换
np.set_printoptions(suppress=True)                                          # numpy显示优化
# endregion
# endregion
