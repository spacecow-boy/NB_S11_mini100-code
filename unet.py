import torch

from torch import optim
from torch.nn import *
from torchsummary import summary

import sys, os, gdal
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tif
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from config import *

# region 数据预处理

def dirs_check(dirs, create=True):
    '''
    检查各个文件夹是否存在
    :param dirs: 需要检查的文件夹名列表
    :param create: 对于不存在的文件夹是否自动创建
    :return:
    '''
    print('dirs check:')
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print('pass:', dir_path)
        else:
            if create:
                os.makedirs(dir_path)
                print('!!!make:', dir_path)
            else:
                print('!!!no exists:', dir_path)
    print('finish!\n')

class MyDataset(torch.utils.data.Dataset):
    '''
    数据集类，根据传入的文件列表、路径，读取数据；
    训练时，返回img、gt
    测试、推理时，返回img、img_name，用于生成对应gt及gt名称
    '''
    def __init__(self, imgs_list=None, gts_list=None, img_transform=None, gt_transform=None,
                 train_mode=True):
        self.imgs_list = imgs_list
        self.gts_list = gts_list
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.size = len(self.imgs_list)
        self.train_mode = train_mode

    def __len__(self):
        return self.size

    def load_img(self, img_path, img_transform):
        img = Image.open(img_path).convert('RGB')
        if img_transform:
            img = self.img_transform(img)
        img = np.array(img, dtype=np.float32)
        return img

    def load_gt(self, gt_path, gt_transform):
        gt = Image.open(gt_path).convert('L')
        if gt_transform:
            gt = self.gt_transform(gt)
        gt = np.array(gt, dtype=np.int64)
        return gt

    def __getitem__(self, index):
        if self.train_mode:
            img = self.load_img(self.imgs_list[index], self.img_transform)
            gt = self.load_gt(self.gts_list[index], self.gt_transform)
            return img, gt
        else:
            img = self.load_img(self.imgs_list[index], self.img_transform)
            img_name = os.path.basename(self.imgs_list[index])
            return img, img_name


def dataset_split(imgs_path, gts_path, split_csv_path):
    '''
    根据img列表，随机划分为不同数据集；
    将划分好的数据集写入dataframe中，并写入csv中
    :param imgs_path: 影像文件夹
    :param gts_path: 标注文件夹
    :param split_csv_path: 数据集划分结果保存路径
    :return:
    '''
    train_rate = 0.8  # 训练数据所占比例
    files_list = os.listdir(gts_path)
    np.random.shuffle(files_list)
    train_num = int(train_rate * len(files_list))
    val_num = len(files_list) - train_num
    train_list = files_list[:train_num]
    val_list = files_list[train_num:]
    train_img_list = [os.path.join(imgs_path,file_name) for file_name in train_list]
    val_img_list = [os.path.join(imgs_path,file_name) for file_name in val_list]
    train_gt_list = [os.path.join(gts_path,file_name) for file_name in train_list]
    val_gt_list = [os.path.join(gts_path,file_name) for file_name in val_list]

    dict = {'img_path': train_img_list + val_img_list,
            'gt_path': train_gt_list + val_gt_list,
            'split': ['train'] * train_num + ['val'] * val_num}
    df = pd.DataFrame(dict)
    df.to_csv(split_csv_path)
    print('train rate %.2f\ntrain %d, val %d, filter %d\nresult save to %s' %
          (train_rate, train_num, val_num, len(files_list), split_csv_path))

# endregion


# region Unet网络搭建

class Conv_group(Module):
    def __init__(self, input_chs, output_chs):
        super(Conv_group, self).__init__()
        self.conv = Sequential(
            Conv2d(input_chs, output_chs, 3, padding=1),
            BatchNorm2d(output_chs),
            ReLU(),
        )
        self.conv2 = Sequential(
            Conv2d(output_chs, output_chs, 3, padding=1),
            BatchNorm2d(output_chs),
            ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        return x


class Unet(Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.layer1 = Conv_group(Img_chs, Layer1_chs, )

        self.layer2 = Sequential(
            MaxPool2d(2, stride=2),
            Conv_group(Layer1_chs, Layer2_chs, ),
        )
        self.layer3 = Sequential(
            MaxPool2d(2, stride=2),
            Conv_group(Layer2_chs, Layer3_chs, ),
        )
        self.layer4 = Sequential(
            MaxPool2d(2, stride=2),
            Conv_group(Layer3_chs, Layer4_chs, ),
        )

        self.layer5 = Sequential(
            MaxPool2d(5, stride=5),
            Conv_group(Layer4_chs, Layer5_chs, ),
            ConvTranspose2d(Layer5_chs, Layer4_chs, kernel_size=5, stride=5),
        )

        self.layer6 = Sequential(
            Conv_group(Layer5_chs, Layer4_chs, ),
            ConvTranspose2d(Layer4_chs, Layer3_chs, kernel_size=2, stride=2),
        )
        self.layer7 = Sequential(
            Conv_group(Layer4_chs, Layer3_chs, ),
            ConvTranspose2d(Layer3_chs, Layer2_chs, kernel_size=2, stride=2),
        )
        self.layer8 = Sequential(
            Conv_group(Layer3_chs, Layer2_chs, ),
            ConvTranspose2d(Layer2_chs, Layer1_chs, kernel_size=2, stride=2),
        )
        self.layer9 = Conv_group(Layer2_chs, Layer1_chs)

        self.output = Sequential(
            Conv2d(Layer1_chs, Labels_nums, 1, ),
            Softmax(dim=1),
        )

    def forward(self, input):
        x1 = self.layer1(input)             # 200 * 200 * 64
        x2 = self.layer2(x1)                # 100 * 100 * 128
        x3 = self.layer3(x2)                # 50  * 50  * 256
        x4 = self.layer4(x3)                # 25  * 25  * 512

        x5 = self.layer5(x4)                # 25  * 25  * 512
        x5 = torch.cat([x4, x5], dim=1)     # 25  * 25  * 1024

        x6 = self.layer6(x5)                # 50  * 50  * 256
        x6 = torch.cat([x3, x6], dim=1)     # 56  * 56  * 512
        x7 = self.layer7(x6)                # 100 * 100 * 128
        x7 = torch.cat([x2, x7], dim=1)     # 112 * 112 * 256
        x8 = self.layer8(x7)                # 200 * 200 * 64
        x8 = torch.cat([x1, x8], dim=1)     # 200 * 200 * 128
        x9 = self.layer9(x8)                # 200 * 200 * 64
        output = self.output(x9)
        return output

# endregion


# region 训练、测试、推理

# region 训练过程，包括train、val，及训练过程的记录输出
def train():
    print('data load...')
    # 读取数据集
    df = pd.read_csv(Split_csv_path, header=0, index_col=0)
    train_imgs_list = df[df['split'] == 'train']['img_path'].tolist()
    train_gts_list = df[df['split'] == 'train']['gt_path'].tolist()
    val_imgs_list = df[df['split'] == 'val']['img_path'].tolist()
    val_gts_list = df[df['split'] == 'val']['gt_path'].tolist()
    # 创建数据集实例
    train_dataset = MyDataset(train_imgs_list, train_gts_list,
                              img_transform=img_transform,
                              gt_transform=gt_transform)
    val_dataset = MyDataset(val_imgs_list, val_gts_list,
                            img_transform=img_transform,
                            gt_transform=gt_transform)
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Batch_size, shuffle=True)

    print('model load...')
    # 加载模型
    model = Unet().to(device)
    # 打印网络
    if Net_show:
        print(summary(model, (Img_chs, Img_size, Img_size)))
    # 加载忘了参数
    if ReLoad and os.path.exists(Model_file_torch):
        model.load_state_dict(torch.load(Model_file_torch))
        print('!!!get model from', Model_file_torch)
    else:
        print('!!!start train')

    print('optimizer load...')
    # 优化器定义
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9, weight_decay=5e-4)

    # 训练过程记录
    record = {
        'train_loss': np.zeros(Epochs),
        'train_acc': np.zeros(Epochs),
        'val_loss': np.zeros(Epochs),
        'val_acc': np.zeros(Epochs),
    }
    best_acc = 0
    best_acc_epoch = 0

    # 训练过程
    for epoch in range(Epochs):
        print('Epoch %d/%d:' % (epoch, Epochs))
        # train
        with torch.set_grad_enabled(True):
            for batch_num, (images, gts) in enumerate(train_loader):
                images, gts = images.to(device), gts.to(device)

                # 单次训练
                outputs = model(images)

                # loss计算及反向传播
                loss = criterion(outputs, gts)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # acc计算
                gts = gts.data.cpu().numpy()
                pres = np.argmax(outputs.data.cpu().numpy(), axis=1)
                correct = (pres == gts).sum()
                acc = correct / (gts.shape[0] * gts.shape[1] * gts.shape[2])

                # 可视化过程
                record['train_loss'][epoch] += loss.item()
                record['train_acc'][epoch] += acc
                process_show(batch_num, len(train_loader), acc, loss.data, prefix='train:')
            print()

        # validate
        with torch.set_grad_enabled(False):
            for batch_num, (images, gts) in enumerate(val_loader):
                images, gts = images.to(device), gts.to(device)

                # 单次训练
                output = model(images)

                # loss计算
                loss = criterion(output, gts)

                # acc计算
                gts = gts.data.cpu().numpy()
                pres = np.argmax(outputs.data.cpu().numpy(), axis=1)
                correct = (pres == gts).sum()
                acc = correct / (gts.shape[0] * gts.shape[1] * gts.shape[2])

                # 可视化过程
                record['val_loss'][epoch] += loss.item()
                record['val_acc'][epoch] += acc
                process_show(batch_num, len(val_loader), acc, loss, prefix='val:')
            print()

        # 单次epoch平均汇总
        record['train_loss'][epoch] /= train_dataset.size
        record['train_acc'][epoch] /= train_dataset.size
        record['val_loss'][epoch] /= val_dataset.size
        record['val_acc'][epoch] /= val_dataset.size
        # 单次 汇总输出
        print('average summary:\ntrain acc %.4f, loss %.4f ; val acc %.4f, loss %.4f'
              % (record['train_acc'][epoch], record['train_loss'][epoch],
                 record['val_acc'][epoch], record['val_loss'][epoch]))

        # best更新及模型保存
        if record['val_acc'][epoch] > best_acc:
            print('val_acc improve from %.4f to %.4f, model save to %s ! \n' %
                  (best_acc, record['val_acc'][epoch], Model_file_torch))
            best_acc_epoch = epoch
            best_acc = record['val_acc'][epoch]
            torch.save(model.state_dict(), Model_file_torch)
        else:
            print('val_acc do not improve from %.4f \n' % (best_acc))

    # Record 保存及输出
    print('best acc %.4f at epoch %d \n' % (best_acc, best_acc_epoch))
    df = pd.DataFrame(record)
    df.to_csv(Record_csv_path)  # 保存训练指标
    draw_loss_acc(record, save_path=Record_csv_path.replace('.csv','.png'))# 保存训练指标曲线


def draw_loss_acc(record, save_path=None):
    '''
    绘制每轮train、val过程中的loss、acc
    :param record: loss、acc字典，来自train过程
    :param save_path: 是否保存，如果传入None，不保存
    :return:
    '''
    x = [epoch for epoch in range(len(record['train_loss']))]

    plt.subplot(2, 1, 1)
    plt.plot(x, record['train_acc'], 'b-')
    plt.plot(x, record['val_acc'], 'r--')
    plt.legend(['train_acc', 'val_acc'])
    plt.ylabel('accuracy')
    plt.title('train&val accuracy per epoch')

    plt.subplot(2, 1, 2)
    plt.plot(x, record['train_loss'], 'b-')
    plt.plot(x, record['val_loss'], 'r--')
    plt.legend(['train_loss', 'val_loss'])
    plt.ylabel('loss')
    plt.title('train&val loss per epoch')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def process_show(num, nums, acc, loss, prefix='', suffix=''):
    '''
    按进度条输出每个batch的acc、loss
    :param num: 当前batch数
    :param nums: 总batch数
    :param acc:
    :param loss:
    :param prefix:
    :param suffix:
    :return:
    '''
    rate = num / nums
    ratenum = round(rate, 3) * 100
    bar = '\r%s batch %3d/%d:train acc %.4f, train loss %00.4f [%s%s]%.1f%% %s; ' % (
        prefix, num, nums, acc, loss, '#' * (int(ratenum) // 5), '_' * (20 - int(ratenum) // 5), ratenum,
        suffix)
    sys.stdout.write(bar)
    sys.stdout.flush()

# endregion


# region 测试过程，包括图片推理、生成混淆矩阵、mIoU等

def test():
    '''
    测试过程，包括图片推理、生成混淆矩阵、mIoU等
    :param pres_path:
    :param imgs_path:
    :param gts_path:
    :return:
    '''
    # 读取数据集
    df = pd.read_csv(Split_csv_path, header=0, index_col=0)
    test_imgs_list = df[df['split'] == 'val']['img_path'].tolist()
    test_gts_list = df[df['split'] == 'val']['gt_path'].tolist()
    test_pres_list = [gt_path.replace(GTs_path,Pres_path) for gt_path in test_gts_list]

    # 模型加载
    model = Unet().to(device)
    model.load_state_dict(torch.load(Model_file_torch))
    print('!!!get model from', Model_file_torch)

    # 数据加载
    test_dataset = MyDataset(test_imgs_list, None, img_transform=img_transform, gt_transform=gt_transform,
                             train_mode=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)

    # 预测输出
    with torch.set_grad_enabled(False):
        with tqdm(test_loader) as pbar:
            for batch_num, (images, imgs_name) in enumerate(pbar):
                images = images.to(device)

                # 单次训练
                outputs = model(images)

                pres = np.argmax(outputs.data.cpu().numpy(), axis=1)
                # pre保存
                for index in range(pres.shape[0]):
                    pre_save_path = os.path.join(Pres_path, imgs_name[index])
                    tif.imwrite(pre_save_path, pres[index])
                pbar.set_description('test')

    # 指标计算
    cfM = mean_confusionMaxtrix(test_gts_list, test_pres_list, save_csv_path=Test_cfm_path,
                                save_pic_path=Test_cfm_path.replace('.csv', '.png'))
    print(cfM)
    mIou = mean_Intersection_over_Union(cfM,Test_mIoU_path)
    print(mIou)


def mean_confusionMaxtrix(gts_list, pres_list, save_csv_path, save_pic_path=None):
    '''
    根据gt和pre生成混淆矩阵，并生成可视化的热力图
    :param gts_list:
    :param pres_list:
    :param save_csv_path:
    :param save_pic_path:
    :return:
    '''

    def get_confusionMaxtrix(label_vector, pre_vector, class_num):
        '''
        单个样本的gt、pre比较，生成混淆矩阵
        :param label_vector:
        :param pre_vector:
        :param class_num:
        :return:
        '''
        mask = (label_vector >= 0) & (label_vector < class_num)
        return np.bincount(class_num * label_vector[mask].astype(int) + pre_vector[mask],
                           minlength=class_num ** 2).reshape(
            class_num, class_num)
    cfM = np.zeros((Labels_nums, Labels_nums))
    with tqdm(list(zip(gts_list, pres_list))) as pbar:
        for index, (gt_path, pre_path) in enumerate(pbar):
            gt = tif.imread(gt_path)
            pre = tif.imread(pre_path)
            cfM += get_confusionMaxtrix(gt.flatten(), pre.flatten(), Labels_nums)
            pbar.set_description('compare')
    cn = [cls.name for cls in Clss]

    cfM_df = pd.DataFrame(cfM, index=cn, columns=cn)
    cfM_df.to_csv(save_csv_path)
    print('cfm save to', save_csv_path)

    sns.heatmap(cfM_df, annot=True, fmt='.20g', cmap='Blues')
    if save_pic_path:
        plt.savefig(save_pic_path)
    return cfM_df


def mean_Intersection_over_Union(confusion_matrix, save_path):
    '''
    根据混淆矩阵计算mIoU
    :param confusion_matrix:
    :param save_path:
    :return:
    '''
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
    mIoU = {}
    for cls in Clss:
        mIoU[cls.name] = MIoU[cls.id]
    mIoU['Mean:'] = np.nanmean(MIoU)
    mIoU = pd.Series(mIoU)
    mIoU.to_csv(save_path)
    print('cfm save to', save_path)
    return mIoU

# endregion


# region 推理过程，生成预测结果

def infer(imgs_list, save_path, model_path=Model_file_torch):
    '''
    模型推理
    :param imgs_list: 图像路径列表
    :param save_path: 推理结果保存文件夹
    :param model_path: 模型参数路径
    :return:
    '''
    # 模型加载
    model = Unet().to(device)
    model.load_state_dict(torch.load(model_path))
    print('!!!get model from', model_path)

    # 数据加载
    infer_dataset = MyDataset(imgs_list, None,img_transform=img_transform,gt_transform=gt_transform,
                             train_mode=False)
    infer_loader = torch.utils.data.DataLoader(infer_dataset, batch_size=Batch_size, shuffle=False)

    # 预测输出
    with torch.set_grad_enabled(False):
        with tqdm(infer_loader) as pbar:
            for batch_num, (images, imgs_name) in enumerate(pbar):
                images = images.to(device)

                # 单次预测
                outputs = model(images)
                pres = np.argmax(outputs.data.cpu().numpy(), axis=1)

                # pre保存
                for index in range(pres.shape[0]):
                    pre_save_path = os.path.join(save_path, imgs_name[index])
                    tif.imwrite(pre_save_path, pres[index])
                pbar.set_description('infer')


def image_infer_app(proj=True):
    '''
    影像裁剪、预测、拼接,复制地理坐标系等
    :param proj: 是否对预测结果复制原始影像的坐标系
    :return:
    '''

    img_clip(Infer_image_path, Infer_imgs_path, InferSize_csv_path, Img_size)

    infer_imgs_list = [os.path.join(Infer_imgs_path, file_name) for file_name in os.listdir(Infer_imgs_path)]

    infer(infer_imgs_list, Infer_pres_path, model_path=Model_file_torch)

    pre_merge(Infer_predict_path, Infer_pres_path, InferSize_csv_path)

    if proj:
        copy_geoCoordSys(Infer_image_path, Infer_predict_path)

# endregion

# endregion


# region 影像裁剪、预测、拼接,复制地理坐标系等

def img_clip(tif_img_path, imgs_path, size_csv_path, img_size):
    '''
    大幅影像的裁剪，裁剪为指定尺寸，并按序号保存，最右侧和左下侧补（0，0，0）
    :param tif_img_path: 大幅影像路径
    :param imgs_path: 裁剪出的影像存放文件夹路径
    :param size_csv_path:
    :param img_size:
    :return:
    '''
    # 读取影像
    print(tif_img_path,'reading...')
    img = tif.imread(tif_img_path)

    # 信息统计并保存
    cut_nums = [img.shape[0] // img_size + 1, img.shape[1] // img_size + 1]
    size_df = pd.Series([img.shape[0], img.shape[1], img.shape[2], img_size, cut_nums[0], cut_nums[1]],
                        index=['img_height', 'img_height', 'img_channel', 'cut_img_size', 'cut_nums0', 'cut_nums1'])
    print(size_df)
    size_df.to_csv(size_csv_path)

    # 影像裁剪及保存
    # region method1 fast
    for i in range(cut_nums[0]):
        for j in range(cut_nums[1]):
            x0, x1 = i * img_size, (i + 1) * img_size
            y0, y1 = j * img_size, (j + 1) * img_size
            cut_file_name = str(i * cut_nums[1] + j).zfill(6) + '.tif'
            cut_img_path = os.path.join(imgs_path, cut_file_name)

            if i < cut_nums[0] - 1 and j < cut_nums[1] - 1:
                cut_img = img[x0:x1, y0:y1]
            else:
                cut_img = np.zeros((Img_size, Img_size, img.shape[2]), np.uint8)
                if i == cut_nums[0] - 1:
                    x1 = img.shape[0]
                if j == cut_nums[1] - 1:
                    y1 = img.shape[1]
                cut_img[0:x1 - x0, 0:y1 - y0] = img[x0:x1, y0:y1]

            tif.imwrite(cut_img_path, cut_img)
        print('clip:',i + 1,':', cut_nums[0])
    print('image clip finish!')
    # endregion


def pre_merge(tif_pre_path, pres_path, size_csv_path):
    '''
    预测结果拼接
    :param tif_pre_path: 大图预测结果
    :param pres_path:小图预测结果
    :param size_csv_path: 大图尺寸信息
    :return:
    '''
    # 读取信息
    size_df = pd.read_csv(size_csv_path, index_col=0, names=['0'])
    print(size_df)
    cut_nums = [int(size_df.at['cut_nums0', '0']), int(size_df.at['cut_nums1', '0'])]
    img_size = int(size_df.at['cut_img_size', '0'])

    # 创建预测结果矩阵，并逐个拼接
    pre_result = np.zeros((img_size * cut_nums[0], img_size * cut_nums[1]), np.uint8)
    for i in range(cut_nums[0]):
        for j in range(cut_nums[1]):
            cur_num = i * cut_nums[1] + j
            cur_file = str(cur_num).zfill(6) + '.tif'
            cur_img = tif.imread(os.path.join(pres_path, cur_file))
            pre_result[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = cur_img
        print('merge:',i + 1,':', cut_nums[0])
    print('predict merge finish!')
    # 结果保存
    print('saving...')
    tif.imwrite(tif_pre_path, pre_result)
    print('save to', tif_pre_path)


def copy_geoCoordSys(img_proj_path, img_noProj_path):
    '''
    获取img_pos坐标，并赋值给img_none
    :param img_proj_path: 带有坐标的图像
    :param img_noProj_path: 不带坐标的图像
    :return:
    '''
    dataset = gdal.Open(img_proj_path)
    print(img_proj_path, 'geoCoordSys get!')# 打开文件
    img_transf = dataset.GetGeoTransform()                      # 仿射矩阵
    img_proj = dataset.GetProjection()                          # 地图投影信息

    array_dataset = gdal.Open(img_noProj_path)
    img_array = array_dataset.ReadAsArray(0, 0, array_dataset.RasterXSize, array_dataset.RasterYSize)
    if 'int8' in img_array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_array.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img_array.shape) == 3:
        img_bands, im_height, im_width = img_array.shape
    else:
        img_bands, (im_height, im_width) = 1, img_array.shape

    proj_file_path = os.path.join(os.path.dirname(img_noProj_path),'proj_'+os.path.basename(img_noProj_path))
    driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
    dst_dataset = driver.Create(proj_file_path, im_width, im_height, img_bands, datatype)
    dst_dataset.SetGeoTransform(img_transf)  # 写入仿射变换参数
    dst_dataset.SetProjection(img_proj)  # 写入投影

    # 写入影像数据
    if img_bands == 1:
        dst_dataset.GetRasterBand(1).WriteArray(img_array)
    else:
        for i in range(img_bands):
            dst_dataset.GetRasterBand(i + 1).WriteArray(img_array[i])
    print(proj_file_path, 'geoCoordSys project success!')


def gray2color_dir(grays_path, colors_path, clss=Clss):
    '''
    灰度转彩色，按文件夹进行
    :param grays_path: 灰度图路径
    :param colors_path: 彩色图路径
    :return:
    '''
    with tqdm(os.listdir(grays_path)) as pbar:
        for index, file_name in enumerate(pbar):
            gray_path = os.path.join(grays_path,file_name)
            color_path = os.path.join(colors_path,file_name)
            gray2color(gray_path, color_path, clss=clss)
            pbar.set_description('gray 2 color')

def gray2color(gray_path,color_path,clss=Clss):
    '''
    灰度转彩色，按文件进行
    :param gray_path:
    :param color_path:
    :param clss:
    :return:
    '''
    def getDict_id2color(clss):
        '''
        灰度gt, 映射为 彩色gt, 辅助函数，生成id2color颜色映射表
        :param Clss:
        :return:
        '''
        id2color = {}
        for cls in clss:
            id2color[cls.id] = cls.color
        return id2color

    gray_array = tif.imread(gray_path)
    h,w = gray_array.shape
    gray2color_map = getDict_id2color(clss)
    color_array = np.zeros((h,w,3),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel_gray = gray_array[i][j]
            pixel_color = gray2color_map[pixel_gray]
            color_array[i][j] = pixel_color
    tif.imwrite(color_path,color_array)

# endregion



if __name__ == '__main__':
    pass
    # 检查文件夹结构
    # dirs_check(Dir_structure)

    # 划分数据集
    # dataset_split(Imgs_path, GTs_path,Split_csv_path)

    # 训练
    # train()

    # 测试
    # test()
    # 颜色映射
    # gray2color_dir(Pres_path,Pres_color_path)

    # 大幅遥感影像推理预测
    # image_infer_app()
    # 颜色映射
    # gray2color(Infer_predict_path,Infer_predict_color_path)
