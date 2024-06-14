import os
import os.path
import random

import torch.utils.data as data
from PIL import Image

import numpy as np


def npy_open(npy_path):
    npy = np.load(npy_path)
    depth = (npy * 255.0).astype(np.uint8)
    img = Image.fromarray(depth)
    return img


def get_dataset(root, is_train):
    """
    根据训练数据集根目录root，生成训练和验证图像列表。
    Train：
    -- original: XXXX_NUMB.jpg
    -- haze:     XXXX_NUMB_A_B.jpg
    -- depth:
       -- XXXX
         --depth_npy
           -- XXXX_NUMB_pred.npy

    参数：
    root: 训练数据集根目录。
    is_train: 训练集/测试集。

    返回值：
    train_list: 训练图像列表，每个元素是一个包含 原始图像 、雾霾图像 和 深度数据 路径的列表。
    val_list: 验证图像列表，每个元素是一个包含 原始图像 、雾霾图像 和 深度数据 路径的列表。
    """
    original_path = root + 'original/'
    haze_path = root + 'haze/'
    depth_path = root + 'depth/'

    train_list = []  # 训练图像列表
    val_list = []  # 验证图像列表

    # 获取雾天图像列表
    haze_list = [os.path.join(haze_path, name)
                 for name in os.listdir(haze_path)]

    tmp_dict = {}  # 临时字典，用于存储同一张原始图像对应的雾霾图像列表

    for haze in haze_list:

        haze = haze.replace("\\", "/")
        haze = haze.split('/')[-1]
        key = '_'.join(haze.split("_")[:2])  # 获取原始图像文件名前缀作为键.即XXXX_NUMB

        if key in tmp_dict.keys():
            tmp_dict[key].append(haze)  # 若 键 已经存在于字典，则向其 值 内继续添加同名文件组成list
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(haze)  # 若 键 未存在于字典，则向其 值 内创建全新list用于储存同名文件

    for key in list(tmp_dict.keys()):
        if is_train:
            for haze_img in tmp_dict[key]:
                orig_img = original_path + key + '.jpg'
                haze_img = haze_path + haze_img
                depth_folder = key.split('_')[0]
                dep_npy = depth_path + depth_folder + '/depth_npy/' + key + '_pred.npy'

                train_list.append([orig_img, haze_img, dep_npy])

            random.shuffle(train_list)
            return train_list
        else:
            for haze_img in tmp_dict[key]:
                orig_img = original_path + key + '.jpg'
                haze_img = haze_path + haze_img
                depth_folder = key.split('_')[0]
                dep_npy = depth_path + depth_folder + '/depth_npy/' + key + '_pred.npy'

                val_list.append([orig_img, haze_img, dep_npy])

            random.shuffle(val_list)
            return val_list


class ImageFolder(data.Dataset):
    def __init__(self, root, triple_transform=None, transform=None, target_transform=None, is_train=True):
        self.root = root
        self.imgs = get_dataset(root, is_train)
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        orig_path, haze_path, depth_path = self.imgs[index]
        original = Image.open(orig_path)
        haze = Image.open(haze_path)
        depth = npy_open(depth_path)

        if self.triple_transform is not None:
            original, haze, depth = self.triple_transform(original, haze, depth)
        if self.transform is not None:
            original = self.transform(original)
        if self.target_transform is not None:
            haze = self.target_transform(haze)
            depth = self.target_transform(depth)

        return original, haze, depth

    def __len__(self):
        return len(self.imgs)
