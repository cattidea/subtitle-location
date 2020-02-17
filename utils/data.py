import os
import random
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import cv2
from utils.config import Config
from utils.ocr import baiduOCR

W = 128
H = 128
C = 3

CONFIG = Config()
IMG_DIR = CONFIG["img_dir"]
TEST_IMG_DIR = CONFIG["test_img_dir"]
LABEL_PATH = CONFIG["label_path"]
H5_PATH = CONFIG["h5_path"]
SEED = 1111
random.seed(SEED)
np.random.seed(SEED)


def image2array(img_path):
    """ 将图片转为 numpy array
    注意 cv2 接口的路径不要含中文
    """
    img_path = os.path.normpath(img_path)
    color_mode = cv2.COLOR_BGR2GRAY if C == 1 else cv2.COLOR_BGR2RGB
    img_arr = cv2.cvtColor(cv2.imread(img_path), color_mode)
    img_arr = cv2.resize(img_arr, (W, H)).reshape((H, W, C))
    return img_arr


def img_plot(img_array, shape=(H, W, C)):
    """ 显示处理过的图片 array """
    if shape:
        if shape[-1] == 1:
            img_array = np.reshape(img_array, shape[: 2])
        else:
            img_array = np.reshape(img_array, shape)
    cmap = 'gray' if shape and shape[-1] == 1 else None
    plt.imshow(img_array, cmap=cmap)
    plt.show()


def img_save(img_array, file_path):
    """ 保存图片 array """
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, img_array)


def img_read(img_path):
    img_path = os.path.normpath(img_path)
    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return img_arr


def img_add_label(img, label, color=(0, 0, 255), noise=(0, 0)):
    """ 根据 label 将图片的目标区域圈出来 """
    has_subtitle, cx, cy, rh, rw = label
    rh = min(rh + noise[0], 1.)
    rw = min(rw + noise[1], 1.)
    if has_subtitle < 0.5:
        return img
    h, w, _ = img.shape
    pt1 = (int((cx-rw/2)*w), int((cy-rh/2)*h))
    pt2 = (int((cx+rw/2)*w), int((cy+rh/2)*h))
    img_with_label = cv2.rectangle(img, pt1, pt2, color, 2)
    return img_with_label


def img_crop_by_label(img, label, noise=(0, 0)):
    """ 根据 label 将图片的目标区域切分出来 """
    has_subtitle, cx, cy, rh, rw = label
    rh = min(rh + noise[0], 1.)
    rw = min(rw + noise[1], 1.)
    if has_subtitle < 0.5:
        return None
    h, w, _ = img.shape
    start_x, start_y = int((cx-rw/2)*w), int((cy-rh/2)*h)
    end_x, end_y = int((cx+rw/2)*w), int((cy+rh/2)*h)
    crop_area = img[start_y: end_y, start_x: end_x]
    return crop_area


def read_labels():
    """ 读取标签数据 """
    labels = {}
    with open(LABEL_PATH, "r", encoding="gbk") as f:
        for line in f:
            items = line.split(",")
            assert len(items) >= 6
            img_name = os.path.basename(",".join(items[: -5]))
            img_name = img_name.split('\\')[-1].split('/')[-1]
            labels[img_name] = np.array(items[-5:], dtype=np.float32)
    return labels


def get_data_set():
    """ 将数据格式化为标准的 array ，并且将 X 与 Y 对齐 """
    img_names = os.listdir(IMG_DIR)
    labels = read_labels()
    X, Y = [], []
    data_set = {}
    for i, img_name in enumerate(img_names):
        print("get data set {}/{} ".format(i, len(img_names)), end='\r')
        if img_name not in labels:
            continue
        if os.path.splitext(img_name)[-1] not in [".png", ".jpg", ".jpeg"]:
            continue
        img_path = os.path.join(IMG_DIR, img_name)
        img_arr = image2array(img_path)
        X.append(img_arr)
        Y.append(labels[img_name])

    assert len(X) == len(Y)
    data_set["X"], data_set["Y"] = np.array(
        X, dtype=np.uint8), np.array(Y, dtype=np.float32)
    return data_set


def ramdom_divide_data_set(data_set, train_proportion=0.95):
    """ 随机将数据划分为训练集与测试集 """
    data_size = len(data_set["X"])
    train_size = int(data_size * train_proportion)
    permutation = list(np.random.permutation(data_size))
    train_permutation, dev_permutation = permutation[:
                                                     train_size], permutation[train_size:]
    train_data_set, dev_data_set = {}, {}
    train_data_set["X"], dev_data_set["X"] = data_set["X"][train_permutation], data_set["X"][dev_permutation]
    train_data_set["Y"], dev_data_set["Y"] = data_set["Y"][train_permutation], data_set["Y"][dev_permutation]
    return train_data_set, dev_data_set


def data_import():
    """ 导入数据，自动缓存为 h5 文件"""
    data_set = {}
    if not os.path.exists(H5_PATH):
        print("未发现处理好的数据文件，正在处理...")

        data_set = get_data_set()

        h5f = h5py.File(H5_PATH, 'w')
        h5f["X"] = data_set["X"]
        h5f["Y"] = data_set["Y"]
        h5f.close()
    else:
        h5f = h5py.File(H5_PATH, 'r')
        data_set["X"] = h5f["X"][:]
        data_set["Y"] = h5f["Y"][:]
        h5f.close()
        print("发现处理好的数据文件，正在读取...")
    return data_set


def test_data_import(test_img_dir=TEST_IMG_DIR, cache=None):
    """ 导入测试数据 """
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f:
            test_data_set = pickle.load(f)
    else:
        test_data_set = {}
        img_names = os.listdir(test_img_dir)
        X, img_paths = [], []
        for i, img_name in enumerate(img_names):
            print("get test data set {}/{} ".format(i, len(img_names)), end='\r')
            img_path = os.path.join(test_img_dir, img_name)
            img_arr = image2array(img_path)
            X.append(img_arr)
            img_paths.append(img_path)
        test_data_set["X"] = np.array(X, dtype=np.uint8)
        test_data_set["img_paths"] = img_paths
        if cache:
            with open(cache, 'wb') as f:
                pickle.dump(test_data_set, f)
    return test_data_set


def subtitle_recognition(crop_imgs_dir):
    """ 识别字幕 """
    crop_img_names = os.listdir(crop_imgs_dir)
    subtitles = []
    subtitle_cur = ""
    GAMMA = 0.3
    for i, crop_img_name in enumerate(crop_img_names):
        print("recognition {}/{}".format(i, len(crop_img_names)), end="\r")
        crop_img_path = os.path.join(crop_imgs_dir, crop_img_name)
        try:
            subtitle = baiduOCR(crop_img_path)
        except:
            continue
        subtitle_set = set(subtitle)
        subtitle_cur_set = set(subtitle_cur)
        if subtitle and \
           len(subtitle_set & subtitle_cur_set) / len(subtitle_set | subtitle_cur_set) < GAMMA:
            subtitle_cur = subtitle
            subtitles.append(subtitle_cur)
    return subtitles


def batch_loader(X, y, batch_size=64):
    data_size = X.shape[0]
    permutation = np.random.permutation(data_size)
    batch_permutation_indices = [permutation[i: i + batch_size]
                                 for i in range(0, data_size, batch_size)]
    for batch_permutation in batch_permutation_indices:
        yield X[batch_permutation], y[batch_permutation]


def change_range(X):
    """ 将 [0, 255] 变为 [-1, 1]"""
    X = np.divide(X, 127.5, dtype=np.float32) - 1
    return X
