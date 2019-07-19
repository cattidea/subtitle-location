import os
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from utils.config import Config

W = 96
H = 96
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

def plot(img_array, shape=(H, W, C)):
    """ 显示处理过的图片 array """
    if C == 1:
        plt.imshow(np.reshape(img_array, shape[: 2]), cmap='gray')
    else:
        plt.imshow(np.reshape(img_array, shape))
    plt.show()

def plot_with_label(img_path, label):
    img_path = os.path.normpath(img_path)
    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    has_danmaku, cx, cy, rh, rw = label
    if has_danmaku > 0.5:
        h, w, _ = img_arr.shape
        pt1 = (int((cx-rw/2)*w), int((cy-rh/2)*h))
        pt2 = (int((cx+rw/2)*w), int((cy+rh/2)*h))
        img_arr = cv2.rectangle(img_arr, pt1, pt2, (0,0,255), 3)
    plt.imshow(img_arr)
    plt.show()

def read_labels():
    """ 读取标签数据 """
    labels = {}
    with open(LABEL_PATH, "r", encoding="gbk") as f:
        for line in f:
            items = line.split(",")
            assert len(items) >= 6
            img_name = os.path.basename(",".join(items[: -5]))
            img_name = img_name.split('\\')[-1].split('/')[-1]
            labels[img_name] = np.array(items[-5: ], dtype=np.float64)
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
        img_arr = image2array(img_path) / 255
        X.append(img_arr)
        Y.append(labels[img_name])

    assert len(X) == len(Y)
    data_set["X"], data_set["Y"] = np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)
    return data_set

def ramdom_divide_data_set(data_set, train_proportion=0.95):
    """ 随机将数据划分为训练集与测试集 """
    data_size = len(data_set["X"])
    train_size = int(data_size * train_proportion)
    permutation = list(np.random.permutation(data_size))
    train_permutation, dev_permutation = permutation[: train_size], permutation[train_size: ]
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
        data_set["X"] = h5f["X"][: ]
        data_set["Y"] = h5f["Y"][: ]
        h5f.close()
        print("发现处理好的数据文件，正在读取...")
    return data_set

def test_data_import():
    """ 导入测试数据 """
    test_data_set = {}
    img_names = os.listdir(TEST_IMG_DIR)
    X, img_paths = [], []
    for i, img_name in enumerate(img_names):
        print("get test data set {}/{} ".format(i, len(img_names)), end='\r')
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img_arr = image2array(img_path) / 255
        X.append(img_arr)
        img_paths.append(img_path)
    test_data_set["X"] = np.array(X, dtype=np.float64)
    test_data_set["img_paths"] = img_paths
    return test_data_set
