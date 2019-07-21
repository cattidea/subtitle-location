import operator
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

"""
ref: https://blog.csdn.net/qq_21997625/article/details/81285096
"""

frame_diffs = []
THRESH = lambda frames, keyframe_id_set: _thresh(frames, keyframe_id_set, thresh=0.6)
LOCAL_MAXIMA = lambda frames, keyframe_id_set: _local_maxima(frames, keyframe_id_set, frame_diffs, smoothing_window_size=50)
TOP_ORDER = lambda frames, keyframe_id_set: _top_order(frames, keyframe_id_set, num_top_frames=50)

def smooth(x, window_len=13, window='hanning'):
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


class Frame:
    """class to hold information about each frame

    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
    if max(a, b) == 0:
        x = 0
    else:
        x = (b - a) / max(a, b)
    return x

def _top_order(frames, keyframe_id_set, num_top_frames=50):
    frames.sort(key=operator.attrgetter("diff"), reverse=True)
    for keyframe in frames[:num_top_frames]:
        keyframe_id_set.add(keyframe.id)

def _thresh(frames, keyframe_id_set, thresh=0.6):
    for i in range(1, len(frames)):
        if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= thresh):
            keyframe_id_set.add(frames[i].id)

def _local_maxima(frames, keyframe_id_set, frame_diffs, smoothing_window_size=50):
    diff_array = np.array(frame_diffs)
    sm_diff_array = smooth(diff_array, smoothing_window_size)
    frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
    for i in frame_indexes:
        keyframe_id_set.add(frames[i - 1].id)

def get_keyframe_id_set(video_path, compute_keyframe=LOCAL_MAXIMA):
    """ 从视频中提取关键帧序号 """
    cap = cv2.VideoCapture(video_path)
    curr_frame = None
    prev_frame = None
    frames = []
    success = cap.isOpened()
    idx = 0
    while(success):
        print("get keyframe id set {}".format(idx), end="\r")
        if curr_frame is not None and prev_frame is not None:
            #logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(idx, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        success, frame = cap.read()
        idx = idx + 1
        if frame is not None:
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
    cap.release()

    # compute keyframe
    keyframe_id_set = set()
    compute_keyframe(frames, keyframe_id_set)
    return keyframe_id_set

def video_split(video_path, imgs_dir, only_keyframes=False, compute_keyframe=LOCAL_MAXIMA):
    """ 将视频切分为图片 """
    cap = cv2.VideoCapture(str(video_path))
    keyframes = []
    success, frame = cap.isOpened(), None
    idx = 0

    if compute_keyframe:
        keyframe_id_set = get_keyframe_id_set(video_path, compute_keyframe=compute_keyframe)
    else:
        keyframe_id_set = set()

    while(success):
        print("video_split {} ".format(idx), end="\r")
        success, frame = cap.read()
        if frame is not None and (not only_keyframes or idx in keyframe_id_set):
            img_name = "{:06d}.jpg".format(idx)
            img_path = os.path.join(imgs_dir, img_name)
            cv2.imwrite(img_path, frame)
        idx = idx + 1
    cap.release()
    return keyframe_id_set

def pics_merge_into_video(video_path, pics_dir):
    """ 将图片合并为视频 """
    img_names = os.listdir(pics_dir)
    shape = cv2.imread(os.path.join(pics_dir, img_names[0])).shape
    size = (shape[1], shape[0])
    fps = 24
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(video_path, fourcc, fps, size)

    for i, img_name in enumerate(img_names):
        print("merge {}/{} ".format(i, len(img_names)), end='\r')
        if img_name.endswith('.jpg'):
            img_path = os.path.join(pics_dir, img_name)
            img = cv2.imread(img_path)
            video.write(img)
    video.release()


if __name__ == "__main__":
    video_path = "test_video.mp4"
    imgs_dir = "key_frames/"
    print(video_split(video_path, imgs_dir, only_keyframes=True, compute_keyframe=THRESH))
