import os

import numpy as np
import tensorflow as tf

from utils.config import Config
from utils.data import SEED
from utils.data import C as IC
from utils.data import H as IH
from utils.data import W as IW
from utils.data import (img_add_label, img_crop_by_label, img_plot, img_save,
                        subtitle_recognition, test_data_import)
from utils.video_editor import THRESH, pics_merge_into_video, video_split

CONFIG = Config()
MODEL_PATH = CONFIG['model_path']
MODEL_DIR = CONFIG['model_dir']
MODEL_META = CONFIG['model_meta']
TEST_IMG_DIR = CONFIG['test_video_imgs_dir']
TEST_VIDEO = CONFIG['test_video']
CROP_IMGS_DIR = CONFIG['crop_imgs_dir']
LABEL_IMGS_DIR = CONFIG['label_imgs_dir']
TEST_CACHE = CONFIG['test_cache']
TEST_OUT_VIDEO = CONFIG['test_out_video']
SUBTITLES_TXT = CONFIG['subtitles_txt']

def test(test_config):
    use_cache = test_config["use_cache"]
    recognition = test_config["recognition"]
    GPU = test_config["use_GPU"]

    test_batch_step = 128
    keyframe_id_set = video_split(TEST_VIDEO, TEST_IMG_DIR, only_keyframes=False, compute_keyframe=THRESH)
    cache = TEST_CACHE if use_cache else None
    test_data_set = test_data_import(test_img_dir=TEST_IMG_DIR, cache=cache)
    num_test_imgs = len(test_data_set["X"])

    # GPU Config
    config = tf.ConfigProto(allow_soft_placement=True)
    if GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(MODEL_META)
        X, Y, keep_prob_op, is_training, Y_, loss, train_op = tf.get_collection("train")

        with tf.Session(graph=graph, config=config) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
            print("成功恢复模型")

            # 前向传播获取编码向量
            encodings = np.zeros((num_test_imgs, *Y_.shape[1: ]), dtype=np.float64)
            for i in range(0, num_test_imgs, test_batch_step):
                print("encoding {}/{} ".format(i, num_test_imgs), end="\r")
                X_batch = test_data_set["X"][i: i+test_batch_step]
                encodings_batch = sess.run(Y_, feed_dict={
                        X: X_batch,
                        keep_prob_op: 1,
                        is_training: False
                    })
                encodings[i: i+test_batch_step] = encodings_batch

            # 对图片打标签并裁剪出字幕区域
            for i in range(num_test_imgs):
                signs = [">>   ", " >>  ", "  >> ", "   >>"]
                print("{} {:6d}/{} ".format(signs[i%len(signs)], i, num_test_imgs), end="\r")
                img_path = test_data_set["img_paths"][i]
                img_name = os.path.basename(img_path)
                crop_img_path = os.path.join(CROP_IMGS_DIR, img_name)
                label_img_path = os.path.join(LABEL_IMGS_DIR, img_name)

                label = encodings[i]
                if i in keyframe_id_set:
                    crop_area = img_crop_by_label(img_path, label)
                    if crop_area is not None:
                        img_save(crop_area, crop_img_path)
                img_save(img_add_label(img_path, label), label_img_path)

            pics_merge_into_video(TEST_OUT_VIDEO, LABEL_IMGS_DIR)
            if recognition:
                subtitles = subtitle_recognition(CROP_IMGS_DIR)
                with open(SUBTITLES_TXT, "w", encoding="utf8") as f:
                    for subtitle in subtitles:
                        f.write(subtitle + '\n')
