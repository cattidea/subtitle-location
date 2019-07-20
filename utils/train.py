import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image

from utils.data import data_import, ramdom_divide_data_set, H as IH, W as IW, C as IC, \
                       SEED, test_data_import, img_plot, img_add_label
from utils.config import Config

CONFIG = Config()
MODEL_PATH = CONFIG['model_path']
MODEL_DIR = CONFIG['model_dir']
MODEL_META = CONFIG['model_meta']

def train(resume=False):
    """ 训练 """
    data_set = data_import()
    train_data_set, dev_data_set = ramdom_divide_data_set(data_set, train_proportion=0.95)
    test_data_set = test_data_import()
    train_size = len(train_data_set["X"])
    dev_size = len(dev_data_set["X"])
    test_size = len(test_data_set["X"])
    print("训练集数据 {} 条，开发集数据 {} 条，测试集数据 {} 条".format(train_size, dev_size, test_size))

    num_epochs = 1
    mini_batch_size = 64
    learning_rate = 0.0001
    keep_prob = 0.97
    GPU = True

    dev_step = 1
    save_step = 10
    test_plot = True
    max_to_keep = 5
    model = model_v4
    k = np.array([1, 1, 4, 1, 3])

    # GPU Config
    config = tf.ConfigProto(allow_soft_placement=True)
    if GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 训练
    graph = tf.Graph()
    with graph.as_default():
        # 训练计算图
        tf.set_random_seed(SEED)
        if resume:
            saver = tf.train.import_meta_graph(MODEL_META)
            X, Y, keep_prob_op, is_training, Y_, loss, train_op = tf.get_collection("train")
        else:
            X = tf.placeholder(dtype=tf.float32, shape=[None, IH, IW, IC], name="X")
            Y = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Y")
            keep_prob_op = tf.placeholder(dtype=tf.float32, name="keep_prob")
            is_training = tf.placeholder(dtype=tf.bool, name="is_training")
            Y_= model(X, keep_prob_op, is_training)
            loss = tf.reduce_sum(
                Y[:, 0] * tf.reduce_sum(tf.square(k * tf.subtract(Y, Y_)), axis=-1) + \
                (1 - Y[:, 0]) * tf.square(tf.subtract(Y[:, 0], Y_[:, 0]))
                )
            # loss = tf.reduce_sum(
            #     Y[:, 0] * (k[0] * -Y[:, 0] * tf.log(Y_[:, 0]) + tf.reduce_sum(tf.square(k[1: ] * tf.subtract(Y[:, 1:], Y_[:, 1:])), axis=-1)) + \
            #     (1 - Y[:, 0]) * (k[0] * -Y[:, 0] * tf.log(Y_[:, 0]))
            #     )
            # learning_rate = tf.train.exponential_decay(
            #     learning_rate,
            #     global_step = num_epochs * mini_batch_size,
            #     decay_steps = mini_batch_size,
            #     decay_rate = 0.999)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)

            tf.add_to_collection("train", X)
            tf.add_to_collection("train", Y)
            tf.add_to_collection("train", keep_prob_op)
            tf.add_to_collection("train", is_training)
            tf.add_to_collection("train", Y_)
            tf.add_to_collection("train", loss)
            tf.add_to_collection("train", train_op)


        with tf.Session(graph=graph, config=config) as sess:
            if resume:
                saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
                print("成功恢复模型")
            else:
                saver = tf.train.Saver(max_to_keep=max_to_keep)
                sess.run(tf.global_variables_initializer())
            for epoch in range(1, num_epochs+1):
                # train
                train_costs = []
                for mini_batch_X, mini_batch_Y in random_mini_batches(train_data_set, mini_batch_size=mini_batch_size, seed=SEED):
                    _, temp_cost = sess.run([train_op, loss], feed_dict={
                        X: mini_batch_X,
                        Y: mini_batch_Y,
                        keep_prob_op: keep_prob,
                        is_training: True
                    })
                    print("{} mini-batch >> cost: {} ".format(epoch, temp_cost / mini_batch_size), end="\r")
                    train_costs.append(temp_cost)
                train_cost = sum(train_costs) / train_size

                log_str = "{}/{} {} train cost is {} ".format(
                    epoch, num_epochs, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), train_cost)

                if epoch % dev_step == 0:
                    # dev
                    dev_cost = sess.run(loss, feed_dict={
                        X: dev_data_set["X"],
                        Y: dev_data_set["Y"],
                        keep_prob_op: 1,
                        is_training: False
                    })
                    dev_cost /= dev_size
                    log_str += "dev cost is {}".format(dev_cost)
                print(log_str)

                # save model
                if epoch % save_step == 0:
                    saver.save(sess, MODEL_PATH)

            if test_plot:
                encodings = sess.run(Y_, feed_dict={
                        X: test_data_set["X"],
                        keep_prob_op: 1,
                        is_training: False
                    })
                for i in range(test_size):
                    img_path = test_data_set["img_paths"][i]
                    label = encodings[i]
                    print(label)
                    img_plot(img_add_label(img_path, label), shape=None)


def model_v1(X, keep_prob, is_training):
    """ CNN model """
    A0 = X
    print("A0: {}".format(A0.shape))

    # CONV1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1")
    A1 = tf.nn.relu(A1)
    A1 = tf.layers.max_pooling2d(A1, pool_size=2, strides=2)
    print("A1: {}".format(A1.shape))

    # CONV2
    A2 = tf.layers.conv2d(inputs=A1, filters=16, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2")
    A2 = tf.nn.relu(A2)
    A2 = tf.layers.max_pooling2d(A2, pool_size=2, strides=2)
    print("A2: {}".format(A2.shape))

    # CONV3
    A3 = tf.layers.conv2d(inputs=A2, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV3")
    A3 = tf.nn.relu(A3)
    A3 = tf.layers.max_pooling2d(A3, pool_size=2, strides=2)
    print("A3: {}".format(A3.shape))

    # CONV4
    A4 = tf.layers.conv2d(inputs=A3, filters=64, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV4")
    A4 = tf.nn.relu(A4)
    A4 = tf.layers.max_pooling2d(A4, pool_size=2, strides=2)
    print("A4: {}".format(A4.shape))

    # flatten
    A5 = tf.layers.flatten(A4)
    print("A5: {}".format(A5.shape))

    # FC L1
    A6 = tf.layers.dense(inputs=A5, units=1024, name="FC1")
    if keep_prob != 1:
        A6 = tf.nn.dropout(A6, keep_prob)
    A6 = tf.nn.relu(A6)
    print("A6: {}".format(A6.shape))

    # FC L2
    A7 = tf.layers.dense(inputs=A6, units=256, name="FC2")
    if keep_prob != 1:
        A7 = tf.nn.dropout(A7, keep_prob)
    A7 = tf.nn.relu(A7)
    print("A7: {}".format(A7.shape))

    # FC L3
    A8 = tf.layers.dense(inputs=A7, units=5, name="FC3")
    if keep_prob != 1:
        A8 = tf.nn.dropout(A8, keep_prob)
    A8 = tf.nn.sigmoid(A8)
    print("A8: {}".format(A8.shape))

    Y = A8
    return Y

def model_v2(X, keep_prob, is_training):
    """ CNN model """
    A0 = X
    print("A0: {}".format(A0.shape))

    # CONV1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1")
    A1 = tf.layers.batch_normalization(inputs=A1, training=is_training, name="BN1")
    A1 = tf.nn.relu(A1)
    A1 = tf.layers.max_pooling2d(A1, pool_size=3, strides=3)
    print("A1: {}".format(A1.shape))

    # # CONV2
    # A2 = tf.layers.conv2d(inputs=A1, filters=16, kernel_size=3, strides=1, padding='same',
    #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2")
    # A2 = tf.layers.batch_normalization(inputs=A2, training=is_training, name="BN2")
    # A2 = tf.nn.relu(A2)
    # A2 = tf.layers.max_pooling2d(A2, pool_size=2, strides=2)
    # print("A2: {}".format(A2.shape))

    # INCEPTION1
    A3 = inception_v2(input=A1, scope="INCEPTION1", filters=32, is_training=is_training)
    print("A3: {}".format(A3.shape))

    # INCEPTION2
    A4 = inception_v2(input=A3, scope="INCEPTION2", filters=32, is_training=is_training)
    print("A3: {}".format(A3.shape))

    # CONV4
    A5 = tf.layers.average_pooling2d(A4, pool_size=32, strides=32)
    A5 = tf.layers.flatten(A5)
    print("A5: {}".format(A5.shape))

    # FC L1
    A6 = tf.layers.dense(inputs=A5, units=5, name="FC1")
    if keep_prob != 1:
        A6 = tf.nn.dropout(A6, keep_prob)
    # A6 = maxout(A6, 5)
    A6 = tf.nn.sigmoid(A6)
    print("A6: {}".format(A6.shape))

    Y = A6
    return Y

def model_v3(X, keep_prob, is_training):
    """ CNN model """
    A0 = X
    print("A0: {}".format(A0.shape))

    # CONV1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=3, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1")
    print("A1: {}".format(A1.shape))

    # INCEPTION1
    A2 = inception_v2(input=A1, scope="INCEPTION1", filters=32, is_training=is_training)
    print("A2: {}".format(A2.shape))

    # INCEPTION2
    A3 = inception_v2(input=A2, scope="INCEPTION2", filters=32, is_training=is_training)
    print("A3: {}".format(A3.shape))

    # CONV4
    A4 = tf.layers.conv2d(inputs=A3, filters=32, kernel_size=2, strides=2, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2")
    A4 = maxout(A4, 8)
    A4 = tf.layers.flatten(A4)
    print("A4: {}".format(A4.shape))

    # FC L1
    A5 = tf.layers.dense(inputs=A4, units=2048, name="FC1")
    if keep_prob != 1:
        A5 = tf.nn.dropout(A5, keep_prob)
    A5 = tf.nn.relu(A5)
    # A5 = maxout(A5, 512)
    print("A5: {}".format(A5.shape))

    # FC L2
    A6 = tf.layers.dense(inputs=A5, units=512, name="FC2")
    if keep_prob != 1:
        A6 = tf.nn.dropout(A6, keep_prob)
    A6 = tf.nn.relu(A6)
    # A6 = maxout(A6, 128)
    print("A6: {}".format(A6.shape))

    # FC L3
    A7 = tf.layers.dense(inputs=A6, units=5, name="FC3")
    if keep_prob != 1:
        A7 = tf.nn.dropout(A7, keep_prob)
    A7 = tf.nn.sigmoid(A7)
    print("A7: {}".format(A7.shape))

    Y = A7
    return Y

def model_v4(X, keep_prob, is_training):
    """ CNN model """
    A0 = X
    print("A0: {}".format(A0.shape))

    # CONV1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1")
    A1 = tf.nn.relu(A1)
    A1 = tf.layers.max_pooling2d(A1, pool_size=3, strides=3)
    print("A1: {}".format(A1.shape))

    # CONV2
    A2 = tf.layers.conv2d(inputs=A1, filters=16, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV3")
    A2 = tf.nn.relu(A2)
    A2 = tf.layers.max_pooling2d(A2, pool_size=2, strides=2)
    print("A2: {}".format(A2.shape))

    # CONV3
    A3 = tf.layers.conv2d(inputs=A2, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV4")
    A3 = tf.nn.relu(A3)
    A3 = tf.layers.max_pooling2d(A3, pool_size=2, strides=2)
    print("A3: {}".format(A3.shape))

    # flatten
    A4 = tf.layers.flatten(A3)
    print("A4: {}".format(A4.shape))

    # FC L1
    A5 = tf.layers.dense(inputs=A4, units=2048, name="FC1")
    if keep_prob != 1:
        A5 = tf.nn.dropout(A5, keep_prob)
    # A6 = tf.nn.relu(A6)
    A5 = maxout(A5, 512)
    print("A5: {}".format(A5.shape))

    # FC L2
    A6 = tf.layers.dense(inputs=A5, units=512, name="FC2")
    if keep_prob != 1:
        A6 = tf.nn.dropout(A6, keep_prob)
    # A6 = tf.nn.relu(A6)
    A6 = maxout(A6, 128)
    print("A6: {}".format(A6.shape))

    # FC L3
    A7 = tf.layers.dense(inputs=A6, units=5, name="FC3")
    if keep_prob != 1:
        A7 = tf.nn.dropout(A7, keep_prob)
    A7 = tf.nn.sigmoid(A7)
    print("A7: {}".format(A7.shape))

    Y = A7
    return Y


def model_v5(X, keep_prob, is_training):
    """ CNN model """
    A0 = X
    print("A0: {}".format(A0.shape))

    # CONV1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1_1")
    # A1 = tf.layers.batch_normalization(inputs=A1, training=is_training, name="BN1_1")
    A1 = tf.nn.relu(A1)

    A1 = tf.layers.conv2d(inputs=A1, filters=16, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1_2")
    # A1 = tf.layers.batch_normalization(inputs=A1, training=is_training, name="BN1_2")
    A1 = tf.nn.relu(A1)

    A1 = tf.layers.conv2d(inputs=A1, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1_3")
    # A1 = tf.layers.batch_normalization(inputs=A1, training=is_training, name="BN1_3")
    A1 = tf.nn.relu(A1)
    A1 = tf.layers.max_pooling2d(A1, pool_size=2, strides=2)
    print("A1: {}".format(A1.shape))

    # CONV2
    A2 = tf.layers.conv2d(inputs=A1, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2_1")
    # A2 = tf.layers.batch_normalization(inputs=A2, training=is_training, name="BN2_1")
    A2 = tf.nn.relu(A2)

    A2 = tf.layers.conv2d(inputs=A2, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2_2")
    # A2 = tf.layers.batch_normalization(inputs=A2, training=is_training, name="BN2_2")
    A2 = tf.nn.relu(A2)

    A2 = tf.layers.conv2d(inputs=A2, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2_3")
    # A2 = tf.layers.batch_normalization(inputs=A2, training=is_training, name="BN2_3")
    A2 = tf.nn.relu(A2)
    A2 = tf.layers.max_pooling2d(A2, pool_size=2, strides=2)
    print("A2: {}".format(A2.shape))

    # CONV3
    A3 = tf.layers.conv2d(inputs=A2, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV3_1")
    # A3 = tf.layers.batch_normalization(inputs=A3, training=is_training, name="BN3_1")
    A3 = tf.nn.relu(A3)

    A3 = tf.layers.conv2d(inputs=A3, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV3_2")
    # A3 = tf.layers.batch_normalization(inputs=A3, training=is_training, name="BN3_2")
    A3 = tf.nn.relu(A3)

    A3 = tf.layers.conv2d(inputs=A3, filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV3_3")
    # A3 = tf.layers.batch_normalization(inputs=A3, training=is_training, name="BN3_3")
    A3 = tf.nn.relu(A3)
    A3 = tf.layers.max_pooling2d(A3, pool_size=2, strides=2)
    print("A3: {}".format(A3.shape))

    # FLATTEN
    A4 = tf.layers.flatten(A3)
    print("A4: {}".format(A4.shape))

    # FC L1
    A5 = tf.layers.dense(inputs=A4, units=2048, name="FC1")
    if keep_prob != 1:
        A5 = tf.nn.dropout(A5, keep_prob)
    A5 = tf.nn.relu(A5)
    # A5 = maxout(A5, 512)
    print("A5: {}".format(A5.shape))

    # FC L2
    A6 = tf.layers.dense(inputs=A5, units=512, name="FC2")
    if keep_prob != 1:
        A6 = tf.nn.dropout(A6, keep_prob)
    A6 = tf.nn.relu(A6)
    # A6 = maxout(A6, 128)
    print("A6: {}".format(A6.shape))

    # FC L3
    A7 = tf.layers.dense(inputs=A6, units=5, name="FC3")
    if keep_prob != 1:
        A7 = tf.nn.dropout(A7, keep_prob)
    A7 = tf.nn.sigmoid(A7)
    print("A7: {}".format(A7.shape))

    Y = A7
    return Y

def model_v6(X, keep_prob, is_training):
    """ CNN model """
    A0 = X
    print("A0: {}".format(A0.shape))

    # CONV1
    A1 = tf.layers.conv2d(inputs=A0, filters=8, kernel_size=3, strides=3, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV1")
    print("A1: {}".format(A1.shape))

    # INCEPTION1
    A2 = inception_v2(input=A1, scope="INCEPTION1", filters=32, is_training=is_training)
    print("A2: {}".format(A2.shape))

    # INCEPTION2
    A3 = inception_v2(input=A2, scope="INCEPTION2", filters=32, is_training=is_training)
    print("A3: {}".format(A3.shape))

    # INCEPTION3
    A4 = inception_v2(input=A3, scope="INCEPTION3", filters=32, is_training=is_training)
    print("A4: {}".format(A4.shape))

    # INCEPTION4
    A5 = inception_v2(input=A4, scope="INCEPTION4", filters=32, is_training=is_training)
    print("A5: {}".format(A5.shape))

    # INCEPTION5
    A6 = inception_v2(input=A5, scope="INCEPTION5", filters=32, is_training=is_training)
    print("A6: {}".format(A6.shape))

    # INCEPTION6
    A7 = inception_v2(input=A6, scope="INCEPTION6", filters=32, is_training=is_training)
    print("A7: {}".format(A7.shape))

    # CONV4
    A8 = tf.layers.conv2d(inputs=A7, filters=32, kernel_size=2, strides=2, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="CONV2")
    A8 = maxout(A8, 8)
    A8 = tf.layers.flatten(A8)
    print("A8: {}".format(A8.shape))

    # FC L1
    A9 = tf.layers.dense(inputs=A8, units=2048, name="FC1")
    if keep_prob != 1:
        A9 = tf.nn.dropout(A9, keep_prob)
    # A9 = tf.nn.relu(A9)
    A9 = maxout(A9, 512)
    print("A9: {}".format(A9.shape))

    # FC L2
    A10 = tf.layers.dense(inputs=A9, units=512, name="FC2")
    if keep_prob != 1:
        A10 = tf.nn.dropout(A10, keep_prob)
    # A10 = tf.nn.relu(A10)
    A10 = maxout(A10, 128)
    print("A10: {}".format(A10.shape))

    # FC L3
    A11 = tf.layers.dense(inputs=A10, units=5, name="FC3")
    if keep_prob != 1:
        A11 = tf.nn.dropout(A11, keep_prob)
    A11 = tf.nn.sigmoid(A11)
    print("A11: {}".format(A11.shape))

    Y = A11
    return Y


def random_mini_batches(data_set, mini_batch_size=64, seed=0):
    """ 随机切分训练集为 mini_batch """
    np.random.seed(seed)
    data_size = len(data_set["X"])
    permutation = list(np.random.permutation(data_size))
    batch_permutation_indices = [permutation[i: i + mini_batch_size] for i in range(0, data_size, mini_batch_size)]
    for batch_permutation in batch_permutation_indices:
        mini_batch_X = data_set["X"][batch_permutation]
        mini_batch_Y = data_set["Y"][batch_permutation]
        yield mini_batch_X, mini_batch_Y

def maxout(inputs, num_units, axis=None):
    """ 将前层部分参数作为 maxout 的参数进行处理 """
    shape = inputs.get_shape().as_list()
    if axis is None:
        # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'
             .format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = -1
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs


def inception_v2(input, filters, scope, is_training):
    """ inception_v2 网络
    ref: https://blog.csdn.net/loveliuzz/article/details/79135583
    """
    assert filters % 32 == 0
    k = filters // 32
    res_1_1 = tf.layers.conv2d(inputs=input, filters=8*k, kernel_size=1, strides=1, padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1", reuse=tf.AUTO_REUSE)
    res_1_1_t3 = tf.layers.conv2d(inputs=input, filters=12*k, kernel_size=1, strides=1, padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1_t3", reuse=tf.AUTO_REUSE)
    res_3_3 = tf.layers.conv2d(inputs=res_1_1_t3, filters=16*k, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_3_3", reuse=tf.AUTO_REUSE)
    res_1_1_t5 = tf.layers.conv2d(inputs=input, filters=k, kernel_size=1, strides=1, padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_1_1_t5", reuse=tf.AUTO_REUSE)
    res_3_3_t5 = tf.layers.conv2d(inputs=res_1_1_t5, filters=2*k, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_3_3_t5", reuse=tf.AUTO_REUSE)
    res_5_5 = tf.layers.conv2d(inputs=res_3_3_t5, filters=4*k, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_5_5", reuse=tf.AUTO_REUSE)
    res_pool_t = tf.layers.max_pooling2d(input, pool_size=3, strides=1, padding='same')
    res_pool = tf.layers.conv2d(inputs=res_pool_t, filters=4*k, kernel_size=1, strides=1, padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name=scope+"_pool", reuse=tf.AUTO_REUSE)
    res = tf.concat([res_1_1, res_3_3, res_5_5, res_pool], axis=-1)
    # res = tf.layers.batch_normalization(inputs=res, training=is_training, name=scope+"_BN", reuse=tf.AUTO_REUSE)
    res = tf.nn.relu(res)
    return res
