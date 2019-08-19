# coding:utf-8
import tensorflow as tf
import numpy as np
import gc
import os
import cv2
import tifffile as tif
import psutil
from tools import read_img, write_img
from smooth_tiled_predictions import predict_img_with_smooth_windowing


def predict(data_dir, output_dir, X, mode, pred):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(0, 15000, 5000):
        for j in range(0, 15000, 5000):
            one_data = '%s/tile_%d_%d.tif' % (data_dir, i, j)
            one_data_output = '%s/pred_tile_%d_%d.tif' % (output_dir, i, j)
            print ("predict tile_%d_%d.tif" % (i, j))
            proj, geotrans, input_img = read_img(one_data)
            label_pred_tiny = predict_img_with_smooth_windowing(
                input_img,
                window_size=768,
                subdivisions=2,
                batch_size=8,
                pred_func=(
                    lambda img: sess.run(pred, feed_dict={X: img, mode: False})
                )
            )
            label_pred_tiny = label_pred_tiny[:, :, 0]
            # label_pred_tiny[np.where(label_pred_tiny >= 0.3)] = 255
            label_pred_tiny[label_pred_tiny<0.1] = 0
            label_pred_tiny = label_pred_tiny.astype(np.float32)
            write_img(one_data_output, proj, geotrans, label_pred_tiny)


if __name__ == '__main__':
    gpu = 1
    model_dir = '/workspace/UNetXS/modelhp2/05_29_08_57_09/model20.ckpt'
    data_dir = ['/workspace/UNetXS/INPUT3']
    output_dir = ['./output17_hp']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    saver = tf.train.import_meta_graph(model_dir + ".meta")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver.restore(sess, model_dir)
    X, mode = tf.get_collection("inputs")
    pred = tf.get_collection("outputs")[0]

    for i in range(len(data_dir)):
        predict(data_dir[i], output_dir[i], X, mode, pred)

