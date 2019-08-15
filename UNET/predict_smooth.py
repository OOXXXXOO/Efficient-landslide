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

    print ("reading image...")
    proj, geotrans, input_img = read_img(data_dir)
    # input_img[:, :, [2, 0]] = input_img[:, :, [0, 2]]
    input_img = input_img[:, :, :3]
    print (input_img.shape)
    input_img = np.asarray(input_img, dtype='uint8')

    step = 32
    height, width, _ = input_img.shape
    print (height // step)
    print (width // step)
    for i in range(height//step):
        if i == height//step and height//step != 0:
            continue
        for j in range(width//step + 1):
            if j == width//step and width//step != 0:
                continue
            last_height_value = (i + 1) * step if i + 1 != height // step else height
            last_width_value = (j + 1) * step if j + 1 != width // step else width
            cv2.imwrite('%s/tmp_%d_%d.jpg' % (tmp_dir, i, j), input_img[i*step:last_height_value, j*step:last_width_value, :])
            print ('%s/tmp_%d_%d.jpg write done' % (tmp_dir, i, j))

    del input_img
    gc.collect()
    print ("done1")
    label_pred = np.zeros((height, width), dtype=np.uint8)
    for i in range(height//step):
        if i == height//step and height//step != 0:
            continue
        for j in range(width//step + 1):
            if j == width//step and width//step != 0:
                continue
            last_height_value = (i+1)*step if i+1 != height//step else height
            last_width_value = (j+1)*step if j+1 != width//step else width
            print(last_height_value, last_width_value)
            print('%s/tmp_%d_%d.jpg' % (tmp_dir, i, j))
            img_tiny = cv2.imread('%s/tmp_%d_%d.jpg' % (tmp_dir, i, j))
            print(img_tiny)
            label_pred_tiny = predict_img_with_smooth_windowing(
                img_tiny,
                window_size=32,
                subdivisions=2,
                batch_size=64,
                pred_func=(
                    lambda img: sess.run(pred, feed_dict={X: img, mode: False})
                )
            )
            label_pred_tiny = label_pred_tiny[:, :, 0]
            label_pred_tiny[np.where(label_pred_tiny >= 0.5)] = 1
            label_pred_tiny[np.where(label_pred_tiny < 0.5)] = 0
            label_pred_tiny = label_pred_tiny.astype(np.uint8)
            label_pred[i * step:last_height_value, j * step:last_width_value] = label_pred_tiny

    gc.collect()
    print ("done2")
    # cv2.imwrite(output_dir, label_pred)
    write_img(output_dir, proj, geotrans, label_pred)

    print ("done3")


def predictwhole(image,X,mode,pred):
    proj, geotrans, input_img = read_img(image)
    predfunc=(lambda img: sess.run(pred, feed_dict={X: img, mode: False}))
    onebatch=[]
    onebatch.append(input_img)
    one_batch = np.array(onebatch)
    pred_result=predfunc(onebatch)
    print(pred_result)



if __name__ == '__main__':
    tmp_dir = './tmp'
    gpu = 1
    model_dir = '/workspace/UNetXS/modeldem/06_13_03_24_02/model998.ckpt'
    data_dir = ['/workspace/UNetXS/DEMdata/image/0.tif','/workspace/UNetXS/DEMdata/image/1.tif']
    postfix = '_dem'
    output_dir = ['/workspace/UNetXS/predictresources/0.tif','/workspace/UNetXS/predictresources/1.tif']


    output_dir = [os.path.splitext(one_dir)[0]+postfix+os.path.splitext(one_dir)[1] for one_dir in output_dir]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    saver = tf.train.import_meta_graph(model_dir + ".meta")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver.restore(sess, model_dir)
    X, mode = tf.get_collection("inputs")
    pred = tf.get_collection("outputs")[0]

    for i in range(len(data_dir)):
        # predict(data_dir[i], output_dir[i], X, mode, pred)
        predictwhole(data_dir[i],X,mode,pred)