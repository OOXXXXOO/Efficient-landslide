# coding:utf-8

import os
import cv2
import glob
import tensorflow as tf
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--rgb',
                    type=str,
                    default='./test.csv')
parser.add_argument('--model_dir',
                    type=str,
                    default='models_aug/08_28_15_45_18')
parser.add_argument('--disk_save_dir',
                    type=str,
                    default='pred')
parser.add_argument('--disk_save_name',
                    type=str,
                    default='2.jpg')
parser.add_argument('--gpu',
                    type=int,
                    default=1)
parser.add_argument('--gpu_fraction', type=float,
                    help='how much gpu is needed,usually 4G is enough', default=0.4)
parser.add_argument("--image-size",
                        default=768,
                        type=int,
                        help="image size (default: 256)")

flags = parser.parse_args()


def load_model():
    file_meta = os.path.join(flags.model_dir, 'model48.ckpt.meta')
    file_ckpt = os.path.join(flags.model_dir, 'model48.ckpt')


    saver = tf.train.import_meta_graph(file_meta)
    # tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags.gpu_fraction)
    sess = tf.InteractiveSession()
    saver.restore(sess, file_ckpt)
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
    return sess


def read_image(image_path, gray=False):
    # image_path = flags.rgb

    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        print(image_path)
        image = cv2.imread(image_path)
        # print(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main(flags):
    sess = load_model()
    X, mode = tf.get_collection('inputs')
    pred = tf.get_collection('outputs')[0]
    y = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 1], name="y")
    # y = tf.add_collection(y)
    csv_file = csv.reader(open(flags.rgb, 'r'))
    image_list= csv_file
    Iou = []
    precision = []
    recall = []
    # 计算Iou

    xpred = tf.nn.sigmoid(pred, name='sigmoid')

    pred_label = tf.cast(tf.greater(xpred, 0.5), dtype=tf.float32)
    H, W = pred_label.get_shape().as_list()[1:3]

    pred_flat = tf.reshape(pred_label, [-1, H * W])
    true_flat = tf.reshape(y, [-1, H * W])
    true_flat = tf.cast(true_flat, dtype=tf.float32)

    intersection = pred_flat * true_flat
    pre = tf.reduce_sum(intersection) / (tf.reduce_sum(pred_flat) + 1e-7)
    rec = tf.reduce_sum(intersection) / (tf.reduce_sum(true_flat) + 1e-7)
    f1_score = 2 / (1 / pre + 1 / rec + 1e-7)

    # pre = sess.run(pre)
    # rec = sess.run(rec)
    # f1_score = sess.run(f1_score)




    for item in image_list:

        pic= item[0]
        image = read_image(pic)
        Y = read_image(item[1], gray=True)
        Y = np.expand_dims(Y,-1)
        # Y= sess.run(Y)
        # image = cv2.resize(image, (128, 128))
        # sess=tf.InteractiveSession()

        label_pred,pre_value,rec_value,f1_score_value = sess.run(
            [pred, pre, rec, f1_score],
            feed_dict={X: np.expand_dims(image, 0), mode: False, y: np.expand_dims(Y,0)})
        # merged = np.squeeze(label_pred) * 255
        #merged = cv2.resize(merged, (4032, 3024))
        # save_name = os.path.join(flags.disk_save_dir, item[0])
        # cv2.imwrite(save_name, merged)
        precision.append(pre_value)
        recall.append(rec_value)
        Iou.append(f1_score_value)




    Prec = sum (precision) / len(precision)
    Rec = sum (recall) / len(recall)
    f1 = sum(Iou) / len(Iou)
    print Prec, Rec, f1

if __name__ == '__main__':
    main(flags)


