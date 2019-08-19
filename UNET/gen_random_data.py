# coding:utf-8

from __future__ import print_function
import numpy as np
import random
import time
import os
import argparse
# from PIL import Image
# from skimage import io
import tifffile as tif
import cv2
import shutil
import glob
from tqdm import tqdm
import tools.read_img as readtif
_SIZE = 32
_CHANNELS = 1

# 后缀名，供进行多次测试数据集使用
_POST = '_deme2e'


def read_flags():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=('./data' + _POST))
    return parser.parse_args()



def cut_patches(index, image_path, label_path, rate, mode, csv_,directory='.'):


    if not os.path.exists(directory):
        os.makedirs(directory)

    print('tifname is ',image_path)
    image = tif.imread(image_path)
    print('label name is ',label_path)
    label = tif.imread(label_path)
    print('image is ', image.shape, '\nlabel is ', label.shape)
    print(image)


    print('image is ', image.shape, '\nlabel is ', label.shape)


    cv2.imwrite(image_path+'.png',image)
    cv2.imwrite(label_path+'.png',label)

    list_dir = os.path.join('{:s}.csv'.format(mode + _POST))
    with open(csv_, 'a') as f:
        _h = image.shape[0]
        _w = image.shape[1]
        num = int((_h * _w)/(_SIZE*_SIZE * rate))
        print('image shape is ',image.shape,'\nlabel shape is ',label.shape)

        print("tif %d generate %d images" % (index, num))
        print('index ',index,'\n num',num)

        for i in tqdm(range(num)):
            xs = random.randint(0, _h - _SIZE)
            ys = random.randint(0, _w - _SIZE)



            im = image[xs:xs + _SIZE, ys:ys + _SIZE]
            lb = label[xs:xs + _SIZE, ys:ys + _SIZE]
            fim_name = '%s_%02d_%05d_im' % (mode, index, i) + '.npy'
            flb_name = '%s_%02d_%05d_lb' % (mode, index, i) + '.npy'
            fim = os.path.join(directory, fim_name)
            flb = os.path.join(directory, flb_name)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # image = np.expand_dims(a, axis=2)
            # image = np.concatenate((image, image, image), axis=-1)
            # cv2.imwrite(fim, im)
            # cv2.imwrite(flb, lb)
            np.save(fim,im)
            np.save(flb,lb)
            f.write('data' + _POST + '/'+fim_name + ',' + 'data' + _POST + '/'+flb_name + '\n')


def main():

    flags = read_flags()

  
    train_path_hp='/workspace/DatasetWenchuan/DemComplate/e2e/train/'
    train_hp=glob.glob(os.path.join(train_path_hp,"*.tif"))

    train_path_label_hp='/workspace/DatasetWenchuan/DemComplate/e2e/label/'
    train_label_hp=glob.glob(os.path.join(train_path_label_hp, "*.tif"))

    train=train_hp
    train_label=train_label_hp

    print(train,'\n\n',train_label,'\n\n')
    # for i in range(len(train)):
    #     print(train[i] + '\t' + train_label[i])
    #
    # print(flags.save_dir)
    #
        # if os.path.exists(flags.save_dir):
        #     pass
        #     # shutil.rmtree(flags.save_dir)
        # else:
        #     os.makedirs(flags.save_dir)

    # 创建两个csv文件，其中存放着切图后的路径
    train_csv = '%s.csv' % ('train' + _POST)
    # test_csv = '%s.csv' % ('test' + _POST)
    if not os.path.exists(train_csv):
        os.system('touch '+train_csv)
    #     os.remove(train_csv)
    #
    # if os.path.exists(test_csv):
    #     os.remove(test_csv)

    # 构建训练集
    # for i in range(0,1):
    cut_patches(0, train[0], train_label[0], 0.01, mode='train', csv_=train_csv,directory='./data'+_POST)
        # cut_patches(i, train[i], train_label[i], 0.999, mode='test', directory=flags.save_dir)

    # 构建验证集
    # cut_patches(0, train[len(train)-1], train_label[len(train)-1], 1, mode='test', directory=flags.save_dir)

    # 进行shuffle
    for filename in [train_csv]:#, test_csv]:
        with open(filename, 'r') as f:
            lines = f.readlines()
        print("csv length: %d" % len(lines))
        random.shuffle(lines)
        with open(filename, 'w') as f:
            for i in lines:
                f.write(i)

if __name__ == '__main__':
    gpu = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    main()
