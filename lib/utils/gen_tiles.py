# !coding:utf-8
from __future__ import print_function
import numpy as np
import os
import argparse
from tqdm import tqdm
import tifffile as tif
import gc
import time


def cut_tiles(img_num, image_path, label_path, directory, h_n, w_n):
    start = time.clock()
    print("Reading image and label...")
    image = tif.imread(image_path)
    label = tif.imread(label_path)
    elapsed = (time.clock() - start)
    print("Read tif time used: %fs" % elapsed)
    height, width, _ = image.shape
    h_step = height//h_n
    w_step = width//w_n
    for i in tqdm(range(h_n)):
        for j in range(w_n):
            last_height_value = (i + 1) * h_step if i + 1 != h_n else height
            last_width_value = (j + 1) * w_step if j + 1 != w_n else width
            tif.imsave('%s/tile_im_%02d_%02d_%02d.tif' % (directory, img_num, i, j), image[i*h_step:last_height_value, j*w_step:last_width_value, :])
            tif.imsave('%s/tile_lb_%02d_%02d_%02d.tif' % (directory, img_num, i, j), label[i*h_step:last_height_value, j*w_step:last_width_value])
    del image
    del label
    gc.collect()


def read_flags():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/orange')
    parser.add_argument('--save_dir', type=str, default='pretrain_data')
    return parser.parse_args()


def main():

    flags = read_flags()

    img_zone = ["信丰县.tif",
                "安远县.tif",
                "瑞金市.tif",
                '蒲江县.tif',
                '于都县.tif',
                '宁都县.tif',
                '会昌县.tif',
                '东坡区.tif',
                '寻乌县.tif'
                ]
    lb_zone = ["信丰县_classes.tif",
               "安远县_classes.tif",
               "瑞金市_classes.tif",
               '蒲江县_classes.tif',
               '于都县_classes.tif',
               '宁都县_classes.tif',
               '会昌县_classes.tif',
               '东坡区_classes.tif',
               '寻乌县_classes.tif'
               ]
    image_path = [os.path.join(flags.data_dir, zone) for zone in img_zone]
    label_path = [os.path.join(flags.data_dir, zone) for zone in lb_zone]

    print(image_path)
    print(label_path)

    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)

    for i in range(len(image_path)):
        cut_tiles(i, image_path[i], label_path[i], flags.save_dir, 10, 10)


if __name__ == '__main__':
    main()
