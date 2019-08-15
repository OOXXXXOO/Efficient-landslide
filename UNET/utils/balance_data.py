import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import random

save_dir = 'data'

count_ground = 0
count_front = 0
train_list_dir = os.path.join('train1.csv')
with open(train_list_dir, 'w') as f:
    train_list = glob.glob(os.path.join(save_dir, '*train*im.png'))
    train_list.sort()
    train_lb_list = glob.glob(os.path.join(save_dir, '*train*lb.png'))
    train_lb_list.sort()
    for i in tqdm(range(len(train_list))):
        train_lb = cv2.imread(train_lb_list[i])
        if not train_lb.any():
            if np.random.random() < 0.2:
                f.write(train_list[i] + ',' + train_lb_list[i] + '\n')
                count_ground += 1
        else:
            f.write(train_list[i] + ',' + train_lb_list[i] + '\n')
            count_front += 1
print "train all ground %d, has front %d" % (count_ground, count_front)

count_test = 0
test_list_dir = os.path.join('test1.csv')
with open(test_list_dir, 'w') as f:
    test_list = glob.glob(os.path.join(save_dir, '*test*im.png'))
    test_list.sort()
    test_lb_list = glob.glob(os.path.join(save_dir, '*test*lb.png'))
    test_lb_list.sort()
    for i in tqdm(range(len(test_list))):
        test_lb = cv2.imread(test_lb_list[i])
        if test_lb.any():
            f.write(test_list[i] + ',' + test_lb_list[i] + '\n')
            count_test += 1
            if count_test >= 1000:
                break
print "test count %d" % count_test

filelist = ['train1.csv', 'test1.csv']

for filename in filelist:
    with open(filename, 'r') as f:
        lines = f.readlines()

    print len(lines)
    random.shuffle(lines)
    with open(filename, 'w') as f:
        for i in lines:
            f.write(i)

print "shuffle done"
