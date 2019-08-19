import glob
import os
import numpy as np
import tifffile as tif
from tqdm import tqdm

pre_train_dir = "./pretrain_data"
im_list = glob.glob(os.path.join(pre_train_dir, 'tile_im*.tif'))
im_list.sort()
lb_list = glob.glob(os.path.join(pre_train_dir, '*tile_lb*.tif'))
lb_list.sort()

used_im = 0
f_all = open("all_im_list.csv", 'w')
f_used = open("used_im_list.csv", 'w')
for i in tqdm(range(len(im_list))):
    im = tif.imread(im_list[i])
    shape = im.shape
    if np.count_nonzero(im) < shape[0]*shape[1]*shape[2] * 0.7:
        continue
    lb = tif.imread(lb_list[i])
    if np.count_nonzero(lb) < shape[0] * shape[1] * 0.1:
        f_all.write(im_list[i]+'\n')
        continue
    f_all.write(im_list[i]+'\n')
    f_used.write(im_list[i]+'\n')
f_all.close()
f_used.close()
