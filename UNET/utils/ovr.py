# !coding:utf-8
import gdal
import argparse
import time
import os
# image_list = ['../nas/orange/jx_label/瑞金市_classes.tif',
#               '../nas/orange/jx_label/东坡区_classes.tif',
#               '../nas/orange/jx_label/信丰县_classes.tif',
#               '../nas/orange/jx_label/寻乌县_classes.tif',
#               '../nas/orange/jx_label/于都县_classes.tif',
#               '../nas/orange/jx_label/宁都县_classes.tif',
#               '../nas/orange/jx_label/会昌县_classes.tif',
#               '../nas/orange/jx_label/安远县_classes.tif',
#               '../nas/orange/jx_label/蒲江县_classes.tif']
file_list = ['../datagroup/广深-2018/pin']
image_list = os.listdir(file_list)

for image in image_list:
    start = time.clock()
    print("Generate ovr...")
    ds = gdal.Open(image)
    ovlist = [2**a for a in range(1, 6)]
    ds.BuildOverviews(overviewlist=ovlist)
    elapsed = (time.clock() - start)
    print("Read tif time used: %fs" % elapsed)
