# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  :2017-05-23
# @Email :hanlixiao@gagogroup.com
# 主要功能：矢量图（shp）转化为对应分类图
import gdal
import os
import gc


def shp_to_imgclass(input_img_path, input_shp_path):
    """
    :param input_img_path: 输入图像路径
    :param input_shp_path: 栅格图像路径
    :return: 空
    """
    # 读取文件坐标信息
    dataset = gdal.Open(input_img_path, gdal.GA_ReadOnly)
    geo_transform = dataset.GetGeoTransform()
    raster_shape = dataset.GetRasterBand(1).ReadAsArray().shape
    Xmin = geo_transform[0]
    Xmax = geo_transform[0]+geo_transform[1]*raster_shape[1]
    Ymax = geo_transform[3]
    Ymin = geo_transform[3]+geo_transform[5]*raster_shape[0]
    out_raster_path = os.path.splitext(input_img_path)[0]+'_classes'+os.path.splitext(input_img_path)[1]
    os.system('/usr/bin/gdal_rasterize -a TYPE -ot Byte -te %s %s %s %s -ts %s %s %s %s' % (Xmin, Ymin, Xmax, Ymax, raster_shape[1], raster_shape[0], input_shp_path, out_raster_path))
    del dataset
    gc.collect()
    return


def main():
    input_img_path = ['/home/dk/data/pine/GF2_PMS2_E124.3_N42.0_20150530_L1A0001092183-MSS2_rpcortho.tif']
    input_shp_path = ['/home/dk/data/pine/样本数据/抚顺结果1.shp']
    for i in range(len(input_img_path)):
        shp_to_imgclass(input_img_path[i], input_shp_path[i])


if __name__ == "__main__":
    main()

