# coding: utf-8
import gdal
import cv2
import os
import numpy
import tifffile
import random
from PIL import Image
import numpy as np


# 读图像文件
def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    if im_data.shape[0]==4:
        im_data=im_data[0:3]
    if len(im_data.shape) == 3:
        im_data = im_data.transpose(1, 2, 0)

    del dataset
    return im_proj, im_geotrans, im_data


# 写文件，以写成tif为例
def write_img(filename, im_proj, im_geotrans, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'uint8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
        # print 'uint8'
    elif 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_data = im_data.transpose(2, 0, 1)
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(
        filename,
        im_width,
        im_height,
        im_bands,
        datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def shuffle():
    filelist = ['train.csv', 'test.csv']
    for filename in filelist:
        with open(filename, 'r') as f:
            lines = f.readlines()

        print (len(lines))
        random.shuffle(lines)
        with open(filename, 'w') as f:
            for i in lines:
                f.write(i)


if __name__ == "__main__":
    proj, geotrans, _ = read_img('../data/haidian_data/data/2003.73.tif')  # 读数据
    # print proj
    # print geotrans
    # print data
    data2 = tifffile.imread('../data/haidian_data/label/2003.73.tif')
    data2.astype('uint8')
    # print data2.shape
    write_img('abc.tif', proj, geotrans, data2)  # 写数据
    data = tifffile.imread('abc.tif')
    # print data.shap
