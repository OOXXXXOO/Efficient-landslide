# !coding:utf-8
import os, gdal

in_path = '/workspace/01_汶川_九寨沟地震相关数据/汶川高分二号数据/'
input_filename = 'GF2_PMS1_E103.5_N31.5_20161230_L1A0002079855-PAN1.tif'

out_path = '../datasetGF'

if os.path.exists(out_path):
    os.makedirs(out_path)
output_filename = 'tile_'

tile_size_x = 5000
tile_size_y = 5000

ds = gdal.Open(in_path + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

# print xsize
# print ysize

gpu = 1

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)
