import tifffile as tif
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
from tools import read_img, write_img

if __name__ == "__main__":
    pred_label_file = "../data/pine/pred_classes.tif"
    test_label_file = "../data/pine/test_classes.tif"
    pred_label_file_png = "../data/pine/pred_classes.png"
    proj, geotrans, input_img = read_img(test_label_file)
    pred = cv2.imread(pred_label_file)
    write_img(pred_label_file, proj, geotrans, pred)
