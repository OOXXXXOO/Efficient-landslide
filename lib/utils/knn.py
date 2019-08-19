import tifffile as tif
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


if __name__ == "__main__":
    train_file = "../data/pine/train.tif"
    train_label_file = "../data/pine/train_classes.tif"
    test_file = "../data/pine/test.tif"
    test_label_file = "../data/pine/test_classes.tif"
    pred_label_file = "../data/pine/pred_classes.tif"

    train = tif.imread(train_file)
    train_label = tif.imread(train_label_file)
    test = tif.imread(test_file)
    test_label = tif.imread(test_label_file)
    width, height, _ = test.shape
    train = np.reshape(train, [-1, 4])
    test = np.reshape(test, [-1, 4])
    train_label = np.reshape(train_label, [-1])
    test_label = np.reshape(test_label, [-1])

    knn = KNeighborsClassifier()
    print "predict start:"
    knn.fit(train, train_label)
    test_pred = knn.predict(test)
    test_pred = np.reshape(test_pred, [width, height])
    tif.imsave(pred_label_file, test_pred)
    score = knn.score(test, test_label, sample_weight=None)

    print test_pred
    print score

