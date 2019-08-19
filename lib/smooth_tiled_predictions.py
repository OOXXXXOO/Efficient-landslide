# coding:utf-8
import numpy as np
import scipy.signal
import gc
# import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import os

def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2

    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 3), 3)
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind


def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    gc.collect()
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    gc.collect()
    return ret


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, batch_size, pred_func):
    padded_img = _pad_img(input_img, window_size, subdivisions)

    window_spline_2d = _window_2D(window_size=window_size, power=2)

    step = int(window_size / subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]

    result = np.zeros((padx_len, pady_len,1), dtype=np.float32)

    x_len = (padx_len-window_size)//step + 1
    y_len = (pady_len-window_size)//step + 1

    for batch_first in tqdm(range(0, x_len*y_len, batch_size)):
        batch_last = batch_first + batch_size
        # print batch_first
        if batch_last > x_len*y_len:
            batch_last = x_len*y_len
        one_batch =[]
        for index in range(batch_first, batch_last):
            x = (index // y_len)*step
            y = (index % y_len)*step
            one_batch.append(padded_img[x:x + window_size, y:y + window_size, :])
        one_batch = np.array(one_batch)
        pred = pred_func(one_batch)
        # pred = 1/(1+np.exp(-pred))
        for i, index in enumerate(range(batch_first, batch_last)):
            x = (index // y_len) * step
            y = (index % y_len) * step
            result[x:x + window_size, y:y + window_size] = \
                result[x:x + window_size, y:y + window_size] + pred[i] * window_spline_2d
        # print batch_first

    gc.collect()
    # print "done1"
    result = result / (subdivisions ** 2)

    gc.collect()
    # print "done2"
    prd = _unpad_img(result, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    return prd


if __name__ == '__main__':
    window_spline_2d = _window_2D(window_size=512, power=2)
    window_spline_2d = window_spline_2d.reshape((512,512))
    # plt.imshow(window_spline_2d, cmap="viridis")
    # plt.show()


