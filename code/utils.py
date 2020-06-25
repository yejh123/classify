import numpy as np


def cal_ssr(y_predict, y_mean):
    return np.sum((y_predict-y_mean)**2)

