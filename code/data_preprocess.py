import os
import re
import math
import glob
from functools import cmp_to_key
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from scipy.signal import find_peaks

from config import Config


def file_compare(x, y):
    i1, i2 = 0, 0
    if mode == "sample":
        i1 = int(re.split('[e.]', x[-8:])[1])
        i2 = int(re.split('[e.]', y[-8:])[1])
    elif mode == "test":
        i1 = int(re.split('[t.]', x[-8:])[1])
        i2 = int(re.split('[t.]', y[-8:])[1])
    return i1 - i2


config = Config()

"""读取数据文件
    mode可以设置读取的是 sample 的数据还是 test 的数据
"""
mode = config.mode
files, class_map = [], []
if mode == "sample":
    # sample文件夹下的所有数据文件
    files = glob.glob(os.path.join(config.sample_data_dir, "sample*.txt"))
    files = sorted(files, key=cmp_to_key(file_compare))
    class_map = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'] + ['Unknown'] * 5
elif mode == "test":
    files = glob.glob(os.path.join(config.test_data_dir, "test*.txt"))
    files = sorted(files, key=cmp_to_key(file_compare))
    class_map = ['Unknown'] * 15
print(files)

data = []
for file in files:
    sample = pd.read_csv(file, sep="\t", header=None)
    sample.columns = ['x', 'y']
    data.append(sample)

if __name__ == '__main__':

    # 绘制趋势图
    if config.plot:
        row_num = 3
        column_cum = math.ceil(len(data) / row_num)
        fig, axs = plt.subplots(3, column_cum, figsize=(15, 8), constrained_layout=True)
        fig.suptitle(f"{mode} data 0-14")
        for i, sample in enumerate(data):
            axs[i % row_num][i // row_num].plot(sample.iloc[:, 0], sample.iloc[:, 1])
            axs[i % row_num][i // row_num].set_title(class_map[i])
            # axs[i // column_cum][i % column_cum].set_xlabel('自变量')
            # axs[i // column_cum][i % column_cum].set_ylable('因变量')
        plt.show()

    """
    用多项式回归模型拟合样本点
    计算峰值
    计算SSR
    """
    peaks_result, ssr = [], []
    if config.fit_with_linear_regression:
        row_num = 3
        column_cum = math.ceil(len(data) / row_num)
        fig, axs = plt.subplots(3, column_cum, figsize=(15, 8), constrained_layout=True)
        fig.suptitle(f"fit_with_linear_regression_{mode}")
        for i, sample in enumerate(data):
            # 对原数据进行平滑化处理，取每stride个数据点平均后作为一个新的数据点
            stride = 8
            num = len(sample) // stride
            end = num * stride
            x_stride = np.ndarray((stride, num))
            y_stride = np.ndarray((stride, num))
            x_process = np.ndarray((num,))
            y_process = np.ndarray((num,))

            for j in range(stride):
                x_stride[j] = sample.iloc[j:end:stride, 0].values
                y_stride[j] = sample.iloc[j:end:stride, 1].values

            x_process = x_stride.mean(axis=0)
            y_process = y_stride.mean(axis=0)

            z1 = np.polyfit(x_process, y_process, 7)  # 用7次多项式拟合
            poly_model = np.poly1d(z1)  # 多项式系数
            print(poly_model)  # 在屏幕上打印拟合多项式

            # 用拟合的曲线去预测y坐标
            y_pred = poly_model(sample.iloc[:, 0])
            sample['y_pred'] = pd.DataFrame(y_pred)
            sample.to_csv(os.path.join(config.prediction_dir, f"{mode}/{mode}_prediction_{i + 1}"))
            # 计算SSR
            # ssr.append(cal_ssr(y_pred, np.mean(sample.iloc[:, 1])))

            # 搜寻原数据的峰值
            peaks, _ = find_peaks(y_process, height=y_process.mean(), threshold=2, distance=100, width=1)
            assert len(peaks) > 0
            peaks_result.append([x_process[peaks[0]], y_process[peaks[0]]])
            # peaks, _ = find_peaks(y_pred)

            axs[i % row_num][i // row_num].plot(x_process, y_process, 'b.')
            axs[i % row_num][i // row_num].plot(sample.iloc[:, 0], y_pred, c='g')
            axs[i % row_num][i // row_num].plot(x_process[peaks], y_process[peaks], 'rx')
            axs[i % row_num][i // row_num].set_title(class_map[i])
        plt.show()

    # 保存ssr
    # ssr = pd.DataFrame(ssr, columns=['ssr'], index=range(1, len(ssr) + 1))
    # ssr.to_csv(os.path.join(config.prediction_dir, f"{mode}/{mode}_ssr"))

    """统计数据特征信息
    平均值
    标准差
    最大值
    最小值
    峰值x坐标
    峰值y坐标
    """
    avg = []
    standard = []
    max_value = []
    min_value = []

    for i, sample in enumerate(data):
        avg.append(sample['y'].mean())
        standard.append(sample['y'].std())
        max_value.append(sample['y'].max())
        min_value.append(sample['y'].min())

    analysis = pd.DataFrame(
        {'avg': avg, 'standard': standard, 'max_value': max_value, 'min_value': min_value})
    # analysis.to_csv(os.path.join(config.data_dir, f"analysis_{mode}.csv"))
    analysis = pd.concat([analysis, pd.DataFrame(peaks_result, columns=['peak_x', 'peak_y'])], axis=1)
    print(analysis)
    analysis.to_csv(os.path.join(config.data_dir, f"analysis_{mode}.csv"))
