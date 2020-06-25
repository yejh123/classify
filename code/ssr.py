import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from data_preprocess import data
from config import Config


def cal_ssr(y_predict, y_mean):
    return np.sum((y_predict - y_mean) ** 2)


if __name__ == '__main__':
    config = Config()
    assert config.mode == "sample"

    num_known = 9
    num_unknown = 5
    ssr_known_to_unknown = []
    known_index = list(range(1, num_known + 1))
    unknown_index = list(range(num_known + 1, num_known + num_unknown + 1))
    known_index = known_index * len(unknown_index)
    unknown_index = sorted(unknown_index * int(len(known_index) / len(unknown_index)))
    for i in range(num_known):
        for j in range(num_known, num_known + num_unknown):
            combined_data = np.concatenate([data[i].iloc[:, 0:2].values, data[j].iloc[:, 0:2].values], axis=0)
            z1 = np.polyfit(combined_data[:, 0], combined_data[:, 1], 7)  # 用7次多项式拟合
            poly_model = np.poly1d(z1)  # 多项式系数

            print(f"sample{i + 1}与sample{j + 1}拟合曲线方程：{poly_model}")
            y_pred = poly_model(combined_data[:, 0])
            ssr_known_to_unknown.append(cal_ssr(y_pred, np.mean(combined_data[:, 1])))

            if i == 0:
                plt.figure()
                plt.plot(combined_data[:, 0], combined_data[:, 1], 'b.')
                plt.plot(combined_data[:, 0], y_pred, 'r.')
                plt.show()

    ssr_pd = pd.DataFrame({'known_index':known_index, 'unknown_index':unknown_index, "ssr":ssr_known_to_unknown})
    ssr_pd.to_csv(os.path.join(config.prediction_dir, f"sample/ssr_known_to_unknown"))

