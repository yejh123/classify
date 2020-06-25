"""
分析统计特征
    avg
    standard
    max_value
    min_value
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt

from data_preprocess import data
from config import Config


config = Config()
analysis = pd.read_csv(os.path.join(config.data_dir, f"analysis_{config.mode}.csv"))
print(analysis)

analysis = analysis.values

# 正则化
scaler = StandardScaler()
analysis = scaler.fit_transform(analysis)  # 注意維度

pca = PCA(n_components=2)
analysis_low_dim = pca.fit_transform(analysis)
print('analysis_low_dim: ' , analysis_low_dim)


print('explained_variance_ratio_: ', pca.explained_variance_ratio_)
cum_explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(cum_explained_var_ratio)
plt.xlabel('# principal components')
plt.ylabel('cumulative explained variance')

if config.mode == 'sample':
    plt.figure()
    plt.scatter(analysis_low_dim[:3, 0], analysis_low_dim[:3, 1], c='r', label='A')
    plt.scatter(analysis_low_dim[3:6, 0], analysis_low_dim[3:6, 1], c='b', label='B')
    plt.scatter(analysis_low_dim[6:9, 0], analysis_low_dim[6:9, 1], c='g', label='C')
    plt.scatter(analysis_low_dim[9:14, 0], analysis_low_dim[9:14, 1], c='y', label='Unknown')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
plt.show()
