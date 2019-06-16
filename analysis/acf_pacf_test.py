# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

自相关函数检验，确定嵌入延迟
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa import stattools
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.data_filtering import savitzky_golay_filtering


def acf_pacf_test(data, columns):
    """
	对数据表中各字段数据逐一进行acf和pacf检验
	:param data: pd.DataFrame, 待检验变量数据表
	:param columns: list of strings, 选择字段名列表
	"""
    samples = np.array(data[columns])
    for i in range(samples.shape[1]):
        fig = plt.figure('acf & pacf test for %s' % columns[i], figsize=[12, 10])
        acf_fig = fig.add_subplot(2, 1, 1)
        sm.graphics.tsa.plot_acf(samples[:, i], lags = 200, ax = acf_fig)
        pacf_fig = fig.add_subplot(2, 1, 2)
        sm.graphics.tsa.plot_pacf(samples[:, i], lags = 200, ax = pacf_fig)
        plt.tight_layout()


if __name__ == '__main__':
    # 载入数据
    file_name = '../tmp/total_implemented_normalized_data.csv'
    data = pd.read_csv(file_name)
    
    # 数据滤波
    data = savitzky_golay_filtering(data)

    # 计算自相关函数
    columns = list(set([config.conf['model_params']['target_column']] + config.conf['model_params']['selected_columns']))
    data = data[columns]

    for col in columns:
        acf = []
        start_locs = range(1000, 25000, 100)
        for loc in start_locs:
            time_series = data.loc[loc:loc + 1000, col]
            acf.append(stattools.acf(time_series, nlags = 60))

        acf = np.array(acf).T
        labels = []
        for i in range(acf.shape[0]):
            labels.append(i * np.ones(acf.shape[1]))
        labels = np.array(labels)
        acf = np.hstack((acf.reshape(-1, 1), labels.reshape(-1, 1)))
        acf = pd.DataFrame(acf, columns=['acf_value', 'labels'])
        plt.figure(figsize=[20, 3])
        sns.boxplot(x = 'labels', y = 'acf_value', data = acf)
        plt.xlabel('acf lag')
        plt.ylabel('acf value for {}'.format(col))
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
