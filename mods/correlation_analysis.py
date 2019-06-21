# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

数据相关性分析
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

sys.path.append('../')

from mods.config_loader import config


def cross_correlation(time_series_0, time_series_1, d):
    """
    在延迟为d上的互相关分析, 固定time_series_0, 移动time_series_1: d > 0向左移，d < 0向右移
    :param time_series_0: np.ndarray, 目标变量
    :param time_series_1: np.ndarray, 外生变量
    :param d: 延迟阶数
    :return:
    """
    # 数据截断
    if d > 0:
        x_series = time_series_0.flatten()[d:]
        y_series = time_series_1.flatten()[:-d]
    elif d == 0:
        x_series = time_series_0.flatten()
        y_series = time_series_1.flatten()
    elif d < 0:
        x_series = time_series_0.flatten()[:d]
        y_series = time_series_1.flatten()[-d:]

    mean_x, mean_y = np.mean(x_series), np.mean(y_series)
    numerator = np.sum((x_series - mean_x) * (y_series - mean_y))
    denominator = np.sqrt(np.sum(np.power((x_series - mean_x), 2))) * np.sqrt(
        np.sum(np.power((y_series - mean_y), 2)))

    return numerator / denominator


def cross_correlation_analysis(target_column, selected_columns, data, half_range = None):
    """
    互相关分析, 保存各字段对目标字段再不通时间延迟上的ccf函数计算结果
    :param target_column: str, 目标变量名
    :param selected_columns: list of strings, 选取变量名
    :param data: pd.DataFrame, 数据集
    """
    if half_range is None:
        half_range = 500
        
    target_series = np.array(data[target_column]).flatten()
    time_lags_dict = {}
    for column in selected_columns:
        print('processing column {}'.format(column))
        exogen_series = np.array(data[column]).flatten()

        cross_correlation_results = []
        for d in range(-half_range, half_range + 1):
            cross_correlation_results.append([d, cross_correlation(target_series, exogen_series, d)])
        peak_loc, peak_value = peak_loc_and_value([p[1] for p in cross_correlation_results])
        time_lags_dict[column] = [peak_loc - half_range, peak_value]
    return time_lags_dict


def peak_loc_and_value(ccf_values):
    """从一段ccf计算结果中找出峰值位置和对应的值"""
    mean, sigma = np.mean(ccf_values), np.power(np.var(ccf_values), 0.5)
    pos_corr_values, neg_corr_values = [p for p in ccf_values if (p > mean + 3.0 * sigma)], [p for p in ccf_values if (p < mean - 3.0 * sigma)]
    if (len(pos_corr_values) == 0) & (len(neg_corr_values) == 0):
        peak_value = 0
        peak_loc = int((len(ccf_values) - 1) / 2)
    else:
        if len(pos_corr_values) >= len(neg_corr_values):
            peak_value = max(pos_corr_values) - mean
            peak_loc = ccf_values.index(max(pos_corr_values))
        else:
            peak_value = mean - min(neg_corr_values)
            peak_loc = ccf_values.index(min(neg_corr_values))
    
    return peak_loc, peak_value
    

if __name__ == '__main__':
    # 载入数据
    file_name = '../tmp/total_encoded_data.csv'
    data = pd.read_csv(file_name)
    target_column = config.conf['model_params']['target_column']
    columns = list(data.columns[2:])

    # 计算外生变量影响
    time_lags_dict = cross_correlation_analysis(target_column, columns, data)

    # 联合分布
    # for column in columns:
    #     sns.jointplot(x = column, y = target_column, data = data, kind = 'hex', space = 0, height = 3)
    #     plt.xlabel(column)
    #     plt.ylabel(target_column)
    #     plt.xlim([0, 1])
    #     plt.ylim([0, 1])
    #     plt.tight_layout()
    #     plt.show()
    #     plt.pause(1.0)
        