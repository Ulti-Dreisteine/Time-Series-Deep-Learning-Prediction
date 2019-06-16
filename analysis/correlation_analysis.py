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
from mods.data_filtering import savitzky_golay_filtering


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


def cross_correlation_analysis(target_column, selected_columns, data):
    """
    互相关分析
    :param target_column: str, 目标变量名
    :param selected_columns: list of strings, 选取变量名
    :param data: pd.DataFrame, 数据集
    """
    plt.figure(figsize=[6, 8])
    for exogen_column in selected_columns:
        plt.subplot(len(selected_columns), 1,
                    selected_columns.index(exogen_column) + 1)

        if exogen_column == selected_columns[0]:
            plt.title(
                'cross correlation analysis: exogeneous variable -> target variable'
            )

        target_series = np.array(data[target_column]).flatten()
        exogen_series = np.array(data[exogen_column]).flatten()

        cross_correlation_results = []
        for d in range(-2000, 2001):
            cross_correlation_results.append(
                [d, cross_correlation(target_series, exogen_series, d)])
        cross_correlation_results = np.array(cross_correlation_results)

        plt.plot(cross_correlation_results[:, 0],
                 cross_correlation_results[:, 1])
        plt.plot([0, 0], [
            np.min(cross_correlation_results[:, 1]),
            np.max(cross_correlation_results[:, 1])
        ], 'r--')
        plt.ylabel(exogen_column)

        if exogen_column == selected_columns[-1]:
            plt.xlabel('time lag')

        plt.tight_layout()
    plt.tight_layout()
    plt.show()
    plt.pause(1.0)


if __name__ == '__main__':
    # 载入数据
    data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
    target_column = config.conf['model_params']['target_column']
    selected_columns = config.conf['model_params']['selected_columns']

    # 带通滤波
    # data = data[list(set([target_column] + selected_columns))]
    data_filtered = savitzky_golay_filtering(data)

    # 计算外生变量影响
    cross_correlation_analysis(target_column, selected_columns[5:], data_filtered)

    # # 联合分布
    # for col in selected_columns:
    #     sns.jointplot(x = col, y = target_column, data = data, kind = 'hex', space = 0, size = 3)
    #     plt.xlabel(col)
    #     plt.ylabel(target_column)
    #     plt.xlim([0, 1])
    #     plt.ylim([0, 1])
    #     plt.tight_layout()
    #     plt.show()
    #     plt.pause(1.0)
    #
    # # 对比降噪前后效果
    # data = data.loc[10000:12000, :]
    # data_filtered = data_filtered.loc[10000:12000, :]
    # plt.figure(figsize = [8, 4])
    # plt.plot(list(data['pm10']))
    # plt.plot(list(data_filtered['pm10']), '--')
    # plt.legend(['true', 'filtered'])
    # plt.xlim([0, 2000])
    # plt.xlabel('time (hour)')
    # plt.ylabel('pm10 value')
    # plt.grid(True)
    # plt.tight_layout()
