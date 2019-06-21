# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

自相关函数检验，确定嵌入延迟
"""
import pandas as pd
import numpy as np
from statsmodels.tsa import stattools
import sys

sys.path.append('../')

from mods.config_loader import config


def cal_acf_lag(acf_results, thresholds = None):
    """
    从计算所得acf结果中得出各acf阈值对应的lag
    :param acf_results: np.array, 一维acf计算结果，shape = (-1,)
    :param thresholds: list of floats, acf阈值
    :return:
    """
    if thresholds is None:
        thresholds = [0.8, 0.05]
    
    try:
        lags = {}
        for thres in thresholds:
            lags[thres] = []
        
        for i in range(len(acf_results)):
            for thres in thresholds:
                if acf_results[i] <= thres:
                    lags[thres].append(i)
        
        for thres in thresholds:
            lags[thres] = min(lags[thres])
        
        return lags
    except Exception as e:
        raise RuntimeError(e)
    

if __name__ == '__main__':
    # 设定参数
    continuous_columns = config.conf['model_params']['continuous_columns']
    
    # 载入数据
    file_name = '../tmp/total_encoded_data.csv'
    data = pd.read_csv(file_name)[continuous_columns]

    # 计算自相关函数
    columns = list(data.columns)  # 不包括'city'和'time_stamp'字段
    series_len = 1000
    start_locs = range(100, 25000, 10)
    acf_results = {}
    for column in columns:
        print('\nprocessing column {}'.format(column))
        acf_column_results = []
        for loc in start_locs:
            time_series = data.loc[loc: loc + series_len, column]
            acf_column_results.append(stattools.acf(time_series, nlags = 500))
        acf_column_results = np.mean(np.array(acf_column_results), axis = 0)
        
        acf_results[column] = cal_acf_lag(acf_column_results)