# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

计算变量作用时滞
"""
import json
import pandas as pd
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.correlation_analysis import cross_correlation_analysis


if __name__ == '__main__':
	# 载入数据
	file_name = '../tmp/total_encoded_data.csv'
	data = pd.read_csv(file_name)
	target_column = config.conf['model_params']['target_column']
	columns = list(data.columns[2:])
	
	# 计算外生变量影响
	time_lags_dict = cross_correlation_analysis(target_column, columns, data)
	
	for key in time_lags_dict.keys():
		if time_lags_dict[key][0] < 0:
			time_lags_dict[key] = 0
		elif time_lags_dict[key][0] > 10:
			time_lags_dict[key] = 10
		else:
			time_lags_dict[key] = time_lags_dict[key][0]
	
	# 保存结果
	with open('../tmp/time_lags.json', 'w') as f:
		json.dump(time_lags_dict, f)



