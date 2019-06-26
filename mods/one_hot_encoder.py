# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

one-hot编码
"""
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.data_filtering import savitzky_golay_filtering


def one_hot_encoding(data, save = True):
	"""对数据中离散变量进行one-hot编码"""
	continuous_columns = config.conf['model_params']['continuous_columns']
	discrete_columns = config.conf['model_params']['discrete_columns']
	
	continuous_data, discrete_data = data[continuous_columns], data[discrete_columns]
	
	for column in discrete_columns:
		discrete_data = discrete_data.copy()
		discrete_data[column] = discrete_data[column].apply(lambda x: int(x))
		
	# 对离散数据执行onehot编码
	total_data = pd.concat([data[['city', 'time_stamp']], continuous_data], axis = 1)
	enc = OneHotEncoder(categories = 'auto')
	for column in discrete_columns:
		enc.fit(discrete_data[[column]])
		encoded_data = enc.transform(discrete_data[[column]]).toarray()
		encoded_data = pd.DataFrame(encoded_data, columns = [column + '_{}'.format(p) for p in range(encoded_data.shape[1])])
		total_data = pd.concat([total_data, encoded_data], axis = 1)
	
	if save:
		total_data.to_csv('../tmp/total_encoded_data.csv', index = False)
	
	return total_data


if __name__ == '__main__':
	# 读取数据
	file = '../tmp/total_implemented_normalized_data.csv'
	data = pd.read_csv(file)
	
	# 数据滤波
	data = savitzky_golay_filtering(data)
	
	# 数据编码
	total_one_hot_encoded_data = one_hot_encoding(data, save = True)

