# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""
import copy
import time
from scipy.ndimage.interpolation import shift
import numpy as np
import pandas as pd
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.data_filtering import savitzky_golay_filtering


def build_single_dim_manifold(time_series, embed_dim, lag, time_lag, direc = 1):
	"""
	构建一维时间序列嵌入流形
	:param time_lag: int, 对应的时间延迟
	:param direc: int, 平移方向，1为向下，-1为向上
	:param time_series: np.ndarray or pd.DataFrame, 一维时间序列, shape = (-1,)
	:param embed_dim: int, 嵌入维数
	:param lag: int, 嵌入延迟
	:return: manifold: np.ndarray, 嵌入流形数组, shape = (-1, embed_dim)
	"""
	time_series_copy = copy.deepcopy(time_series)
	manifold = []
	for dim in range(embed_dim):
		manifold.append(shift(shift(time_series_copy, direc * dim * lag), time_lag))
	manifold = np.array(manifold).T
	return manifold


def build_samples_data_frame(data):
	"""
	构建样本集
	:param data: pd.DataFrame, 数据表
	:return:
		data_new: pd.DataFrame, 构建的全量数据集，每个字段对应向量从前往后对应时间戳降序排列
	"""
	selected_columns = config.conf['model_params']['selected_columns']
	embed_lags = config.conf['model_params']['embed_lags']
	acf_lags = config.conf['model_params']['acf_lags']
	time_lags = config.conf['model_params']['time_lags']
	embed_dims = dict()
	for col in selected_columns:
		embed_dims[col] = int(np.floor(acf_lags[col] / embed_lags[col]))
		print('embed_dim for {} is {}'.format(col, embed_dims[col]))
	
	data_new = data[['time_stamp']]
	for col in selected_columns:
		samples = build_single_dim_manifold(data.loc[:, col], embed_dims[col], embed_lags[col], time_lags[col])
		columns = [col + '_{}'.format(i) for i in range(samples.shape[1])]
		samples = pd.DataFrame(samples, columns = columns)
		data_new = pd.concat([data_new, samples], axis = 1, sort = True)
		
	return data_new


def build_targets_data_frame(data):
	"""
	构建目标数据集
	:param data: pd.DataFrame, 数据表
	:return:
	"""
	target_column = config.conf['model_params']['target_column']
	embed_lag = 1
	pred_dim = config.conf['model_params']['pred_dim']
	
	data_new = data[['time_stamp']]
	samples = build_single_dim_manifold(data.loc[:, target_column], pred_dim, embed_lag, direc = -1, time_lag = 0)
	columns = [target_column + '_{}'.format(i) for i in range(samples.shape[1])]
	samples = pd.DataFrame(samples, columns = columns)
	data_new = pd.concat([data_new, samples], axis = 1, sort = True)
	
	return data_new


def build_train_samples_and_targets():
	"""
	获取训练集的样本和目标数据集
	:param samples_df: pd.DataFrame
	:return:
		X_train: np.ndarray, 训练集样本
		y_train: np.ndarray, 训练集目标
	"""
	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	
	# 滤波
	data_filtered = savitzky_golay_filtering(data)
	# data_filtered = data
	
	# 构建样本的目标数据集
	samples_df = build_samples_data_frame(data_filtered)
	targets_df = build_targets_data_frame(data_filtered)
	
	exist_record_time = config.conf['exist_record_time']
	samples_len = config.conf['model_params']['samples_len']
	hr = 3600
	exist_time_stamp = int(time.mktime(time.strptime(str(exist_record_time), "%Y%m%d%H")))
	
	X_train_df = samples_df[(samples_df.time_stamp >= exist_time_stamp - samples_len * hr) & (samples_df.time_stamp < exist_time_stamp)]
	y_train_df = targets_df[(targets_df.time_stamp >= exist_time_stamp - samples_len * hr + hr) & (targets_df.time_stamp < exist_time_stamp + hr)]
	
	X_train = np.array(X_train_df.iloc[:, 1:])
	y_train = np.array(y_train_df.iloc[:, 1:])
	
	return X_train, y_train


def build_test_samples_and_targets():
	"""
	获取训练集的样本和目标数据集
	:param samples_df: pd.DataFrame
	:return:
		X_test: np.ndarray, 测试集样本
		y_test: np.ndarray, 测试集目标
	"""
	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	
	# 滤波
	data_filtered = savitzky_golay_filtering(data)
	# data_filtered = data
	
	samples_df = build_samples_data_frame(data_filtered)
	targets_df = build_targets_data_frame(data_filtered)
	
	# 数据集构建
	exist_record_time = config.conf['exist_record_time']
	pred_horizon_len = config.conf['model_params']['pred_horizon_len']
	hr = 3600
	exist_time_stamp = int(time.mktime(time.strptime(str(exist_record_time), "%Y%m%d%H")))
	
	X_test_df = samples_df[(samples_df.time_stamp >= exist_time_stamp) & (samples_df.time_stamp < exist_time_stamp + pred_horizon_len * hr)]
	y_test_df = targets_df[(targets_df.time_stamp >= exist_time_stamp + hr) & (targets_df.time_stamp < exist_time_stamp + pred_horizon_len * hr + hr)]
	
	X_test = np.array(X_test_df.iloc[:, 1:])
	y_test = np.array(y_test_df.iloc[:, 1:])
	
	return X_test, y_test
	

if __name__ == '__main__':
	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	
	# 滤波
	data_filtered = savitzky_golay_filtering(data)

	# 进行样本构建
	samples_df = build_samples_data_frame(data_filtered)

	# 进行目标值样本构建
	targets_df = build_targets_data_frame(data_filtered)
	
	# 训练集
	X_train, y_train = build_train_samples_and_targets()
	
	# 测试集
	X_test, y_test = build_test_samples_and_targets()
	
	#
	import matplotlib.pyplot as plt
	plt.plot(samples_df['time_stamp'])
	plt.plot(targets_df['time_stamp'])
	

	
	
	