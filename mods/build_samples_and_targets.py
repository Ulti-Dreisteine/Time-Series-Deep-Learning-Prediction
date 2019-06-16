# -*- coding: utf-8 -*-
"""
Created on 2019/6/7 19:14
@author: luolei

构建lstm模型样本
"""
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np
import copy
from scipy.ndimage.interpolation import shift
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.data_filtering import savitzky_golay_filtering


def build_single_series_manifold(time_series, embed_dim, embed_lag, time_lag, direc = 1):
	"""
	构建一维时间序列嵌入流形
	:param time_lag: int, 对应的时间延迟
	:param direc: int, 平移方向，1为向下，-1为向上
	:param time_series: np.ndarray or pd.DataFrame, 一维时间序列, shape = (-1,)
	:param embed_dim: int, 嵌入维数
	:param embed_lag: int, 嵌入延迟
	:return: manifold: np.ndarray, 嵌入流形数组, shape = (-1, embed_dim)
	"""
	time_series_copy = copy.deepcopy(time_series)
	manifold = []
	for dim in range(embed_dim):
		manifold.append(shift(shift(time_series_copy, direc * dim * embed_lag), time_lag))
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
		samples = build_single_series_manifold(data.loc[:, col], embed_dims[col], embed_lags[col], time_lags[col])
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
	embed_dim = config.conf['model_params']['pred_dim']

	data_new = data[['time_stamp']]
	samples = build_single_series_manifold(data.loc[:, target_column], embed_dim, embed_lag, direc = -1, time_lag = 0)
	columns = [target_column + '_{}'.format(i) for i in range(samples.shape[1])]
	samples = pd.DataFrame(samples, columns = columns)
	data_new = pd.concat([data_new, samples], axis = 1, sort = True)

	return data_new


def build_train_samples_dict():
	"""
	获取样本字典
	:return:
		samples_dict: dict, {'pm10': np.ndarray(shape = (samples_len, embed_dim, 1)), ...}
	"""
	# 设定参数
	exist_record_time = config.conf['exist_record_time']
	exist_time_stamp = int(time.mktime(time.strptime(str(exist_record_time), '%Y%m%d%H')))
	selected_columns = config.conf['model_params']['selected_columns']
	samples_len = config.conf['model_params']['samples_len']
	hr = config.conf['model_params']['hr']

	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')

	# 滤波
	data = savitzky_golay_filtering(data)

	# 生成样本数据
	samples_df = build_samples_data_frame(data)
	samples_columns = samples_df.columns

	# 构造样本
	samples_dict = {}
	for column in selected_columns:
		data_columns = [p for p in samples_columns if column in p]

		samples = samples_df[['time_stamp'] + data_columns]

		start_time_stamp = exist_time_stamp - samples_len * hr
		end_time_stamp = exist_time_stamp - hr
		samples = samples[(samples.time_stamp >= start_time_stamp) & (samples.time_stamp <= end_time_stamp)]

		samples = np.array(samples.iloc[:, 1:])
		samples = samples[:, :, np.newaxis]

		samples_dict[column] = samples

	return samples_dict


def build_test_samples_dict():
	"""
	获取样本字典
	:return:
		samples_dict: dict, {'pm10': np.ndarray(shape = (samples_len, embed_dim, 1)), ...}
	"""
	# 设定参数
	exist_record_time = config.conf['exist_record_time']
	exist_time_stamp = int(time.mktime(time.strptime(str(exist_record_time), '%Y%m%d%H')))
	selected_columns = config.conf['model_params']['selected_columns']
	pred_horizon_len = config.conf['model_params']['pred_horizon_len']
	hr = config.conf['model_params']['hr']

	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')

	# 滤波
	data = savitzky_golay_filtering(data)

	# 生成样本数据
	samples_df = build_samples_data_frame(data)
	samples_columns = samples_df.columns

	# 构造样本
	samples_dict = {}
	for column in selected_columns:
		data_columns = [p for p in samples_columns if column in p]

		samples = samples_df[['time_stamp'] + data_columns]

		start_time_stamp = exist_time_stamp
		end_time_stamp = exist_time_stamp + (pred_horizon_len - 1) * hr
		samples = samples[(samples.time_stamp >= start_time_stamp) & (samples.time_stamp <= end_time_stamp)]

		samples = np.array(samples.iloc[:, 1:])
		samples = samples[:, :, np.newaxis]

		samples_dict[column] = samples

	return samples_dict


def build_train_targets_array():
	"""
	获取样本字典
	:return:
		samples_dict: dict, {'pm10': np.ndarray(shape = (samples_len, embed_dim, 1)), ...}
	"""
	# 设定参数
	exist_record_time = config.conf['exist_record_time']
	exist_time_stamp = int(time.mktime(time.strptime(str(exist_record_time), '%Y%m%d%H')))
	target_column = config.conf['model_params']['target_column']
	samples_len = config.conf['model_params']['samples_len']
	hr = config.conf['model_params']['hr']

	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')

	# 滤波
	data = savitzky_golay_filtering(data)

	# 生成样本数据
	targets_df = build_targets_data_frame(data)
	targets_columns = targets_df.columns

	# 构造样本
	data_columns = [p for p in targets_columns if target_column in p]

	targets = targets_df[['time_stamp'] + data_columns]

	start_time_stamp = exist_time_stamp - samples_len * hr + hr
	end_time_stamp = exist_time_stamp
	targets = targets[(targets.time_stamp >= start_time_stamp) & (targets.time_stamp <= end_time_stamp)]

	targets = np.array(targets.iloc[:, 1:])
	targets = targets[:, :, np.newaxis]

	return targets


def build_test_targets_array():
	"""
	获取样本字典
	:return:
		samples_dict: dict, {'pm10': np.ndarray(shape = (samples_len, embed_dim, 1)), ...}
	"""
	# 设定参数
	exist_record_time = config.conf['exist_record_time']
	exist_time_stamp = int(time.mktime(time.strptime(str(exist_record_time), '%Y%m%d%H')))
	target_column = config.conf['model_params']['target_column']
	pred_horizon_len = config.conf['model_params']['pred_horizon_len']
	hr = config.conf['model_params']['hr']

	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')

	# 滤波
	data = savitzky_golay_filtering(data)

	# 生成样本数据
	targets_df = build_targets_data_frame(data)
	targets_columns = targets_df.columns

	# 构造样本
	data_columns = [p for p in targets_columns if target_column in p]

	targets = targets_df[['time_stamp'] + data_columns]

	start_time_stamp = exist_time_stamp + hr
	end_time_stamp = exist_time_stamp + pred_horizon_len * hr
	targets = targets[(targets.time_stamp >= start_time_stamp) & (targets.time_stamp <= end_time_stamp)]

	targets = np.array(targets.iloc[:, 1:])
	targets = targets[:, :, np.newaxis]

	return targets


def build_train_and_verify_datasets():
	"""构建训练和验证数据集"""
	pred_dim = config.conf['model_params']['pred_dim']
	batch_size = config.conf['model_params']['batch_size']
	use_cuda = config.conf['model_params']['train_use_cuda']
	
	# 载入训练样本和目标数据集
	train_samples_dict = build_train_samples_dict()
	train_targets_arr = build_train_targets_array()
	
	# 构造训练集
	X = np.concatenate([train_samples_dict[col] for col in train_samples_dict.keys()], axis = 2).astype(np.float32)
	X = np.hstack((X, np.zeros([X.shape[0], pred_dim, X.shape[2]]).astype(np.float32)))
	y = train_targets_arr.astype(np.float32)
	
	# shuffle操作
	id_list = np.random.permutation(range(X.shape[0]))
	X, y = X[list(id_list), :, :], y[list(id_list), :, :]
	
	# 划分训练集和验证集
	split_num = int(0.9 * X.shape[0])
	X_train, y_train = X[:split_num, :, :], y[:split_num, :, :]
	X_verify, y_verify = X[split_num:, :, :], y[split_num:, :, :]
	
	train_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
	trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	verify_dataset = Data.TensorDataset(torch.from_numpy(X_verify), torch.from_numpy(y_verify))
	verifyloader = DataLoader(verify_dataset, batch_size = X_verify.shape[0])
	
	if use_cuda:
		torch.cuda.empty_cache()
		trainloader = [(train_x.cuda(), train_y.cuda()) for (train_x, train_y) in trainloader]
		verifyloader = [(verify_x.cuda(), verify_y.cuda()) for (verify_x, verify_y) in verifyloader]
	
	return trainloader, verifyloader, X_train, y_train, X_verify, y_verify


if __name__ == '__main__':
	# 设定参数
	selected_columns = config.conf['model_params']['selected_columns']

	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')

	# 生成样本数据
	samples_df = build_samples_data_frame(data)

	# # 构建样本字典
	# samples_dict = build_train_samples_dict()

	# 构建训练目标数据集
	targets_arr = build_train_targets_array()

