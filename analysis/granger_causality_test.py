# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

格兰杰检验
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../')

from mods.granger_causality import granger_causality
from mods.config_loader import config
from mods.data_filtering import savitzky_golay_filtering


def load_raw_data():
	"""
	载入原数据并返回数据信息
	:return:
		data: pd.DataFrame, 载入的数据表
		describe: pd.DataFrame, 对数据表的描述性统计表
	"""
	data = pd.read_csv('../tmp/PRSA_data.csv')
	describe = data.describe()
	return data, describe


def replace_nans(granger_results):
	"""
	将格兰杰检验计算结果中的nan替换为0
	:param granger_results: list, 原始的格兰杰检验结果
	:return:
		granger_results, np.ndarray, 替换后的格兰杰检验结果
	"""
	granger_results = pd.DataFrame(granger_results)
	granger_results = granger_results.fillna(0)
	granger_results = np.array(granger_results)
	return granger_results


def cal_mean_granger_results(total_results_copy):
	"""
	计算各变量上与pm2.5相关的的平均格兰杰检测值
	:param total_results_copy: list, 各变量总体格兰杰检测结果
	:return:
		total_results: list, 平均的格兰杰检测值
	"""
	total_results = []
	for i in range(len(total_results_copy[0])):
		total_results_i = []
		for j in range(len(total_results_copy)):
			total_results_i.append(total_results_copy[j][i, :])
		total_results_i = np.array(total_results_i)
		total_results.append(np.mean(total_results_i, axis = 0))
	return total_results


def cal_mean_granger_causality(data, start_locs, sample_length, target_column, selected_columns, max_lag = 50):
	"""
	计算平均采样后的格兰杰检验结果
	:param data: pd.DataFrame, 待检验数据表
	:param start_locs: list of integers, 起始位置列表，通过选取不同起始位置提高格兰杰检验的准确度
	:param sample_length: int, 每条序列样本长度
	:param selected_columns: list of strings, 选择的字段列表
	:param max_lag: 待检测最大时间延迟范围
	:return:
		lags: list, 所有的时滞列表
		mean_forward_results: list, 其他字段对目标字段在时滞区间上的影响结果记录
		mean_reverse_results: list, 目标字段对其他字段在时滞区间上的影响结果记录
	"""
	total_forward_results = []
	total_reverse_results = []
	for start_loc in start_locs:
		print('start_loc = %s' % start_loc)
		data_seg = data.iloc[start_loc: start_loc + sample_length]
		lags, forward_results, reverse_results = granger_causality(
			data_seg, target_column, selected_columns, max_lag = max_lag
		)
		total_forward_results.append(forward_results)
		total_reverse_results.append(reverse_results)

	# 将格兰杰检验结果中的nan替换为0
	total_forward_results_copy = []
	total_reverse_results_copy = []
	for i in range(len(total_reverse_results)):
		total_forward_results_copy.append(replace_nans(total_forward_results[i]))
		total_reverse_results_copy.append(replace_nans(total_reverse_results[i]))

	# 计算平均化后的格兰杰检验结果
	mean_forward_results = cal_mean_granger_results(total_forward_results_copy)
	mean_reverse_results = cal_mean_granger_results(total_reverse_results_copy)

	plt.figure('forward causality', figsize = [6, 8])
	for i in range(len(selected_columns)):
		plt.subplot(len(selected_columns), 1, i + 1)
		plt.plot(lags, mean_forward_results[i])
		plt.ylabel(selected_columns[i])
	plt.xlabel('time lag')
	plt.tight_layout()

	plt.figure('reverse causality', figsize = [6, 8])
	for i in range(len(selected_columns)):
		plt.subplot(len(selected_columns), 1, i + 1)
		plt.plot(lags, mean_reverse_results[i], 'r')
		plt.ylabel(selected_columns[i])
	plt.xlabel('time lag')
	plt.tight_layout()

	return lags, mean_forward_results, mean_reverse_results


if __name__ == '__main__':
	# 载入数据
	file_name = '../tmp/total_implemented_normalized_data.csv'
	data = pd.read_csv(file_name)
	data_filtered = savitzky_golay_filtering(data)

	# Granger因果性检测
	start_locs = np.arange(10000, 11200, 50)  # 注意这一步要避开异常值
	sample_length = 2000
	target_column = config.conf['model_params']['target_column']
	selected_columns = config.conf['model_params']['selected_columns'][:6]

	lags, mean_forward_results, mean_reverse_results = cal_mean_granger_causality(
		data_filtered, start_locs, sample_length, target_column, list(set([target_column] + selected_columns)), max_lag = 80)

	time_lags_dict = {}
	for col in selected_columns:
		time_lags_dict[col] = lags[list(mean_forward_results[selected_columns.index(col)]).index(np.max(mean_forward_results[selected_columns.index(col)]))] - 1

