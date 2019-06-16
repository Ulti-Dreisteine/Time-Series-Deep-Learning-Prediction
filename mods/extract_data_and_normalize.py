# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""
import pandas as pd
import numpy as np
import time
import copy
import sys

sys.path.append('../')

from mods.config_loader import config


def _find_closest_element_in_list(lst, x):
	"""
	寻找x在list中的最近邻值
	:param lst: list of numbers, 列表
	:param x: float, 目标值
	:return:
		min_dist: float, 最小值
		min_index: int, 最小值对应index
	"""
	dist = list(np.power(np.array(lst) - x, 2))
	min_dist = min(dist)
	return min_dist, dist.index(min_dist)


def implement_time_records(data, start_time_stamp, end_time_stamp):
	"""
	根据当前时间和样本长度在data中筛选出连续的天气数据记录
	:param save: bool, 保留至本地
	:param data: pd.DataFrame, 待筛选数据
	:param start_time_stamp: int, 起始时间戳
	:param end_time_stamp: int, 结束时间戳
	:return:
		hourly_data: pd.DataFrame, 筛选出的小时级天气数据记录
	"""
	hr = 3600
	exist_hour_data_time_stamps = list(data['time_stamp'])
	total_hour_data_time_stamps = list(np.arange(start_time_stamp, end_time_stamp + hr, hr))

	hourly_data = copy.deepcopy(data)
	for time_stamp in total_hour_data_time_stamps:
		print('time_stamp = {}'.format(time_stamp))
		if time_stamp not in exist_hour_data_time_stamps:
			row2insert = pd.DataFrame(data.loc[_find_closest_element_in_list(exist_hour_data_time_stamps, time_stamp)[1]]).T
			row2insert.iloc[0]['time_stamp'] = time_stamp
			hourly_data = pd.concat([hourly_data, row2insert])

	hourly_data = hourly_data[hourly_data.time_stamp.isin(total_hour_data_time_stamps)].sort_values(by = 'time_stamp', ascending = True)
	hourly_data.reset_index(drop = True, inplace = True)
	hourly_data['time_stamp'] = hourly_data.loc[:, 'time_stamp'].apply(lambda x: int(x))

	if (len(hourly_data) != len(total_hour_data_time_stamps)) | (len(hourly_data.drop_duplicates(['time_stamp'])) != len(total_hour_data_time_stamps)):
		raise ValueError(('筛选所得记录长度与设定的样本长度不符合',))

	if set(list(hourly_data['time_stamp'])) != set(total_hour_data_time_stamps):
		raise ValueError(('筛选所得记录时间不连续',))

	return hourly_data


def normalize(data, columns):
	"""
	对数据指定列进行归一化
	:param data:
	:return:
		data_copy: pd.DataFrame, 归一化后的数据
	"""
	data_copy = data.copy()
	bounds = config.conf['model_params']['variable_bounds']

	for col in columns:
		normalize = lambda x: (x - bounds[col][0]) / (bounds[col][1] - bounds[col][0])
		data_copy.loc[:, col] = data_copy.loc[:, col].apply(normalize)

	return data_copy


def extract_implemented_data(raw_data_file, use_local = True, save = True):
	"""
	提取时间戳连续化填补后的数据
	:param raw_data_file: str, 原始数据名
	:param use_local: bool, 使用本地数据文件
	:param save: bool, 保存到本地
	:return:
		total_implemented_normalized_data: pd.DataFrame, 时间连续性填补和归一化处理后的数据
	"""
	if use_local:
		total_implemented_normalized_data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	else:
		full_data = pd.read_csv('../tmp/{}'.format(raw_data_file))

		# 时间戳
		full_data['time_stamp'] = full_data.loc[:, 'ptime'].apply(lambda x: int(time.mktime(time.strptime(str(int(x)), "%Y%m%d%H"))))

		# 时间戳连续化
		record_start_time = config.conf['record_start_time']
		record_end_time = config.conf['record_end_time']

		start_time_stamp = int(time.mktime(time.strptime(record_start_time, '%Y%m%d%H')))
		end_time_stamp = int(time.mktime(time.strptime(record_end_time, '%Y%m%d%H')))
		implemented_data = implement_time_records(full_data, start_time_stamp, end_time_stamp)

		# 字段值转换
		weather_code_dict = {}
		weathers = list(implemented_data['weather'].drop_duplicates())
		for weather in weathers:
			weather_code_dict[weather] = weathers.index(weather)
		implemented_data['weather'] = implemented_data.loc[:, 'weather'].apply(lambda x: weather_code_dict[x])

		# 归一化
		target_column = config.conf['model_params']['target_column']
		selected_columns = config.conf['model_params']['selected_columns']
		
		print('\n')
		for column in [target_column] + selected_columns:
			print('max {}: {}'.format(column, np.max(implemented_data[column])))
		
		total_implemented_normalized_data = normalize(implemented_data, [target_column] + selected_columns)
		total_implemented_normalized_data = total_implemented_normalized_data[['city', 'ptime', 'time_stamp'] + [target_column] + selected_columns]

	# 异常值替换
	total_implemented_normalized_data.replace(np.nan, 0.0, inplace = True)
	total_implemented_normalized_data.replace(np.inf, 1.0, inplace = True)

	if save:
		total_implemented_normalized_data.to_csv('../tmp/total_implemented_normalized_data.csv', index = False)

	return total_implemented_normalized_data


if __name__ == '__main__':
	total_implemented_normalized_data = extract_implemented_data('taiyuan_cityHour.csv', use_local = False)


