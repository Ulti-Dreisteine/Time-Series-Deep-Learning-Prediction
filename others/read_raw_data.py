# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

查看数据
"""
import sys
from pypinyin import lazy_pinyin
import numpy as np
import pandas as pd
import bson
import os

sys.path.append('../')


def extract_city_data_from_raw(file_name, city_name):
	"""
	提取某城市的数据
	:param file_name: str, 文件名
	:param city_name: str, 城市中文名
	:return:
	"""
	part_num = 0
	data = []
	with open('./env_data/{}'.format(file_name), 'rb') as f:
		bson_data = bson.decode_file_iter(f)

		record_num = 0
		for item in enumerate(bson_data):
			data.append(item[1])
			record_num += 1

			if record_num % 10000 == 0:
				print('record_num = {}'.format(record_num))

			if record_num % 1e6 == 0:
				data = pd.DataFrame(data)
				data = data[data.city == city_name].drop_duplicates('ptime').sort_values(by = 'ptime', ascending = True)
				data.to_csv('./tmp/{}_{}.csv'.format(file_name[:-5], part_num), index = False)
				data = []
				part_num += 1


def merge_city_data(use_local = True, **kwargs):
	"""
	将某城市数据融合起来
	:param use_local: bool, 使用本地提取的数据
	:return:
		data, pd.DataFrame, 该城市的数据整合列表
	"""
	if not use_local:
		extract_city_data_from_raw(kwargs['file_name'], kwargs['city_name'])

	path = os.getcwd() + '/tmp/'
	file_names = [p for p in os.listdir(path) if file_name[:-5] in p]
	for name in file_names:
		if file_names.index(name) == 0:
			data = pd.read_csv(path + name)
		else:
			file_data = pd.read_csv(path + name)
			data = pd.concat([data, file_data], axis = 0, sort = False)
	data = data.drop_duplicates('ptime').sort_values(by = 'ptime', ascending = True).reset_index(drop = True)
	data = data[np.isnan(data.temp) == False]
	return data


if __name__ == '__main__':
	# 载入数据
	file_name = 'cityHour.bson'
	city_name = '北京'
	data = merge_city_data(use_local = False, file_name = file_name, city_name = city_name)

	description = data.describe()

	# 保存数据
	city = ''
	for p in lazy_pinyin(city_name):
		city += p
	data.to_csv('{}_{}.csv'.format(city, file_name[:-5]), index = False)


