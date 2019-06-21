# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

构建离散样本
"""
import pandas as pd
import sys

sys.path.append('../')

from mods.build_train_and_test_samples import build_samples_data_frame, build_targets_data_frame, build_train_samples_and_targets, \
	build_test_samples_and_targets


if __name__ == '__main__':
	
	# 载入数据
	data = pd.read_csv('../tmp/total_encoded_data.csv')
	
	# 构建总体样本数据集
	samples_df = build_samples_data_frame(data)
	
	# 构建总体目标数据集
	targets_df = build_targets_data_frame(data)
	
	# 训练集
	X_train, y_train = build_train_samples_and_targets()
	
	# 测试集
	X_test, y_test = build_test_samples_and_targets()
	


