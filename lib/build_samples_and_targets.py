# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

构建样本和目标数据集
"""
import pandas as pd
import sys

sys.path.append('../')

from mods.build_train_and_test_samples_for_nn import build_samples_data_frame, build_targets_data_frame, build_train_samples_and_targets, \
	build_test_samples_and_targets


if __name__ == '__main__':
	# 载入数据
	data = pd.read_csv('../tmp/total_encoded_data.csv')
	
	# 构建样本和目标值数据
	samples_df, continuous_columns_num, _ = build_samples_data_frame(data)
	targets_df = build_targets_data_frame(data)
	
	# 训练集样本和目标数据集
	X_train, y_train, continuous_columns_num = build_train_samples_and_targets()
	
	# 测试集样本和目标数据集
	X_test, y_test, continuous_columns_num = build_test_samples_and_targets()

