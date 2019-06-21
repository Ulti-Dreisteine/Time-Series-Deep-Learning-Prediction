# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

特征重要性
"""
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import sys

sys.path.append('../')

from mods.build_train_and_test_samples import build_samples_data_frame, build_test_samples_and_targets
from mods.models import load_models


if __name__ == '__main__':
	# 载入训练好的模型
	[nn, continuous_encoder, discrete_encoder] = load_models()

	# 构造测试数据
	X_test, y_test, continuous_columns_num = build_test_samples_and_targets()

	# 摄动求解偏导数
	eps = 1e-5
	total_partial_derives = []
	
	for i in range(X_test.shape[0]):		# 逐样本计算
		print('processing row {}'.format(i))
		x_test = X_test[i, :].reshape(1, -1)
		var_x_test = Variable(torch.from_numpy(x_test.astype(np.float32)))
		
		con_x, dis_x = var_x_test[:, :continuous_columns_num], var_x_test[:, continuous_columns_num:]
		con_encoded_x = continuous_encoder(con_x)
		dis_encoded_x = discrete_encoder(dis_x)
		encoded_x = torch.cat((con_encoded_x, dis_encoded_x), dim = 1)
		y_test_model = nn(encoded_x).detach().cpu().numpy().flatten()
		
		partial_derives = []
		for j in range(X_test.shape[1]):
			x_test_pert = copy.deepcopy(x_test)
			x_test_pert[0, j] += eps
			var_x_test_pert = Variable(torch.from_numpy(x_test_pert.astype(np.float32)))
			
			con_x, dis_x = var_x_test_pert[:, :continuous_columns_num], var_x_test_pert[:, continuous_columns_num:]
			con_encoded_x = continuous_encoder(con_x)
			dis_encoded_x = discrete_encoder(dis_x)
			encoded_x = torch.cat((con_encoded_x, dis_encoded_x), dim = 1)
			y_test_model_pert = nn(encoded_x).detach().cpu().numpy().flatten()
			
			partial_derives.append(np.abs(np.linalg.norm(y_test_model_pert - y_test_model, 2) / eps))
		total_partial_derives.append(partial_derives)
	
	total_partial_derives = np.array(total_partial_derives)
	mean_partial_derives = np.mean(total_partial_derives, axis = 0).reshape(-1, 1)

	# 计算特征重要性
	data = pd.read_csv('../tmp/total_encoded_data.csv')
	_, _, columns = build_samples_data_frame(data)
	columns = np.array(columns[1:]).reshape(-1, 1)
	
	partial_derivatives_table = np.hstack((columns, mean_partial_derives))
	feature_importances_dict = dict()
	for i in range(partial_derivatives_table.shape[0]):
		column = partial_derivatives_table[i, 0].split('_')[0]
		if column not in feature_importances_dict.keys():
			feature_importances_dict[column] = float(partial_derivatives_table[i, 1])
		else:
			feature_importances_dict[column] += float(partial_derivatives_table[i, 1])
	
	# 数据可视化
	plt.figure(figsize = [12, 5])
	plt.title('feature importance table')
	plt.bar(feature_importances_dict.keys(), feature_importances_dict.values(), label="rainfall")
	plt.xlabel('feature')
	plt.ylabel('importance score')
	plt.tight_layout()



