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

from mods.config_loader import config
from mods.build_train_and_test_samples import build_test_samples_and_targets, build_samples_data_frame
from mods.models import load_models
from mods.partial_derivative_matrix import partial_derivative_matrix
from mods.project_graph import callgraph


def model_prediction(X, continuous_columns_num):
	"""模型预测"""
	[nn, continuous_encoder, discrete_encoder] = load_models()
	var_x = Variable(torch.from_numpy(X.astype(np.float32)))
	con_x, dis_x = var_x[:, :continuous_columns_num], var_x[:, continuous_columns_num:]
	con_encoded_x = continuous_encoder(con_x)
	dis_encoded_x = discrete_encoder(dis_x)
	encoded_x = torch.cat((con_encoded_x, dis_encoded_x), dim = 1)
	y = nn(encoded_x).detach().cpu().numpy()
	return y


@callgraph
def feature_importance_results():
	# 设定计算参数
	pred_dim = config.conf['model_params']['pred_dim']
	target_columns = config.conf['model_params']['target_columns']
	
	# 计算特征重要性
	data = pd.read_csv('../tmp/total_encoded_data.csv')
	_, _, columns = build_samples_data_frame(data)
	columns = np.array(columns[1:]).reshape(-1, 1)
	
	# 构造测试数据
	X, _, continuous_columns_num = build_test_samples_and_targets()
	
	# 计算偏导数矩阵
	p_deriv_matrix = partial_derivative_matrix(X, model_prediction, continuous_columns_num, eps = 1e-6)
	
	# 对各个污染物进行区分
	p_deriv_matrix = np.abs(p_deriv_matrix)
	
	plt.figure(figsize = [6, 12])
	plt.title('feature importance table')
	p_deriv_dict = {}
	for i in range(len(target_columns)):
		p_deriv_dict[target_columns[i]] = p_deriv_matrix[:, i * pred_dim: (i + 1) * pred_dim]
		partial_derivatives_table = np.hstack((columns, p_deriv_dict[target_columns[i]]))
		feature_importances_dict = dict()
		for j in range(partial_derivatives_table.shape[0]):
			column = partial_derivatives_table[j, 0].split('_')[0]
			if column not in feature_importances_dict.keys():
				feature_importances_dict[column] = float(partial_derivatives_table[j, 1])
			else:
				feature_importances_dict[column] += float(partial_derivatives_table[j, 1])
		
		# 数据可视化
		plt.subplot(len(target_columns), 1, i + 1)
		plt.bar(feature_importances_dict.keys(), feature_importances_dict.values(), label = "rainfall")
		plt.xticks(fontsize = 6)
		plt.yticks(fontsize = 6)
		plt.ylabel('importance score', fontsize = 6)
		plt.ylim([0, 1])
		plt.legend([target_columns[i]], fontsize = 6)
		plt.tight_layout()
	
	
if __name__ == '__main__':
	feature_importance_results()
	
	# # 设定计算参数
	# pred_dim = config.conf['model_params']['pred_dim']
	# target_columns = config.conf['model_params']['target_columns']
	#
	# # 计算特征重要性
	# data = pd.read_csv('../tmp/total_encoded_data.csv')
	# _, _, columns = build_samples_data_frame(data)
	# columns = np.array(columns[1:]).reshape(-1, 1)
	#
	# # 构造测试数据
	# X, _, continuous_columns_num = build_test_samples_and_targets()
	#
	# # 计算偏导数矩阵
	# p_deriv_matrix = partial_derivative_matrix(X, model_prediction, continuous_columns_num, eps = 1e-6)
	#
	# # 对各个污染物进行区分
	# p_deriv_matrix = np.abs(p_deriv_matrix)
	#
	# plt.figure(figsize = [6, 12])
	# plt.title('feature importance table')
	# p_deriv_dict = {}
	# for i in range(len(target_columns)):
	# 	p_deriv_dict[target_columns[i]] = p_deriv_matrix[:, i * pred_dim: (i + 1) * pred_dim]
	# 	partial_derivatives_table = np.hstack((columns, p_deriv_dict[target_columns[i]]))
	# 	feature_importances_dict = dict()
	# 	for j in range(partial_derivatives_table.shape[0]):
	# 		column = partial_derivatives_table[j, 0].split('_')[0]
	# 		if column not in feature_importances_dict.keys():
	# 			feature_importances_dict[column] = float(partial_derivatives_table[j, 1])
	# 		else:
	# 			feature_importances_dict[column] += float(partial_derivatives_table[j, 1])
	#
	# 	# 数据可视化
	# 	plt.subplot(len(target_columns), 1, i + 1)
	# 	plt.bar(feature_importances_dict.keys(), feature_importances_dict.values(), label = "rainfall")
	# 	plt.xticks(fontsize = 6)
	# 	plt.yticks(fontsize = 6)
	# 	plt.ylabel('importance score', fontsize = 6)
	# 	plt.ylim([0, 1])
	# 	plt.legend([target_columns[i]], fontsize = 6)
	# 	plt.tight_layout()



