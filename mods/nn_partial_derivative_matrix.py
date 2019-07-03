# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

计算偏导数矩阵
"""
import numpy as np


def func(x):
	"""输入为一维向量，输出也为一维向量"""
	return np.array([np.sum(x), np.linalg.norm(x, 3)])


def single_sample_partial_derives(x, func, eps = 1e-2):
	"""
	单个样本的偏导矩阵
	:return:
	"""
	x = x.flatten()
	y = func(x)
	
	p_deriv_matrix = np.zeros([len(x), len(y)])
	for i in range(len(x)):
		x_per = x.copy()
		x_per[i] += x[i] * eps
		y_per = func(x_per)
		p_deriv_matrix[i, :] = (y_per - y) / (x[i] * eps)
	
	return p_deriv_matrix


def nn_partial_derivative_matrix(X, func, *args, eps = 1e-5):
	"""计算偏导数矩阵"""
	# 摄动求解
	y = func(X, *args)
	
	p_deriv_matrix = np.zeros([X.shape[1], y.shape[1]])
	for i in range(X.shape[1]):
		print('total rows: {}, processing {}'.format(X.shape[1], i))
		X_per = X.copy()
		X_per[:, i] += X[:, i] * eps
		y_per = func(X_per, *args)
		p_deriv_matrix[i, :] = np.dot(np.linalg.pinv(X_per - X), (y_per - y))[i, :]
	
	return p_deriv_matrix
	

if __name__ == '__main__':
	# 单样本计算
	x = np.array([100, 200, 300])
	f = func
	
	p_deriv_matrix = single_sample_partial_derives(x, f)
	
	# 多样本计算
	from mods.build_train_and_test_samples_for_nn import build_test_samples_and_targets
	from analysis.nn_feature_importance import model_prediction
	
	X, _, continuous_columns_num = build_test_samples_and_targets()
	
	# 计算偏导数矩阵
	p_deriv_matrix = nn_partial_derivative_matrix(X, model_prediction, continuous_columns_num)
			
	


