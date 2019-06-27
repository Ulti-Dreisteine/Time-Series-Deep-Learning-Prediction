# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

计算偏导数矩阵
"""
import numpy as np


def partial_derivative_matrix(X, func, *args, eps = 1e-5):
	"""摄动求解偏导数矩阵"""
	y = func(X, *args)
	
	p_deriv_matrix = np.zeros([X.shape[1], y.shape[1]])
	for i in range(X.shape[1]):
		print('total rows: {}, processing {}'.format(X.shape[1], i))
		X_per = X.copy()
		X_per[:, i] += X[:, i] * eps
		y_per = func(X_per, *args)
		p_deriv_matrix[i, :] = np.dot(np.linalg.pinv(X_per - X), (y_per - y))[i, :]
	
	return p_deriv_matrix
			
	


