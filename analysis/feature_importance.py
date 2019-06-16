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
import copy
import seaborn as sns
import sys

sys.path.append('../')

# from trash.build_samples import build_test_samples_and_targets
# from trash.pred_local_model import load_model
# from mods.config_loader import config
#
#
# if __name__ == '__main__':
# 	# 载入训练好的模型
# 	use_cuda = config.conf['model_params']['use_cuda']
# 	nn = load_model()
# 	if use_cuda:
# 		nn = nn.cuda()
#
# 	# 构造测试数据
# 	X_test, y_test = build_test_samples_and_targets()
#
# 	# 摄动求解偏导数
# 	eps = 1e-5
# 	total_partial_derives = []
# 	for i in range(X_test.shape[0]):
# 		print('processing row {}'.format(i))
# 		x_test = X_test[i, :]
# 		var_x_test = Variable(torch.from_numpy(x_test.astype(np.float32)))
# 		if use_cuda:
# 			var_x_test = var_x_test.cuda()
# 		y_test_model = nn(var_x_test).detach().cpu().numpy().flatten()[0]
# 		partial_derives = []
# 		for j in range(X_test.shape[1]):
# 			x_test_pert = copy.deepcopy(x_test)
# 			# x_test_pert[j] = (1 + eps) * x_test_pert[j]
# 			x_test_pert[j] += eps
# 			var_x_test_pert = Variable(torch.from_numpy(x_test_pert.astype(np.float32)))
# 			if use_cuda:
# 				var_x_test_pert = var_x_test_pert.cuda()
# 			y_test_model_pert = nn(var_x_test_pert).detach().cpu().numpy().flatten()[0]
# 			# partial_derives.append(np.abs((y_test_model_pert - y_test_model) / (eps * x_test[j])))
# 			partial_derives.append(np.abs((y_test_model_pert - y_test_model) / eps))
# 		total_partial_derives.append(partial_derives)
#
# 	total_partial_derives = np.array(total_partial_derives)
# 	mean_partial_derives = np.mean(total_partial_derives, axis = 0)
#
# 	# 计算特征重要性
# 	selected_columns = config.conf['model_params']['selected_columns']
# 	embed_lags = config.conf['model_params']['embed_lags']
# 	acf_lags = config.conf['model_params']['acf_lags']
# 	embed_dims = dict()
# 	for col in selected_columns:
# 		embed_dims[col] = int(np.floor(acf_lags[col] / embed_lags[col]))
#
# 	feature_importances = dict()
# 	selected_columns = config.conf['model_params']['selected_columns']
# 	for col in selected_columns:
# 		feature_importances[col] = mean_partial_derives[:embed_dims[col]]
# 		mean_partial_derives = mean_partial_derives[embed_dims[col]:]
#
# 	max_feature_dim = np.max([len(p) for p in feature_importances.values()])
# 	feature_importances_implemented = []
# 	for col in selected_columns:
# 		if len(feature_importances[col]) < max_feature_dim:
# 			feature_importances[col] = np.hstack((feature_importances[col], np.zeros(max_feature_dim - len(feature_importances[col]))))
# 		feature_importances_implemented.append(feature_importances[col])
# 	feature_importances_implemented = np.array(feature_importances_implemented)
#
# 	# 可视化
# 	plt.figure(figsize = [5, 12])
# 	plt.title('feature importance')
# 	sns.heatmap(feature_importances_implemented, cmap = 'Blues', vmin = np.min(feature_importances_implemented), vmax = 0.2)
# 	selected_columns = config.conf['model_params']['selected_columns']
# 	plt.yticks(np.arange(len(selected_columns)) + 0.5, selected_columns)
# 	plt.xlabel('embed num')




