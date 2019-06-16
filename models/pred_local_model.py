# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

模型预测
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
from torch.autograd import Variable
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.model_evaluations import rmse, smape, mae, r2
from mods.build_samples import build_test_samples_and_targets, build_targets_data_frame
from mods.nn_model import load_model


if __name__ == '__main__':
	# 载入训练好的模型
	use_cuda = config.conf['model_params']['pred_use_cuda']
	nn = load_model()
	if use_cuda:
		nn = nn.cuda()

	# 构造测试数据
	X_test, y_test = build_test_samples_and_targets()
	
	# 真实目标值
	exist_record_time = config.conf['exist_record_time']
	pred_horizon_len = config.conf['model_params']['pred_horizon_len']
	hr = 3600
	exist_time_stamp = int(time.mktime(time.strptime(str(exist_record_time), "%Y%m%d%H")))
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	y_test_raw = build_targets_data_frame(data)
	y_test_raw = y_test_raw[(y_test_raw.time_stamp >= exist_time_stamp + hr) & (y_test_raw.time_stamp < exist_time_stamp + pred_horizon_len * hr + hr)]
	y_test_raw = np.array(y_test_raw.iloc[:, 1:])
	
	# 模型预测
	var_x_test = Variable(torch.from_numpy(X_test.astype(np.float32)))
	if use_cuda:
		var_x_test = var_x_test.cuda()
	y_test_model = nn(var_x_test).detach().cpu().numpy()

	# 还原为真实值
	target_column = config.conf['model_params']['target_column']
	bounds = config.conf['model_params']['variable_bounds'][target_column]
	y_test_raw = y_test_raw * (bounds[1] - bounds[0]) + bounds[0]
	y_test_model = y_test_model * (bounds[1] - bounds[0]) + bounds[0]

	# 模型结果评估
	steps = [0, 7, 11, 23]
	rmse_results, smape_results, mae_results, r2_results = [], [], [], []
	for i in range(y_test_raw.shape[1]):
		rmse_results.append(rmse(y_test_raw[:, i], y_test_model[:, i]))
		smape_results.append(smape(y_test_raw[:, i], y_test_model[:, i]))
		mae_results.append(mae(y_test_raw[:, i], y_test_model[:, i]))
		r2_results.append(r2(y_test_raw[:, i], y_test_model[:, i]))
	
	print('\n========== PREDICTION EFFECTS ===========')
	for step in [0, 3, 7, 11, 23, 47, 71]:
		print('{} hr: rmse {:4f}, smape {:4f}, mae {:4f}, r2 {:4f}'.format(
			step, rmse_results[step], smape_results[step], mae_results[step], r2_results[step])
		)
	print('=========================================')

	plt.figure(figsize = [5, 10])
	for step in steps:
		plt.subplot(len(steps), 1, steps.index(step) + 1)
		if steps.index(step) == 0:
			plt.title('fitting results at different pred steps')
		plt.plot(y_test_raw[:, step])
		plt.plot(y_test_model[:, step], 'r')
		plt.ylabel(target_column)
		plt.legend(['step = {}'.format(step + 1)], loc = 'upper right')
		if steps.index(step) == len(steps) - 1:
			plt.xlabel('time step')
			plt.tight_layout()
	
	plt.figure(figsize = [8, 10])
	plt.suptitle('comparison of true and pred values at different predicting time steps')
	for step in steps:
		plt.subplot(2, len(steps) / 2, steps.index(step) + 1)
		plt.scatter(y_test_raw[:, step], y_test_model[:, step], s = 1)
		plt.plot([0, 300], [0, 300], 'k--')
		plt.xlim([0, 300])
		plt.ylim([0, 300])
		plt.xlabel('true value')
		plt.ylabel('pred value')
		plt.legend(['step = {}, r2_score: {:.2f}'.format(step, r2_results[step])], loc = 'upper right')
		if steps.index(step) == len(steps) - 1:
			plt.xlabel('time step')
			plt.tight_layout()
	
	eval_methods = ['mae', 'smape', 'rmse', 'r2']
	plt.figure(figsize = [5, 6])
	for method in eval_methods:
		plt.subplot(len(eval_methods), 1, eval_methods.index(method) + 1)
		if eval_methods.index(method) == 0:
			plt.title('model evaluations with different methods')
		plt.plot(eval(method + '_results'))
		plt.xlim([0, 72])
		plt.ylabel(method)
		if eval_methods.index(method) == len(eval_methods) - 1:
			plt.ylim([-0.2, 1.0])
			plt.xlabel('time step')
			plt.tight_layout()
	
	plt.figure(figsize = [12, 6])
	plt.subplot(1, 2, 1)
	sns.heatmap(y_test_raw, cmap = 'Blues')
	plt.subplot(1, 2, 2)
	sns.heatmap(y_test_model, cmap = 'Blues')
	plt.tight_layout()






