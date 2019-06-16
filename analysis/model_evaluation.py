# -*- coding: utf-8 -*-
"""
Created on 2019/6/8 14:14
@author: luolei

模型评估
"""
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import numpy as np
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.models import EncoderLSTM
from mods.build_samples_and_targets import build_train_samples_dict, build_train_targets_array, build_test_samples_dict, build_test_targets_array
from mods.model_evaluations import smape, mae, rmse, r2
from models.train_lstm import cal_attention_outputs


def load_models():
	"""载入模型"""
	target_column = config.conf['model_params']['target_column']
	
	# 载入模型和参数文件
	with open('../tmp/model_struc_params.pkl', 'r') as f:
		model_struc_params = json.load(f)
	encoder_state_dict = torch.load('../tmp/encoder_state_dict_{}.pth'.format(target_column), map_location = 'cpu')
	
	# 初始化模型
	encoder = EncoderLSTM(input_size = model_struc_params['encoder']['input_size'])
	
	# 载入模型参数
	encoder.load_state_dict(encoder_state_dict, strict = False)
	
	return encoder


if __name__ == '__main__':
	# 设定参数
	pred_dim = config.conf['model_params']['pred_dim']
	use_cuda = config.conf['model_params']['pred_use_cuda']
	target_column = config.conf['model_params']['target_column']

	# 载入训练好的模型 ———————————————————————————————————————————————————————————————————————————————————————————
	encoder = load_models()
	
	# 训练集效果 ————————————————————————————————————————————————————————————————————————————————————————————————
	# 载入训练样本和目标数据集
	train_samples_dict = build_train_samples_dict()
	train_targets_arr = build_train_targets_array()

	# 划分训练集
	X_train = np.concatenate([train_samples_dict[col] for col in train_samples_dict.keys()], axis = 2).astype(np.float32)
	X_train = np.hstack((X_train, np.zeros([X_train.shape[0], pred_dim, X_train.shape[2]]).astype(np.float32)))
	y_train = train_targets_arr.astype(np.float32)

	# 训练效果
	y_train_raw = y_train[:, :, 0]

	X_train = torch.from_numpy(X_train)

	if use_cuda:
		encoder = encoder.cuda()
	
	weighten_targets = cal_attention_outputs(encoder, X_train)
	y_train_model = weighten_targets[:, :, 0]
	y_train_model = y_train_model.detach().cpu().numpy()

	# 还原为真实值
	target_column = config.conf['model_params']['target_column']
	bounds = config.conf['model_params']['variable_bounds'][target_column]
	y_train_raw = y_train_raw * (bounds[1] - bounds[0]) + bounds[0]
	y_train_model = y_train_model * (bounds[1] - bounds[0]) + bounds[0]

	# 模型训练结果评估
	rmse_results, smape_results, mae_results, r2_results = [], [], [], []
	for i in range(y_train_raw.shape[1]):
		rmse_results.append(rmse(y_train_raw[:, i], y_train_model[:, i]))
		smape_results.append(smape(y_train_raw[:, i], y_train_model[:, i]))
		mae_results.append(mae(y_train_raw[:, i], y_train_model[:, i]))
		r2_results.append(r2(y_train_raw[:, i], y_train_model[:, i]))

	print('\n===========TRAINING EFFECTS==============')
	for step in [0, 3, 7, 11, 23, 47, 71]:
		print('{} hr: rmse {:4f}, smape {:4f}, mae {:4f}, r2 {:4f}'.format(
			step, rmse_results[step], smape_results[step], mae_results[step], r2_results[step])
		)
	print('=========================================')

	plt.figure(figsize = [8, 4])
	plt.subplot(1, 2, 1)
	sns.heatmap(y_train_raw)
	plt.subplot(1, 2, 2)
	sns.heatmap(y_train_model)
	
	# 查看训练loss曲线
	with open('../tmp/train_loss.pkl', 'r') as f:
		train_loss_record = json.load(f)
	with open('../tmp/verify_loss.pkl', 'r') as f:
		verify_loss_record = json.load(f)
	plt.figure('loss curve', figsize = [4, 8])
	plt.plot(train_loss_record)
	plt.plot(verify_loss_record, 'r')
	plt.legend(['train set', 'verify set'])
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.tight_layout()

	# 测试集效果 ————————————————————————————————————————————————————————————————————————————————————————————————
	# 载入训练样本和目标数据集
	test_samples_dict = build_test_samples_dict()
	test_targets_arr = build_test_targets_array()
	
	# 划分训练集
	X_test = np.concatenate([test_samples_dict[col] for col in test_samples_dict.keys()], axis = 2).astype(np.float32)
	X_test = np.hstack((X_test, np.zeros([X_test.shape[0], pred_dim, X_test.shape[2]]).astype(np.float32)))
	y_test = test_targets_arr.astype(np.float32)
	
	# 训练效果
	y_test_raw = y_test[:, :, 0]
	
	X_test = torch.from_numpy(X_test)
	
	if use_cuda:
		encoder = encoder.cuda()
		X_test = X_test.cuda()
	
	weighten_targets = cal_attention_outputs(encoder, X_test)
	y_test_model = weighten_targets[:, :, 0]
	y_test_model = y_test_model.detach().cpu().numpy()
	
	# 还原为真实值
	target_column = config.conf['model_params']['target_column']
	bounds = config.conf['model_params']['variable_bounds'][target_column]
	y_test_raw = y_test_raw * (bounds[1] - bounds[0]) + bounds[0]
	y_test_model = y_test_model * (bounds[1] - bounds[0]) + bounds[0]
	
	# 模型训练结果评估
	rmse_results, smape_results, mae_results, r2_results = [], [], [], []
	for i in range(y_test_raw.shape[1]):
		rmse_results.append(rmse(y_test_raw[:, i], y_test_model[:, i]))
		smape_results.append(smape(y_test_raw[:, i], y_test_model[:, i]))
		mae_results.append(mae(y_test_raw[:, i], y_test_model[:, i]))
		r2_results.append(r2(y_test_raw[:, i], y_test_model[:, i]))
	
	print('\n===========TESTING EFFECTS==============')
	for step in [0, 3, 7, 11, 23, 47, 71]:
		print('{} hr: rmse {:4f}, smape {:4f}, mae {:4f}, r2 {:4f}'.format(
			step, rmse_results[step], smape_results[step], mae_results[step], r2_results[step])
		)
	print('=========================================')
	
	plt.figure(figsize = [8, 4])
	plt.subplot(1, 2, 1)
	sns.heatmap(y_test_raw)
	plt.subplot(1, 2, 2)
	sns.heatmap(y_test_model)
	
	# 不同时间预测步上的效果
	plt.figure('pred effects at different steps', figsize = [4, 8])
	steps = [0, 40, 65, 71]
	for step in steps:
		plt.subplot(len(steps), 1, steps.index(step) + 1)
		if steps.index(step) == 0:
			plt.title('fitting effects at different time steps')
		plt.plot(y_test_raw[:, step])
		plt.plot(y_test_model[:, step], 'r')
		plt.ylabel('pm25')
		plt.legend(['step = {}'.format(step)], loc = 'upper right')
		if steps.index(step) == len(steps) - 1:
			plt.xlabel('time step')
		plt.tight_layout()
			
		
