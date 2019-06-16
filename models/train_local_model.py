# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

训练模型
"""
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time
import seaborn as sns
import numpy as np
import pandas as pd
import sys

sys.path.append('../')

from mods.nn_model import NN, initialize_model_params
from mods.build_samples import build_train_samples_and_targets, build_targets_data_frame
from mods.config_loader import config
from mods.loss_criterion import criterion
from mods.model_evaluations import smape, mae, rmse, r2


def func(x):
	# return pow(x, 0.5)
	return 1.0


if __name__ == '__main__':
	# 载入训练数据集
	use_cuda = config.conf['model_params']['train_use_cuda']
	X_train, y_train = build_train_samples_and_targets()
	
	# 真实目标值
	exist_record_time = config.conf['exist_record_time']
	samples_len = config.conf['model_params']['samples_len']
	hr = 3600
	exist_time_stamp = int(time.mktime(time.strptime(str(exist_record_time), "%Y%m%d%H")))
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	y_train_raw = build_targets_data_frame(data)
	y_train_raw = y_train_raw[(y_train_raw.time_stamp >= exist_time_stamp - samples_len * hr + hr) & (y_train_raw.time_stamp < exist_time_stamp + hr)]
	y_train_raw = np.array(y_train_raw.iloc[:, 1:])

	# 构造神经网络模型
	input_size = X_train.shape[1]
	hidden_size = [int(np.floor(X_train.shape[1] / 2)), int(np.floor(X_train.shape[1] / 4))]
	output_size = y_train.shape[1]
	nn = NN(input_size, hidden_size, output_size)

	# 初始化模型参数
	nn = initialize_model_params(nn, input_size, hidden_size, output_size)

	if use_cuda:
		nn = nn.cuda()

	# 模型训练参数
	lr = config.conf['model_params']['lr']
	epochs = config.conf['model_params']['epochs']
	batch_size = config.conf['model_params']['batch_size']['local']

	# 指定优化器
	optimizer = torch.optim.Adam(nn.parameters(), lr = lr)

	# 划分数据集
	X_train_model, y_train_model = torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))
	torch_dataset = Data.TensorDataset(X_train_model, y_train_model)
	loader = Data.DataLoader(dataset = torch_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
	train_datasets = [p[1] for p in enumerate(loader)]
	for i in range(len(train_datasets)):
		train_datasets[i][0] = Variable(train_datasets[i][0])
		train_datasets[i][1] = Variable(train_datasets[i][1])
		if use_cuda:
			train_datasets[i][0] = train_datasets[i][0].cuda()
			train_datasets[i][1] = train_datasets[i][1].cuda()
	
	# 模型训练
	train_loss_record = []  # 训练集loss
	verify_loss_record = []  # 验证集loss

	weights_len = y_train_raw.shape[1]
	weights = [func(1)]
	for i in range(1, weights_len):
		weights.append(func(i + 1))

	weights = np.array(weights).astype(np.float32).reshape(1, -1)
	weights = torch.from_numpy(weights)
	if use_cuda:
		weights = weights.cuda()

	for epoch in range(epochs):
		for i in range(len(train_datasets)):
			var_x, var_y = train_datasets[i][0], train_datasets[i][1]
			if i != len(train_datasets) - 1:  # 训练集
				# 前向传播
				y_train_p = nn(var_x)
				y_train_t = var_y
				train_loss_fn = criterion(y_train_p, y_train_t, weights)

				# 反向传播
				optimizer.zero_grad()  # 梯度清零
				train_loss_fn.backward()
				optimizer.step()
				if i == 0:
					train_loss_epoch = train_loss_fn.reshape(1, 1)
				else:
					train_loss_epoch = torch.cat((train_loss_epoch, train_loss_fn.reshape(1, 1)))
			elif i == len(train_datasets) - 1:  # 验证集
				y_verify_p = nn(var_x)
				y_verify_t = var_y
				verify_loss_fn = criterion(y_verify_p, y_verify_t, weights)
				verify_loss_epoch = verify_loss_fn

		if (epoch + 1) % 100 == 0:  # 间隔打印训练结果
			train_loss_epoch = torch.mean(train_loss_epoch)
			verify_loss_epoch = verify_loss_epoch
			train_loss_record.append([epoch + 1, train_loss_epoch.detach()])
			verify_loss_record.append([epoch + 1, verify_loss_epoch.detach()])
			print('epoch: {}, train_loss: {:6f}, verify_loss: {:6f}'.format(epoch + 1, train_loss_epoch, verify_loss_epoch))

		# 早停
		if epoch > 100 * 10:
			last_verify_losses = [p[1].cpu().numpy() for p in verify_loss_record[-10:]]
			if np.max(last_verify_losses) - np.min(last_verify_losses) <= 1e-4:
				break

	if use_cuda:
		torch.save(nn.state_dict(), '../tmp/model_state_dict_{}.pth'.format(config.conf['model_params']['target_column']))
	else:
		torch.save(nn, '../tmp/model_{}.pkl'.format(config.conf['model_params']['target_column']))

	# 训练集上预测值
	X_train = torch.from_numpy(X_train.astype(np.float32))
	var_x_train = Variable(X_train)
	if use_cuda:
		var_x_train = var_x_train.cuda()
	y_train_model = nn(var_x_train).detach().cpu().numpy()

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

	# 模型效果可视化
	plt.figure(figsize = [4, 3])
	plt.title('loss curve')
	plt.plot([p[0] for p in train_loss_record[10:]], [p[1].cpu().numpy() for p in train_loss_record[10:]])
	plt.plot([p[0] for p in verify_loss_record[10:]], [p[1].cpu().numpy() for p in verify_loss_record[10:]], 'r--')
	plt.legend(['train set loss', 'verify set loss'])
	plt.xlabel('epoch')
	plt.ylabel('loss')

	steps = [0, 7, 11, 23, 47]
	plt.figure(figsize = [5, 10])
	for step in steps:
		plt.subplot(len(steps), 1, steps.index(step) + 1)
		if steps.index(step) == 0:
			plt.title('fitting results at different time steps')
		plt.plot(y_train_raw[:, step])
		plt.plot(y_train_model[:, step], 'r--')
		plt.ylabel(target_column)
		plt.legend(['step = {}'.format(step)], loc = 'upper right')
		if steps.index(step) == len(steps) - 1:
			plt.xlabel('time step')
			plt.tight_layout()

	eval_methods = ['mae', 'smape', 'rmse', 'r2']
	plt.figure(figsize = [5, 10])
	for method in eval_methods:
		plt.subplot(len(eval_methods), 1, eval_methods.index(method) + 1)
		if eval_methods.index(method) == 0:
			plt.title('model evaluations with different methods')
		plt.plot(eval(method + '_results'))
		plt.ylabel(method)
		if eval_methods.index(method) == len(eval_methods) - 1:
			plt.ylim([-0.1, 1.0])
			plt.xlabel('time step')
			plt.tight_layout()

	plt.figure(figsize = [12, 6])
	plt.subplot(1, 2, 1)
	sns.heatmap(y_train_raw, cmap = 'Blues', vmin = 0, vmax = 500)
	plt.subplot(1, 2, 2)
	sns.heatmap(y_train_model, cmap = 'Blues', vmin = 0, vmax = 500)
	plt.tight_layout()

	plt.figure(figsize = [5, 10])
	plt.subplot(2, 1, 1)
	sns.heatmap(y_train_raw[0, :].reshape(1, -1), cmap = 'Blues', vmin = 0, vmax = 100)
	plt.subplot(2, 1, 2)
	sns.heatmap(y_train_model[0, :].reshape(1, -1), cmap = 'Blues', vmin = 0, vmax = 100)

	plt.figure(figsize = [12, 6])
	loc = 6000
	plt.plot(y_train_raw[loc, :])
	plt.plot(y_train_model[loc, :])
	
	
	
	
	
	
	
	
	
	


