# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

训练模型
"""
import torch
import numpy as np
import json
import sys

sys.path.append('../')

from mods.extract_data_and_normalize import extract_implemented_data
from mods.models import NN, ContinuousEncoder, DiscreteEncoder, initialize_nn_params, initialize_continuous_encoder_params, \
	initialize_discrete_encoder_params
from mods.build_train_and_test_samples import build_train_and_verify_datasets
from mods.config_loader import config
from mods.loss_criterion import criterion
from mods.one_hot_encoder import one_hot_encoding
from mods.data_filtering import savitzky_golay_filtering


def save_models(nn, continuous_encoder, discrete_encoder, train_loss_record, verify_loss_record):
	"""保存模型文件"""
	target_columns = config.conf['model_params']['target_columns']
	
	# 保存模型文件
	torch.save(nn.state_dict(), '../tmp/nn_state_dict_{}.pth'.format(target_columns))
	torch.save(continuous_encoder.state_dict(), '../tmp/continuous_encoder_state_dict_{}.pth'.format(target_columns))
	torch.save(discrete_encoder.state_dict(), '../tmp/discrete_encoder_state_dict_{}.pth'.format(target_columns))
	
	# 保存模型结构参数
	model_struc_params = {
		'nn': {
			'input_size': nn.input_size,
			'hidden_size': nn.hidden_size,
			'output_size': nn.output_size
		},
		'continuous_encoder': {
			'input_size': continuous_encoder.input_size,
			'output_size': continuous_encoder.output_size
		},
		'discrete_encoder': {
			'input_size': discrete_encoder.input_size,
			'output_size': discrete_encoder.output_size
		}
	}
	
	with open('../tmp/model_struc_params.json', 'w') as f:
		json.dump(model_struc_params, f)
		
	# 损失函数记录
	train_loss_list = [float(p.detach().cpu().numpy()) for p in train_loss_record]
	verify_loss_list = [float(p.cpu().numpy()) for p in verify_loss_record]
	
	with open('../tmp/train_loss.json', 'w') as f:
		json.dump(train_loss_list, f)
	with open('../tmp/verify_loss.json', 'w') as f:
		json.dump(verify_loss_list, f)
	

if __name__ == '__main__':
	# 读取原始数据并整理成表 ————————————————————————————————————————————————————————————————————————————————————————————————————————————
	file_name = '../tmp/taiyuan_cityHour.csv'
	total_implemented_normalized_data = extract_implemented_data(file_name, use_local = False, save = True)
	
	# 数据滤波和编码
	data = savitzky_golay_filtering(total_implemented_normalized_data)
	_ = one_hot_encoding(data, save = True)
	
	# 设定参数 ————————————————————————————————————————————————————————————————————————————————————————-——————————————————————————————
	use_cuda = config.conf['model_params']['train_use_cuda']
	batch_size = config.conf['model_params']['batch_size']
	lr = config.conf['model_params']['lr']
	epochs = config.conf['model_params']['epochs']

	# 载入数据集，构建训练和验证集样本 ————————————————————————————————————————————————————————————————————————————————————————————————————
	trainloader, verifyloader, X_train, y_train, X_verify, y_verify, continuous_columns_num = build_train_and_verify_datasets()

	# 构造神经网络模型 —————————————————————————————————————————————————————————————————————————————————————————————————————————————-———
	input_size = continuous_columns_num
	output_size = 20
	continuous_encoder = ContinuousEncoder(input_size, output_size)

	input_size = X_train.shape[1] - continuous_columns_num
	output_size = 20
	discrete_encoder = DiscreteEncoder(input_size, output_size)

	input_size = continuous_encoder.connect_1.out_features + discrete_encoder.connect_0.out_features
	hidden_size = [int(np.floor(input_size) / 2), int(np.floor(y_train.shape[1]))]
	output_size = y_train.shape[1]
	nn = NN(input_size, hidden_size, output_size)

	# 初始化模型参数
	continuous_encoder = initialize_continuous_encoder_params(continuous_encoder)
	discrete_encoder = initialize_discrete_encoder_params(discrete_encoder)
	nn = initialize_nn_params(nn)

	if use_cuda:
		torch.cuda.empty_cache()
		trainloader = [(train_x.cuda(), train_y.cuda()) for (train_x, train_y) in trainloader]
		verifyloader = [(verify_x.cuda(), verify_y.cuda()) for (verify_x, verify_y) in verifyloader]
		continuous_encoder = continuous_encoder.cuda()
		discrete_encoder = discrete_encoder.cuda()
		nn = nn.cuda()

	# 指定优化器 —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	optimizer = torch.optim.Adam(
		[
			{'params': nn.parameters()},
			{'params': continuous_encoder.parameters()},
			{'params': discrete_encoder.parameters()}
		],
		lr = lr
	)

	# 模型训练和保存 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	train_loss_record, verify_loss_record = [], []
	early_stop_steps = 200
	sum = torch.tensor(early_stop_steps - 50).int()
	stop_criterion = torch.tensor(1).byte()

	if use_cuda:
		sum = sum.cuda()
		stop_criterion = stop_criterion.cuda()

	for epoch in range(epochs):
		# 训练集
		for train_x, train_y in trainloader:
			con_x, dis_x = train_x[:, :continuous_columns_num], train_x[:, continuous_columns_num:]
			con_encoded_x = continuous_encoder(con_x)
			dis_encoded_x = discrete_encoder(dis_x)
			encoded_x = torch.cat((con_encoded_x, dis_encoded_x), dim = 1)
			y_train_p = nn(encoded_x)
			y_train_t = train_y
			train_loss_fn = criterion(y_train_p, y_train_t)

			optimizer.zero_grad()
			train_loss_fn.backward()
			optimizer.step()

		train_loss_record.append(train_loss_fn)

		with torch.no_grad():
			for verify_x, verify_y in verifyloader:
				con_x, dis_x = verify_x[:, :continuous_columns_num], verify_x[:, continuous_columns_num:]
				con_encoded_x = continuous_encoder(con_x)
				dis_encoded_x = discrete_encoder(dis_x)
				encoded_x = torch.cat((con_encoded_x, dis_encoded_x), dim = 1)
				y_verify_p = nn(encoded_x)
				y_verify_t = verify_y
				verify_loss_fn = criterion(y_verify_p, y_verify_t)
			verify_loss_record.append(verify_loss_fn)

		if epoch % 100 == 0:
			print(epoch, train_loss_fn, verify_loss_fn)

		# 保存模型
		if epoch % 1000 == 0:
			save_models(nn, continuous_encoder, discrete_encoder, train_loss_record, verify_loss_record)

	save_models(nn, continuous_encoder, discrete_encoder, train_loss_record, verify_loss_record)
	
	
	
	
	
	
	
	
	
	


