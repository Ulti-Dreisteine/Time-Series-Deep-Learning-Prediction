# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

训练lstm模型
"""
import numpy as np
import torch
import json
import torch.nn as nn
import torch.nn.functional as f
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.build_samples_and_targets import build_train_and_verify_datasets
from mods.models import EncoderLSTM
from mods.loss_criterion import criterion


def cal_attention_outputs(encoder, train_x):
	"""计算lstm加入attention后的输出"""
	pred_dim = config.conf['model_params']['pred_dim']
	
	encoder_out = encoder(train_x)
	encoded_samples, encoded_targets = encoder_out[:, :-pred_dim, :], encoder_out[:, -pred_dim:, :]
	
	# 计算隐含层对输出层的点积
	dot_products = torch.bmm(encoded_targets, encoded_samples.permute(0, 2, 1))
	
	# 获得了softmax归一化后所有输入对各个输出的权重
	attention_weights = f.softmax(dot_products, dim = 2)
	
	# 获得经过attention权重加权后的输出
	weighten_targets = torch.bmm(attention_weights, encoded_samples)
	
	return weighten_targets


def save_models_and_records(train_loss_record, verify_loss_record, encoder):
	"""保存模型和相应记录"""
	# 参数
	target_column = config.conf['model_params']['target_column']
	
	# 损失函数记录
	train_loss_list = [float(p.detach().cpu().numpy()) for p in train_loss_record]
	verify_loss_list = [float(p.cpu().numpy()) for p in verify_loss_record]
	
	with open('../tmp/train_loss.pkl', 'w') as f:
		json.dump(train_loss_list, f)
	with open('../tmp/verify_loss.pkl', 'w') as f:
		json.dump(verify_loss_list, f)
	
	# 保存模型文件
	torch.save(encoder.state_dict(), '../tmp/encoder_state_dict_{}.pth'.format(target_column))
	
	# 保存模型结构参数
	model_struc_params = {'encoder': {'input_size': encoder.input_size}}
	with open('../tmp/model_struc_params.pkl', 'w') as f:
		json.dump(model_struc_params, f)
	

if __name__ == '__main__':
	# 设定参数
	target_column = config.conf['model_params']['target_column']
	selected_columns = config.conf['model_params']['selected_columns']
	pred_dim = config.conf['model_params']['pred_dim']
	use_cuda = config.conf['model_params']['train_use_cuda']
	lr = config.conf['model_params']['lr']
	epochs = config.conf['model_params']['epochs']
	batch_size = config.conf['model_params']['batch_size']
	
	# 构建训练和验证数据集 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————
	trainloader, verifyloader, X_train, y_train, X_verify, y_verify = build_train_and_verify_datasets()
	
	# 构建模型 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	encoder_input_size = X_train.shape[2]
	encoder = EncoderLSTM(encoder_input_size)  # 样本的特征数
		
	# 设定优化器 —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	optimizer = torch.optim.Adam(
		encoder.parameters(),
		lr = lr
	)
	
	# 模型训练和保存 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	train_loss_record, verify_loss_record = [], []
	early_stop_steps = 200
	sum = torch.tensor(early_stop_steps - 50).int()
	stop_criterion = torch.tensor(1).byte()
	
	# CUDA设置 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	if use_cuda:
		encoder = encoder.cuda()
		sum = sum.cuda()
		stop_criterion = stop_criterion.cuda()
		
	for epoch in range(epochs):
		# 训练集
		for train_x, train_y in trainloader:
			weighten_targets = cal_attention_outputs(encoder, train_x)
			train_loss = criterion(weighten_targets[:, :, 0], train_y[:, :, 0])
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()
		train_loss_record.append(train_loss)
		
		# 验证集
		with torch.no_grad():
			for verify_x, verify_y in verifyloader:
				weighten_targets = cal_attention_outputs(encoder, verify_x)
				verify_loss = criterion(weighten_targets[:, :, 0], verify_y[:, :, 0])
			verify_loss_record.append(verify_loss)
		
		if epoch % 5 == 0:
			print(epoch, train_loss, verify_loss)
			
		if epoch % 500 == 0:
			save_models_and_records(train_loss_record, verify_loss_record, encoder)
	
	save_models_and_records(train_loss_record, verify_loss_record, encoder)

	



