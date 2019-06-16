# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

pytorch的神经网络模型
"""
import sys
import numpy as np
import torch
from torch import nn

sys.path.append('../')

from mods.config_loader import config
from mods.build_samples import build_train_samples_and_targets, build_test_samples_and_targets


class NN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(NN, self).__init__()
		self.connec_0 = nn.Linear(input_size, hidden_size[0])
		self.act_0 = nn.Sigmoid()
		self.connec_1 = nn.Linear(hidden_size[0], hidden_size[1])
		self.act_1 = nn.Sigmoid()
		self.connec_2 = nn.Linear(hidden_size[1], output_size)
		self.act_2 = nn.ReLU()

	def forward(self, x):
		x = self.connec_0(x)
		x = self.act_0(x)
		x = self.connec_1(x)
		x = self.act_1(x)
		x = self.connec_2(x)
		x = self.act_2(x)
		return x
	

def initialize_model_params(nn, input_size, hidden_size, output_size):
	"""
	初始化模型参数
	:param nn: NN(), 神经网络模型
	:param input_size: int, 输入维数
	:param hidden_size: list of ints, 中间隐含层维数
	:param output_size: int, 输出维数
	:return:
		nn, NN(), 参数初始化后的神经网络模型
	"""
	nn.connec_0.weight.data = torch.rand(hidden_size[0], input_size)  # attention: 注意shape是转置关系
	nn.connec_0.bias.data = torch.rand(hidden_size[0])
	nn.connec_1.weight.data = torch.rand(hidden_size[1], hidden_size[0])
	nn.connec_1.bias.data = torch.rand(hidden_size[1])
	nn.connec_2.weight.data = torch.rand(output_size, hidden_size[1])
	nn.connec_2.bias.data = torch.rand(output_size)
	return nn


def load_model():
	"""载入已经训练好的模型"""
	target_column = config.conf['model_params']['target_column']
	# model_path = '../tmp/model_{}.pkl'.format(target_column)
	model_path = '../tmp/model_state_dict_{}.pth'.format(target_column)
	use_cuda = config.conf['model_params']['pred_use_cuda']
	
	if '.pth' in model_path:
		if use_cuda:
			pretrained_model_dict = torch.load(model_path)
		else:
			pretrained_model_dict = torch.load(model_path, map_location = 'cpu')
		X_train, y_train = build_train_samples_and_targets()
		input_size = X_train.shape[1]
		hidden_size = [int(np.floor(X_train.shape[1] / 2)), int(np.floor(X_train.shape[1] / 4))]
		output_size = y_train.shape[1]
		model = NN(input_size, hidden_size, output_size)
		model.load_state_dict(pretrained_model_dict, strict = False)
		return model
	elif '.pkl' in model_path:
		model = torch.load(model_path)
		return model
	else:
		raise RuntimeError('file path incorrect', )
