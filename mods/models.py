# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

pytorch的神经网络模型
"""
import json
import sys
import numpy as np
import torch
from torch import nn

sys.path.append('../')

from mods.config_loader import config


class ContinuousEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(ContinuousEncoder, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.connect_0 = nn.Linear(self.input_size, int(self.input_size / 2))
		self.act_0 = nn.Sigmoid()
		self.connect_1 = nn.Linear(int(self.input_size / 2), self.output_size)
		self.act_1 = nn.Sigmoid()
	
	def forward(self, x):
		x = self.connect_0(x)
		x = self.act_0(x)
		x = self.connect_1(x)
		x = self.act_1(x)
		return x


def initialize_continuous_encoder_params(continuous_encoder):
	"""
	初始化模型参数
	:param nn: NN(), 神经网络模型
	:param input_size: int, 输入维数
	:param hidden_size: list of ints, 中间隐含层维数
	:param output_size: int, 输出维数
	:return:
		nn, NN(), 参数初始化后的神经网络模型
	"""
	continuous_encoder.connect_0.weight.data = torch.rand(int(continuous_encoder.input_size / 2), continuous_encoder.input_size)
	continuous_encoder.connect_0.bias.data = torch.rand(int(continuous_encoder.input_size / 2))
	continuous_encoder.connect_1.weight.data = torch.rand(continuous_encoder.output_size, int(continuous_encoder.input_size / 2))
	continuous_encoder.connect_1.bias.data = torch.rand(continuous_encoder.output_size)
	return continuous_encoder


class DiscreteEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(DiscreteEncoder, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.connect_0 = nn.Linear(self.input_size, self.output_size)
		self.act_0 = nn.Sigmoid()
	
	def forward(self, x):
		x = self.connect_0(x)
		x = self.act_0(x)
		return x
	

def initialize_discrete_encoder_params(discrete_encoder):
	"""
	初始化模型参数
	:param nn: NN(), 神经网络模型
	:param input_size: int, 输入维数
	:param hidden_size: list of ints, 中间隐含层维数
	:param output_size: int, 输出维数
	:return:
		nn, NN(), 参数初始化后的神经网络模型
	"""
	discrete_encoder.connect_0.weight.data = torch.rand(discrete_encoder.output_size, discrete_encoder.input_size)
	discrete_encoder.connect_0.bias.data = torch.rand(discrete_encoder.output_size)
	return discrete_encoder
	

class NN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(NN, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.connec_0 = nn.Linear(self.input_size, self.hidden_size[0])
		self.act_0 = nn.Sigmoid()
		self.connec_1 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
		self.act_1 = nn.Sigmoid()
		self.connec_2 = nn.Linear(self.hidden_size[1], self.output_size)
		self.act_2 = nn.ReLU()

	def forward(self, x):
		x = self.connec_0(x)
		x = self.act_0(x)
		x = self.connec_1(x)
		x = self.act_1(x)
		x = self.connec_2(x)
		x = self.act_2(x)
		return x
	

def initialize_nn_params(nn):
	"""
	初始化模型参数
	:param nn: NN(), 神经网络模型
	:param input_size: int, 输入维数
	:param hidden_size: list of ints, 中间隐含层维数
	:param output_size: int, 输出维数
	:return:
		nn, NN(), 参数初始化后的神经网络模型
	"""
	nn.connec_0.weight.data = torch.rand(nn.hidden_size[0], nn.input_size)  # attention: 注意shape是转置关系
	nn.connec_0.bias.data = torch.rand(nn.hidden_size[0])
	nn.connec_1.weight.data = torch.rand(nn.hidden_size[1], nn.hidden_size[0])
	nn.connec_1.bias.data = torch.rand(nn.hidden_size[1])
	nn.connec_2.weight.data = torch.rand(nn.output_size, nn.hidden_size[1])
	nn.connec_2.bias.data = torch.rand(nn.output_size)
	return nn


def load_models():
	"""载入已经训练好的模型"""
	target_column = config.conf['model_params']['target_column']
	
	with open('../tmp/model_struc_params.json', 'r') as f:
		model_struc_params = json.load(f)
	
	model_paths = [
		'../tmp/nn_state_dict_{}.pth'.format(target_column),
		'../tmp/continuous_encoder_state_dict_{}.pth'.format(target_column),
		'../tmp/discrete_encoder_state_dict_{}.pth'.format(target_column)
	]
	
	model_classes = [NN, ContinuousEncoder, DiscreteEncoder]
	model_names = ['nn', 'continuous_encoder', 'discrete_encoder']
	models = []
	for i in range(len(model_paths)):
		pretrained_model_dict = torch.load(model_paths[i], map_location = 'cpu')
		if i == 0:
			input_size = model_struc_params[model_names[i]]['input_size']
			hidden_size = model_struc_params[model_names[i]]['hidden_size']
			output_size = model_struc_params[model_names[i]]['output_size']
			model = model_classes[i](input_size, hidden_size, output_size)
			model.load_state_dict(pretrained_model_dict, strict = False)
			models.append(model)
		else:
			input_size = model_struc_params[model_names[i]]['input_size']
			output_size = model_struc_params[model_names[i]]['output_size']
			model = model_classes[i](input_size, output_size)
			model.load_state_dict(pretrained_model_dict, strict = False)
			models.append(model)
	return models
