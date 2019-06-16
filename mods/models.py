# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

各模型
"""
import sys
import numpy as np
import torch
from torch import nn

sys.path.append('../')


class EncoderLSTM(nn.Module):
	"""lstm模型"""
	
	def __init__(self, input_size):
		super(EncoderLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = int(np.floor(input_size) * 2)
		self.num_layers = 2
		
		self.lstm = nn.LSTM(
			input_size = self.input_size,
			hidden_size = self.hidden_size,
			num_layers = self.num_layers,
			batch_first = True
		)
		
		self.connect_0 = nn.Linear(self.hidden_size, 1)
	
	def forward(self, x):
		x, _ = self.lstm(x)
		x = self.connect_0(x)
		return x
	

def initialize_lstm_params(lstm):
	"""
	初始化模型参数
	:param lstm: nn.LSTM(), 神经网络模型
	:return:
		lstm, nn.LSTM(), 参数初始化后的神经网络模型
	"""
	lstm.connect_0.weight.data = torch.rand(1, lstm.hidden_size)  # attention: 注意shape是转置关系
	lstm.connect_0.bias.data = torch.rand(1)
	return lstm


