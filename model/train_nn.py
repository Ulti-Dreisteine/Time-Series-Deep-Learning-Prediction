# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

训练模型
"""
import torch
import json
import sys

sys.path.append('../')

from mods.extract_data_and_normalize import extract_implemented_data
from mods.nn_models import NN
from mods.build_train_and_test_samples_for_nn import build_train_and_verify_datasets
from mods.config_loader import config
from mods.loss_criterion import criterion
from mods.one_hot_encoder import one_hot_encoding
from mods.data_filtering import savitzky_golay_filtering


def save_models(nn_model, train_loss_record, verify_loss_record):
	"""保存模型文件"""
	target_columns = config.conf['model_params']['target_columns']
	
	# 保存模型文件
	torch.save(nn_model.state_dict(), '../tmp/nn_state_dict_{}.pth'.format(target_columns))
	
	# 保存模型结构参数
	model_struc_params = {
		'nn_model': {
			'input_size': nn_model.input_size,
			'hidden_sizes': nn_model.hidden_sizes,
			'output_size': nn_model.output_size
		}
	}
	
	with open('../tmp/nn_struc_params.json', 'w') as f:
		json.dump(model_struc_params, f)
		
	# 损失函数记录
	train_loss_list = [float(p.detach().cpu().numpy()) for p in train_loss_record]
	verify_loss_list = [float(p.cpu().numpy()) for p in verify_loss_record]
	
	with open('../tmp/nn_train_loss.json', 'w') as f:
		json.dump(train_loss_list, f)
	with open('../tmp/nn_verify_loss.json', 'w') as f:
		json.dump(verify_loss_list, f)
	

if __name__ == '__main__':
	# 读取原始数据并整理成表 ————————————————————————————————————————————————————————————————————————————————————————————————————————————
	file_name = '../tmp/taiyuan_cityHour.csv'
	total_implemented_normalized_data = extract_implemented_data(file_name, use_local = True, save = True)
	
	# 数据滤波和编码
	data = savitzky_golay_filtering(total_implemented_normalized_data)
	_ = one_hot_encoding(data, save = True)
	
	# 设定参数 ————————————————————————————————————————————————————————————————————————————————————————-——————————————————————————————
	use_cuda = torch.cuda.is_available()
	batch_size = config.conf['model_params']['batch_size']
	lr = config.conf['model_params']['lr']
	epochs = config.conf['model_params']['epochs']

	# 载入数据集，构建训练和验证集样本 ————————————————————————————————————————————————————————————————————————————————————————————————————
	trainloader, verifyloader, X_train, y_train, X_verify, y_verify, continuous_columns_num = build_train_and_verify_datasets()

	# 构造神经网络模型 —————————————————————————————————————————————————————————————————————————————————————————————————————————————-———
	input_size = X_train.shape[1]
	output_size = y_train.shape[1]
	hidden_sizes = [2 * input_size, 2 * input_size, 2 * input_size, 2 * output_size, 2 * output_size, 2 * output_size]
	nn_model = NN(input_size, hidden_sizes, output_size)
	
	if use_cuda:
		torch.cuda.empty_cache()
		trainloader = [(train_x.cuda(), train_y.cuda()) for (train_x, train_y) in trainloader]
		verifyloader = [(verify_x.cuda(), verify_y.cuda()) for (verify_x, verify_y) in verifyloader]
		nn_model = nn_model.cuda()

	# 指定优化器 —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	optimizer = torch.optim.Adam(
		nn_model.parameters(),
		lr = lr,
		weight_decay = 1e-3
	)

	# 模型训练和保存 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	train_loss_record, verify_loss_record = [], []

	for epoch in range(epochs):
		# 训练集
		nn_model.train()
		for train_x, train_y in trainloader:
			y_train_p = nn_model(train_x)
			y_train_t = train_y
			train_loss_fn = criterion(y_train_p, y_train_t)

			optimizer.zero_grad()
			train_loss_fn.backward()
			optimizer.step()

		train_loss_record.append(train_loss_fn)

		nn_model.eval()
		with torch.no_grad():
			for verify_x, verify_y in verifyloader:
				y_verify_p = nn_model(verify_x)
				y_verify_t = verify_y
				verify_loss_fn = criterion(y_verify_p, y_verify_t)
			verify_loss_record.append(verify_loss_fn)

		if epoch % 100 == 0:
			print(epoch, train_loss_fn, verify_loss_fn)

		# 保存模型
		if epoch % 500 == 0:
			save_models(nn_model, train_loss_record, verify_loss_record)

	save_models(nn_model, train_loss_record, verify_loss_record)
	
	
	
	
	
	
	
	
	
	


