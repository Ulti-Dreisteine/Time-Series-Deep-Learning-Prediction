# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

对数据中的离散变量进行编码
"""
import pandas as pd
import sys

sys.path.append('../')

from mods.one_hot_encoder import one_hot_encoding
from mods.data_filtering import savitzky_golay_filtering


if __name__ == '__main__':
	# 读取数据
	file = '../tmp/total_implemented_normalized_data.csv'
	data = pd.read_csv(file)
	
	# 数据滤波
	data = savitzky_golay_filtering(data)
	
	# 数据编码
	total_one_hot_encoded_data = one_hot_encoding(data, save = True)
		
	
	


