# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

获取原数据并归一化
"""
import sys

sys.path.append('../')

from mods.extract_data_and_normalize import extract_implemented_data


if __name__ == '__main__':
	file_name = '../tmp/taiyuan_cityHour.csv'
	total_implemented_normalized_data = extract_implemented_data(file_name, use_local = False, save = True)

