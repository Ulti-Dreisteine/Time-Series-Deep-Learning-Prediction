# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

不同变量间的互相关函数计算
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.data_filtering import band_pass_filtering, savitzky_golay_filtering


def cross_correlation(x_series, y_series, d):
	"""
	在延迟为d上的互相关分析, x_series, 移动y_series: d > 0向右移，d < 0向左移
	:param time_series_0: np.ndarray, 目标变量
	:param time_series_1: np.ndarray, 外生变量
	:param d: 延迟阶数
	:return:
	"""
	len_y = len(y_series)
	y_series = np.hstack((y_series[-(d % len_y):], y_series[: -(d % len_y)]))
	
	mean_x, mean_y = np.mean(x_series), np.mean(y_series)
	numerator = np.sum((x_series - mean_x) * (y_series - mean_y))
	denominator = np.sqrt(np.sum(np.power((x_series - mean_x), 2))) * np.sqrt(
		np.sum(np.power((y_series - mean_y), 2)))
	
	return numerator / denominator


def peak_loc_and_value(x_series, y_series, detect_len):
	"""
	检测峰值
	:param x_series:
	:param y_series:
	:param detect_len: 检测半长
	:return:
	"""
	ccf_results = []
	series_len = len(x_series)
	for d in range(-detect_len, detect_len + 1):
		ccf_results.append(cross_correlation(x_series, y_series, d))
	
	mean, sigma = np.mean(ccf_results), np.power(np.var(ccf_results), 0.5)
	pos_corr_values = [p for p in ccf_results if (p > mean + 3.0 * sigma)]
	neg_corr_values = [p for p in ccf_results if (p < mean - 3.0 * sigma)]
	
	start_loc = detect_len - series_len // 2
	end_loc = detect_len + series_len // 2 + 1
	ccf_seg = ccf_results[start_loc: end_loc]
	if (len(pos_corr_values) == 0) & (len(neg_corr_values) == 0):
		peak_value = 0
		peak_loc = 0
	else:
		if len(pos_corr_values) >= len(neg_corr_values):
			peak_value = max(ccf_seg) - mean
			peak_loc = ccf_seg.index(max(ccf_seg)) + start_loc
		else:
			peak_value = mean - min(ccf_seg)
			peak_loc = ccf_seg.index(min(ccf_seg)) + start_loc
	
	time_lag = peak_loc - detect_len
	
	return ccf_seg, start_loc, end_loc, detect_len, time_lag, peak_value


if __name__ == '__main__':
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	data = savitzky_golay_filtering(data)
	
	columns = config.conf['model_params']['continuous_columns']
	diagnosis_locs = np.arange(0, 20000, 1000)
	series_len = 200
	detect_len = 5000
	
	ccf_and_time_lags = {}
	plt.figure(figsize = [14, 14])
	for col_x in columns:
		ccf_and_time_lags[col_x] = {}
		for col_y in columns:
			print('processing col_x: {}, col_y: {}'.format(col_x, col_y))
			ccf_segs, time_lags, peak_values = [], [], []
			for loc in diagnosis_locs:
				x_series = np.array(data[col_x]).flatten()[loc: loc + series_len]
				y_series = np.array(data[col_y]).flatten()[loc: loc + series_len]
				ccf_seg, start_loc, end_loc, detect_len, time_lag, peak_value = peak_loc_and_value(x_series, y_series, detect_len)
				ccf_segs.append(ccf_seg)
				if (time_lag != -detect_len) & (peak_value != 0):
					time_lags.append(time_lag)
					peak_values.append(peak_value)
			mean_ccf_seg = np.mean(np.array(ccf_segs), axis = 0)
			mean_time_lag = np.mean(np.array(time_lags), axis = 0)
			mean_peak_value = np.mean(np.array(peak_values), axis = 0)
			
			ccf_and_time_lags[col_x][col_y] = {'ccf_value': mean_peak_value, 'time_lag': mean_time_lag}
			
			plt.subplot(len(columns), len(columns), len(columns) * columns.index(col_x) + columns.index(col_y) + 1)
			plt.plot(range(start_loc - detect_len, end_loc - detect_len), mean_ccf_seg)
			plt.fill_between(range(start_loc - detect_len, end_loc - detect_len), mean_ccf_seg)
			plt.plot([start_loc - detect_len, end_loc - detect_len], [0, 0], 'k--', linewidth = 0.3)
			plt.plot([0, 0], [-1.0, 1.0], 'k-', linewidth = 0.3)
			plt.xlim([-series_len // 2, series_len // 2])
			plt.ylim([-1, 1])
			plt.xticks(fontsize = 6)
			plt.yticks(fontsize = 6)
			if col_x == columns[0]:
				plt.title(col_y, fontsize = 8)
			
			if col_y == columns[0]:
				plt.ylabel(col_x, fontsize = 8)
			
			plt.tight_layout()
			plt.show()
			plt.pause(1.0)
	
	plt.tight_layout()
	plt.savefig('../graphs/ccf_analysis_on_continuous.png')
