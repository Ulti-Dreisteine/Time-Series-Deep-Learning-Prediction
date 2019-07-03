# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

不同变量间的互相关函数计算
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from math import factorial
import sys

sys.path.append('../')

from mods.config_loader import config


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


def savitzky_golay(y, window_size, order, deriv = 0, rate = 1):
	r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
	The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techniques.
	Parameters
	----------
	y : array_like, shape (N,)
		the values of the time history of the signal.
	window_size : int
		the length of the window. Must be an odd integer number.
	order : int
		the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1.
	deriv: int
		the order of the derivative to compute (default = 0 means only smoothing)
	rate: int
		[unknown]
	Returns
	-------
	ys : ndarray, shape (N)
		the smoothed signal (or it's n-th derivative).
	Notes
	-----
	The Savitzky-Golay is a type of low-pass filter, particularly
	suited for smoothing noisy data. The main idea behind this
	approach is to make for each point a least-square fit with a
	polynomial of high order over a odd-sized window centered at
	the point.
	Examples
	--------
	t = np.linspace(-4, 4, 500)
	y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
	ysg = savitzky_golay(y, window_size=31, order=4)
	import matplotlib.pyplot as plt
	plt.plot(t, y, label='Noisy signal')
	plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
	plt.plot(t, ysg, 'r', label='Filtered signal')
	plt.legend()
	plt.show()
	References
	----------
	.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
	   Data by Simplified Least Squares Procedures. Analytical
	   Chemistry, 1964, 36 (8), pp 1627-1639.
	.. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
	   W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
	   Cambridge University Press ISBN-13: 9780521880688
	"""
	
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except Exception:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order + 1)
	half_window = (window_size - 1) // 2
	
	# precompute coefficients
	b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
	m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
	
	# pad the signal at the extremes with values taken from the signal itself
	firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
	lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve(m[::-1], y, mode = 'valid')


def savitzky_golay_filtering(data, window_size = 11, order = 2):
	"""
	使用savitzky-golay滤波
	:param data: pd.DataFrame, 带滤波数据表, columns = [target_column, selected_columns_0, selected_column_1, ...]
	:param window_size: int, 选定的相邻窗口长度
	:param order: int, 用于滤波的多项式的阶数
	:return:
		data_filtered: pd.DataFrame, 滤波处理后的数据表, columns = [target_column, selected_columns_0, selected_column_1, ...]
	"""
	data_copy = copy.deepcopy(data)
	columns = list(set(config.conf['model_params']['target_columns'] + config.conf['model_params']['continuous_columns']))
	data_copy = data_copy[columns]
	filtered_results = []
	for column in columns:
		y = np.array(data_copy.loc[:, column]).flatten()
		y_filtered = savitzky_golay(y, window_size = window_size, order = order)
		filtered_results.append(y_filtered)
	filtered_results = np.array(filtered_results).T
	filtered_results = pd.DataFrame(filtered_results, columns = columns)
	
	data_filtered = pd.concat([data[['city', 'ptime', 'time_stamp']], filtered_results], axis = 1, sort = False).reset_index(
		drop = True)
	data_filtered = pd.concat([data_filtered, data[config.conf['model_params']['discrete_columns']]], axis = 1, sort = False)
	
	return data_filtered


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
	
	plt.figure(figsize = [14, 14])
	columns = config.conf['model_params']['continuous_columns']
	columns_num = len(columns)
	ccf_and_time_lag = {}
	for col_x in columns:
		ccf_and_time_lag[col_x] = {}
		for col_y in columns:
			print('processing col_x: {}, col_y: {}'.format(col_x, col_y))
			x_series = np.array(data[col_x]).flatten()[1000:5000]
			y_series = np.array(data[col_y]).flatten()[1000:5000]
			series_len = len(x_series)
			detect_len = 5000
			
			ccf_seg, start_loc, end_loc, detect_len, time_lag, peak_value = peak_loc_and_value(x_series, y_series, detect_len)
			ccf_and_time_lag[col_x][col_y] = {'ccf_value': peak_value, 'time_lag': time_lag}
			
			plt.subplot(columns_num, columns_num, columns_num * columns.index(col_x) + columns.index(col_y) + 1)
			plt.plot(range(start_loc - detect_len, end_loc - detect_len), ccf_seg)
			plt.fill_between(range(start_loc - detect_len, end_loc - detect_len), ccf_seg)
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
	plt.savefig('../tmp/ccf_analysis_on_continuous.png')
