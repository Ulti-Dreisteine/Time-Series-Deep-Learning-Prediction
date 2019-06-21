# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

带通滤波
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from math import factorial
import sys

sys.path.append('../')

from mods.config_loader import config


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
	except:
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


def band_pass_filtering(data):
	"""
	带通滤波
	:param data: pd.DataFrame, 带滤波数据表, columns = [target_column, selected_columns_0, selected_column_1, ...]
	:return:
		data_filtered: pd.DataFrame, 滤波处理后的数据表, columns = [target_column, selected_columns_0, selected_column_1, ...]
	"""
	data_copy = copy.deepcopy(data)
	embed_lags = config.conf['model_params']['embed_lags']
	samples_len = len(data_copy)

	columns = [config.conf['model_params']['target_column']] + config.conf['model_params']['continuous_columns']
	filtered_results = []
	for column in columns:
		lag = embed_lags[column]
		time_series = np.array(data_copy[column])
		fft_results = np.fft.fft(time_series)
		fft_real, fft_imag = fft_results.real, fft_results.imag

		frequencies_to_keep = [0, round(samples_len / lag)]
		kept_frequencies_list = list(
			np.arange(
				frequencies_to_keep[0],
				frequencies_to_keep[1] + 1
			)
		) + list(
			np.arange(
				samples_len - frequencies_to_keep[1] - 1,
				samples_len - frequencies_to_keep[0]
			)
		)

		kept_frequencies_list = [int(p) for p in kept_frequencies_list]  # 注意一定要是整数值
		for i in range(len(fft_real)):
			if i not in kept_frequencies_list:
				fft_real[i] = 0
				fft_imag[i] = 0

		fft_results.real, fft_results.imag = fft_real, fft_imag
		filtered_series = np.fft.ifft(fft_results).real

		filtered_results.append(filtered_series)
	filtered_results = pd.DataFrame(np.array(filtered_results).T, columns = columns)
	
	data_filtered = pd.concat([data_copy[['city', 'ptime', 'time_stamp']], filtered_results], axis = 1, sort = False).reset_index(drop = True)

	return data_filtered


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
	columns = list(set([config.conf['model_params']['target_column']] + config.conf['model_params']['continuous_columns']))
	data_copy = data_copy[columns]
	filtered_results = []
	for column in columns:
		y = np.array(data_copy.loc[:, column]).flatten()
		y_filtered = savitzky_golay(y, window_size = window_size, order = order)
		filtered_results.append(y_filtered)
	filtered_results = np.array(filtered_results).T
	filtered_results = pd.DataFrame(filtered_results, columns = columns)
	
	data_filtered = pd.concat([data[['city', 'ptime', 'time_stamp']], filtered_results], axis = 1, sort = False).reset_index(drop = True)
	data_filtered = pd.concat([data_filtered, data[config.conf['model_params']['discrete_columns']]], axis = 1, sort = False)
	
	return data_filtered


if __name__ == '__main__':
	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	
	# 带通滤波
	# filtered_results = band_pass_filtering(data)
	# plt.plot(data['pm25'])
	# plt.plot(filtered_results['pm25'])
	
	# Savitzky-Golay滤波
	filtered_results = savitzky_golay_filtering(data)
	plt.figure(figsize = [12, 6])
	plt.plot(data['temp'])
	plt.plot(filtered_results['temp'], 'r')
	plt.legend(['true', 'filtered'])
	plt.xlabel('time (hour)')
	plt.ylabel('temperature (C)')
	plt.grid(True)
	plt.tight_layout()

