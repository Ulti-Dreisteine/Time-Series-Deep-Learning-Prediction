# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests


def granger_causality(data, target_column, selected_columns, max_lag, show_plots = False):
	"""
	格兰杰检验pm25与data中选定字段间的因果关系
	:param target_column: str, 目标字段
	:param data: pd.DataFrame, 数据表
	:param selected_columns: list of strings, 选定字段列表
	:return:
		lags: list, 所有的时滞列表
		forward_results: list, 其他字段对目标字段在时滞区间上的影响结果记录
		reverse_results: list, 目标字段对其他字段在时滞区间上的影响结果记录
	"""
	forward_results = []
	reverse_results = []
	
	if show_plots:
		plt.figure('granger-causality', figsize = [6, 6])
		plt.title('granger causality')
		
	for column in selected_columns:
		granger_test_result = grangercausalitytests(data[[target_column, column]], maxlag = max_lag, verbose = False)
		optimal_lag = -1
		F_test = -1.0
		for key in granger_test_result.keys():
			_F_test_ = granger_test_result[key][0]['params_ftest'][0]
			if _F_test_ > F_test:
				F_test = _F_test_
				optimal_lag = key
		forward_results.append([granger_test_result[key][0]['params_ftest'][0] for key in granger_test_result.keys()])
		
		if show_plots:
			plt.subplot(len(selected_columns), 1, selected_columns.index(column) + 1)
			plt.plot(
				granger_test_result.keys(),
				[granger_test_result[key][0]['params_ftest'][0] for key in granger_test_result.keys()]
			)
			
			plt.ylabel(column)
			plt.grid(True)
	
	if show_plots:
		plt.xlabel('hour')
		plt.tight_layout()
		plt.show()
	
	if show_plots:
		plt.figure('granger-causality reverse', figsize = [6, 6])
		plt.title('granger causality reverse')
		
	for column in selected_columns:
		granger_test_result = grangercausalitytests(data[[column, target_column]], maxlag = max_lag, verbose = False)
		optimal_lag = -1
		F_test = -1.0
		for key in granger_test_result.keys():
			_F_test_ = granger_test_result[key][0]['params_ftest'][0]
			if _F_test_ > F_test:
				F_test = _F_test_
				optimal_lag = key
		reverse_results.append([granger_test_result[key][0]['params_ftest'][0] for key in granger_test_result.keys()])
		
		if show_plots:
			plt.subplot(len(selected_columns), 1, selected_columns.index(column) + 1)
			plt.plot(
				granger_test_result.keys(),
				[granger_test_result[key][0]['params_ftest'][0] for key in granger_test_result.keys()],
				'r'
			)
			plt.ylabel(column)
			plt.grid(True)
	
	if show_plots:
		plt.xlabel('hour')
		plt.tight_layout()
		plt.show()
	
	lags = list(granger_test_result.keys())
	return lags, forward_results, reverse_results


