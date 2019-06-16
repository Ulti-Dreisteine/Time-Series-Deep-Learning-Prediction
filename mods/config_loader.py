# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: lichenarthurdata@gmail.com

载入配置
"""
import os
import yaml
import sys
import logging
import logging.config
import lake

import lake.decorator
import lake.data
import lake.dir

sys.path.append(os.path.join(os.path.dirname(__file__), '../config/'))

if len(logging.getLogger().handlers) == 0:
	logging.basicConfig(level = logging.DEBUG)


@lake.decorator.singleton
class ConfigLoader(object):
	def __init__(self, config_path=None):
		self._config_path = config_path or self._absolute_path('../config/config.yml')
		self._load()

	def _absolute_path(self, path):
		return os.path.join(os.path.dirname(__file__), path)

	def _load(self):
		with open(self._config_path, 'r') as f:
			self._conf = yaml.load(f, Loader = yaml.Loader)  # yaml.FullLoader

	@property
	def conf(self):
		return self._conf

	def set_logging(self):
		"""
		配制logging文件
		"""
		log_dir = self._absolute_path('../logs/')
		lake.dir.mk(log_dir)
		log_config = self.conf['logging']
		update_filename(log_config)
		logging.config.dictConfig(log_config)


def update_filename(log_config):
	"""
	更新logging中filename的配置
	:param log_config: dict, 日志配置
	"""
	to_log_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), '../', x))
	if 'filename' in log_config:
		log_config['filename'] = to_log_path(log_config['filename'])
	for key, value in log_config.items():
		if isinstance(value, dict):
			update_filename(value)


config = ConfigLoader()

