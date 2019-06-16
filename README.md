## time_series_forecast_app
时间序列预测app

### 项目目的
使用某固定站点的历史污染物浓度、温度、压力、湿度等数据对未来污染物浓度变化进行预测

### 项目依赖
```markdown
python>=3.6
pandas>=0.24.2
statsmodels>=0.9.0
matplotlib>=3.0.3
numpy>=1.16.3
seaborn>=0.9.0
scipy>=1.2.1
lake==1.0.1
scikit-learn==0.20.3
```

### 代码结构
```markdown
|--config
    |--config.yml                           # 项目配置文件
|--analysis
    |--acf_pacf_test.py                     # 自相关和偏自相关分析，确定时间序列动态特性参数
    |--correlation_analysis.py              # 时间序列相关分析
    |--feature_importance.py                # 特征重要性分析
    |--granger_causality_test               # 格兰杰检验, 确定时间序列因果性和时间延迟
|--model
    |--train_local_model.py                 # 训练模型
    |--pred_local_model.py                  # 模型预测
|--logs
    |--web_error.log                        # error日志
    |--web_info.log                         # info日志
|--mods
    |--config_loader.py                     # 配置加载器
    |--nn_model.py                          # 基于pytorch的神经网络训练和预测模型
    |--granger_causality.py                 # 格兰杰检验
    |--build_samples.py                     # 构建样本和目标数据集
    |--data_filtering.py                    # 数据滤波
    |--extract_data_and_normailze.py        # 提取数据并归一化
    |--granger_causality.py                 # 格兰杰检验
    |--loss_criterion.py                    # 模型损失函数
    |--model_evaluations.py                 # 模型评估
|--others
    |--read_raw_data.py                     # 读取原始数据
    |--get_raw_data_and_normalize.py        # 读取原始数据并归一化
    |--data_correlation_analysis.py         # 数据相关性分析
|--tmp                                      # 数据和模型文件

```
