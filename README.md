## time_series_forecast_app
时间序列预测app

### 项目目的
使用某固定站点的历史污染物浓度、温度、压力、湿度等数据对未来污染物浓度变化进行预测

### TODO
```markdown
1.  神经网络中引入类似lstm和attention的机制以提高模型精度
2.  计算不同城市、不同污染物的预测效果并进行对比
```

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
    |--config.yml                       # 项目配置文件
|--bin
    |--app.py                           # web服务代码
|--model
    |--train_lstm.py                    # 训练LSTM模型
|--mods
    |--config_loader.py                 # 配置加载器
    |--models.py                        # 长短期记忆模型以及基于pytorch的神经网络训练和预测模型
    |--data_filtering.py                # 清理数据
    |--granger_causality.py             # 格兰杰检验，用于确定变量时滞
    |--build_samples_and_targets.py     # 构建样本和目标数据集 
    |--extract_data_and_normalize.py    # 清理异常值，缺省值并进行归一化
    |--loss_criterion.py                # 损失函数
    |--model_evaluations.py             # 模型评估
|--others
    |--get_raw_data_and_normalize.py    # 获取原始数据并进行归一化
    |--data_correlation_and_analysis.py # 数据相关性分析
|--analysis 
    |--acf_pacf_test.py                 # acf和pacf检验, 确定数据时间动态参数
    |--correlation_analysis.py          # 数据相关性分析, 确定数据时滞
    |--feature_importance.py            # 特征重要性分析
    |--granger_causality_test.py        # 格兰杰检验, 确定数据相关性和时滞
|--tmp                                  # 资源文档

```

### 个别文档说明
* config.yml
    * `record_start_time`     # 数据记录起始时间
    * `record_end_time`       # 数据记录结束时间
    * `exist_record_time`     # 当前时间 
    * `model_params`          # 模型参数
        * `samples_len`         # 训练样本长度
        * `hr`                  # 每小时秒数
        * `train_use_cuda`      # 训练使用cuda
        * `pred_use_cuda`       # 预测使用cuda
        * `lr`                  # 学习率
        * `epochs`              # 训练代数
        * `batch_size`          # batch数
        * `pred_horizon_len`    # 预测样本长度
        * `pred_dim`            # 预测维数（单个样本预测窗口长度）
        * `target_column`       # 预测目标
        * `selected_columns`    # 预测被选择的使用历史变量
        * `variable_bounds`     # 变量取值上下界
        * `time_lags`           # 各被选择变量相对于目标变量的时滞
        * `embed_lags`          # 各被选择变量构造样本时前后时间间隔, 设定为1
        * `acf_lags`            # 使用自相关函数分析计算得出的被选择变量样本窗口长度
        
* mods
    * models.py                         
        * LSTM 
            * `input_size`                  # 输入数据维度, 对应于被选择变量个数
            * `batch_first`                 # 设定为True则使用batch优先批式训练
            * `hidden_size`                 # 隐藏层层数
            * `num_layers`                  # LSTM网络层数
        * initialize_lstm_params            # 初始化lstm参数
        * AlphaLayer                        # attention中对输入加权层
            * `features_num`                # 每个时刻输入中特征维数，对应于被选择变量个数
            * `input_weight`                # 该时刻输入样本映射成的权重
        * initialize_alpha_layer_params     # 初始化alpha_layer模型参数
        * WeightsLayer                      # attention中对输出加权层
            * `weights_num`                 # 输入编码的权重个数
            * `output_size`                 # 输出序列长度
        * initialize_weights_layer_params   # 初始化weights_layer参数
        
    * data_filtering.py                     # 数据降噪
        * savitzky_golay_filtering          # Savitzky-Golay滤波法，用线性最小二乘法把相邻数据点fit到低阶多项式 
        * band_pass_filtering               # 带通滤波，去掉高频低频数据      
    * build_test_samples_and_targets.py     # 构建样本和目标数据集
    * granger_causality.py                  # 格兰杰检验
                                
