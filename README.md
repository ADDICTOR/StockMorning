# StockMorning

StockMorning 是一个基于 Python 的股票量化交易系统，支持技术指标分析、股票筛选和持仓管理等功能。

## 功能特点

- 多维度技术指标分析（MA、MACD、RSI、KDJ等12个指标）
- 基于多线程的高效数据获取
- 完整的回测系统
- 实时风险监控和持仓管理
- 详细的交易分析报告

## 系统架构

StockMorning/
├── src/ # 源代码目录
│ ├── main.py # 主程序入口
│ ├── core/ # 核心功能模块
│ │ ├── backtest.py # 回测系统
│ │ ├── strategy.py # 交易策略
│ │ ├── risk_manager.py# 风险管理
│ │ ├── data_fetcher.py# 数据获取
│ │ └── position.py # 持仓管理
│ ├── analysis/ # 分析模块
│ │ └── performance.py # 性能分析
│ └── utils/ # 工具函数
│ ├── config.py # 配置加载
│ └── logger.py # 日志管理
├── configs/ # 配置文件目录
│ └── default.yaml # 默认配置
├── data/ # 数据目录
│ ├── input/ # 输入数据
│ └── output/ # 输出结果
└── tests/ # 测试目录
└── data/ # 测试数据

## 当前功能

### 1. 交易策略

- 12个技术指标组合
- 可配置的指标权重
- 灵活的信号生成机制

### 2. 风险管理

- 止损止盈机制（当前止损5%，止盈10%）
- 持仓限制（单股最大30%）
- 回撤控制（最大回撤15%）

### 3. 回测系统

- 完整的交易成本模拟（佣金万三、印花税千分之一）
- 考虑滑点影响（默认0.1%）
- 详细的回测报告

### 4. 性能分析

- 收益率分析（总收益率、年化收益率）
- 风险指标（夏普比率、最大回撤）
- 交易统计（胜率、盈亏比）

## 使用说明

### 安装依赖

bash
pip install -r requirements.txt

### 运行回测

bash
python -m src.main

### 配置修改

修改 `configs/default.yaml` 文件来调整：

- 交易策略参数
- 风险控制阈值
- 回测系统参数

### 数据准备

- 将交易记录放在 `tests/data/sample_trades.csv`
- 确保数据格式符合系统要求

## 开发环境

- Python 3.8+
- 主要依赖包：
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - akshare >= 1.0.0
  - PyYAML >= 5.4.0

## 优化路线图

### 1. 回测系统完善

- [ ] 添加关键回测指标
  - 夏普比率(Sharpe Ratio)
  - 最大回撤(Maximum Drawdown)
  - 年化收益率(Annual Return)
  - 信息比率(Information Ratio)
- [ ] 实现交易成本模拟
  - 手续费
  - 印花税
  - 滑点
- [ ] 开发回测可视化模块
  - 收益曲线
  - 持仓分布
  - 交易记录

### 2. 风险控制增强

- [ ] 完善仓位管理
  - 单一股票最大持仓限制
  - 行业敞口控制
  - 整体仓位动态调整
- [ ] 实现风险预警系统
  - 波动率监控
  - 流动性风险控制
  - 集中度风险管理
- [ ] 添加组合风险分析
  - Beta值控制
  - 相关性分析
  - VaR计算

### 3. 策略维度扩展

- [ ] 基本面分析模块
  - 财务指标分析
  - 估值指标评估
  - 行业地位分析
- [ ] 市场情绪分析
  - 资金流向监控
  - 机构持仓跟踪
  - 市场情绪指标
- [ ] 宏观因素整合
  - 行业景气度分析
  - 政策影响评估
  - 市场周期判断

### 4. 数据源优化

- [ ] 多数据源支持
  - Wind接口集成
  - 东方财富数据对接
  - 其他备选数据源
- [ ] 实时行情系统
  - Level-2行情接入
  - 盘口数据分析
  - 实时指标计算
- [ ] 另类数据整合
  - 舆情数据分析
  - 行业研报解析
  - 消息面监控

### 5. 机器学习应用

- [ ] 预测模型开发
  - LSTM价格预测
  - XGBoost选股模型
  - 随机森林分类器
- [ ] 因子挖掘系统
  - 多维度因子库
  - 因子有效性检验
  - 因子组合优化
- [ ] 强化学习策略
  - Q-Learning交易策略
  - Policy Gradient应用
  - 多因子强化学习

### 6. 系统架构优化

- [ ] Web界面开发
  - 策略监控面板
  - 回测结果展示
  - 实时交易界面
- [ ] 性能优化
  - 数据处理加速
  - 内存使用优化
  - 计算效率提升
- [ ] 模块化重构
  - 策略模块解耦
  - 接口标准化
  - 插件系统支持

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目维护者：[Your Name]
- 邮箱：[Your Email]
- 项目主页：[GitHub Repository URL]
