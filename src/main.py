import os
from datetime import datetime, timedelta
from typing import Dict
import pandas as pd

from .core.data_fetcher import DataFetcher
from .core.strategy import Strategy
from .core.backtest import BacktestSystem
from .analysis.performance import PerformanceAnalyzer
from .analysis.visualization import Visualizer
from .utils.config import load_config
from .utils.logger import setup_logger

class StockMorning:
    """主程序类"""
    def __init__(self, config_path: str = 'configs/default.yaml'):
        # 加载配置
        self.config = load_config(config_path)
        self.logger = setup_logger(__name__)
        
        # 初始化组件
        self.data_fetcher = DataFetcher(self.config)
        self.strategy = Strategy(self.config)
        self.backtest = BacktestSystem(self.config, self.strategy)
        self.analyzer = PerformanceAnalyzer()
        self.visualizer = Visualizer()
        
    def run_backtest(self, 
                    start_date: str,
                    end_date: str,
                    stock_list: list = None) -> Dict:
        """运行回测"""
        try:
            # 获取���票列表
            if stock_list is None:
                df_stocks = self.data_fetcher.get_stock_list()
                stock_list = df_stocks['code'].tolist()[:10]  # 默认取前10只股票
            
            # 获取股票数据
            data = {}
            for code in stock_list:
                df = self.data_fetcher.get_stock_data(code, start_date, end_date)
                # 添加数据验证
                if df is not None and not df.empty and all(df.index.notna()):
                    data[code] = df
                else:
                    self.logger.warning(f"股票{code}数据无效或为空，已跳过")
                    continue
            
            if not data:
                self.logger.error("未获取到任何有效的股票数据")
                return {}
            
            # 运行回测
            results = self.backtest.run(data, start_date, end_date)
            
            if not results:
                self.logger.error("回测结果为空")
                return {}
            
            # 分析结果
            report = self.analyzer.generate_report(
                metrics=results,
                trade_history=self.backtest.trade_history,
                daily_positions=self.backtest.daily_positions
            )
            
            if not report:
                self.logger.error("无法生成分析报告")
                return {}
            
            # 保存交易记录
            self.backtest.save_trade_records()
            
            return report
            
        except Exception as e:
            self.logger.error(f"回测过程出错: {str(e)}")
            return {}
            
    def run_live(self):
        """运行实盘交易"""
        # TODO: 实现实盘交易逻辑
        pass

def load_trade_records(file_path: str) -> pd.DataFrame:
    """加载交易记录"""
    try:
        trades_df = pd.read_csv(file_path, encoding='utf-8-sig')
        # 确保股票代码为字符串格式
        trades_df['股票代码'] = trades_df['股票代码'].astype(str).str.zfill(6)  # 补齐6位
        trades_df['交易时间'] = pd.to_datetime(trades_df['交易时间'])
        return trades_df
    except Exception as e:
        print(f"加载交易记录时出错: {str(e)}")
        return pd.DataFrame()

def analyze_trade_records(trades_df: pd.DataFrame) -> Dict:
    """分析实际交易记录的性能指标"""
    if trades_df.empty:
        return {}
        
    # 按日期分组计算每日盈亏和资金占用
    trades_df['日期'] = pd.to_datetime(trades_df['交易时间']).dt.date
    daily_pnl = []
    daily_capital = []  # 记录每日资金占用
    capital = 1000000  # 初始资金
    positions = {}     # 持仓记录
    
    # 按时间顺序处理交易
    for _, trade in trades_df.sort_values('交易时间').iterrows():
        amount = trade['成交价格'] * trade['成交数量']
        cost = trade['手续费'] + trade['印花税']
        
        if trade['交易方向'] == 'BUY':
            # 买入，更新持仓
            if trade['股票代码'] not in positions:
                positions[trade['股票代码']] = {
                    '数量': 0, 
                    '成本': 0,
                    '最新价格': trade['成交价格']
                }
            positions[trade['股票代码']]['数量'] += trade['成交数量']
            positions[trade['股票代码']]['成本'] += amount + cost
            positions[trade['股票代码']]['最新价格'] = trade['成交价格']
            capital -= (amount + cost)
        else:
            # 卖出，计算盈亏
            if trade['股票代码'] in positions:
                avg_cost = positions[trade['股票代码']]['成本'] / positions[trade['股票代码']]['数量']
                profit = (trade['成交价格'] - avg_cost) * trade['成交数量'] - cost
                daily_pnl.append({
                    '日期': trade['日期'],
                    '盈亏': profit
                })
                # 更新持仓和资金
                positions[trade['股票代码']]['数量'] -= trade['成交数量']
                remaining_ratio = positions[trade['股票代码']]['数量'] / (positions[trade['股票代码']]['数量'] + trade['成交数量'])
                positions[trade['股票代码']]['成本'] *= remaining_ratio
                positions[trade['股票代码']]['最新价格'] = trade['成交价格']
                if positions[trade['股票代码']]['数量'] == 0:
                    del positions[trade['股票代码']]
                capital += (amount - cost)
        
        # 记录当日资金占用
        occupied_capital = sum(pos['成本'] for pos in positions.values())
        daily_capital.append({
            '日期': trade['日期'],
            '占用资金': occupied_capital
        })
    
    # 计算当前持仓的浮动盈亏
    total_floating_pnl = 0
    current_occupied_capital = 0
    for code, pos in positions.items():
        avg_cost = pos['成本'] / pos['数量']
        floating_pnl = (pos['最新价格'] - avg_cost) * pos['数量']
        total_floating_pnl += floating_pnl
        current_occupied_capital += pos['成本']
    
    # 计算性能指标
    daily_returns = pd.DataFrame(daily_pnl).groupby('日期')['盈亏'].sum()
    daily_capital_df = pd.DataFrame(daily_capital).groupby('日期')['占用资金'].mean()
    
    # 计算平均资金占用
    avg_occupied_capital = daily_capital_df.mean() if not daily_capital_df.empty else current_occupied_capital
    
    total_days = (trades_df['日期'].max() - trades_df['日期'].min()).days
    total_pnl = daily_returns.sum() + total_floating_pnl
    
    # 计算收益率
    if avg_occupied_capital > 0:
        total_return = total_pnl / avg_occupied_capital * 100
        annual_return = ((1 + total_pnl / avg_occupied_capital) ** (252 / total_days) - 1) * 100
    else:
        total_return = 0
        annual_return = 0
    
    analysis = {
        '总交易次数': len(trades_df),
        '买入次数': len(trades_df[trades_df['交易方向'] == 'BUY']),
        '卖出次数': len(trades_df[trades_df['交易方向'] == 'SELL']),
        '总手续费': trades_df['手续费'].sum(),
        '总印花税': trades_df['印花税'].sum(),
        '交易股票数': len(trades_df['股票代码'].unique()),
        '最早交易日': trades_df['交易时间'].min().strftime('%Y-%m-%d'),
        '最后交易日': trades_df['交易时间'].max().strftime('%Y-%m-%d'),
        # 性能指标
        '已实现盈亏': daily_returns.sum(),
        '浮动盈亏': total_floating_pnl,
        '总盈亏': total_pnl,
        '总收益率': total_return,  # 相对于平均占用资金的收益率
        '年化收益率': annual_return,
        '胜率': (daily_returns > 0).mean() * 100,
        '日均收益': daily_returns.mean(),
        '收益标准差': daily_returns.std(),
        '最大单日盈利': daily_returns.max(),
        '最大单日亏损': daily_returns.min(),
        '盈亏比': abs(daily_returns[daily_returns > 0].mean() / daily_returns[daily_returns < 0].mean()) if len(daily_returns[daily_returns < 0]) > 0 else float('inf'),
        '当前持仓': {code: {'数量': pos['数量'], '成本价': pos['成本']/pos['数量'], '现价': pos['最新价格']} 
                    for code, pos in positions.items()},
        '平均资金占用': avg_occupied_capital,
        '当前资金占用': current_occupied_capital,
    }
    
    return analysis

def main():
    """主程序入口"""
    # 初始化系统
    system = StockMorning()
    
    # 加载历史交易记录
    trades_df = load_trade_records('tests/data/sample_trades.csv')
    
    if trades_df.empty:
        print("未能加载交易记录，请检查文件路径")
        return
    
    # 获取交易的股票代码列表
    test_stocks = trades_df['股票代码'].unique().tolist()
    
    # 获取时间范围
    start_date = trades_df['交易时间'].min().strftime('%Y%m%d')  # 使用strftime直接格式化
    end_date = trades_df['交易时间'].max().strftime('%Y%m%d')    # 使用strftime直接格式化
    
    print(f"\n开始回测 - 时间区间: {start_date} 至 {end_date}")
    print(f"测试股票: {', '.join(test_stocks)}")
    
    # 运行回测
    results = system.run_backtest(start_date, end_date, stock_list=test_stocks)
    
    # 打印结果
    if results:
        print("\n=== 回测系统结果 ===")  # 修改标题
        print("\n回测性能指标:")  # 明确是回测指标
        for key, value in results['performance_metrics'].items():
            print(f"{key}: {value:.4f}")
        
        print("\n回测交易统计:")  # 明确是回测统计
        for key, value in results['trade_statistics'].items():
            print(f"{key}: {value}")
        
        print("\n回测持仓统计:")  # 明确是回测统计
        for key, value in results['position_statistics'].items():
            print(f"{key}: {value}")
        
        # 分析实际交易记录
        trade_analysis = analyze_trade_records(trades_df)
        print("\n=== 实际交易结果 ===")  # 新的标题
        print("\n基础统计:")
        basic_metrics = ['总交易次数', '买入次数', '卖出次数', '总手续费', '总印花税', 
                        '交易股票数', '最早交易日', '最后交易日']
        for key in basic_metrics:
            print(f"{key}: {trade_analysis[key]}")
            
        print("\n性能指标:")
        performance_metrics = ['已实现盈亏', '浮动盈亏', '总盈亏', '总收益率', '年化收益率', '胜率', '日均收益',
                             '收益标准差', '最大单日盈利', '最大单日亏损', '盈亏比']
        for key in performance_metrics:
            if isinstance(trade_analysis[key], float):
                print(f"{key}: {trade_analysis[key]:.2f}")
            else:
                print(f"{key}: {trade_analysis[key]}")
    else:
        print("\n未获取到回测结果，请检查数据���取是否正常。")

if __name__ == "__main__":
    main() 