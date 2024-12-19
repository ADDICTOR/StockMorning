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
                if df is not None:
                    data[code] = df
            
            if not data:
                self.logger.error("未获取到任何股票数据")
                return {}
            
            # 运行回测
            results = self.backtest.run(data, start_date, end_date)
            
            # 分析结果
            report = self.analyzer.generate_report(
                results,
                self.backtest.trade_history,
                self.backtest.daily_positions
            )
            
            # 生成图表
            dates = pd.date_range(start_date, end_date)
            
            # 创建输出目录
            output_dir = 'data/output/backtest'
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存图表
            self.visualizer.plot_performance(
                self.backtest.daily_capital,
                self.backtest.daily_returns,
                dates,
                os.path.join(output_dir, 'performance.png')
            )
            
            self.visualizer.plot_positions(
                self.backtest.daily_positions,
                dates,
                os.path.join(output_dir, 'positions.png')
            )
            
            # 保存交易记录
            trades_df = pd.DataFrame([vars(trade) for trade in self.backtest.trade_history])
            trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
            
            return report
            
        except Exception as e:
            self.logger.error(f"回测过程出错: {str(e)}")
            return {}
            
    def run_live(self):
        """运行实盘交易"""
        # TODO: 实现实盘交易逻辑
        pass

def main():
    """主程序入口"""
    # 初始化系统
    system = StockMorning()
    
    # 设置回测参数
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # 运行回测
    results = system.run_backtest(start_date, end_date)
    
    # 打印结果
    if results:
        print("\n=== 回测结果 ===")
        print("\n性能指标:")
        for key, value in results['performance_metrics'].items():
            print(f"{key}: {value:.4f}")
            
        print("\n交易统计:")
        for key, value in results['trade_statistics'].items():
            print(f"{key}: {value}")
            
        print("\n持仓统计:")
        for key, value in results['position_statistics'].items():
            print(f"{key}: {value}")
    
if __name__ == "__main__":
    main() 