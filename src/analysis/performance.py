import pandas as pd
import numpy as np
from typing import List, Dict
from ..utils.logger import setup_logger

class PerformanceAnalyzer:
    """绩效分析器"""
    def __init__(self):
        self.logger = setup_logger(__name__)
        
    def calculate_metrics(self, 
                         daily_capital: List[float],
                         daily_returns: List[float],
                         risk_free_rate: float = 0.03) -> Dict:
        """计算绩效指标"""
        try:
            # 转换为numpy数组
            returns = np.array(daily_returns)
            
            # 计算累计收益
            total_return = (daily_capital[-1] / daily_capital[0]) - 1
            
            # 计算年化收益
            days = len(daily_returns)
            annual_return = (1 + total_return) ** (252/days) - 1
            
            # 计算波动率
            volatility = np.std(returns) * np.sqrt(252)
            
            # 计算夏普比率
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            
            # 计算最大回撤
            cumulative = np.array(daily_capital) / daily_capital[0]
            max_drawdown = 0
            peak = cumulative[0]
            
            for value in cumulative[1:]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # 计算胜率
            winning_days = sum(1 for r in returns if r > 0)
            win_rate = winning_days / len(returns)
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
        except Exception as e:
            self.logger.error(f"计算绩效指标时出错: {str(e)}")
            return {}
            
    def generate_report(self, 
                       metrics: Dict,
                       trade_history: List,
                       daily_positions: List[pd.DataFrame]) -> Dict:
        """生成分析报告"""
        try:
            # 交易统计
            trades_df = pd.DataFrame([vars(trade) for trade in trade_history])
            
            if not trades_df.empty:
                # 计算每笔交易的盈亏
                trades_df['profit'] = np.where(
                    trades_df['direction'] == 'SELL',
                    trades_df['price'] * trades_df['shares'] - \
                    trades_df['commission'] - trades_df['tax'],
                    -(trades_df['price'] * trades_df['shares'] + \
                    trades_df['commission'] + trades_df['tax'])
                )
                
                # 统计交易信息
                total_trades = len(trades_df)
                profitable_trades = sum(trades_df['profit'] > 0)
                total_profit = trades_df['profit'].sum()
                
                trade_stats = {
                    'total_trades': total_trades,
                    'profitable_trades': profitable_trades,
                    'trade_win_rate': profitable_trades / total_trades,
                    'total_profit': total_profit,
                    'avg_profit_per_trade': total_profit / total_trades
                }
            else:
                trade_stats = {}
            
            # 持仓分析
            position_stats = {}
            if daily_positions:
                latest_positions = daily_positions[-1]
                if not latest_positions.empty:
                    position_stats = {
                        'total_positions': len(latest_positions),
                        'total_market_value': latest_positions['市值'].sum(),
                        'avg_position_return': latest_positions['收益率'].mean()
                    }
            
            return {
                'performance_metrics': metrics,
                'trade_statistics': trade_stats,
                'position_statistics': position_stats
            }
            
        except Exception as e:
            self.logger.error(f"生成分析报告时出错: {str(e)}")
            return {} 