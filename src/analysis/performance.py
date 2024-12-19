import pandas as pd
import numpy as np
from typing import List, Dict
from ..utils.logger import setup_logger

class PerformanceAnalyzer:
    """性能分析器"""
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
                       daily_positions: Dict) -> Dict:
        """生成分析报告"""
        try:
            if not metrics:
                self.logger.warning("没有可用的回测指标")
                return {}

            report = {
                'performance_metrics': {
                    '总收益率': metrics.get('total_return', 0) * 100,
                    '年化收益率': metrics.get('annual_return', 0) * 100,
                    '夏普比率': metrics.get('sharpe_ratio', 0),
                    '最大回撤': metrics.get('max_drawdown', 0) * 100,
                    '胜率': metrics.get('win_rate', 0) * 100,
                    '盈亏比': metrics.get('profit_factor', 0),
                    '交易天数': metrics.get('total_trading_days', 0),
                    '日均收益率': metrics.get('avg_daily_return', 0) * 100,
                    '收益波动率': metrics.get('return_volatility', 0) * 100
                },
                'trade_statistics': self._calculate_trade_statistics(trade_history),
                'position_statistics': self._calculate_position_statistics(daily_positions)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成分析报告时出错: {str(e)}")
            return {}

    def _calculate_trade_statistics(self, trade_history: List) -> Dict:
        """计算交易统计"""
        try:
            if not trade_history:
                return {
                    '总交易次数': 0,
                    '买入次数': 0,
                    '卖出次数': 0,
                    '平均交易金额': 0,
                    '总手续费': 0,
                    '总印花税': 0
                }

            trades_df = pd.DataFrame([{
                '方向': t.direction,
                '金额': t.price * t.shares,
                '手续费': t.commission,
                '印花税': t.tax
            } for t in trade_history])

            return {
                '总交易次数': len(trades_df),
                '买入次数': len(trades_df[trades_df['方向'] == 'BUY']),
                '卖出次数': len(trades_df[trades_df['方向'] == 'SELL']),
                '平均交易金额': trades_df['金额'].mean(),
                '总手续费': trades_df['手续费'].sum(),
                '总印花税': trades_df['印花税'].sum()
            }
            
        except Exception as e:
            self.logger.error(f"计算交易统计时出错: {str(e)}")
            return {}

    def _calculate_position_statistics(self, daily_positions: Dict) -> Dict:
        """计算持仓统计"""
        try:
            if not daily_positions:
                return {
                    '最大持仓股票数': 0,
                    '平均持仓股票数': 0,
                    '持仓频率': 0
                }

            position_counts = [len(pos) for pos in daily_positions.values()]
            total_days = len(daily_positions)
            days_with_position = sum(1 for count in position_counts if count > 0)

            return {
                '最大持仓股票数': max(position_counts, default=0),
                '平均持仓股票数': sum(position_counts) / total_days if total_days > 0 else 0,
                '持仓频率': days_with_position / total_days if total_days > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"计算持仓统计时出错: {str(e)}")
            return {} 