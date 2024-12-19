from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from .strategy import Strategy
from .risk_manager import RiskManager
from .position import PositionManager
from ..utils.logger import setup_logger
import os

@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: datetime
    code: str
    name: str
    direction: str
    price: float
    shares: int
    commission: float
    tax: float
    reason: str
    profit: float = 0.0  # 添加profit属性，默认为0

class BacktestSystem:
    """回测系统"""
    def __init__(self, config: dict, strategy: Strategy):
        self.config = config
        self.strategy = strategy
        self.risk_manager = RiskManager(config)
        self.position_manager = PositionManager()
        self.logger = setup_logger(__name__)
        
        # 回测参数
        self.initial_capital = config['backtest']['initial_capital']
        self.current_capital = self.initial_capital
        self.commission_rate = config['backtest']['commission_rate']
        self.tax_rate = config['backtest']['tax_rate']
        self.slippage = config['backtest']['slippage']
        
        # 回测结果
        self.trade_history: List[TradeRecord] = []
        self.daily_capital = [self.initial_capital]
        self.daily_returns = [0.0]
        self.daily_positions = []
        
    def calculate_trade_cost(self, 
                           price: float, 
                           shares: int, 
                           direction: str) -> tuple:
        """计算交易成本"""
        amount = price * shares
        
        # 佣金费率 0.0003 (万三)，最低5元
        commission = max(amount * 0.0003, 5.0)
        
        # 印花税（仅卖出收取，税率 0.001）
        tax = amount * 0.001 if direction == 'SELL' else 0
        
        return commission, tax
        
    def execute_trade(self, 
                     code: str,
                     name: str,
                     direction: str,
                     price: float,
                     shares: int,
                     timestamp: datetime,
                     reason: str) -> bool:
        """执行交易"""
        try:
            # 考虑滑点
            if direction == 'BUY':
                actual_price = price * (1 + self.slippage)
            else:
                actual_price = price * (1 - self.slippage)
                
            # 计算交易成本
            commission, tax = self.calculate_trade_cost(
                actual_price, shares, direction
            )
            total_cost = actual_price * shares + commission + tax
            
            # 执行交易
            profit = 0.0
            if direction == 'BUY':
                if total_cost > self.current_capital:
                    self.logger.warning(f"资金不足，无法买入{code}")
                    return False
                    
                self.current_capital -= total_cost
                self.position_manager.add_position(code, name, shares, actual_price)
                
            else:  # SELL
                realized_pnl = self.position_manager.reduce_position(
                    code, shares, actual_price
                )
                if realized_pnl is None:
                    return False
                    
                self.current_capital += actual_price * shares - commission - tax
                profit = realized_pnl - commission - tax
                
            # 记录交易
            self.trade_history.append(
                TradeRecord(
                    timestamp=timestamp,
                    code=code,
                    name=name,
                    direction=direction,
                    price=actual_price,
                    shares=shares,
                    commission=commission,
                    tax=tax,
                    reason=reason,
                    profit=profit  # 设置profit
                )
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"执行交易时出错: {str(e)}")
            return False
        
    def run(self, 
           data: Dict[str, pd.DataFrame],
           start_date: str,
           end_date: str) -> Dict:
        """运行回测"""
        try:
            # 验证输入数据
            if not data:
                self.logger.error("没有输入数据")
                return {}
            
            # 确保所有数据都有必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for code, df in data.items():
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    self.logger.error(f"股票{code}缺少必要的列: {missing_cols}")
                    return {}
            
            # 初始化回测状态
            self.current_capital = self.initial_capital  # 当前资金
            self.daily_capital = [self.initial_capital]  # 初始资金
            self.daily_returns = [0.0]  # 第一天收益率为0
            self.daily_positions = {}  # 清空持仓记录
            self.trade_history = []  # 清空交易记录
            
            # 获取所有交易日期
            dates = pd.date_range(start_date, end_date)
            
            # 记录上一日总资产，用于计算收益率
            previous_total_assets = self.initial_capital
            
            for date in dates:
                date_str = date.strftime('%Y-%m-%d')
                
                # 更新当日价格
                current_prices = {}
                for code, df in data.items():
                    if date_str in df.index:
                        current_prices[code] = df.loc[date_str, 'close']
                
                # 更新持仓市值
                self.position_manager.update_prices(current_prices)
                
                # 计算当前总资产（现金 + 持仓市值）
                total_assets = self.current_capital + sum(
                    pos.market_value for pos in self.position_manager.positions.values()
                )
                
                # 计算日收益率
                daily_return = (total_assets / previous_total_assets - 1) if previous_total_assets > 0 else 0
                
                # 检查风险指标
                stocks_to_sell = self.risk_manager.check_risk_indicators(
                    self.position_manager.positions,
                    current_prices
                )
                
                # 执行卖出
                for code in stocks_to_sell:
                    if code in self.position_manager.positions:
                        pos = self.position_manager.positions[code]
                        self.execute_trade(
                            code=code,
                            name=pos.name,
                            direction='SELL',
                            price=current_prices[code],
                            shares=pos.shares,
                            timestamp=date,
                            reason='风险控制卖出'
                        )
                
                # 寻找买入机会
                for code, df in data.items():
                    if date_str not in df.index:
                        continue
                    
                    # 获取历史数据用于策略分析
                    hist_data = df.loc[:date_str].copy()
                    
                    # 检查是否应该买入
                    should_buy, signal_score, reason = self.strategy.should_buy(hist_data)
                    
                    if should_buy:
                        # 计算购买股数
                        price = current_prices[code]
                        available_capital = self.current_capital * 0.95  # 预留5%现金
                        max_shares = int(available_capital / (price * (1 + self.commission_rate)))
                        
                        # 根据信号强度调整购买数量
                        shares = int(max_shares * signal_score)
                        
                        if shares > 0:
                            self.execute_trade(
                                code=code,
                                name=df['name'].iloc[0] if 'name' in df else code,
                                direction='BUY',
                                price=price,
                                shares=shares,
                                timestamp=date,
                                reason=reason
                            )
                
                # 更新每日状态
                self.daily_capital.append(total_assets)
                self.daily_returns.append(daily_return)
                self.daily_positions[date_str] = {
                    code: {
                        'shares': pos.shares,
                        'market_value': pos.market_value,
                        'cost': pos.cost_basis,
                        'unrealized_pnl': pos.get_unrealized_pnl(),
                        'unrealized_pnl_pct': (pos.price / pos.cost_basis - 1) * 100 if pos.cost_basis > 0 else 0
                    }
                    for code, pos in self.position_manager.positions.items()
                }
                
                # 更新上一日总资产
                previous_total_assets = total_assets
                
                # 输出每日状态日志
                self.logger.debug(f"日期: {date_str}, 总资产: {total_assets:.2f}, "
                                f"收益率: {daily_return*100:.2f}%, "
                                f"持仓数: {len(self.position_manager.positions)}")
            
            return self.calculate_metrics()
            
        except Exception as e:
            self.logger.error(f"运行回测时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def calculate_metrics(self) -> Dict:
        """计算回测指标"""
        try:
            # 确保数据长度一致
            if len(self.daily_returns) != len(self.daily_capital):
                self.logger.warning("日收益率和日资产数据长度不一致，进行调整")
                # 如果收益率少一个，添加一个初始值0
                if len(self.daily_returns) < len(self.daily_capital):
                    self.daily_returns.insert(0, 0.0)
                # 如果收益率多一个，删除最后一个
                elif len(self.daily_returns) > len(self.daily_capital):
                    self.daily_returns.pop()
            
            returns = pd.Series(self.daily_returns)
            
            metrics = {
                'total_return': (self.daily_capital[-1] - self.initial_capital) / self.initial_capital,
                'annual_return': self._calculate_annual_return(),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(),
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor()
            }
            
            # 添加一些统计信息
            metrics.update({
                'total_trading_days': len(self.daily_capital),
                'final_capital': self.daily_capital[-1],
                'avg_daily_return': returns.mean(),
                'return_volatility': returns.std()
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {str(e)}")
            return {}

    def _calculate_annual_return(self) -> float:
        """计算年化收益率"""
        if not self.daily_capital:
            return 0.0
        total_days = len(self.daily_capital)
        if total_days < 1:
            return 0.0
        total_return = (self.daily_capital[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / total_days) - 1
        return annual_return

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        if returns.empty:
            return 0.0
        risk_free_rate = 0.03  # 假设无风险利率3%
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def _calculate_max_drawdown(self) -> float:
        """计算最大回"""
        if not self.daily_capital:
            return 0.0
        capital_series = pd.Series(self.daily_capital)
        rolling_max = capital_series.expanding().max()
        drawdowns = (capital_series - rolling_max) / rolling_max
        return abs(drawdowns.min())

    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.trade_history:
            return 0.0
        winning_trades = sum(1 for trade in self.trade_history if trade.profit > 0)
        return winning_trades / len(self.trade_history)

    def _calculate_profit_factor(self) -> float:
        """计算盈亏比"""
        if not self.trade_history:
            return 0.0
        total_profit = sum(trade.profit for trade in self.trade_history if trade.profit > 0)
        total_loss = abs(sum(trade.profit for trade in self.trade_history if trade.profit < 0))
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        return total_profit / total_loss

    def _calculate_trade_statistics(self) -> Dict:
        """计算交易统计"""
        return {
            'total_trades': len(self.trade_history),
            'winning_trades': sum(1 for trade in self.trade_history if trade.profit > 0),
            'losing_trades': sum(1 for trade in self.trade_history if trade.profit < 0)
        }

    def _calculate_position_statistics(self) -> Dict:
        """计算持仓统计"""
        return {
            'current_positions': len(self.daily_positions.get(list(self.daily_positions.keys())[-1], [])) if self.daily_positions else 0,
            'average_position_size': np.mean([len(pos) for pos in self.daily_positions.values()]) if self.daily_positions else 0
        }

    def _get_trading_dates(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """获取所有交易日期"""
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        return sorted(list(all_dates))

    def _execute_trades(self, signals: Dict, current_data: Dict, date: str):
        """执行交易"""
        # TODO: 实现交易执行逻辑
        pass

    def _record_daily_state(self, date: str):
        """记录每日状态"""
        try:
            # 计算当前总资产
            total_assets = self.current_capital + sum(
                pos.market_value for pos in self.position_manager.positions.values()
            )
            
            # 计算日收益率（相对于前一天的总资产）
            if len(self.daily_capital) > 0:
                daily_return = (total_assets / self.daily_capital[-1]) - 1
            else:
                daily_return = 0.0
            
            # 记录数据
            self.daily_capital.append(total_assets)
            self.daily_returns.append(daily_return)
            
            # 记录持仓
            self.daily_positions[date] = {
                code: pos.shares for code, pos in self.position_manager.positions.items()
            }
            
        except Exception as e:
            self.logger.error(f"记录每日状态时出错: {str(e)}")

    def save_trade_records(self, output_dir: str = 'data/output/backtest'):
        """保存交易记录到CSV文件"""
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建交易记录DataFrame
            trades_data = []
            for trade in self.trade_history:
                trades_data.append({
                    '交易时间': trade.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    '股票代码': trade.code,
                    '股票名称': trade.name,
                    '交易方向': trade.direction,
                    '成交价格': trade.price,
                    '成交数量': trade.shares,
                    '成交金额': trade.price * trade.shares,
                    '手续费': trade.commission,
                    '印花税': trade.tax,
                    '交易原因': trade.reason
                })
            
            # 创建DataFrame并保存
            trades_df = pd.DataFrame(trades_data)
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f'trade_records_{current_time}.csv')
            trades_df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"交易记录已保存到: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"保存交易记录时出错: {str(e)}")
            return None