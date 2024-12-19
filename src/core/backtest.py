from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from .strategy import Strategy
from .risk_manager import RiskManager
from .position import PositionManager
from ..utils.logger import setup_logger

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
        # 计算佣金
        commission = price * shares * self.commission_rate
        commission = max(commission, 5.0)  # 最低佣金5元
        
        # 计算印花税（仅卖出收取）
        tax = price * shares * self.tax_rate if direction == 'SELL' else 0
        
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
                reason=reason
            )
        )
        
        return True
        
    def run(self, 
           data: Dict[str, pd.DataFrame],
           start_date: str,
           end_date: str) -> dict:
        """运行回测"""
        dates = pd.date_range(start_date, end_date)
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # 更新持仓价格
            current_prices = {}
            for code, df in data.items():
                if date_str in df.index:
                    current_prices[code] = df.loc[date_str, 'close']
            
            self.position_manager.update_prices(current_prices)
            
            # 检查风险指标
            stocks_to_sell = self.risk_manager.check_risk_indicators(
                self.position_manager.positions,
                current_prices
            )
            
            # 执行卖出
            for code in stocks_to_sell:
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
                    
                # 检查是否应该买入
                should_buy, signal_score, reason = self.strategy.should_buy(
                    df.loc[:date_str]
                )
                
                if should_buy:
                    # 计算购买股数
                    price = current_prices[code]
                    shares = self.risk_manager.calculate_position_size(
                        self.current_capital,
                        price,
                        signal_score
                    )
                    
                    if shares > 0:
                        self.execute_trade(
                            code=code,
                            name=df['name'].iloc[0],
                            direction='BUY',
                            price=price,
                            shares=shares,
                            timestamp=date,
                            reason=reason
                        )
            
            # 记录每日数据
            total_assets = self.current_capital + sum(
                pos.market_value for pos in self.position_manager.positions.values()
            )
            self.daily_capital.append(total_assets)
            
            # 计算日收益率
            daily_return = (total_assets / self.daily_capital[-2] - 1) \
                if len(self.daily_capital) > 1 else 0
            self.daily_returns.append(daily_return)
            
            # 记录持仓信息
            self.daily_positions.append(
                self.position_manager.get_position_summary()
            )
            
        return self.calculate_metrics() 