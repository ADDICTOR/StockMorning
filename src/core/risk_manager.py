from typing import Dict, List
import pandas as pd
from ..utils.logger import setup_logger

class RiskManager:
    """风险管理器"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(__name__)
        
        # 风险控制参数
        self.stop_loss = config['risk']['stop_loss']  # 止损线
        self.take_profit = config['risk']['take_profit']  # 止盈线
        self.max_position_size = config['risk']['max_position_size']  # 单个持仓最大比例
        self.max_drawdown = config['risk']['max_drawdown']  # 最大回撤限制
        
    def check_risk_indicators(self, 
                            positions: Dict[str, 'Position'],
                            current_prices: Dict[str, float]) -> List[str]:
        """检查风险指标"""
        stocks_to_sell = []
        
        # 计算总资产
        total_assets = sum(pos.market_value for pos in positions.values())
        
        for code, position in positions.items():
            if code not in current_prices:
                continue
                
            current_price = current_prices[code]
            cost_price = position.cost_basis
            
            # 计算收益率
            returns = (current_price - cost_price) / cost_price
            
            # 止损检查
            if returns <= -self.stop_loss:
                self.logger.info(f"{code} 触发止损: 收益率 {returns*100:.2f}%")
                stocks_to_sell.append(code)
                continue
            
            # 止盈检查
            if returns >= self.take_profit:
                self.logger.info(f"{code} 触发止盈: 收益率 {returns*100:.2f}%")
                stocks_to_sell.append(code)
                continue
            
            # 检查持仓规模（相对于总资产的比例）
            if total_assets > 0:
                position_ratio = position.market_value / total_assets
                if position_ratio > self.max_position_size:
                    self.logger.info(f"{code} 触发持仓规模限制: 当前持仓比例 {position_ratio*100:.2f}%")
                    stocks_to_sell.append(code)
                    continue
        
        return stocks_to_sell
        
    def calculate_position_size(self,
                              available_capital: float,
                              price: float,
                              signal_strength: float) -> int:
        """计算仓位大小"""
        # 根据信号强度和可用资金计算目标仓位
        target_value = available_capital * signal_strength * self.max_position_size
        
        # 计算可买入的股数（向下取整到100的倍数）
        shares = int(target_value / price / 100) * 100
        
        return shares 