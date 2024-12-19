from typing import Dict, List
import pandas as pd
from ..utils.logger import setup_logger

class RiskManager:
    """风险管理器"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(__name__)
        
        # 风险控制参数
        self.stop_loss = config['risk']['stop_loss']
        self.take_profit = config['risk']['take_profit']
        self.max_position_per_stock = config['risk']['max_position_per_stock']
        self.max_position_total = config['risk']['max_position_total']
        
    def check_position_limit(self, 
                           current_positions: Dict[str, Dict],
                           total_capital: float,
                           stock_code: str,
                           planned_amount: float) -> bool:
        """检查持仓限制"""
        # 检查单个股票持仓限制
        if planned_amount / total_capital > self.max_position_per_stock:
            self.logger.warning(f"{stock_code}计划持仓超过单股限制")
            return False
            
        # 计算当前总持仓
        current_total = sum(pos['market_value'] 
                          for pos in current_positions.values())
        
        # 检查总持仓限制
        if (current_total + planned_amount) / total_capital > self.max_position_total:
            self.logger.warning("总持仓超过限制")
            return False
            
        return True
        
    def check_risk_indicators(self, 
                            positions: Dict[str, Dict],
                            current_prices: Dict[str, float]) -> List[str]:
        """检查风险指标"""
        stocks_to_sell = []
        
        for code, position in positions.items():
            if code not in current_prices:
                continue
                
            current_price = current_prices[code]
            cost_price = position['cost']
            returns = (current_price - cost_price) / cost_price
            
            # 检查止损
            if returns <= -self.stop_loss:
                self.logger.warning(f"{code}触发止损")
                stocks_to_sell.append(code)
                continue
                
            # 检查止盈
            if returns >= self.take_profit:
                self.logger.info(f"{code}触发止盈")
                stocks_to_sell.append(code)
                
        return stocks_to_sell
        
    def calculate_position_size(self, 
                              total_capital: float,
                              current_price: float,
                              signal_strength: float) -> int:
        """计算仓位大小"""
        # 基础仓位是总资金的5%
        base_position = total_capital * 0.05
        
        # 根据信号强度调整仓位(0.3-1.0)
        adjusted_position = base_position * (signal_strength / 0.3)
        
        # 确保不超过单股持仓限制
        max_position = total_capital * self.max_position_per_stock
        position = min(adjusted_position, max_position)
        
        # 计算股数
        shares = int(position / current_price / 100) * 100
        
        return shares 