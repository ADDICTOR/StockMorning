from typing import Dict, List
import pandas as pd
from ..indicators.base import Indicator
from ..utils.logger import setup_logger

class Strategy:
    """交易策略基类"""
    def __init__(self, config: dict):
        self.config = config
        self.indicators: Dict[str, Indicator] = {}
        self.logger = setup_logger(__name__)
        self.weights = config['strategy']['indicators']
        
    def calculate_signals(self, data: pd.DataFrame) -> float:
        """计算综合信号强度"""
        signal_score = 0.0
        signals = {}
        
        for name, indicator in self.indicators.items():
            try:
                # 获取该指标的权重
                weight = self.weights[name]['weight']
                # 计算信号
                signal = indicator.generate_signals(data)
                signals[name] = signal
                # 加权计算
                signal_score += signal * weight
            except Exception as e:
                self.logger.error(f"计算{name}指标信号时出错: {str(e)}")
                
        return signal_score, signals
    
    def should_buy(self, data: pd.DataFrame) -> tuple:
        """买入决策"""
        signal_score, signals = self.calculate_signals(data)
        
        # 生成买入理由
        reasons = []
        for name, signal in signals.items():
            if signal > 0:
                reasons.append(f"{name}指标触发买入")
                
        return signal_score > 0.3, signal_score, "; ".join(reasons)
    
    def should_sell(self, data: pd.DataFrame, 
                   cost_price: float, 
                   current_price: float) -> tuple:
        """卖出决策"""
        # 计算收益率
        returns = (current_price - cost_price) / cost_price
        
        # 止损检查
        if returns <= -self.config['risk']['stop_loss']:
            return True, "触发止损"
            
        # 止盈检查
        if returns >= self.config['risk']['take_profit']:
            return True, "触发止盈"
            
        # 技术指标检查
        signal_score, signals = self.calculate_signals(data)
        if signal_score < -0.3:
            return True, "技术指标转空"
            
        return False, "" 