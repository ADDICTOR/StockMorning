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
        """判断是否应该买入"""
        try:
            # 计算技术指标
            ma5 = data['close'].rolling(window=5).mean()
            ma20 = data['close'].rolling(window=20).mean()
            
            # 计算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 获取最新值
            current_ma5 = ma5.iloc[-1]
            current_ma20 = ma20.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # 生成买入信号
            ma_cross = current_ma5 > current_ma20
            rsi_oversold = current_rsi < 30
            
            # 计算信号强度 (0-1)
            signal_strength = 0.0
            reasons = []
            
            if ma_cross:
                signal_strength += 0.5
                reasons.append("MA5上穿MA20")
            
            if rsi_oversold:
                signal_strength += 0.5
                reasons.append("RSI超卖")
            
            # 返回买入决策
            should_buy = signal_strength > 0.3
            return should_buy, signal_strength, "；".join(reasons) if reasons else "无买入信号"
            
        except Exception as e:
            self.logger.error(f"计算买入信号时出错: {str(e)}")
            return False, 0.0, "计算错误"
    
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