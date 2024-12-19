from .base import Indicator
import pandas as pd
import numpy as np

class MAIndicator(Indicator):
    """移动平均指标"""
    def calculate(self, data: pd.DataFrame) -> dict:
        """计算MA指标"""
        short_window = self.params.get('short_window', 5)
        long_window = self.params.get('long_window', 20)
        
        # 计算短期和长期MA
        ma_short = data['close'].rolling(window=short_window).mean()
        ma_long = data['close'].rolling(window=long_window).mean()
        
        return {
            'MA_SHORT': ma_short,
            'MA_LONG': ma_long
        }

    def generate_signals(self, data: pd.DataFrame) -> float:
        """生成MA交叉信号"""
        indicators = self.calculate(data)
        ma_short = indicators['MA_SHORT']
        ma_long = indicators['MA_LONG']
        
        # 计算金叉死叉
        cross_over = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        cross_under = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))
        
        # 最近是否出现金叉
        if cross_over.iloc[-1]:
            return 1.0
        elif cross_under.iloc[-1]:
            return -1.0
        return 0.0 