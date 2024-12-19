from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Indicator(ABC):
    """指标基类"""
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算指标"""
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        pass 