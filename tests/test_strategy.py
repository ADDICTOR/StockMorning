import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from src.utils.config import load_config

@pytest.fixture
def strategy():
    """创建策略实例"""
    config = load_config('configs/default.yaml')
    return Strategy(config)

@pytest.fixture
def sample_data():
    """创建测试数据"""
    dates = pd.date_range('2023-01-01', '2023-01-10')
    data = pd.DataFrame({
        'open': np.random.uniform(10, 20, len(dates)),
        'high': np.random.uniform(11, 21, len(dates)),
        'low': np.random.uniform(9, 19, len(dates)),
        'close': np.random.uniform(10, 20, len(dates)),
        'volume': np.random.uniform(1000000, 2000000, len(dates))
    }, index=dates)
    return data

def test_calculate_signals(strategy, sample_data):
    """测试信号计算"""
    signal_score, signals = strategy.calculate_signals(sample_data)
    assert isinstance(signal_score, float)
    assert isinstance(signals, dict)

def test_should_buy(strategy, sample_data):
    """测试买入决策"""
    should_buy, score, reason = strategy.should_buy(sample_data)
    assert isinstance(should_buy, bool)
    assert isinstance(score, float)
    assert isinstance(reason, str)

def test_should_sell(strategy, sample_data):
    """测试卖出决策"""
    should_sell, reason = strategy.should_sell(
        sample_data,
        cost_price=15.0,
        current_price=14.0
    )
    assert isinstance(should_sell, bool)
    assert isinstance(reason, str) 