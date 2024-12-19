import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.core.backtest import BacktestSystem
from src.core.strategy import Strategy
from src.utils.config import load_config

@pytest.fixture
def backtest_system():
    """创建回测系统实例"""
    config = load_config('configs/default.yaml')
    strategy = Strategy(config)
    return BacktestSystem(config, strategy)

@pytest.fixture
def sample_data():
    """创建测试数据"""
    dates = pd.date_range('2023-01-01', '2023-01-10')
    data = {
        '000001': pd.DataFrame({
            'open': np.random.uniform(10, 20, len(dates)),
            'high': np.random.uniform(11, 21, len(dates)),
            'low': np.random.uniform(9, 19, len(dates)),
            'close': np.random.uniform(10, 20, len(dates)),
            'volume': np.random.uniform(1000000, 2000000, len(dates)),
            'name': ['平安银行'] * len(dates)
        }, index=dates)
    }
    return data

def test_calculate_trade_cost(backtest_system):
    """测试交易成本计算"""
    commission, tax = backtest_system.calculate_trade_cost(
        price=10.0,
        shares=1000,
        direction='BUY'
    )
    assert commission >= 5.0  # 最低佣金
    assert tax == 0  # 买入不收印花税

def test_execute_trade(backtest_system):
    """测试交易执行"""
    result = backtest_system.execute_trade(
        code='000001',
        name='平安银行',
        direction='BUY',
        price=10.0,
        shares=1000,
        timestamp=datetime.now(),
        reason='测试买入'
    )
    assert result is True
    assert len(backtest_system.trade_history) == 1

def test_run_backtest(backtest_system, sample_data):
    """测试回测运行"""
    results = backtest_system.run(
        data=sample_data,
        start_date='2023-01-01',
        end_date='2023-01-10'
    )
    assert isinstance(results, dict)
    assert len(backtest_system.daily_capital) > 0
    assert len(backtest_system.daily_returns) > 0 