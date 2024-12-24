# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, List
from ..utils.logger import setup_logger

class TradeEvaluator:
    """交易评估器"""
    def __init__(self):
        self.logger = setup_logger(__name__)
        
    def evaluate_trade_timing(self, trade_df: pd.DataFrame, stock_data: Dict[str, pd.DataFrame]) -> Dict:
        """评估交易时机"""
        timing_analysis = []
        
        for _, trade in trade_df.iterrows():
            code = trade['股票代码']
            if code not in stock_data:
                continue
                
            price_data = stock_data[code]
            trade_date = pd.to_datetime(trade['交易时间']).strftime('%Y-%m-%d')
            trade_price = trade['成交价格']
            
            # 获取交易日前后10个交易日的数据
            trade_idx = price_data.index.get_loc(trade_date)
            window_start = max(0, trade_idx - 10)
            window_end = min(len(price_data), trade_idx + 11)
            window_data = price_data.iloc[window_start:window_end]
            
            # 分析交易时机
            if trade['交易方向'] == 'BUY':
                min_price = window_data['low'].min()
                timing_score = 1 - (trade_price - min_price) / (window_data['high'].max() - min_price)
            else:  # SELL
                max_price = window_data['high'].max()
                timing_score = (trade_price - window_data['low'].min()) / (max_price - window_data['low'].min())
            
            timing_analysis.append({
                '交易时间': trade_date,
                '股票代码': code,
                '交易方向': trade['交易方向'],
                '成交价格': trade_price,
                '时机评分': timing_score,
                '窗口最高价': window_data['high'].max(),
                '窗口最低价': window_data['low'].min(),
                '建议操作': self._get_timing_suggestion(timing_score)
            })
        
        return pd.DataFrame(timing_analysis)
    
    def _get_timing_suggestion(self, score: float) -> str:
        """根据时机评分给出建议"""
        if score >= 0.8:
            return "时机选择非常好"
        elif score >= 0.6:
            return "时机选择较好"
        elif score >= 0.4:
            return "时机选择一般"
        elif score >= 0.2:
            return "时机选择欠佳，建议优化"
        else:
            return "时机选择不佳，建议改进" 