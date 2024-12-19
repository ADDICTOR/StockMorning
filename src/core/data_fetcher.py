import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

class DataFetcher:
    """数据获取基类"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_stock_data(self, 
                      code: str, 
                      start_date: str, 
                      end_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            # 获取股票数据
            df = ak.stock_zh_a_hist(symbol=code, 
                                  start_date=start_date,
                                  end_date=end_date,
                                  adjust="qfq")
            
            # 重命名列
            df.columns = ['date', 'open', 'close', 'high', 'low', 
                         'volume', 'amount', 'amplitude', 'pct_chg', 
                         'change', 'turnover']
            
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票{code}数据时出错: {str(e)}")
            return None

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            # 获取A股列表
            stock_list = ak.stock_zh_a_spot_em()
            
            # 筛选需要的字段
            stock_list = stock_list[['代码', '名称', '总市值']]
            stock_list.columns = ['code', 'name', 'market_value']
            
            return stock_list
            
        except Exception as e:
            self.logger.error(f"获取股票列表时出错: {str(e)}")
            return pd.DataFrame()

    def get_trading_dates(self, 
                         start_date: str, 
                         end_date: str) -> List[str]:
        """获取交易日历"""
        try:
            # 获取交易日历
            calendar = ak.tool_trade_date_hist_sina()
            calendar = pd.to_datetime(calendar['trade_date'])
            
            # 筛选日期范围
            mask = (calendar >= pd.to_datetime(start_date)) & \
                   (calendar <= pd.to_datetime(end_date))
            
            return calendar[mask].strftime('%Y-%m-%d').tolist()
            
        except Exception as e:
            self.logger.error(f"获取交易日历时出错: {str(e)}")
            return [] 