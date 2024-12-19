import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class DataFetcher:
    """数据获取基类"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # 添加数据缓存
        self.data_cache = {}
        
    def get_stock_data(self, 
                      code: str, 
                      start_date: str, 
                      end_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            # 确保日期格式正确
            start_date = start_date.replace('-', '')
            end_date = end_date.replace('-', '')
            
            # 使用缓存键
            cache_key = f"{code}_{start_date}_{end_date}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # 获取数据
            df = ak.stock_zh_a_hist(symbol=code, 
                                  start_date=start_date,
                                  end_date=end_date,
                                  adjust="qfq")
            
            # 确保数据不为空
            if df is None or df.empty:
                self.logger.warning(f"股票{code}在指定时间段内没有数据")
                return None
            
            # 重命名列以匹配预期格式
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 确保数据类型正确
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # 存入缓存
            self.data_cache[cache_key] = df
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票{code}数据时出错: {str(e)}")
            return None

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            # 使用并行处理来提高性能
            with ThreadPoolExecutor() as executor:
                stock_info_future = executor.submit(ak.stock_info_a_code_name)
                stock_real_time_future = executor.submit(ak.stock_zh_a_spot_em)
                
                stock_info = stock_info_future.result()
                stock_real_time = stock_real_time_future.result()
            
            # 提取需要的字段
            market_value_df = stock_real_time[['代码', '总市值']]
            market_value_df.columns = ['code', 'market_value']
            
            # 合并股票信息和市值数据
            stock_list = pd.merge(stock_info, market_value_df, on='code', how='left')
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

    def batch_get_stock_data(self, 
                           stock_codes: List[str], 
                           start_date: str, 
                           end_date: str, 
                           show_progress: bool = True) -> Dict[str, pd.DataFrame]:
        """批量获取股票数据"""
        results = {}
        
        def process_stock(code: str) -> Optional[pd.DataFrame]:
            return self.get_stock_data(code, start_date, end_date)
        
        # 使用线程池并显示进度条
        with ThreadPoolExecutor(max_workers=20) as executor:
            if show_progress:
                futures = {executor.submit(process_stock, code): code for code in stock_codes}
                for future in tqdm(futures, desc="获取股票数据"):
                    code = futures[future]
                    try:
                        data = future.result()
                        if data is not None:
                            results[code] = data
                    except Exception as e:
                        self.logger.error(f"处理股票{code}时出错: {str(e)}")
            else:
                futures = {executor.submit(process_stock, code): code for code in stock_codes}
                for future in futures:
                    code = futures[future]
                    try:
                        data = future.result()
                        if data is not None:
                            results[code] = data
                    except Exception as e:
                        self.logger.error(f"处理股票{code}时出错: {str(e)}")
        
        return results 