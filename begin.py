import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class TradingStrategy:
    def __init__(self, stop_loss=0.07, take_profit=0.15):
        self.stop_loss = stop_loss  # 止损位，默认7%
        self.take_profit = take_profit  # 止盈位，默认15%
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 计算布林带
        df['BB_middle'] = df['MA20']
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        return df
    
    def generate_signals(self, df):
        """生成交易信号"""
        df['signal'] = 0  # 0表示无信号，1表示买入
        
        # 趋势跟踪信号
        df['MA5_cross_MA20'] = (df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1))
        
        # 布林带信号
        df['price_below_lower'] = df['close'] < df['BB_lower']
        
        # 添加信号原因说明
        df['signal_reason'] = ''
        
        # 生成买入信号
        for i in range(1, len(df)):
            # 同时满足两个条件
            if (df['MA5_cross_MA20'].iloc[i] and 
                df['close'].iloc[i] > df['MA20'].iloc[i] and 
                df['price_below_lower'].iloc[i]):
                df.loc[df.index[i], 'signal'] = 2  # 使用2表示同时满足两个条件
                df.loc[df.index[i], 'signal_reason'] = 'MA5上穿MA20且价格在MA20上方，同时触及布林带下轨'
            
            # 满足MA5上穿MA20
            elif (df['MA5_cross_MA20'].iloc[i] and df['close'].iloc[i] > df['MA20'].iloc[i]):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_reason'] = 'MA5上穿MA20，且价格在MA20上方'
            
            # 满足布林带下轨
            elif df['price_below_lower'].iloc[i]:
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_reason'] = '价格触及布林带下轨'
        
        return df
    
    def apply_risk_management(self, df, entry_price):
        """应用风险管理"""
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            
            # 检查止损
            if entry_price is not None:
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct < -self.stop_loss:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_reason'] = f'触发止损（亏损超过{self.stop_loss*100}%）'
                    entry_price = None
                
                # 检查止盈
                elif loss_pct > self.take_profit:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_reason'] = f'触发止盈（盈利超过{self.take_profit*100}%）'
                    entry_price = None
            
            # 更新入场价格
            if df['signal'].iloc[i] == 1:
                entry_price = current_price
                
        return df, entry_price

    def backtest(self, df):
        """回测策略"""
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        
        entry_price = None
        df, _ = self.apply_risk_management(df, entry_price)
        
        return df

class StockDataFetcher:
    def __init__(self):
        self.stock_list = None
        # 添加缓存
        self.data_cache = {}
        
    def get_stock_list(self):
        """获取A股股票列表"""
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
            self.stock_list = pd.merge(stock_info, market_value_df, on='code', how='left')
            return self.stock_list
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return None

    def get_stock_data(self, stock_code, start_date, end_date):
        """获取单个股票的历史数据"""
        # 使用缓存键
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        try:
            # 获取股票日线数据
            df = ak.stock_zh_a_hist(symbol=stock_code, 
                                  start_date=start_date, 
                                  end_date=end_date,
                                  adjust="qfq")  # qfq表示前复权
            
            # 重命名列以匹配策略要求
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'  # 添加成交额
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 存入缓存
            self.data_cache[cache_key] = df
            return df
        except Exception as e:
            print(f"获取股票{stock_code}数据失败: {e}")
            return None

class StockScreener:
    def __init__(self, min_amount=200000000, min_market_value=20000000000):
        """
        初始化筛选器
        :param min_amount: 最小日成交（元），默认2亿
        :param min_market_value: 最小市值（元），默认200亿
        """
        self.min_amount = min_amount
        self.min_market_value = min_market_value
        
    def screen_stocks(self, stock_data, market_value, stock_code):
        """基础股票筛选"""
        if stock_data is None or stock_data.empty:
            return False
            
        try:
            # 排除科创板股票
            if stock_code.startswith('688') or stock_code.startswith('300'):
                return False
                
            # 检查市值条件（单位：亿元转换为元）
            if market_value * 100000000 < self.min_market_value:
                return False
            
            # 检查平均成交额
            avg_amount = stock_data['amount'].mean()
            if avg_amount < self.min_amount:
                return False
            
            return True
        except Exception as e:
            print(f"筛选过程出错: {e}")
            return False

def main():
    # 记录开始时间
    start_time = datetime.now()
    print(f"程序开始运行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化数据获取器和交易策略
    fetcher = StockDataFetcher()
    strategy = TradingStrategy(stop_loss=0.07, take_profit=0.15)
    screener = StockScreener(min_amount=200000000, min_market_value=20000000000)
    
    # 设置时间范围
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
    
    # 获取股票列表
    stock_list = fetcher.get_stock_list()
    
    # 存储符合条件的股票及其信号
    trading_opportunities = {}
    
    print("开始筛选股票...")
    
    # 使用多线程处理股票分析
    def process_stock(stock_info):
        stock_code, stock_name, market_value = stock_info
        
        stock_data = fetcher.get_stock_data(stock_code, start_date, end_date)
        
        if not screener.screen_stocks(stock_data, market_value, stock_code):
            return None
            
        try:
            result_df = strategy.backtest(stock_data)
            latest_signal = result_df['signal'].iloc[-1]
            
            # 返回所有买入信号（包括同时满足两个条件的信号）
            if latest_signal >= 1:
                signal_strength = "同时满足双重买入条件" if latest_signal == 2 else "买入"
                return {
                    'code': stock_code,
                    'name': stock_name,
                    'signal': signal_strength,
                    'price': result_df['close'].iloc[-1],
                    'date': result_df.index[-1],
                    'market_value': market_value,
                    'daily_amount': stock_data['amount'].iloc[-1],
                    'reason': result_df['signal_reason'].iloc[-1],
                    'signal_strength': latest_signal  # 添加信号强度
                }
        except Exception as e:
            print(f"分析股票{stock_code}时出错: {e}")
        return None

    # 创建股票信息列表
    stock_info_list = [(row['code'], row['name'], row['market_value']) 
                       for _, row in stock_list.iterrows()]
    
    # 使用tqdm包装线程池的执行过程
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(process_stock, stock_info_list),
            total=len(stock_info_list),
            desc="股票分析进度"
        ))
    
    # 过滤掉None结果并转换为字典
    trading_opportunities = {r['code']: r for r in results if r is not None}

    # 在结果输出之前记录结束时间
    end_time = datetime.now()
    run_duration = end_time - start_time
    
    # 输出交易机会
    print("\n=== 买入信号汇总 ===")
    print(f"运行时间: {run_duration}")
    print("代码  名称  信号强度  价格  市值(亿)  日成交额(亿)  买入理由")
    print("-" * 100)

    # 先输出同时满足两个条件的股票
    print("\n*** 同时满足双重买入条件的股票 ***")
    for code, info in trading_opportunities.items():
        if info['signal_strength'] == 2:
            print(f"{code} {info['name']} {info['signal']} "
                  f"{info['price']:.2f} {info['market_value']:.2f} "
                  f"{info['daily_amount']/100000000:.2f} "
                  f"{info['reason']}")

    # 再输出满足单个条件的股票
    print("\n*** 满足单个买入条件的股票 ***")
    for code, info in trading_opportunities.items():
        if info['signal_strength'] == 1:
            print(f"{code} {info['name']} {info['signal']} "
                  f"{info['price']:.2f} {info['market_value']:.2f} "
                  f"{info['daily_amount']/100000000:.2f} "
                  f"{info['reason']}")

    # 保存结果到CSV文件
    if trading_opportunities:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'trading_signals_{current_date}.csv'
        
        # 创建DataFrame并只包含需要的列
        df_opportunities = pd.DataFrame([
            {
                '代码': info['code'],
                '名称': info['name'],
                '信号强度': info['signal'],
                '当前价格': info['price'],
                '市值(亿)': info['market_value'],
                '日成交额(亿)': info['daily_amount']/100000000,
                '买入理由': info['reason']
            }
            for info in trading_opportunities.values()
        ])
        
        # 保存为CSV文件
        df_opportunities.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {filename}")
        print(f"总运行时间: {run_duration}")

if __name__ == "__main__":
    main()
