import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class TradingStrategy:
    def __init__(self):
        # 更新权重分配
        self.weights = {
            'ma_cross': 0.12,    # MA交叉
            'bollinger': 0.12,   # 布林带
            'rsi': 0.08,        # RSI
            'macd': 0.12,       # MACD
            'kdj': 0.08,        # KDJ
            'volume': 0.12,      # 成交量
            'dmi': 0.08,        # DMI
            'cci': 0.05,        # CCI
            'trix': 0.05,       # TRIX
            'obv': 0.06,        # OBV（能量潮指标）
            'wr': 0.06,         # WR（威廉指标）
            'emv': 0.06         # EMV（简易波动指标）
        }
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 保留原有指标计算
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 布林带计算
        df['BB_middle'] = df['MA20']
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # RSI计算
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['RSI'] = calculate_rsi(df['close'])
        
        # 添加MACD指标
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # 添加KDJ指标
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        # 计算成交量变化
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA5']
        
        # 添加新的技术指标
        
        # 添加DMI指标
        df['TR'] = pd.DataFrame({
            'HL': df['high'] - df['low'],
            'HC': abs(df['high'] - df['close'].shift(1)),
            'LC': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        
        df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                             np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                             np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        df['+DI'] = 100 * (df['+DM'].rolling(window=14).sum() / df['TR'].rolling(window=14).sum())
        df['-DI'] = 100 * (df['-DM'].rolling(window=14).sum() / df['TR'].rolling(window=14).sum())
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = df['DX'].rolling(window=14).mean()
        
        # 添加CCI指标
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # 添加TRIX指标
        def calc_trix(close, n=12):
            tr = close.ewm(span=n, adjust=False).mean()
            tr2 = tr.ewm(span=n, adjust=False).mean()
            tr3 = tr2.ewm(span=n, adjust=False).mean()
            return (tr3 - tr3.shift(1)) / tr3.shift(1) * 100
        
        df['TRIX'] = calc_trix(df['close'])
        
        # 添加OBV（能量潮指标）
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(window=20).mean()
        
        # 添加威廉指标(WR)
        def calculate_wr(data, periods=14):
            high = data['high'].rolling(periods).max()
            low = data['low'].rolling(periods).min()
            wr = -100 * (high - data['close']) / (high - low)
            return wr
        
        df['WR'] = calculate_wr(df)
        
        # 添加EMV（简易波动指标）
        def calculate_emv(data):
            dm = ((data['high'] + data['low']) / 2) - ((data['high'].shift(1) + data['low'].shift(1)) / 2)
            br = (data['volume'] / 100000000) / ((data['high'] - data['low']))
            emv = dm / br 
            return emv
            
        df['EMV'] = calculate_emv(df)
        df['EMV_MA'] = df['EMV'].rolling(window=14).mean()
        
        return df
    
    def calculate_signal_score(self, df, i):
        """计算信号得分"""
        scores = {}
        
        # MA交叉得分 - 考虑交叉角度
        ma5_slope = (df['MA5'].iloc[i] - df['MA5'].iloc[i-1]) / df['MA5'].iloc[i-1]
        ma20_slope = (df['MA20'].iloc[i] - df['MA20'].iloc[i-1]) / df['MA20'].iloc[i-1]
        scores['ma_cross'] = 1 if (df['MA5'].iloc[i] > df['MA20'].iloc[i] and 
                                 df['MA5'].iloc[i-1] <= df['MA20'].iloc[i-1] and
                                 ma5_slope > ma20_slope) else 0
        
        # 布林带得分 - 优化计算方法
        bb_position = (df['close'].iloc[i] - df['BB_lower'].iloc[i]) / (df['BB_upper'].iloc[i] - df['BB_lower'].iloc[i])
        scores['bollinger'] = max(0, 1 - bb_position * 2) if bb_position < 0.5 else 0
        
        # RSI得分 - 细化区间
        rsi = df['RSI'].iloc[i]
        if rsi < 30:
            scores['rsi'] = 1
        elif rsi < 40:
            scores['rsi'] = 0.5
        else:
            scores['rsi'] = 0
        
        # MACD得分 - 考虑柱状图强度
        macd_strength = df['MACD_Hist'].iloc[i] / df['close'].iloc[i] * 100
        scores['macd'] = min(1, max(0, macd_strength / 0.5)) if df['MACD_Hist'].iloc[i] > 0 else 0
        
        # KDJ得分 - 优化超卖判断
        k = df['K'].iloc[i]
        d = df['D'].iloc[i]
        if k < 20:
            scores['kdj'] = 1
        elif k < 30:
            scores['kdj'] = 0.5
        else:
            scores['kdj'] = 0
        
        # 成交量得分 - 考虑连续性
        volume_ratio = df['Volume_Ratio'].iloc[i]
        volume_trend = (df['volume'].iloc[i] > df['volume'].iloc[i-1])
        scores['volume'] = min(1, volume_ratio / 2) if (volume_ratio > 1 and volume_trend) else 0
        
        # DMI得分 - 优化判断标准
        adx = df['ADX'].iloc[i]
        plus_di = df['+DI'].iloc[i]
        minus_di = df['-DI'].iloc[i]
        if adx > 25 and plus_di > minus_di:
            scores['dmi'] = min(1, (adx - 25) / 25)
        else:
            scores['dmi'] = 0
        
        # CCI得分 - 优化区间判断
        cci = df['CCI'].iloc[i]
        if -100 <= cci <= -80:
            scores['cci'] = 1
        elif -80 < cci <= -50:
            scores['cci'] = 0.5
        else:
            scores['cci'] = 0
        
        # TRIX得分 - 考虑趋势变化
        trix = df['TRIX'].iloc[i]
        trix_prev = df['TRIX'].iloc[i-1]
        if trix > 0 and trix > trix_prev:
            scores['trix'] = min(1, trix / 0.5)
        else:
            scores['trix'] = 0
        
        # OBV得分
        obv_trend = (df['OBV'].iloc[i] > df['OBV_MA'].iloc[i] and 
                    df['OBV'].iloc[i] > df['OBV'].iloc[i-1])
        scores['obv'] = 1 if obv_trend else 0
        
        # WR得分
        wr = df['WR'].iloc[i]
        if wr < -80:
            scores['wr'] = 1
        elif wr < -60:
            scores['wr'] = 0.5
        else:
            scores['wr'] = 0
            
        # EMV得分
        emv = df['EMV'].iloc[i]
        emv_ma = df['EMV_MA'].iloc[i]
        scores['emv'] = 1 if (emv > emv_ma and emv > 0) else 0
        
        # 计算总分
        total_score = sum(score * self.weights[indicator] 
                         for indicator, score in scores.items())
        
        return total_score, scores
    
    def generate_signals(self, df):
        """生成交易信号"""
        df['signal'] = 0
        df['signal_score'] = 0.0
        df['signal_reason'] = ''
        df['detailed_scores'] = ''
        
        for i in range(1, len(df)):
            total_score, detailed_scores = self.calculate_signal_score(df, i)
            
            # 设置信号阈值
            if total_score > 0.3:  # 可以调整阈值
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_score'] = total_score
                
                # 生成详细的信号原因
                reasons = []
                if detailed_scores['ma_cross'] > 0:
                    reasons.append('MA5上穿MA20')
                if detailed_scores['bollinger'] > 0:
                    reasons.append('价格接近布林带下轨')
                if detailed_scores['rsi'] > 0:
                    reasons.append('RSI超卖')
                if detailed_scores['macd'] > 0:
                    reasons.append('MACD金叉')
                if detailed_scores['kdj'] > 0:
                    reasons.append('KDJ超卖')
                if detailed_scores['volume'] > 0:
                    reasons.append('成交量放大')
                if detailed_scores['dmi'] > 0:
                    reasons.append('ADX强势')
                if detailed_scores['cci'] > 0:
                    reasons.append('CCI回归')
                if detailed_scores['trix'] > 0:
                    reasons.append('TRIX上升')
                if detailed_scores['obv'] > 0:
                    reasons.append('OBV上升趋势')
                if detailed_scores['wr'] > 0:
                    reasons.append('WR超卖')
                if detailed_scores['emv'] > 0:
                    reasons.append('EMV上升')
                
                df.loc[df.index[i], 'signal_reason'] = '；'.join(reasons)
                df.loc[df.index[i], 'detailed_scores'] = str(detailed_scores)
        
        return df
    
    def backtest(self, df):
        """回测策略"""
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
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
            # 取股票日线数据
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
            if market_value < self.min_market_value:
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
    strategy = TradingStrategy()
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
            
            if latest_signal >= 1:
                return {
                    'code': stock_code,
                    'name': stock_name,
                    'signal': latest_signal,
                    'signal_score': result_df['signal_score'].iloc[-1],
                    'price': result_df['close'].iloc[-1],
                    'date': result_df.index[-1],
                    'market_value': market_value,
                    'daily_amount': stock_data['amount'].iloc[-1],
                    'reason': result_df['signal_reason'].iloc[-1]
                }
        except Exception as e:
            print(f"分析股票{stock_code}时出错: {e}")
        return None

    # 创建股票信息列表
    stock_info_list = [(row['code'], row['name'], row['market_value']) 
                       for _, row in stock_list.iterrows()]
    
    # 使用tqdm包装线程池的执行过程
    with ThreadPoolExecutor(max_workers=20) as executor:  # 增加线程数
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
    
    # 设置最大推荐股票数量
    MAX_RECOMMENDATIONS = 50
    
    # 修改输出部分
    print("\n=== 买入信号汇总（按信号得分排序）===")
    print(f"运行时间: {run_duration}")
    print(f"筛选出的股票总数: {len(trading_opportunities)}")
    print("代码  名称  信号得分  价格  市值(亿)  日成交额(亿)  买入理由")
    print("-" * 100)

    # 将结果转换为列表并按信号得分排序，只保留前N个最强信号
    sorted_opportunities = sorted(
        trading_opportunities.values(),
        key=lambda x: x['signal_score'],
        reverse=True
    )[:MAX_RECOMMENDATIONS]  # 限制推荐数量

    # 输出排序后的结果，市值转换为亿元
    for info in sorted_opportunities:
        print(f"{info['code']} {info['name']} {info['signal_score']:.2f} "
              f"{info['price']:.2f} {info['market_value']/100000000:.2f} "  # 转换为亿元
              f"{info['daily_amount']/100000000:.2f} "
              f"{info['reason']}")

    # 保存结果到CSV时添加指标说明
    if sorted_opportunities:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'trading_signals_{current_date}.csv'
        
        # 创建技术指标说明
        indicator_descriptions = pd.DataFrame([
            ['技术指标说明', ''],
            ['1. MA交叉(12%)', 'MA5与MA20交叉是重要的趋势转换信号。当5日均线上穿20日均线时，表示可能开始上涨趋势。'],
            ['2. 布林带(12%)', '布林带是以移动平均线为中轨，上下各加减两个标准差形成的带状区域。当价格接近下轨时，可能存在超卖机会。'],
            ['3. RSI(8%)', '相对强弱指标，用于判断超买超卖。RSI低于30表示超卖，低于40表示较弱。'],
            ['4. MACD(12%)', '平滑异同移动平均线，是重要的趋势指标。当MACD柱由负转正时，表示可能形成上涨趋势。'],
            ['5. KDJ(8%)', '随机指标，用于判断超买超卖。K值小于20表示超卖，是买入机会。'],
            ['6. 成交量(12%)', '成交量是价格变动的确认指标。当成交量较前期明显放大时，表示趋势更可靠。'],
            ['7. DMI(8%)', '趋势方向指标，用于判断趋势强度。ADX大于25且+DI大于-DI时，表示上涨趋势强劲。'],
            ['8. CCI(5%)', '顺势指标，用于判断价格偏离程度。CCI在-100到-80之间时，表示可能存在买入机会。'],
            ['9. TRIX(5%)', '三重指数平滑移动平均指标，用于判断中长期趋势。TRIX由负转正且持续上升，表示趋势向好。'],
            ['10. OBV(6%)', '能量潮指标，用于判断趋势的持续性。当OBV上升且大于其MA时，表示趋势向上。'],
            ['11. WR(6%)', '威廉指标，用于判断超买超卖。WR小于-80表示超卖，小于-60表示较弱。'],
            ['12. EMV(6%)', '简易波动指标，用于判断趋势的强度。当EMV大于其MA且大于0时，表示趋势向上。'],
            ['', ''],
            ['信号得分说明', '各指标得分根据其重要性赋予不同权重，总分大于0.3时产生买入信号。得分越高，信号越强。'],
            ['', ''],
            ['筛选结果', '以下是按信号得分排序的股票列表：'],
            ['', '']
        ], columns=['指标', '说明'])
        
        # 创建股票数据DataFrame
        stock_data = pd.DataFrame([
            {
                '代码': info['code'],
                '名称': info['name'],
                '信号得分': info['signal_score'],
                '当前价格': info['price'],
                '市值(亿)': info['market_value']/100000000,  # 转换为亿元
                '日成交额(亿)': info['daily_amount']/100000000,
                '买入理由': info['reason']
            }
            for info in sorted_opportunities
        ])
        
        # 将说明和数据合并保存
        with open(filename, 'w', encoding='utf-8-sig') as f:
            indicator_descriptions.to_csv(f, index=False)
            stock_data.to_csv(f, index=False)
            
        print(f"\n结果已保存到: {filename}")
        print(f"推荐股票数量: {len(sorted_opportunities)}")

if __name__ == "__main__":
    main()
