import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

class PositionManager:
    def __init__(self, stop_loss=0.07, take_profit=0.15):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.positions = None
        self.data_cache = {}
        self.stock_names = None  # 用于存储股票名称字典
        
    def load_positions(self, file_path='store.csv'):
        """加载持仓数据"""
        try:
            self.positions = pd.read_csv(file_path)
            self.positions['code'] = self.positions['code'].astype(str).str.zfill(6)
            self.positions['date'] = pd.to_datetime(self.positions['date'])
            print(f"已加载 {len(self.positions)} 个持仓")
            return True
        except Exception as e:
            print(f"加载持仓数据失败: {e}")
            return False
            
    def get_stock_data(self, stock_code):
        """获取股票最新数据"""
        try:
            stock_code = str(stock_code).zfill(6)
            
            # 添加缓存检查
            cache_key = f"{stock_code}_{datetime.now().strftime('%Y%m%d')}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
                
            # 获取最近30天数据用于计算指标
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=stock_code, 
                                  start_date=start_date, 
                                  end_date=end_date,
                                  adjust="qfq")
            
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 保存到缓存
            self.data_cache[cache_key] = df
            return df
        except Exception as e:
            print(f"获取股票{stock_code}数据失败: {e}")
            return None
            
    def calculate_indicators(self, df):
        """计算技术指标"""
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        df['BB_middle'] = df['MA20']
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        return df
        
    def get_stock_names(self):
        """获取所有股票代码和名称的映射"""
        try:
            stock_info = ak.stock_info_a_code_name()
            self.stock_names = dict(zip(stock_info['code'], stock_info['name']))
            return True
        except Exception as e:
            print(f"获取股票名称失败: {e}")
            return False
            
    def analyze_position(self, position):
        """分析单个持仓"""
        stock_code = position['code']
        entry_price = round(position['price'], 2)
        entry_date = position['date']
        number = position['number']
        
        # 获取股票名称
        stock_name = self.stock_names.get(stock_code, '未知')
        
        df = self.get_stock_data(stock_code)
        if df is None or df.empty:
            return None
            
        df = self.calculate_indicators(df)
        current_price = round(df['close'].iloc[-1], 2)
        current_date = df.index[-1]
        
        profit_pct = round((current_price - entry_price) / entry_price, 4)
        profit_amount = round((current_price - entry_price) * number, 2)
        
        suggestion = self.generate_suggestion(df, profit_pct, entry_price)
        
        return {
            'code': stock_code,
            'name': stock_name,  # 添加股票名称
            'current_price': current_price,
            'entry_price': entry_price,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'hold_days': (current_date - entry_date).days,
            'suggestion': suggestion
        }
        
    def generate_suggestion(self, df, profit_pct, entry_price):
        """生成操作建议"""
        # 获取最新指标值
        latest = df.iloc[-1]
        
        # 止损检查
        if profit_pct < -self.stop_loss:
            return "建议止损卖出"
            
        # 止盈检查
        if profit_pct > self.take_profit:
            return "建议止盈卖出"
            
        # 技术指标检查
        if latest['close'] < latest['MA20'] and latest['close'] < latest['MA5']:
            return "建议观察，可考虑减仓"
            
        if latest['close'] > latest['BB_upper']:
            return "超买区间，注意风险"
            
        return "建议持有"

def main():
    # 初始化持仓管理器
    manager = PositionManager()
    
    # 获取股票名称映射
    if not manager.get_stock_names():
        return
        
    # 加载持仓数据
    if not manager.load_positions():
        return
        
    print("\n开始分析持仓...")
    
    # 分析每个持仓
    results = []
    for _, position in tqdm(manager.positions.iterrows(), total=len(manager.positions)):
        result = manager.analyze_position(position)
        if result:
            results.append(result)
    
    # 输出分析结果
    print("\n=== 持仓分析报告 ===")
    print("代码    名称    现价    成本    收益率    收益额    持有天数  建议")
    print("-" * 85)
    
    for r in results:
        print(f"{r['code']}  "
              f"{r['name']:<6}  "  # 左对齐，最大6个字符
              f"{r['current_price']:>6.2f}  "
              f"{r['entry_price']:>6.2f}  "
              f"{r['profit_pct']*100:>6.2f}%  "
              f"{r['profit_amount']:>8.2f}  "
              f"{r['hold_days']:>4d}天  "
              f"{r['suggestion']}")
    
    # 保存分析结果
    if results:
        df_results = pd.DataFrame(results)
        # 设置DataFrame的显示精度
        for col in ['current_price', 'entry_price', 'profit_amount']:
            df_results[col] = df_results[col].round(2)
        df_results['profit_pct'] = df_results['profit_pct'].round(4)
        
        output_file = f'position_analysis_{datetime.now().strftime("%Y%m%d")}.csv'
        df_results.to_csv(output_file, index=False, float_format='%.2f')
        print(f"\n分析结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 