import pandas as pd
from datetime import datetime

def convert_to_sample_trades(input_file: str, output_file: str):
    """将实际交易记录转换为样本交易格式"""
    # 读取原始数据
    df = pd.read_csv(input_file)
    
    # 计算交易成本
    def calculate_costs(row):
        price = row['price']
        volume = row['volume']
        amount = price * volume
        
        # 佣金费率 0.0003，最低 5 元
        commission = max(amount * 0.0003, 5)
        
        # 印花税（仅卖出收取，税率 0.001）
        tax = amount * 0.001 if row['direction'] == 'SELL' else 0
        
        return pd.Series({
            '成交金额': amount,
            '手续费': commission,
            '印花税': tax
        })
    
    # 计算成本
    costs = df.apply(calculate_costs, axis=1)
    
    # 创建样本交易记录
    sample_trades = pd.DataFrame({
        '交易时间': pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S'),
        '股票代码': df['code'],
        '股票名称': df['name'],
        '交易方向': df['direction'],
        '成交价格': df['price'],
        '成交数量': df['volume'],
        '成交金额': costs['成交金额'],
        '手续费': costs['手续费'],
        '印花税': costs['印花税'],
        '交易原因': df['reason'].fillna('策略交易')
    })
    
    # 保存为样本交易格式
    sample_trades.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"样本交易记录已保存到: {output_file}")
    return sample_trades

if __name__ == "__main__":
    input_file = "old/stock_trades.csv"
    output_file = "tests/data/sample_trades.csv"
    trades_df = convert_to_sample_trades(input_file, output_file)
    
    # 打印统计信息
    print("\n交易记录统计:")
    print(f"总交易次数: {len(trades_df)}")
    print(f"交易股票数: {len(trades_df['股票代码'].unique())}")
    print(f"交易日期范围: {trades_df['交易时间'].min()} 至 {trades_df['交易时间'].max()}")
    print(f"总成交金额: {trades_df['成交金额'].sum():.2f}")
    print(f"总手续费: {trades_df['手续费'].sum():.2f}")
    print(f"总印花税: {trades_df['印花税'].sum():.2f}")