from old.backtest import BacktestSystem
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_test_trades():
    """生成测试用的交易数据"""
    # 模拟6个月的交易
    start_date = datetime(2023, 7, 1)
    
    # 测试股票池
    test_stocks = [
        {'code': '000001', 'name': '平安银行'},
        {'code': '600036', 'name': '招商银行'},
        {'code': '000858', 'name': '五粮液'},
        {'code': '600519', 'name': '贵州茅台'},
        {'code': '000333', 'name': '美的集团'}
    ]
    
    # 模拟价格变动
    def generate_price_series(base_price, days=180):
        """生成模拟价格序列"""
        # 使用随机游走模拟价格变动
        returns = np.random.normal(0.0003, 0.02, days)  # 均值0.03%，标准差2%
        price_series = base_price * np.cumprod(1 + returns)
        return price_series
    
    # 为每只股票生成价格序列
    stock_prices = {}
    for stock in test_stocks:
        base_price = np.random.randint(20, 100)  # 随机基础价格
        stock_prices[stock['code']] = generate_price_series(base_price)

    return test_stocks, stock_prices, start_date

def run_backtest():
    """运行回测"""
    # 初始化回测系统
    backtest = BacktestSystem(initial_capital=1000000)
    
    # 生成测试数据
    test_stocks, stock_prices, start_date = generate_test_trades()
    
    # 模拟180个交易日
    for day in range(180):
        current_date = start_date + timedelta(days=day)
        
        # 每只股票都判断是否需要交易
        for stock in test_stocks:
            code = stock['code']
            current_price = stock_prices[code][day]
            
            # 模拟交易信号
            if day % 20 == 0:  # 每20天判断一次是否交易
                # 如果没有持仓，考虑买入
                if code not in backtest.positions:
                    if np.random.random() > 0.7:  # 30%概率买入
                        shares = int(100000 / current_price)  # 每次买入10万元
                        backtest.execute_trade(
                            code=code,
                            name=stock['name'],
                            direction='BUY',
                            price=current_price,
                            shares=shares,
                            timestamp=current_date,
                            reason='定期买入信号'
                        )
                # 如果有持仓，考虑卖出
                elif code in backtest.positions:
                    if np.random.random() > 0.8:  # 20%概率卖出
                        position = backtest.positions[code]
                        backtest.execute_trade(
                            code=code,
                            name=stock['name'],
                            direction='SELL',
                            price=current_price,
                            shares=position['shares'],
                            timestamp=current_date,
                            reason='定期卖出信号'
                        )
            
            # 止损止盈逻辑
            if code in backtest.positions:
                position = backtest.positions[code]
                profit_rate = (current_price / position['cost']) - 1
                
                # 止损：亏损超过8%
                if profit_rate < -0.08:
                    backtest.execute_trade(
                        code=code,
                        name=stock['name'],
                        direction='SELL',
                        price=current_price,
                        shares=position['shares'],
                        timestamp=current_date,
                        reason='止损卖出'
                    )
                
                # 止盈：盈利超过20%
                elif profit_rate > 0.20:
                    backtest.execute_trade(
                        code=code,
                        name=stock['name'],
                        direction='SELL',
                        price=current_price,
                        shares=position['shares'],
                        timestamp=current_date,
                        reason='止盈卖出'
                    )
        
        # 更新每日资金
        total_value = backtest.current_capital
        for code, position in backtest.positions.items():
            total_value += position['shares'] * stock_prices[code][day]
        backtest.daily_capital.append(total_value)
        
        # 计算每日收益率
        if day > 0:
            daily_return = (total_value / backtest.daily_capital[-2]) - 1
            backtest.daily_returns.append(daily_return)
        else:
            backtest.daily_returns.append(0.0)

    return backtest

def main():
    """主函数"""
    # 运行回测
    backtest = run_backtest()
    
    # 输出回测结果
    metrics = backtest.calculate_metrics()
    print("\n=== 回测结果 ===")
    print(f"初始资金: {backtest.initial_capital:,.2f}")
    print(f"最终资金: {backtest.current_capital:,.2f}")
    print(f"总收益率: {metrics['total_return']*100:.2f}%")
    print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"完整交易对数: {metrics['complete_trades']}")
    print(f"盈利交易数: {metrics['profitable_trades']}")
    print(f"胜率: {metrics['win_rate']*100:.2f}%")
    
    # 生成详细报告
    report = backtest.generate_report()
    
    print("\n=== 交易统计 ===")
    trade_stats = report['trade_stats']
    if not trade_stats.empty:
        print("\n盈利交易统计:")
        profit_trades = trade_stats[trade_stats['profit'] > 0]
        if not profit_trades.empty:
            print(f"盈利交易数: {len(profit_trades)}")
            print(f"平均盈利率: {profit_trades['profit_rate'].mean()*100:.2f}%")
            print(f"平均持仓天数: {profit_trades['hold_days'].mean():.1f}天")
        
        print("\n亏损交易统计:")
        loss_trades = trade_stats[trade_stats['profit'] < 0]
        if not loss_trades.empty:
            print(f"亏损交易数: {len(loss_trades)}")
            print(f"平均亏损率: {loss_trades['profit_rate'].mean()*100:.2f}%")
            print(f"平均持仓天数: {loss_trades['hold_days'].mean():.1f}天")
    
    # 绘制回测结果图表
    backtest.plot_results()
    
    # 保存交易记录到CSV
    trade_stats.to_csv('backtest_results.csv', index=False, encoding='utf-8-sig')
    print("\n交易记录已保存到 backtest_results.csv")

def test_real_trades():
    """测试真实交易数据"""
    # 初始化回测系统
    backtest = BacktestSystem(initial_capital=100000)  # 设置初始资金为10万
    
    # 导入交易记录
    if backtest.import_trades_from_csv('stock_trades.csv'):
        # 计算指标
        metrics = backtest.calculate_metrics()
        
        # 输出回测结果
        print("\n=== 回测结果 ===")
        print(f"初始资金: {backtest.initial_capital:,.2f}")
        print(f"期末现金: {backtest.current_capital:,.2f}")
        print(f"最终总资产: {metrics['final_capital']:,.2f}")  # 添加总资产显示
        print(f"总收益率: {metrics['total_return']*100:.2f}%")
        print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"完整交易��数: {metrics['complete_trades']}")
        print(f"盈利交易数: {metrics['profitable_trades']}")
        print(f"胜率: {metrics['win_rate']*100:.2f}%")
        
        # 输出持仓信息
        if backtest.positions:
            print("\n=== 当前持仓 ===")
            total_position_value = 0
            for code, pos in backtest.positions.items():
                # 获取最新价格
                last_trade = next(
                    (trade for trade in reversed(backtest.trade_history) 
                     if trade.code == code), 
                    None
                )
                if last_trade:
                    position_value = pos['shares'] * last_trade.price
                    total_position_value += position_value
                    print(f"{code} {pos['name']}: {pos['shares']}股, "
                          f"成本价: {pos['cost']:.2f}, "
                          f"现价: {last_trade.price:.2f}, "
                          f"市值: {position_value:,.2f}")
            print(f"总持仓市值: {total_position_value:,.2f}")
        
        # 输出交易成本统计
        total_commission = sum(trade.commission for trade in backtest.trade_history)
        total_tax = sum(trade.tax for trade in backtest.trade_history)
        print(f"\n=== 交易成本统计 ===")
        print(f"总手续费: {total_commission:.2f}")
        print(f"总印花税: {total_tax:.2f}")
        print(f"总交易成本: {(total_commission + total_tax):.2f}")
        
        # 生成图表
        backtest.plot_results()
    
if __name__ == "__main__":
    test_real_trades() 