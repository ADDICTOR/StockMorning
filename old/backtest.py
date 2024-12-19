import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Optional
import matplotlib.dates as mdates
import os
import warnings
import platform
import matplotlib
import matplotlib.gridspec as gridspec

# 禁用所有警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置 matplotlib 为非交互式后端
matplotlib.use('Agg')

@dataclass
class TradeRecord:
    """交易记录数据类"""
    code: str                    # 股票代码
    name: str                    # 股票名称
    direction: str              # 买入/卖出
    price: float                # 交易价格
    shares: int                 # 交易数量
    commission: float           # 手续费
    tax: float                  # 印花税
    timestamp: datetime         # 交易时间
    reason: str                 # 交易原因

class BacktestSystem:
    def __init__(self, 
                 initial_capital: float = 1000000,
                 commission_rate: float = 0.0005,  # 手续���率
                 tax_rate: float = 0.001,          # 印花税率
                 slippage: float = 0.002):         # 滑点率
        """
        初始化回测系统
        :param initial_capital: 初始资金
        :param commission_rate: 手续费（默认万五）
        :param tax_rate: 印花税率（默认千分之一）
        :param slippage: 滑点率（默认0.2%）
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.slippage = slippage
        
        self.positions: Dict[str, Dict] = {}       # 当前持仓
        self.trade_history: List[TradeRecord] = [] # 交易记录
        self.daily_capital: List[float] = []       # 每日资金曲线
        self.daily_returns: List[float] = []       # 每日收益率
        
    def execute_trade(self, 
                     code: str, 
                     name: str,
                     direction: str, 
                     price: float, 
                     shares: int,
                     timestamp: datetime,
                     reason: str) -> bool:
        """
        执行交易
        :return: 交易是否成功
        """
        # 计算实际成交价格（考虑滑点）
        actual_price = price * (1 + self.slippage) if direction == "BUY" else price * (1 - self.slippage)
        
        # 计算交易成本
        commission = actual_price * shares * self.commission_rate
        tax = actual_price * shares * self.tax_rate if direction == "SELL" else 0
        total_cost = actual_price * shares + commission + tax
        
        # 验证资金是否足够
        if direction == "BUY" and total_cost > self.current_capital:
            print(f"资金不足，交易取消: {code}")
            return False
            
        # 更新资金
        self.current_capital -= total_cost if direction == "BUY" else -total_cost
        
        # 更新持仓
        if direction == "BUY":
            if code not in self.positions:
                self.positions[code] = {
                    'name': name,
                    'shares': shares,
                    'cost': actual_price,
                    'timestamp': timestamp
                }
            else:
                # 计算新的持仓成本
                old_cost = self.positions[code]['cost']
                old_shares = self.positions[code]['shares']
                total_shares = old_shares + shares
                new_cost = (old_cost * old_shares + actual_price * shares) / total_shares
                self.positions[code].update({
                    'shares': total_shares,
                    'cost': new_cost,
                    'timestamp': timestamp
                })
        else:  # SELL
            if code not in self.positions or self.positions[code]['shares'] < shares:
                print(f"持仓不足，交易取消: {code}")
                return False
            self.positions[code]['shares'] -= shares
            if self.positions[code]['shares'] == 0:
                del self.positions[code]
                
        # 记录交易
        self.trade_history.append(TradeRecord(
            code=code,
            name=name,
            direction=direction,
            price=actual_price,
            shares=shares,
            commission=commission,
            tax=tax,
            timestamp=timestamp,
            reason=reason
        ))
        
        return True
        
    def calculate_metrics(self) -> Dict:
        """计算回测指标"""
        # 计算最终总资产（现金 + 持仓市值）
        final_total_value = self.current_capital
        
        # 如果还有持仓，计算持仓市值
        if self.positions and self.trade_history:
            for code, position in self.positions.items():
                # 获取该股票最后一次交易的价格
                last_trade = next(
                    (trade for trade in reversed(self.trade_history) 
                     if trade.code == code), 
                    None
                )
                if last_trade:
                    position_value = position['shares'] * last_trade.price
                    final_total_value += position_value
                    print(f"DEBUG: {code} 持仓 {position['shares']}股, "
                          f"最新价 {last_trade.price}, 市值 {position_value}")
        
        # 打印调试信息
        print(f"\nDEBUG: 期末现金: {self.current_capital:,.2f}")
        print(f"DEBUG: 持仓市值: {(final_total_value - self.current_capital):,.2f}")
        print(f"DEBUG: 最终总资产: {final_total_value:,.2f}")
        
        # 计算收益率
        total_return = (final_total_value / self.initial_capital) - 1
        
        # 计算年化收益率
        total_days = len(self.daily_returns)
        annual_return = (1 + total_return) ** (252 / total_days) - 1 if total_days > 0 else 0
        
        # 计算夏普比率
        if self.daily_returns and len(self.daily_returns) > 1:
            returns = pd.Series(self.daily_returns)
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        # 计算最大回撤
        max_drawdown = 0
        peak = self.daily_capital[0]
        for value in self.daily_capital:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算胜率
        trade_pairs = []
        current_buy = None
        
        for trade in sorted(self.trade_history, key=lambda x: x.timestamp):
            if trade.direction == 'BUY':
                current_buy = trade
            elif trade.direction == 'SELL' and current_buy:
                profit = (trade.price - current_buy.price) / current_buy.price
                trade_pairs.append(profit > 0)
                current_buy = None
        
        win_rate = sum(trade_pairs) / len(trade_pairs) if trade_pairs else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history),
            'complete_trades': len(trade_pairs),
            'profitable_trades': sum(trade_pairs),
            'final_capital': final_total_value
        }
        
    def plot_results(self):
        """可视化回测结果"""
        # 确保输出目录存在
        output_dir = './backtest_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 12))
        
        # 创建三个子图
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 2])
        
        # 第一张图：资金曲线
        ax1 = plt.subplot(gs[0])
        
        # 使用日期范围作为x轴
        dates = pd.date_range(
            start=min(trade.timestamp for trade in self.trade_history),
            end=max(trade.timestamp for trade in self.trade_history),
            freq='D'
        )
        
        if len(dates) == len(self.daily_capital):
            # 确保所有数据长度一致
            if hasattr(self, 'daily_cash') and hasattr(self, 'daily_positions'):
                ax1.plot(dates, self.daily_cash, label='Cash', color='green', linewidth=1, linestyle='--')
                ax1.plot(dates, self.daily_positions, label='Position Value', color='red', linewidth=1, linestyle=':')
            
            # 总资产曲线始终显示
            ax1.plot(dates, self.daily_capital, label='Total Assets', color='blue', linewidth=2)
        
        # 标记起始点和结束点
        ax1.scatter([dates[0]], [self.initial_capital], color='green', s=100, label='Initial Capital')
        ax1.scatter([dates[-1]], [self.daily_capital[-1]], color='red', s=100, label='Final Capital')
        
        # 设置图表格式
        ax1.set_title('Capital Curve', fontsize=12, pad=15)
        ax1.grid(True)
        ax1.legend(loc='best')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.set_ylabel('Amount (CNY)')
        
        # 收益率分布图
        ax2 = plt.subplot(gs[1])
        sns.histplot(self.daily_returns, kde=True, ax=ax2, bins=50)
        ax2.axvline(x=np.mean(self.daily_returns), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(self.daily_returns):.4f}')
        ax2.set_title('Return Distribution', fontsize=12, pad=15)
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 交易点位图
        ax3 = plt.subplot(gs[2])
        all_trades = pd.DataFrame([{
            'timestamp': t.timestamp,
            'price': t.price,
            'direction': t.direction,
            'code': t.code,
            'name': t.name
        } for t in self.trade_history])
        
        # 按股票代码分组绘制
        for code in all_trades['code'].unique():
            stock_trades = all_trades[all_trades['code'] == code]
            stock_name = stock_trades.iloc[0]['name']
            label = f"{code} {stock_name}"
            
            # 绘制价格线
            ax3.plot(stock_trades['timestamp'], stock_trades['price'], 
                    label=label, alpha=0.5, linestyle='--')
            
            # 标记买卖点
            buys = stock_trades[stock_trades['direction'] == 'BUY']
            sells = stock_trades[stock_trades['direction'] == 'SELL']
            
            ax3.scatter(buys['timestamp'], buys['price'], 
                       color='green', marker='^', s=100)
            ax3.scatter(sells['timestamp'], sells['price'], 
                       color='red', marker='v', s=100)
        
        ax3.set_title('Trading Points', fontsize=12, pad=15)
        ax3.grid(True)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f'backtest_results_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close('all')
        
        print(f"\n回测结果图表已保存为: {filename}")
        
        # 自动打开图片
        try:
            if platform.system() == 'Darwin':  # macOS
                os.system(f'open "{filename}"')
            elif platform.system() == 'Windows':
                os.system(f'start "" "{filename}"')
            else:  # Linux
                os.system(f'xdg-open "{filename}"')
        except Exception as e:
            print(f"无法自动打开图片: {e}")
        
    def generate_report(self) -> pd.DataFrame:
        """生成交易报告"""
        # 转换交易记录为DataFrame
        trades_df = pd.DataFrame([vars(record) for record in self.trade_history])
        
        # 计算指标
        metrics = self.calculate_metrics()
        
        # 生成每笔交易的收益统计
        trade_stats = []
        for i, trade in trades_df.iterrows():
            if trade.direction == 'SELL':
                # 查找对应的买入记录
                buy_trade = trades_df[(trades_df.code == trade.code) & 
                                    (trades_df.direction == 'BUY') & 
                                    (trades_df.timestamp < trade.timestamp)].iloc[-1]
                profit = (trade.price - buy_trade.price) * trade.shares
                profit_rate = (trade.price / buy_trade.price) - 1
                hold_days = (trade.timestamp - buy_trade.timestamp).days
                trade_stats.append({
                    'code': trade.code,
                    'name': trade.name,
                    'buy_date': buy_trade.timestamp,
                    'sell_date': trade.timestamp,
                    'buy_price': buy_trade.price,
                    'sell_price': trade.price,
                    'shares': trade.shares,
                    'profit': profit,
                    'profit_rate': profit_rate,
                    'hold_days': hold_days
                })
        
        trade_stats_df = pd.DataFrame(trade_stats)
        
        return {
            'metrics': pd.Series(metrics),
            'trades': trades_df,
            'trade_stats': trade_stats_df
        }

    def import_trades_from_csv(self, csv_path: str):
        """从CSV文件导入真实交易记录"""
        try:
            # 读取CSV文件
            trades_df = pd.read_csv(csv_path)
            
            # 检查必要的列
            required_columns = ['date', 'code', 'name', 'direction', 'price', 'volume']
            missing_columns = [col for col in required_columns if col not in trades_df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_columns)}")
            
            # 清空现有的交易记录
            self.trade_history = []
            self.daily_capital = []
            self.daily_returns = []
            self.positions = {}  # 重置持仓信息
            self.current_capital = self.initial_capital  # 重置当前资金
            
            # ���换日期格式并排序
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df = trades_df.sort_values('date')
            
            # 生成日期序列
            date_range = pd.date_range(start=trades_df['date'].min(), end=trades_df['date'].max(), freq='D')
            
            # 初始化资金曲线数据
            daily_cash = []  # 每日现
            daily_positions = []  # 每日持仓市值
            
            for date in date_range:
                # 处理当日交易
                day_trades = trades_df[trades_df['date'].dt.date == date.date()]
                
                for _, trade in day_trades.iterrows():
                    code = str(trade['code']).zfill(6)
                    direction = trade['direction'].upper()
                    price = float(trade['price'])
                    volume = int(trade['volume'])
                    commission = 5.0
                    tax = price * volume * 0.001 if direction == 'SELL' else 0
                    
                    # 创建交易记录
                    trade_record = TradeRecord(
                        code=code,
                        name=trade['name'],
                        direction=direction,
                        price=price,
                        shares=volume,
                        commission=commission,
                        tax=tax,
                        timestamp=trade['date'],
                        reason=trade.get('reason', '')
                    )
                    self.trade_history.append(trade_record)
                    
                    # 更新资金和持仓
                    if direction == 'BUY':
                        cost = price * volume + commission + tax
                        self.current_capital -= cost
                        if code not in self.positions:
                            self.positions[code] = {
                                'shares': volume,
                                'cost': price,
                                'name': trade['name']
                            }
                        else:
                            old_shares = self.positions[code]['shares']
                            old_cost = self.positions[code]['cost']
                            total_shares = old_shares + volume
                            self.positions[code]['cost'] = (old_cost * old_shares + price * volume) / total_shares
                            self.positions[code]['shares'] += volume
                    else:  # SELL
                        self.current_capital += price * volume - commission - tax
                        self.positions[code]['shares'] -= volume
                        if self.positions[code]['shares'] <= 0:
                            del self.positions[code]
                
                # 计算当日持仓市值
                position_value = 0
                for code, pos in self.positions.items():
                    # 获取当日或之前最近的交易价格
                    last_trades = trades_df[
                        (trades_df['date'] <= date) & 
                        (trades_df['code'] == code)
                    ]
                    if not last_trades.empty:
                        last_price = float(last_trades.iloc[-1]['price'])
                        position_value += pos['shares'] * last_price
                
                # 记录当日资金状况
                daily_cash.append(self.current_capital)
                daily_positions.append(position_value)
                self.daily_capital.append(self.current_capital + position_value)
                
                # 计算日收益率
                if len(self.daily_capital) > 1:
                    daily_return = (self.daily_capital[-1] / self.daily_capital[-2]) - 1
                    self.daily_returns.append(daily_return)
                else:
                    self.daily_returns.append(0.0)
            
            print(f"\n成功导入 {len(self.trade_history)} 条交易记录")
            print(f"交易时间范围: {trades_df['date'].min().strftime('%Y-%m-%d')} 至 {trades_df['date'].max().strftime('%Y-%m-%d')}")
            print(f"\n资金状况:")
            print(f"期初资金: {self.initial_capital:,.2f}")
            print(f"期末现金: {self.current_capital:,.2f}")
            print(f"期末持仓市值: {position_value:,.2f}")
            print(f"期末总资产: {(self.current_capital + position_value):,.2f}")
            
            self.daily_cash = daily_cash
            self.daily_positions = daily_positions
            
            return True
            
        except Exception as e:
            print(f"导入交易记录时出错: {str(e)}")
            return False

def main():
    """示例使用"""
    # 初始化回测系统
    backtest = BacktestSystem(initial_capital=1000000)
    
    # 模拟一些交易
    backtest.execute_trade(
        code='000001',
        name='平安银行',
        direction='BUY',
        price=10.0,
        shares=1000,
        timestamp=datetime.now(),
        reason='技术指标买入信号'
    )
    
    # 计算指标
    metrics = backtest.calculate_metrics()
    print("\n回测指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 生成报告
    report = backtest.generate_report()
    print("\n交易统计:")
    print(report['trade_stats'])
    
    # 绘制结果
    backtest.plot_results()

if __name__ == "__main__":
    main() 