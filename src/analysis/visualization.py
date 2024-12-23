import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
import matplotlib.dates as mdates

class Visualizer:
    """可视化工具"""
    def __init__(self):
        # 设置绘图风格
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 设置中文字体，按顺序尝试不同的字体，直到找到可用的
        plt.rcParams['font.sans-serif'] = [
            'Arial Unicode MS',  # macOS 的默认Unicode字体
            'Microsoft YaHei',   # 微软雅黑
            'SimHei',           # 黑体
            'DejaVu Sans',      # Linux常用
            'Noto Sans CJK JP', # Google Noto字体
            'Hiragino Sans GB'  # macOS中文字体
        ]
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 验证字体是否可用
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        font_names = [f.name for f in fm.ttflist]
        
        # 打印可用的字体列表，用于调试
        # print("Available fonts:", [f for f in font_names if any(keyword in f for keyword in ['Arial', 'Microsoft', 'SimHei', 'DejaVu', 'Noto', 'Hiragino'])])
        
    def plot_performance(self, 
                        daily_capital: List[float],
                        daily_returns: List[float],
                        dates: List[datetime],
                        save_path: str = None):
        """绘制绩效图表"""
        fig = plt.figure(figsize=(15, 10))
        
        # 创建子图
        gs = plt.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])  # 净值曲线
        ax2 = fig.add_subplot(gs[1, 0])  # 收益分布
        ax3 = fig.add_subplot(gs[1, 1])  # 回撤图
        
        # 绘制净值曲线
        ax1.plot(dates, daily_capital, 'b-', label='Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.grid(True)
        ax1.legend()
        
        # 格式化x轴日期
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 绘制收益分布
        sns.histplot(daily_returns, kde=True, ax=ax2)
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Daily Returns')
        ax2.set_ylabel('Frequency')
        
        # 绘制回撤图
        cumulative = np.array(daily_capital) / daily_capital[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        
        ax3.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True)
        
        # 格式化x轴日期
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_positions(self, 
                      daily_positions: List[pd.DataFrame],
                      dates: List[datetime],
                      save_path: str = None):
        """绘制持仓分析图表"""
        if not daily_positions:
            return
            
        # 创建持仓市值数据
        position_values = pd.DataFrame()
        for date, pos_df in zip(dates, daily_positions):
            if not pos_df.empty:
                position_values[date] = pos_df.set_index('代码')['市值']
                
        # 绘制堆积图
        plt.figure(figsize=(15, 8))
        position_values.T.plot(kind='area', stacked=True)
        
        plt.title('Position Analysis')
        plt.xlabel('Date')
        plt.ylabel('Market Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 
            
    def plot_trade_analysis(self, trades_df: pd.DataFrame, save_path: str = None):
        """绘制交易分析图表"""
        # 创建子图
        fig = plt.figure(figsize=(15, 12))
        
        # 1. 收益曲线 (上图)
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        self._plot_profit_curve(trades_df, ax1)
        
        # 2. 持仓分布 (左下图)
        ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
        self._plot_position_distribution(trades_df, ax2)
        
        # 3. 交易记录 (右下图)
        ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
        self._plot_trade_records(trades_df, ax3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def _plot_profit_curve(self, trades_df: pd.DataFrame, ax):
        """绘制累计收益曲线（包含持仓市值）"""
        trades_df = trades_df.copy()
        trades_df['日期'] = pd.to_datetime(trades_df['交易时间']).dt.date
        
        # 按日期计算持仓和收益
        daily_data = []
        positions = {}  # 用于跟踪每日持仓
        
        for date in sorted(trades_df['日期'].unique()):
            day_trades = trades_df[trades_df['日期'] == date]
            
            # 更新持仓
            for _, trade in day_trades.iterrows():
                code = trade['股票代码']
                if code not in positions:
                    positions[code] = {
                        '数量': 0,
                        '成本': 0,
                        '现价': trade['成交价格']
                    }
                
                if trade['交易方向'] == 'BUY':
                    positions[code]['数量'] += trade['成交数量']
                    positions[code]['成本'] += trade['成交金额']
                else:
                    # 卖出时按比例减少成本
                    if positions[code]['数量'] > 0:
                        cost_per_share = positions[code]['成本'] / positions[code]['数量']
                        positions[code]['成本'] -= cost_per_share * trade['成交数量']
                        positions[code]['数量'] -= trade['成交数量']
                
                # 更新现价
                positions[code]['现价'] = trade['成交价格']
            
            # 计算当日交易的已实现收益
            day_realized_pnl = day_trades.apply(
                lambda x: (x['成交金额'] - x['手续费'] - x['印花税']) if x['交易方向'] == 'SELL'
                else -(x['成交金额'] + x['手续费'] + x['印花税']),
                axis=1
            ).sum()
            
            # 计算持仓市值
            total_position_value = sum(
                pos['数量'] * pos['现价']
                for pos in positions.values()
                if pos['数量'] > 0
            )
            
            daily_data.append({
                '日期': date,
                '已实现收益': day_realized_pnl,
                '持仓市值': total_position_value
            })
        
        # 转换为DataFrame并计算累计收益
        daily_df = pd.DataFrame(daily_data)
        daily_df['累计已实现收益'] = daily_df['已实现收益'].cumsum()
        daily_df['累计收益'] = daily_df['累计已实现收益'] + daily_df['持仓市值']
        
        # 绘制曲线
        ax.plot(daily_df['日期'], daily_df['累计收益'], marker='o', linewidth=2)
        
        ax.set_title('累计收益曲线')
        ax.set_xlabel('日期')
        ax.set_ylabel('金额(元)')
        ax.grid(True)
        
        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_position_distribution(self, trades_df: pd.DataFrame, ax):
        """绘制持仓分布（按市值）"""
        # 计算当前持仓
        positions = {}
        for _, trade in trades_df.sort_values('交易时间').iterrows():
            code = f"{trade['股票名称']}({trade['股票代码']})"
            if code not in positions:
                positions[code] = {
                    '数量': 0,
                    '成本': 0,
                    '最新价格': trade['成交价格']
                }
            
            if trade['交易方向'] == 'BUY':
                positions[code]['数量'] += trade['成交数量']
                positions[code]['成本'] += trade['成交金额']
            else:
                # 卖出时按比例减少成本
                if positions[code]['数量'] > 0:
                    cost_per_share = positions[code]['成本'] / positions[code]['数量']
                    positions[code]['成本'] -= cost_per_share * trade['成交数量']
                    positions[code]['数量'] -= trade['成交数量']
            
            # 更新最新价格
            positions[code]['最新价格'] = trade['成交价格']
        
        # 计算每个持仓的市值
        position_values = {}
        for code, pos in positions.items():
            if pos['数量'] > 0:  # 只统计还持有的股票
                market_value = pos['数量'] * pos['最新价格']
                if market_value > 0:  # 避免零市值
                    position_values[code] = market_value
        
        if position_values:
            # 计算总市值和百分比
            total_value = sum(position_values.values())
            labels = []
            for code, value in position_values.items():
                percentage = value / total_value * 100
                labels.append(f'{code}\n{value:,.0f}元 ({percentage:.1f}%)')
            
            # 绘制饼图
            ax.pie(position_values.values(), labels=labels, autopct='%1.1f%%')
            ax.set_title('当前持仓分布(按市值)')
        else:
            ax.text(0.5, 0.5, '当前无持仓', ha='center', va='center')
            
    def _plot_trade_records(self, trades_df: pd.DataFrame, ax):
        """绘制交易记录统计"""
        # 按股票统计交易次数
        trade_counts = trades_df.groupby('股票名称').size().sort_values(ascending=True)
        
        # 绘制水平条形图
        bars = ax.barh(range(len(trade_counts)), trade_counts.values)
        ax.set_yticks(range(len(trade_counts)))
        ax.set_yticklabels(trade_counts.index)
        ax.set_title('交易频次统计')
        ax.set_xlabel('交易次数')
        
        # 在条形上标注具体数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, i, f' {int(width)}', va='center')