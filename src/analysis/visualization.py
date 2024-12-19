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