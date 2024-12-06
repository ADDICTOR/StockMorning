import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
import re
from begin import StockDataFetcher
import concurrent.futures
import numpy as np
from functools import partial

def get_date_from_filename(filename):
    """从文件名中提取日期和时间"""
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
    return None

def load_stock_list(csv_file):
    """加载股票列表并按信号得分排序"""
    try:
        # 先读取整个文件内容
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到股票列表开始的位置
        start_idx = -1
        for i, line in enumerate(lines):
            if '代码,名称,信号得分,当前价格' in line:
                start_idx = i
                break
        
        if start_idx == -1:
            print(f"在文件 {csv_file} 中未找到股票列表起始位置")
            return None
        
        # 只读取股股票列表部分
        stock_data = []
        headers = lines[start_idx].strip().split(',')[:4]  # 只取前4列
        
        for line in lines[start_idx + 1:]:
            if line.strip():  # 跳过空行
                fields = line.strip().split(',')
                if len(fields) >= 4:  # 确保至少有需要的4列
                    stock_data.append(fields[:4])  # 只取前4列
        
        # 转换为DataFrame
        df = pd.DataFrame(stock_data, columns=headers)
        
        # 转换数据类型
        df['信号得分'] = pd.to_numeric(df['信号得分'], errors='coerce')
        df['当前价格'] = pd.to_numeric(df['当前价格'], errors='coerce')
        df['代码'] = df['代码'].astype(str).str.zfill(6)
        
        # 删除无效数据
        result = df.dropna()
        
        print(f"成功加载 {len(result)} 只股票")
        return result
        
    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return None

def get_stock_data_batch(stock_codes, start_date, end_date, fetcher, batch_size=5):
    """批量获取股票数据"""
    results = {}
    failed_codes = set()  # 记录获取失败的股票代码
    total_batches = (len(stock_codes) + batch_size - 1) // batch_size
    
    with tqdm(total=len(stock_codes), desc="获取股票数据") as pbar:
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(fetcher.get_stock_data, code, start_date, end_date): code 
                    for code in batch_codes
                }
                for future in concurrent.futures.as_completed(futures):
                    code = futures[future]
                    try:
                        data = future.result()
                        if data is not None:
                            results[code] = data
                        else:
                            failed_codes.add(code)
                        pbar.update(1)
                    except Exception as e:
                        print(f"获取股票 {code} 数据失败: {e}")
                        failed_codes.add(code)
                        pbar.update(1)
    return results, failed_codes

def calculate_index(stock_list, start_date, end_date=None, fetcher=None):
    """计算指数涨跌幅"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    # 初始化结果DataFrame
    total_weight = len(stock_list)  # 等权重
    
    # 批量获取股票数据
    stock_codes = stock_list['代码'].tolist()
    start_date_str = start_date.strftime('%Y%m%d')
    stock_data_dict, failed_codes = get_stock_data_batch(
        stock_codes, 
        start_date_str, 
        end_date, 
        fetcher, 
        batch_size=5
    )
    
    # 如果没有获取到任何数据，返回空结果
    if not stock_data_dict and not failed_codes:
        print("未能获取任何股票数据")
        return pd.DataFrame(), {}
    
    # 找到所有数据的共同日期
    common_dates = None
    for data in stock_data_dict.values():
        if common_dates is None:
            common_dates = set(data.index)
        else:
            common_dates &= set(data.index)
    
    common_dates = sorted(list(common_dates))
    
    if not common_dates:
        print("没有找到共同的交易日期")
        return pd.DataFrame(), {}
    
    # 使用numpy进行快速计算
    dates = np.array(common_dates)
    returns_matrix = np.zeros((len(stock_list), len(dates)-1))  # 注意：使用总股票数量
    
    # 计算每只股票的收益率
    for i, code in enumerate(stock_codes):
        if code in stock_data_dict:
            data = stock_data_dict[code]
            stock_data = data.loc[common_dates]
            returns_matrix[i] = np.diff(stock_data['close'].values) / stock_data['close'].values[:-1]
        else:
            # 对于获取失败的股票，设置收益率为0
            returns_matrix[i] = np.zeros(len(dates)-1)
    
    # 计算等权重组合收益率
    portfolio_returns = np.mean(returns_matrix, axis=0)  # 使用numpy的mean代替循环
    
    # 计算指数点位
    index_points = 1000 * np.cumprod(1 + portfolio_returns)  # 基点1000
    
    # 创建结果DataFrame
    index_data = pd.DataFrame({
        '日期': dates[1:],  # 跳过第一天
        '指数涨跌幅': portfolio_returns,
        '指数点位': index_points
    })
    
    # 计算统计指标
    stats = {}
    if len(index_data) > 0:
        # 计算最大回撤
        cummax = np.maximum.accumulate(index_points)
        drawdown = (cummax - index_points) / cummax
        stats['最大回撤'] = np.max(drawdown)
        
        # 添加数据获取成功率信息
        stats['数据获取成功率'] = (len(stock_codes) - len(failed_codes)) / len(stock_codes)
        stats['数据缺失股票'] = list(failed_codes)
    
    return index_data, stats

def get_latest_files_by_date(folder_path):
    """获取每个日期最新的文件，排除当天的文件"""
    files = {}
    today = datetime.now().strftime('%Y%m%d')
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            date_time = get_date_from_filename(filename)
            if date_time:
                date_str = date_time.strftime('%Y%m%d')
                # 跳过当天的文件
                if date_str == today:
                    continue
                if date_str not in files or date_time > get_date_from_filename(files[date_str]):
                    files[date_str] = filename
    return sorted(files.values())  # 返回排序后的文件列表

def main():
    # 初始化数据获取器
    fetcher = StockDataFetcher()
    
    # 文件夹路径
    folder_path = 'trading_signals'
    output_folder = 'index_performance'
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建统计信息汇总文件
    summary_file = os.path.join(output_folder, 'performance_summary.csv')
    summary_data = []
    
    # 获取每个日期最新的文件（排除当天的文件）
    latest_files = get_latest_files_by_date(folder_path)
    
    # 处理每个文件
    for signal_file in latest_files:
        try:
            full_path = os.path.join(folder_path, signal_file)
            print(f"\n处理文件: {signal_file}")
            
            # 获取开始日期
            start_date = get_date_from_filename(signal_file)
            if not start_date:
                continue
                
            # 加载股票列表
            stock_list = load_stock_list(full_path)
            if stock_list is None or len(stock_list) == 0:
                continue
            
            # 计算指数
            index_data, stats = calculate_index(stock_list, start_date, fetcher=fetcher)
            
            if len(index_data) > 0:
                # 保存指数数据
                output_file = os.path.join(output_folder, f'index_performance_{start_date.strftime("%Y%m%d")}.csv')
                index_data.to_csv(output_file, index=False)
                print(f"指数数据已保存到: {output_file}")
                
                # 计算统计指标
                total_return = (index_data['指数点位'].iloc[-1] / 1000 - 1) * 100
                
                # 收集统计信息
                summary = {
                    '信号日期': start_date.strftime('%Y-%m-%d'),
                    '起始日期': index_data['日期'].iloc[0],
                    '结束日期': index_data['日期'].iloc[-1],
                    '总收益率': f"{total_return:.2f}%",
                    '成分股数量': len(stock_list),
                    '最大回撤': f"{stats.get('最大回撤', 0)*100:.2f}%",
                    '累计点位': f"{index_data['指数点位'].iloc[-1]:.2f}",
                    '信号文件': signal_file
                }
                summary_data.append(summary)
                
                # 输出简要统计
                print(f"\n指数表现统计:")
                print(f"起始日期: {summary['起始日期']}")
                print(f"结束日期: {summary['结束日期']}")
                print(f"总收益率: {summary['总收益率']}")
                print(f"成分股数量: {summary['成分股数量']}")
                print(f"最大回撤: {summary['最大回撤']}")
                print(f"累计点位: {summary['累计点位']}")
            
        except Exception as e:
            print(f"处理文件 {signal_file} 时出错: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
    
    # 保存汇总统计信息
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('信号日期', ascending=False)  # 按日期降序排序
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n汇总统计信息已保存到: {summary_file}")

if __name__ == "__main__":
    main()
