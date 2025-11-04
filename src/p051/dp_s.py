import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats
from scipy.interpolate import interp1d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


def remove_outliers_iqr(df, column, window_size=10, k=1.8):
    """
    使用滑动窗口IQR方法识别异常值
    返回异常值索引和清理后的数据
    
    参数:
    df: 输入DataFrame
    column: 需要检测异常值的列名
    window_size: 滑动窗口大小
    k: IQR倍数，控制异常值检测严格度
    """
    if len(df) < 2:
        print("数据点太少，无法进行异常值检测")
        return pd.Index([]), df.copy()
    
    # 创建异常值标记数组
    outlier_mask = pd.Series(False, index=df.index)
    
    # 对每个点计算其局部窗口的IQR边界
    for i in range(len(df)):
        # 确定窗口范围
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(df), i + window_size // 2 + 1)
        
        # 获取窗口数据
        window_data = df[column].iloc[start_idx:end_idx]
        
        # 确保窗口内有足够的数据点
        if len(window_data) >= 2:  # 至少需要2个点来计算IQR
            Q1 = window_data.quantile(0.25)
            Q3 = window_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # 处理IQR为0的情况（所有值相同）
            if IQR == 0:
                # 如果所有值相同，检查当前点是否明显偏离
                mean_val = window_data.mean()
                std_val = window_data.std()
                if std_val == 0:  # 所有值完全相同
                    # 如果所有值完全相同，任何点都不是异常值
                    continue
                else:
                    # 使用标准差检测异常值
                    if abs(df[column].iloc[i] - mean_val) > 3 * std_val:
                        outlier_mask.iloc[i] = True
            else:
                # 正常IQR计算
                lower_bound = Q1 - k * IQR
                upper_bound = Q3 + k * IQR
                
                if df[column].iloc[i] < lower_bound or df[column].iloc[i] > upper_bound:
                    outlier_mask.iloc[i] = True
    
    # 获取异常值索引
    outlier_indices = df[outlier_mask].index
    
    # 返回清理后的数据
    cleaned_df = df[~outlier_mask].copy()
    
    return outlier_indices, cleaned_df

# def remove_outliers_iqr(df, column):
#     """
#     使用IQR方法识别异常值
#     返回异常值索引和清理后的数据
#     """
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
    
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
    
#     # 识别异常值索引
#     outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    
#     # 返回清理后的数据
#     cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    
#     return outlier_indices, cleaned_df

def interpolate_outliers(df, x_col, y_col, outlier_indices):
    """
    对异常值进行插值补全
    """
    df_interpolated = df.copy()
    
    # 如果没有异常值，直接返回
    if len(outlier_indices) == 0:
        return df_interpolated
    
    # 创建插值函数
    valid_indices = df.index.difference(outlier_indices)
    
    if len(valid_indices) < 2:
        # 如果有效数据点太少，使用前后值的平均值
        for idx in outlier_indices:
            prev_val = df_interpolated.loc[df_interpolated.index < idx, y_col].tail(1)
            next_val = df_interpolated.loc[df_interpolated.index > idx, y_col].head(1)
            
            if not prev_val.empty and not next_val.empty:
                df_interpolated.loc[idx, y_col] = (prev_val.values[0] + next_val.values[0]) / 2
            elif not prev_val.empty:
                df_interpolated.loc[idx, y_col] = prev_val.values[0]
            elif not next_val.empty:
                df_interpolated.loc[idx, y_col] = next_val.values[0]
    else:
        # 使用线性插值
        valid_df = df_interpolated.loc[valid_indices].sort_values(by=x_col)
        x_valid = valid_df[x_col].values
        y_valid = valid_df[y_col].values
        
        # 确保x值是单调递增的
        if len(x_valid) > 1 and np.all(np.diff(x_valid) > 0):
            f_interp = interp1d(x_valid, y_valid, kind='linear', fill_value='extrapolate')
            
            # 对异常值进行插值
            for idx in outlier_indices:
                x_val = df_interpolated.loc[idx, x_col]
                if x_valid[0] <= x_val <= x_valid[-1]:
                    df_interpolated.loc[idx, y_col] = f_interp(x_val)
                else:
                    # 如果超出插值范围，使用最近的有效值
                    nearest_idx = valid_indices[np.argmin(np.abs(df_interpolated.loc[valid_indices, x_col] - x_val))]
                    df_interpolated.loc[idx, y_col] = df_interpolated.loc[nearest_idx, y_col]
        else:
            # 如果x不是单调的，使用索引顺序插值
            valid_df = df_interpolated.loc[valid_indices].sort_index()
            f_interp = interp1d(valid_df.index, valid_df[y_col], kind='linear', fill_value='extrapolate')
            
            for idx in outlier_indices:
                df_interpolated.loc[idx, y_col] = f_interp(idx)
    
    return df_interpolated

def process_and_plot_single_csv(file_path, output_dir, plot_dir):
    """
    处理单个CSV文件并绘制折线图对比
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        print(f"正在处理: {filename}")
        print(f"原始数据量: {len(df)}")
        
        # 检查数据列数
        if len(df.columns) < 2:
            print(f"警告: {filename} 中列数不足2列，跳过处理")
            return
        
        # 使用第一列作为横轴，第二列作为纵轴
        x_col = df.columns[0]
        y_col = df.columns[1]
        
        print(f"横轴列: {x_col}, 纵轴列: {y_col}")
        
        # 识别异常值
        outlier_indices, df_cleaned = remove_outliers_iqr(df, y_col)
        print(f"识别到异常值数量: {len(outlier_indices)}")
        
        # 对异常值进行插值补全
        df_interpolated = interpolate_outliers(df, x_col, y_col, outlier_indices)
        
        # 绘制对比图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'数据异常值处理对比 - {filename}', fontsize=16)
        
        # 原始数据折线图
        ax1.plot(df[x_col], df[y_col], 'b-', linewidth=1, alpha=0.7, label='原始数据')
        ax1.scatter(df.loc[outlier_indices, x_col], df.loc[outlier_indices, y_col], 
                   color='red', s=30, zorder=5, label='异常值')
        ax1.set_title('原始数据（红色点为异常值）')
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 清理后数据折线图
        ax2.plot(df_interpolated[x_col], df_interpolated[y_col], 'g-', linewidth=1.5, label='清理后数据')
        ax2.scatter(df_interpolated.loc[outlier_indices, x_col], df_interpolated.loc[outlier_indices, y_col], 
                   color='orange', s=30, zorder=5, label='插值点')
        ax2.set_title('清理后数据（橙色点为插值点）')
        ax2.set_xlabel(x_col)
        ax2.set_ylabel(y_col)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plot_filename = os.path.join(plot_dir, f'line_cleaned_{filename[:-4]}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"对比图已保存: {plot_filename}")
        
        # 保存清理后的数据
        output_filename = os.path.join(output_dir, f'cleaned_{filename}')
        if len(df_interpolated) > 510:
            df_interpolated[:601].to_csv(output_filename, index=False)
            print(f"清理后数据已保存: {output_filename}")
        else:
            df_interpolated[:501].to_csv(output_filename, index=False)
            print(f"清理后数据已保存: {output_filename}")
        # 显示异常值统计
        if len(outlier_indices) > 0:
            print("异常值详情:")
            outlier_details = df.loc[outlier_indices, [x_col, y_col]]
            print(outlier_details)
            print(f"异常值占比: {len(outlier_indices)/len(df)*100:.2f}%")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # 配置路径
    input_folder = "./src/p051/data_parse"        # 输入文件夹路径
    output_folder = "./src/p051/data_parse"     # 清理后数据保存路径
    plot_folder = "./src/p051/images"   # 对比图保存路径
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"在文件夹 {input_folder} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        process_and_plot_single_csv(csv_file, output_folder, plot_folder)
    
    print("所有文件处理完成！")
    
    # 生成处理摘要
    generate_summary(input_folder, output_folder)

def generate_summary(input_folder, output_folder):
    """
    生成处理摘要
    """
    input_files = glob.glob(os.path.join(input_folder, "*.csv"))
    output_files = glob.glob(os.path.join(output_folder, "cleaned_*.csv"))
    
    summary_data = []
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_folder, f'cleaned_{filename}')
        
        if os.path.exists(output_file):
            df_original = pd.read_csv(input_file)
            df_cleaned = pd.read_csv(output_file)
            
            # 计算异常值数量
            x_col = df_original.columns[0]
            y_col = df_original.columns[1]
            outlier_indices, _ = remove_outliers_iqr(df_original, y_col)
            
            summary_data.append({
                '文件名': filename,
                '横轴列': x_col,
                '纵轴列': y_col,
                '原始数据量': len(df_original),
                '异常值数量': len(outlier_indices),
                '异常值占比': f"{(len(outlier_indices) / len(df_original) * 100):.2f}%"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_folder, "处理摘要.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n处理摘要已保存: {summary_file}")
        print("\n处理摘要:")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()