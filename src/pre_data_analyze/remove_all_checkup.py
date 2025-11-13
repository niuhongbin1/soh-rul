"""
剔除所有Check-up测试段，只保留规律的充放电循环
输出与原始数据结构完全相同的CSV文件
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
FILE_PATHS = [
    './data/cell_log_age_30s_P051_1_S10_C08.csv',
    './data/cell_log_age_30s_P051_2_S11_C05.csv',
    './data/cell_log_age_30s_P051_3_S12_C05.csv',
]

OUTPUT_DIR = './data_parse/no_checkup'

# Check-up段识别参数
CHECKUP_TIME_WINDOW_BEFORE = 3600    # check-up前扩展时间（秒），约1小时
CHECKUP_TIME_WINDOW_AFTER = 7200     # check-up后扩展时间（秒），约2小时
# ===============================================


def identify_checkup_segments(df):
    """
    识别所有check-up段
    
    返回：需要剔除的时间段列表 [(start_idx, end_idx), ...]
    """
    # 找到所有check-up点（cap_aged_est_Ah有值的行）
    checkup_mask = df['cap_aged_est_Ah'].notna()
    checkup_indices = df[checkup_mask].index.tolist()
    
    if len(checkup_indices) == 0:
        print("  未发现check-up点")
        return []
    
    print(f"  发现 {len(checkup_indices)} 个check-up测量点")
    
    # 获取check-up点的时间戳
    checkup_timestamps = df.loc[checkup_indices, 'timestamp_s'].values
    
    # 为每个check-up点确定需要剔除的时间范围
    segments_to_remove = []
    
    for i, (idx, timestamp) in enumerate(zip(checkup_indices, checkup_timestamps)):
        # 确定时间窗口
        start_time = timestamp - CHECKUP_TIME_WINDOW_BEFORE
        end_time = timestamp + CHECKUP_TIME_WINDOW_AFTER
        
        # 找到对应的索引范围
        start_idx = df[df['timestamp_s'] >= start_time].index.min()
        end_idx = df[df['timestamp_s'] <= end_time].index.max()
        
        # 处理边界情况
        if pd.isna(start_idx):
            start_idx = df.index.min()
        if pd.isna(end_idx):
            end_idx = df.index.max()
        
        segments_to_remove.append((start_idx, end_idx))
        
        print(f"    Check-up {i+1}: timestamp={timestamp:.0f}s, "
              f"剔除范围: [{start_idx}, {end_idx}] "
              f"({end_idx - start_idx + 1}行)")
    
    return segments_to_remove


def merge_overlapping_segments(segments):
    """
    合并重叠的时间段
    """
    if not segments:
        return []
    
    # 按起始索引排序
    segments = sorted(segments, key=lambda x: x[0])
    
    merged = [segments[0]]
    
    for current in segments[1:]:
        last = merged[-1]
        
        # 如果当前段与上一段重叠或相邻，合并
        if current[0] <= last[1] + 1:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def remove_checkup_segments(df, segments):
    """
    剔除指定的时间段，返回剩余数据
    """
    if not segments:
        return df
    
    # 创建保留的掩码（初始全为True）
    keep_mask = pd.Series(True, index=df.index)
    
    # 将需要剔除的段标记为False
    for start_idx, end_idx in segments:
        keep_mask.loc[start_idx:end_idx] = False
    
    # 返回保留的数据
    df_filtered = df[keep_mask].copy()
    
    return df_filtered


def visualize_removal(df_original, df_filtered, output_path):
    """
    可视化剔除前后的数据对比
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 原始数据（灰色背景）
    axes[0].plot(df_original['timestamp_s'], df_original['soc_est'], 
                color='lightgray', alpha=0.5, linewidth=1, label='原始数据（包含check-up）')
    # 过滤后数据（蓝色）
    axes[0].plot(df_filtered['timestamp_s'], df_filtered['soc_est'], 
                color='blue', linewidth=1.5, label='过滤后数据（纯循环）')
    axes[0].set_ylabel('SOC (%)', fontsize=12)
    axes[0].set_title('SOC对比 - 剔除Check-up前后', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 电流
    axes[1].plot(df_original['timestamp_s'], df_original['i_raw_A'], 
                color='lightgray', alpha=0.5, linewidth=1)
    axes[1].plot(df_filtered['timestamp_s'], df_filtered['i_raw_A'], 
                color='red', linewidth=1.5)
    axes[1].set_ylabel('电流 (A)', fontsize=12)
    axes[1].set_title('电流对比', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 温度
    axes[2].plot(df_original['timestamp_s'], df_original['t_cell_degC'], 
                color='lightgray', alpha=0.5, linewidth=1)
    axes[2].plot(df_filtered['timestamp_s'], df_filtered['t_cell_degC'], 
                color='green', linewidth=1.5)
    axes[2].set_xlabel('时间戳 (s)', fontsize=12)
    axes[2].set_ylabel('温度 (°C)', fontsize=12)
    axes[2].set_title('温度对比', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  可视化图已保存: {output_path}")


def process_file(file_path, output_dir):
    """
    处理单个文件
    """
    print(f"\n{'='*70}")
    print(f"处理文件: {os.path.basename(file_path)}")
    print(f"{'='*70}")
    
    # 读取数据
    print("1. 读取数据...")
    df = pd.read_csv(file_path, delimiter=';')
    print(f"   原始数据: {len(df)} 行, {df.shape[1]} 列")
    print(f"   时间范围: {df['timestamp_s'].min():.0f}s - {df['timestamp_s'].max():.0f}s")
    
    # 识别check-up段
    print("\n2. 识别Check-up段...")
    segments = identify_checkup_segments(df)
    
    if not segments:
        print("   未发现check-up段，保存原始数据")
        df_filtered = df
    else:
        # 合并重叠段
        print("\n3. 合并重叠的时间段...")
        segments_merged = merge_overlapping_segments(segments)
        print(f"   合并后 {len(segments_merged)} 个独立的check-up段:")
        
        total_removed = 0
        for i, (start, end) in enumerate(segments_merged):
            n_rows = end - start + 1
            total_removed += n_rows
            print(f"     段 {i+1}: 索引 [{start}, {end}], {n_rows} 行")
        
        # 剔除check-up段
        print(f"\n4. 剔除Check-up段...")
        df_filtered = remove_checkup_segments(df, segments_merged)
        print(f"   剔除: {total_removed} 行 ({total_removed/len(df)*100:.2f}%)")
        print(f"   保留: {len(df_filtered)} 行 ({len(df_filtered)/len(df)*100:.2f}%)")
    
    # 保存过滤后的数据
    print(f"\n5. 保存过滤后的数据...")
    output_filename = os.path.join(output_dir, 
                                   f"no_checkup_{os.path.basename(file_path)}")
    df_filtered.to_csv(output_filename, sep=';', index=False)
    print(f"   ✓ 已保存: {output_filename}")
    
    # 生成对比图
    print(f"\n6. 生成可视化对比图...")
    plot_filename = os.path.join(output_dir, 
                                 f"comparison_{os.path.basename(file_path).replace('.csv', '.png')}")
    visualize_removal(df, df_filtered, plot_filename)
    
    # 统计信息
    print(f"\n7. 统计信息:")
    print(f"   EFC范围: {df_filtered['EFC'].min():.2f} - {df_filtered['EFC'].max():.2f}")
    print(f"   SOC范围: {df_filtered['soc_est'].min():.2f}% - {df_filtered['soc_est'].max():.2f}%")
    print(f"   温度范围: {df_filtered['t_cell_degC'].min():.2f}°C - {df_filtered['t_cell_degC'].max():.2f}°C")
    
    return df_filtered


def main():
    """
    主程序
    """
    print("="*70)
    print(" 剔除Check-up段工具 - 保留纯充放电循环数据".center(70))
    print("="*70)
    print(f"\n配置:")
    print(f"  Check-up前扩展时间: {CHECKUP_TIME_WINDOW_BEFORE}s ({CHECKUP_TIME_WINDOW_BEFORE/3600:.2f}小时)")
    print(f"  Check-up后扩展时间: {CHECKUP_TIME_WINDOW_AFTER}s ({CHECKUP_TIME_WINDOW_AFTER/3600:.2f}小时)")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  输出目录: {OUTPUT_DIR}")
    
    # 处理所有文件
    results = {}
    
    for file_path in FILE_PATHS:
        if not os.path.exists(file_path):
            print(f"\n✗ 文件不存在: {file_path}")
            continue
        
        try:
            df_filtered = process_file(file_path, OUTPUT_DIR)
            results[os.path.basename(file_path)] = df_filtered
        except Exception as e:
            print(f"\n✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print(f"\n{'='*70}")
    print(" 处理完成！".center(70))
    print(f"{'='*70}")
    print(f"\n处理文件数: {len(results)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n输出文件:")
    print(f"  - CSV: no_checkup_*.csv (与原始数据结构完全相同)")
    print(f"  - 图片: comparison_*.png (剔除前后对比)")
    
    print(f"\n说明:")
    print(f"  ✓ 已剔除所有check-up测试段")
    print(f"  ✓ 只保留规律的充放电循环数据")
    print(f"  ✓ 输出CSV与原始数据列结构完全相同")
    print(f"  ✓ 可用于后续的寿命预测建模")
    
    print(f"\n提示:")
    print(f"  - 如果剔除范围不合适，可调整时间窗口参数")
    print(f"  - CHECKUP_TIME_WINDOW_BEFORE: check-up前扩展时间")
    print(f"  - CHECKUP_TIME_WINDOW_AFTER: check-up后扩展时间")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

