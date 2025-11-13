"""
智能识别并剔除Check-up段
策略：基于数据特征自动识别check-up边界，而非固定时间窗口
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

OUTPUT_DIR = './data_parse/no_checkup_smart'

# 智能识别参数
SOC_CHANGE_THRESHOLD = 5.0        # SOC变化阈值(%)，用于识别完整充放电
CURRENT_LOW_THRESHOLD = 0.05      # 低电流阈值(A)，识别静置期
TEMPERATURE_CHANGE_THRESHOLD = 2.0 # 温度变化阈值(°C)
MIN_CYCLE_DURATION = 1800         # 最小循环时长(秒)，30分钟
SAFETY_MARGIN_BEFORE = 600        # 前向安全边界(秒)，10分钟
SAFETY_MARGIN_AFTER = 600         # 后向安全边界(秒)，10分钟
# ===============================================


def detect_full_cycles(df, start_idx, end_idx):
    """
    检测指定范围内是否有完整的0-100%充放电循环
    Check-up通常包含完整的容量测试循环
    """
    segment = df.loc[start_idx:end_idx]
    
    # 检查SOC是否经历了接近0和接近100的值
    soc_min = segment['soc_est'].min()
    soc_max = segment['soc_est'].max()
    
    has_full_cycle = (soc_min < 10) and (soc_max > 90)
    
    return has_full_cycle, soc_min, soc_max


def detect_rest_periods(df, start_idx, end_idx):
    """
    检测静置期（低电流期）
    Check-up前后通常有静置期
    """
    segment = df.loc[start_idx:end_idx]
    
    # 计算低电流比例
    low_current_mask = segment['i_raw_A'].abs() < CURRENT_LOW_THRESHOLD
    low_current_ratio = low_current_mask.sum() / len(segment)
    
    return low_current_ratio


def find_checkup_boundaries(df, checkup_idx):
    """
    智能寻找单个check-up的实际边界
    
    策略：
    1. 从check-up点向前搜索，找到上一个循环结束点（SOC回到稳态）
    2. 从check-up点向后搜索，找到下一个循环开始点（SOC恢复正常循环）
    3. 识别check-up相关的完整充放电过程
    """
    checkup_time = df.loc[checkup_idx, 'timestamp_s']
    
    # ========== 向前搜索起始边界 ==========
    print(f"    向前搜索起始边界...")
    
    # 搜索窗口：check-up前12小时
    search_start_time = checkup_time - 12 * 3600
    search_start_idx = df[df['timestamp_s'] >= search_start_time].index.min()
    if pd.isna(search_start_idx):
        search_start_idx = df.index.min()
    
    # 从check-up点向前搜索
    start_boundary = search_start_idx
    
    # 方法1：找到最后一个"正常循环结束"的点
    # 特征：SOC达到高位(>80%)或低位(<20%)后开始静置
    for idx in range(checkup_idx - 1, search_start_idx, -100):  # 每100行检查一次
        if idx < df.index.min():
            break
        
        # 检查前面一段时间(1小时)的数据特征
        window_start = max(search_start_idx, idx - 120)  # 120行约1小时
        
        # 如果这段时间内有完整循环，且之后开始异常，说明这是边界
        has_cycle, soc_min, soc_max = detect_full_cycles(df, window_start, idx)
        
        if has_cycle:
            # 检查之后是否开始出现异常（接近check-up）
            rest_ratio = detect_rest_periods(df, idx, min(checkup_idx, idx + 120))
            
            if rest_ratio > 0.3:  # 如果之后有较多静置期
                start_boundary = idx
                print(f"      找到起始边界: idx={idx}, 之后静置比例={rest_ratio:.2%}")
                break
    
    # 应用安全边界
    start_boundary = max(search_start_idx, start_boundary - 20)  # 向前扩展20行
    
    # ========== 向后搜索结束边界 ==========
    print(f"    向后搜索结束边界...")
    
    # 搜索窗口：check-up后12小时
    search_end_time = checkup_time + 12 * 3600
    search_end_idx = df[df['timestamp_s'] <= search_end_time].index.max()
    if pd.isna(search_end_idx):
        search_end_idx = df.index.max()
    
    # 从check-up点向后搜索
    end_boundary = search_end_idx
    
    # 方法：找到"恢复正常循环"的点
    # 特征：SOC开始正常的周期性变化
    for idx in range(checkup_idx + 1, search_end_idx, 100):
        if idx > df.index.max():
            break
        
        # 检查后面一段时间的数据特征
        window_end = min(search_end_idx, idx + 240)  # 240行约2小时
        
        # 如果这段时间内开始出现规律循环
        segment = df.loc[idx:window_end]
        
        # 检查电流是否恢复正常工作状态（非静置）
        high_current_ratio = (segment['i_raw_A'].abs() > 0.5).sum() / len(segment)
        
        # 检查SOC是否开始周期性变化
        soc_std = segment['soc_est'].rolling(window=20).std().mean()
        
        if high_current_ratio > 0.5 and soc_std > 5:  # 正常工作且有变化
            end_boundary = idx
            print(f"      找到结束边界: idx={idx}, 电流活跃={high_current_ratio:.2%}, SOC变化={soc_std:.2f}")
            break
    
    # 应用安全边界
    end_boundary = min(search_end_idx, end_boundary + 20)  # 向后扩展20行
    
    return start_boundary, end_boundary


def identify_checkup_segments_smart(df):
    """
    智能识别所有check-up段（基于数据特征）
    """
    # 找到所有check-up点
    checkup_mask = df['cap_aged_est_Ah'].notna()
    checkup_indices = df[checkup_mask].index.tolist()
    
    if len(checkup_indices) == 0:
        print("  未发现check-up点")
        return []
    
    print(f"  发现 {len(checkup_indices)} 个check-up测量点")
    print(f"  使用智能特征识别策略...")
    
    segments_to_remove = []
    
    for i, idx in enumerate(checkup_indices):
        timestamp = df.loc[idx, 'timestamp_s']
        print(f"\n  Check-up {i+1}/{len(checkup_indices)}: timestamp={timestamp:.0f}s")
        
        # 智能寻找边界
        start_idx, end_idx = find_checkup_boundaries(df, idx)
        
        # 计算时间范围
        start_time = df.loc[start_idx, 'timestamp_s']
        end_time = df.loc[end_idx, 'timestamp_s']
        duration = end_time - start_time
        
        segments_to_remove.append((start_idx, end_idx))
        
        print(f"    剔除范围: [{start_idx}, {end_idx}] ({end_idx - start_idx + 1}行)")
        print(f"    时间范围: {start_time:.0f}s - {end_time:.0f}s (持续 {duration/3600:.2f}小时)")
    
    return segments_to_remove


def merge_overlapping_segments(segments):
    """合并重叠的时间段"""
    if not segments:
        return []
    
    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]
    
    for current in segments[1:]:
        last = merged[-1]
        
        if current[0] <= last[1] + 1:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def remove_checkup_segments(df, segments):
    """剔除指定的时间段"""
    if not segments:
        return df
    
    keep_mask = pd.Series(True, index=df.index)
    
    for start_idx, end_idx in segments:
        keep_mask.loc[start_idx:end_idx] = False
    
    df_filtered = df[keep_mask].copy()
    
    return df_filtered


def visualize_removal_detailed(df_original, df_filtered, segments, output_path):
    """详细可视化剔除过程"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # SOC
    axes[0].plot(df_original['timestamp_s'], df_original['soc_est'], 
                color='lightgray', alpha=0.5, linewidth=1, label='原始数据')
    axes[0].plot(df_filtered['timestamp_s'], df_filtered['soc_est'], 
                color='blue', linewidth=1.5, label='保留数据')
    
    # 标记剔除的区域
    for start_idx, end_idx in segments:
        seg = df_original.loc[start_idx:end_idx]
        axes[0].axvspan(seg['timestamp_s'].min(), seg['timestamp_s'].max(), 
                       alpha=0.2, color='red', label='Check-up段' if start_idx == segments[0][0] else '')
    
    axes[0].set_ylabel('SOC (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('智能识别Check-up段 - SOC视图', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 电流
    axes[1].plot(df_original['timestamp_s'], df_original['i_raw_A'], 
                color='lightgray', alpha=0.5, linewidth=1)
    axes[1].plot(df_filtered['timestamp_s'], df_filtered['i_raw_A'], 
                color='red', linewidth=1.5)
    
    for start_idx, end_idx in segments:
        seg = df_original.loc[start_idx:end_idx]
        axes[1].axvspan(seg['timestamp_s'].min(), seg['timestamp_s'].max(), 
                       alpha=0.2, color='red')
    
    axes[1].set_ylabel('电流 (A)', fontsize=12, fontweight='bold')
    axes[1].set_title('电流视图', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Delta Q
    axes[2].plot(df_original['timestamp_s'], df_original['delta_q_Ah'], 
                color='lightgray', alpha=0.5, linewidth=1)
    axes[2].plot(df_filtered['timestamp_s'], df_filtered['delta_q_Ah'], 
                color='purple', linewidth=1.5)
    
    for start_idx, end_idx in segments:
        seg = df_original.loc[start_idx:end_idx]
        axes[2].axvspan(seg['timestamp_s'].min(), seg['timestamp_s'].max(), 
                       alpha=0.2, color='red')
    
    axes[2].set_ylabel('Delta Q (Ah)', fontsize=12, fontweight='bold')
    axes[2].set_title('累积电量视图', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # 温度
    axes[3].plot(df_original['timestamp_s'], df_original['t_cell_degC'], 
                color='lightgray', alpha=0.5, linewidth=1)
    axes[3].plot(df_filtered['timestamp_s'], df_filtered['t_cell_degC'], 
                color='green', linewidth=1.5)
    
    for start_idx, end_idx in segments:
        seg = df_original.loc[start_idx:end_idx]
        axes[3].axvspan(seg['timestamp_s'].min(), seg['timestamp_s'].max(), 
                       alpha=0.2, color='red')
    
    axes[3].set_xlabel('时间戳 (s)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('温度 (°C)', fontsize=12, fontweight='bold')
    axes[3].set_title('温度视图', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 详细对比图已保存: {output_path}")


def analyze_removed_segments(df_original, segments):
    """分析被剔除的段落特征"""
    print(f"\n  剔除段落分析:")
    
    for i, (start_idx, end_idx) in enumerate(segments):
        seg = df_original.loc[start_idx:end_idx]
        
        print(f"\n    段落 {i+1}:")
        print(f"      行数: {len(seg)}")
        print(f"      时长: {(seg['timestamp_s'].max() - seg['timestamp_s'].min())/3600:.2f} 小时")
        print(f"      SOC范围: {seg['soc_est'].min():.1f}% - {seg['soc_est'].max():.1f}%")
        print(f"      电流范围: {seg['i_raw_A'].min():.2f}A - {seg['i_raw_A'].max():.2f}A")
        print(f"      温度范围: {seg['t_cell_degC'].min():.1f}°C - {seg['t_cell_degC'].max():.1f}°C")
        print(f"      Delta Q变化: {seg['delta_q_Ah'].max() - seg['delta_q_Ah'].min():.4f} Ah")


def process_file(file_path, output_dir):
    """处理单个文件"""
    print(f"\n{'='*70}")
    print(f"处理文件: {os.path.basename(file_path)}")
    print(f"{'='*70}")
    
    # 1. 读取数据
    print("\n[1/6] 读取数据...")
    df = pd.read_csv(file_path, delimiter=';')
    print(f"  原始数据: {len(df)} 行, {df.shape[1]} 列")
    
    # 2. 智能识别check-up段
    print("\n[2/6] 智能识别Check-up段...")
    segments = identify_checkup_segments_smart(df)
    
    if not segments:
        print("  未发现check-up段，保存原始数据")
        df_filtered = df
        segments_merged = []
    else:
        # 3. 合并重叠段
        print("\n[3/6] 合并重叠的时间段...")
        segments_merged = merge_overlapping_segments(segments)
        print(f"  合并后 {len(segments_merged)} 个独立的check-up段")
        
        # 4. 分析被剔除的段
        print("\n[4/6] 分析被剔除的段落...")
        analyze_removed_segments(df, segments_merged)
        
        # 5. 剔除check-up段
        print(f"\n[5/6] 剔除Check-up段...")
        df_filtered = remove_checkup_segments(df, segments_merged)
        
        total_removed = sum(end - start + 1 for start, end in segments_merged)
        print(f"  总计剔除: {total_removed} 行 ({total_removed/len(df)*100:.2f}%)")
        print(f"  保留数据: {len(df_filtered)} 行 ({len(df_filtered)/len(df)*100:.2f}%)")
    
    # 6. 保存结果
    print(f"\n[6/6] 保存结果...")
    
    # 保存CSV
    output_filename = os.path.join(output_dir, 
                                   f"no_checkup_{os.path.basename(file_path)}")
    df_filtered.to_csv(output_filename, sep=';', index=False)
    print(f"  ✓ CSV已保存: {output_filename}")
    
    # 生成详细对比图
    plot_filename = os.path.join(output_dir, 
                                 f"smart_removal_{os.path.basename(file_path).replace('.csv', '.png')}")
    visualize_removal_detailed(df, df_filtered, segments_merged, plot_filename)
    
    return df_filtered


def main():
    """主程序"""
    print("="*70)
    print(" 智能Check-up段识别与剔除工具".center(70))
    print(" (基于数据特征的自适应策略)".center(70))
    print("="*70)
    
    print(f"\n识别策略:")
    print(f"  1. 基于 cap_aged_est_Ah 定位check-up测量点")
    print(f"  2. 分析SOC、电流、温度特征自动识别边界")
    print(f"  3. 识别完整的0-100%充放电循环")
    print(f"  4. 检测静置期和异常工作模式")
    print(f"  5. 自适应确定每个check-up的实际范围")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
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
    
    print(f"\n策略优势:")
    print(f"  ✓ 自动识别check-up边界，无需手动设置时间窗口")
    print(f"  ✓ 基于实际数据特征，剔除更精确")
    print(f"  ✓ 适应不同的check-up流程")
    print(f"  ✓ 生成详细的可视化分析图")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
