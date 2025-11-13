"""
最优Check-up剔除策略 - 增强版 v2
1. 标记点识别 + 时间窗口扩展
2. SOC异常检测（变化快慢、突变）
3. delta_q异常检测（变化、幅值）
4. 短间隔检测：检测删除段之间的短间隔并标记（统计阈值法）
5. 一次性剔除所有标记段
6. 详细删除报告
7. 20份等分局部对比图（SOC、delta_q、EFC）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
FILE_PATHS = [
    './data/cell_log_age_30s_P051_1_S10_C08.csv',
    './data/cell_log_age_30s_P051_2_S11_C05.csv',
    './data/cell_log_age_30s_P051_3_S12_C05.csv',
]

OUTPUT_DIR = './data_parse/no_checkup_optimal'

# 扩展参数
EXPAND_BEFORE_HOURS = 10  # 向前扩展10小时
EXPAND_AFTER_HOURS = 12   # 向后扩展12小时

# 异常检测参数
DETECTION_WINDOW = 120           # 滑动窗口大小（约1小时）
SOC_STD_THRESHOLD = 2.0          # SOC标准差阈值（<2说明几乎不变）
SOC_JUMP_THRESHOLD = 15.0        # SOC突变阈值（单次变化>15%异常）
DELTA_Q_JUMP_THRESHOLD = 0.8     # delta_q跳变阈值（Ah）
DELTA_Q_STD_THRESHOLD = 0.01     # delta_q标准差阈值（几乎不变）
MIN_ANOMALY_DURATION = 60        # 最小异常持续时间（行数，约30分钟）

# 局部图配置
NUM_LOCAL_PLOTS = 20             # 生成20张局部图
RATED_CAPACITY_AH = 2.0          # 额定容量（用于计算EFC）

# 短间隔检测参数
ENABLE_SHORT_GAP_REMOVAL = True  # 是否启用短间隔剔除
MIN_GAPS_FOR_CLUSTERING = 3      # 最少需要多少个间隔才进行聚类
# ===============================================


def find_all_markers(df):
    """找到所有check-up标记点"""
    markers = set()
    
    cap_mask = df['cap_aged_est_Ah'].notna()
    cap_indices = df[cap_mask].index.tolist()
    markers.update(cap_indices)
    
    r0_mask = df['R0_mOhm'].notna()
    r0_indices = df[r0_mask].index.tolist()
    markers.update(r0_indices)
    
    r1_mask = df['R1_mOhm'].notna()
    r1_indices = df[r1_mask].index.tolist()
    markers.update(r1_indices)
    
    markers = sorted(list(markers))
    
    print(f"  标记点总数: {len(markers)}")
    print(f"    - 容量测试: {len(cap_indices)}")
    print(f"    - R0测试: {len(r0_indices)}")
    print(f"    - R1测试: {len(r1_indices)}")
    
    return markers


def expand_marker_to_segment(df, marker_idx, hours_before, hours_after):
    """将标记点扩展为时间段"""
    marker_time = df.loc[marker_idx, 'timestamp_s']
    
    start_time = marker_time - hours_before * 3600
    end_time = marker_time + hours_after * 3600
    
    start_candidates = df[df['timestamp_s'] >= start_time].index
    start_idx = start_candidates[0] if len(start_candidates) > 0 else df.index.min()
    
    end_candidates = df[df['timestamp_s'] <= end_time].index
    end_idx = end_candidates[-1] if len(end_candidates) > 0 else df.index.max()
    
    return start_idx, end_idx


def detect_soc_anomalies(df, window_size, std_threshold, jump_threshold):
    """
    检测SOC异常段
    - SOC持续不变（std < threshold）
    - SOC突然跳变（单次变化 > threshold）
    """
    anomaly_segments = []
    
    # 计算滑动标准差
    soc_std = df['soc_est'].rolling(window=window_size, center=True).std()
    
    # 计算单步变化
    soc_diff = df['soc_est'].diff().abs()
    
    # 标记异常点
    low_std_mask = soc_std < std_threshold
    high_jump_mask = soc_diff > jump_threshold
    
    anomaly_mask = low_std_mask | high_jump_mask
    
    # 连续异常段
    anomaly_indices = df[anomaly_mask].index.tolist()
    
    if not anomaly_indices:
        return []
    
    # 将连续的异常点合并为段
    current_start = anomaly_indices[0]
    current_end = anomaly_indices[0]
    
    for idx in anomaly_indices[1:]:
        if idx == current_end + 1:
            current_end = idx
        else:
            # 段结束，保存
            if current_end - current_start + 1 >= MIN_ANOMALY_DURATION:
                anomaly_segments.append((current_start, current_end, 'SOC异常'))
            current_start = idx
            current_end = idx
    
    # 最后一段
    if current_end - current_start + 1 >= MIN_ANOMALY_DURATION:
        anomaly_segments.append((current_start, current_end, 'SOC异常'))
    
    return anomaly_segments


def detect_delta_q_anomalies(df, window_size, std_threshold, jump_threshold):
    """
    检测delta_q异常段
    - delta_q持续不变（std < threshold）
    - delta_q突然跳变（单次变化 > threshold）
    """
    anomaly_segments = []
    
    # 计算滑动标准差
    dq_std = df['delta_q_Ah'].rolling(window=window_size, center=True).std()
    
    # 计算单步变化
    dq_diff = df['delta_q_Ah'].diff().abs()
    
    # 标记异常点
    low_std_mask = dq_std < std_threshold
    high_jump_mask = dq_diff > jump_threshold
    
    anomaly_mask = low_std_mask | high_jump_mask
    
    # 连续异常段
    anomaly_indices = df[anomaly_mask].index.tolist()
    
    if not anomaly_indices:
        return []
    
    # 合并为段
    current_start = anomaly_indices[0]
    current_end = anomaly_indices[0]
    
    for idx in anomaly_indices[1:]:
        if idx == current_end + 1:
            current_end = idx
        else:
            if current_end - current_start + 1 >= MIN_ANOMALY_DURATION:
                anomaly_segments.append((current_start, current_end, 'delta_q异常'))
            current_start = idx
            current_end = idx
    
    # 最后一段
    if current_end - current_start + 1 >= MIN_ANOMALY_DURATION:
        anomaly_segments.append((current_start, current_end, 'delta_q异常'))
    
    return anomaly_segments


def merge_segments_with_labels(segments):
    """合并重叠段（保留标签）"""
    if not segments:
        return []
    
    # 标准化：将所有label统一转换为列表格式
    normalized_segments = []
    for start, end, label in segments:
        if isinstance(label, list):
            normalized_segments.append((start, end, label))
        else:
            normalized_segments.append((start, end, [label]))
    
    # 按起始索引排序
    normalized_segments = sorted(normalized_segments, key=lambda x: x[0])
    
    merged = [normalized_segments[0]]
    
    for start, end, labels in normalized_segments[1:]:
        last_start, last_end, last_labels = merged[-1]
        
        if start <= last_end + 1:
            # 合并
            new_end = max(last_end, end)
            new_labels = last_labels + labels
            merged[-1] = (last_start, new_end, new_labels)
        else:
            merged.append((start, end, labels))
    
    return merged


def calculate_efc(df, rated_capacity):
    """
    计算等效循环次数 (EFC)
    EFC = 累积Ah吞吐量 / (2 * 额定容量)
    """
    # 累积绝对delta_q
    cumulative_throughput = df['delta_q_Ah'].abs().cumsum()
    
    # EFC = 累积吞吐量 / (2 * 额定容量)
    efc = cumulative_throughput / (2 * rated_capacity)
    
    return efc


def detect_short_gaps_between_removals(df_orig, removal_segments, rated_capacity):
    """
    检测删除段之间的短间隔，并标记为需要删除
    
    策略：
    1. 计算相邻删除段之间的时间间隔和EFC间隔
    2. 使用统计阈值（中位数）将间隔分为"短"和"长"两类
    3. 返回需要剔除的短间隔段（在删除段之间的短间隔）
    """
    if len(removal_segments) < 2:
        return []
    
    print(f"\n  检测删除段之间的短间隔:")
    print(f"    删除段数量: {len(removal_segments)}")
    
    # 计算EFC
    df_efc = calculate_efc(df_orig, rated_capacity)
    
    # 计算相邻删除段之间的间隔
    gap_info = []
    for i in range(len(removal_segments) - 1):
        seg1_end = removal_segments[i][1]
        seg2_start = removal_segments[i + 1][0]
        
        # 间隔区域的索引
        gap_start = seg1_end + 1
        gap_end = seg2_start - 1
        
        if gap_end < gap_start:
            # 两个删除段相邻或重叠，没有间隔
            continue
        
        # 时间间隔（小时）
        time_gap = (df_orig.loc[gap_end, 'timestamp_s'] - 
                   df_orig.loc[gap_start, 'timestamp_s']) / 3600
        
        # EFC间隔
        efc_gap = df_efc.loc[gap_end] - df_efc.loc[gap_start]
        
        gap_info.append({
            'gap_index': i,
            'gap_start_idx': gap_start,
            'gap_end_idx': gap_end,
            'time_gap_hours': time_gap,
            'efc_gap': efc_gap,
            'n_rows': gap_end - gap_start + 1
        })
    
    if len(gap_info) < MIN_GAPS_FOR_CLUSTERING:
        print(f"    间隔数量 {len(gap_info)} < {MIN_GAPS_FOR_CLUSTERING}，跳过短间隔检测")
        return []
    
    # 统计间隔
    time_gaps = np.array([g['time_gap_hours'] for g in gap_info])
    efc_gaps = np.array([g['efc_gap'] for g in gap_info])
    
    print(f"\n    间隔统计（共{len(gap_info)}个间隔）:")
    print(f"      时间间隔: 最小={time_gaps.min():.1f}h, 最大={time_gaps.max():.1f}h, "
          f"中位数={np.median(time_gaps):.1f}h, 均值={time_gaps.mean():.1f}h")
    print(f"      EFC间隔:  最小={efc_gaps.min():.2f}, 最大={efc_gaps.max():.2f}, "
          f"中位数={np.median(efc_gaps):.2f}, 均值={efc_gaps.mean():.2f}")
    
    # 使用双阈值策略识别短间隔
    # 阈值1: 时间间隔 < 中位数 * 0.3
    # 阈值2: EFC间隔 < 中位数 * 0.3
    # 满足任一条件即认为是短间隔
    
    time_median = np.median(time_gaps)
    efc_median = np.median(efc_gaps)
    
    time_threshold =8
    efc_threshold = efc_median * 0.3
    
    print(f"\n    短间隔阈值:")
    print(f"      时间阈值: < {time_threshold:.1f}h (中位数的30%)")
    print(f"      EFC阈值:  < {efc_threshold:.2f} (中位数的30%)")
    
    # 识别短间隔
    short_gaps = []
    long_gaps = []
    
    for g in gap_info:
        is_short_time = g['time_gap_hours'] < time_threshold
        is_short_efc = g['efc_gap'] < efc_threshold
        
        # 任一维度是短间隔，则认为是短间隔
        if is_short_time or is_short_efc:
            short_gaps.append(g)
        else:
            long_gaps.append(g)
    
    print(f"\n    分类结果:")
    print(f"      短间隔: {len(short_gaps)} 个")
    if short_gaps:
        short_time = [g['time_gap_hours'] for g in short_gaps]
        short_efc = [g['efc_gap'] for g in short_gaps]
        print(f"        时间范围: {min(short_time):.1f}h - {max(short_time):.1f}h "
              f"(均值={np.mean(short_time):.1f}h)")
        print(f"        EFC范围: {min(short_efc):.2f} - {max(short_efc):.2f} "
              f"(均值={np.mean(short_efc):.2f})")
        
        # 显示具体的短间隔段
        print(f"\n        短间隔详情:")
        for g in short_gaps:
            print(f"          间隔{g['gap_index']+1}: 索引[{g['gap_start_idx']}, {g['gap_end_idx']}], "
                  f"时间={g['time_gap_hours']:.1f}h, EFC={g['efc_gap']:.2f}, 行数={g['n_rows']}")
    
    print(f"      长间隔: {len(long_gaps)} 个")
    if long_gaps:
        long_time = [g['time_gap_hours'] for g in long_gaps]
        long_efc = [g['efc_gap'] for g in long_gaps]
        print(f"        时间范围: {min(long_time):.1f}h - {max(long_time):.1f}h "
              f"(均值={np.mean(long_time):.1f}h)")
        print(f"        EFC范围: {min(long_efc):.2f} - {max(long_efc):.2f} "
              f"(均值={np.mean(long_efc):.2f})")
    
    # 返回需要剔除的短间隔段
    short_gap_segments = [(g['gap_start_idx'], g['gap_end_idx'], '短间隔') 
                          for g in short_gaps]
    
    print(f"\n    将剔除 {len(short_gap_segments)} 个短间隔段")
    
    return short_gap_segments


def remove_segments(df, segments):
    """剔除段落"""
    if not segments:
        return df.copy()
    
    keep_mask = pd.Series(True, index=df.index)
    
    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[1]
        keep_mask.loc[start_idx:end_idx] = False
    
    return df[keep_mask].copy()


def print_removal_report(df, segments):
    """打印详细的删除报告"""
    print(f"\n{'='*80}")
    print("删除段落详细报告".center(80))
    print(f"{'='*80}")
    
    if not segments:
        print("  未检测到需要删除的段落")
        return
    
    total_removed = 0
    
    for i, (start, end, labels) in enumerate(segments):
        seg = df.loc[start:end]
        
        # 时间信息
        start_time = seg['timestamp_s'].iloc[0]
        end_time = seg['timestamp_s'].iloc[-1]
        duration_hours = (end_time - start_time) / 3600
        
        # 数据特征
        n_rows = end - start + 1
        total_removed += n_rows
        
        soc_range = seg['soc_est'].max() - seg['soc_est'].min()
        soc_mean = seg['soc_est'].mean()
        soc_std = seg['soc_est'].std()
        
        current_range = seg['i_raw_A'].max() - seg['i_raw_A'].min()
        current_mean = seg['i_raw_A'].mean()
        
        dq_range = seg['delta_q_Ah'].max() - seg['delta_q_Ah'].min()
        dq_std = seg['delta_q_Ah'].std()
        
        temp_range = seg['t_cell_degC'].max() - seg['t_cell_degC'].min()
        
        # 标记
        has_cap = seg['cap_aged_est_Ah'].notna().any()
        has_r0 = seg['R0_mOhm'].notna().any()
        has_r1 = seg['R1_mOhm'].notna().any()
        
        # 处理labels（可能是嵌套列表）
        if isinstance(labels, list):
            flat_labels = []
            for item in labels:
                if isinstance(item, list):
                    flat_labels.extend(item)
                else:
                    flat_labels.append(item)
            reason = ', '.join(set(flat_labels))
        else:
            reason = str(labels)
        
        print(f"\n段落 {i+1}/{len(segments)}:")
        print(f"  {'─'*76}")
        print(f"  索引范围:    [{start:8d}, {end:8d}]  ({n_rows:6d} 行)")
        print(f"  时间范围:    {start_time:12.0f}s - {end_time:12.0f}s")
        print(f"  持续时长:    {duration_hours:6.2f} 小时")
        print(f"  删除原因:    {reason}")
        
        print(f"\n  数据特征:")
        print(f"    SOC:       范围={soc_range:5.1f}%  均值={soc_mean:5.1f}%  标准差={soc_std:5.2f}%")
        print(f"    电流:      范围={current_range:5.2f}A  均值={current_mean:6.3f}A")
        print(f"    Delta Q:   范围={dq_range:6.3f}Ah  标准差={dq_std:7.4f}Ah")
        print(f"    温度:      范围={temp_range:5.1f}°C")
        
        print(f"\n  测试标记:")
        print(f"    容量测试:  {'✓' if has_cap else '✗'}")
        print(f"    R0测试:    {'✓' if has_r0 else '✗'}")
        print(f"    R1测试:    {'✓' if has_r1 else '✗'}")
    
    print(f"\n{'='*80}")
    print(f"总计删除: {total_removed} 行 ({total_removed/len(df)*100:.2f}%)")
    print(f"{'='*80}\n")


def visualize_global(df_orig, df_filt, segments, path):
    """全局对比图"""
    fig, axes = plt.subplots(4, 1, figsize=(20, 14))
    
    # SOC
    axes[0].plot(df_orig['timestamp_s'], df_orig['soc_est'], 
                'lightgray', alpha=0.4, linewidth=0.8, label='原始数据')
    axes[0].plot(df_filt['timestamp_s'], df_filt['soc_est'], 
                'blue', linewidth=1.8, label='保留数据', zorder=5)
    
    for seg in segments:
        start, end, labels = seg
        seg_data = df_orig.loc[start:end]
        axes[0].axvspan(seg_data['timestamp_s'].min(), seg_data['timestamp_s'].max(),
                       alpha=0.2, color='red')
    
    axes[0].set_ylabel('SOC (%)', fontsize=14, fontweight='bold')
    axes[0].set_title('全局对比 - Check-up剔除效果', fontsize=16, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 电流
    axes[1].plot(df_orig['timestamp_s'], df_orig['i_raw_A'], 
                'lightgray', alpha=0.4, linewidth=0.8)
    axes[1].plot(df_filt['timestamp_s'], df_filt['i_raw_A'], 
                'red', linewidth=1.8, zorder=5)
    
    for seg in segments:
        start, end, labels = seg
        seg_data = df_orig.loc[start:end]
        axes[1].axvspan(seg_data['timestamp_s'].min(), seg_data['timestamp_s'].max(),
                       alpha=0.2, color='red')
    
    axes[1].set_ylabel('电流 (A)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    
    # Delta Q
    axes[2].plot(df_orig['timestamp_s'], df_orig['delta_q_Ah'], 
                'lightgray', alpha=0.4, linewidth=0.8)
    axes[2].plot(df_filt['timestamp_s'], df_filt['delta_q_Ah'], 
                'purple', linewidth=1.8, zorder=5)
    
    for seg in segments:
        start, end, labels = seg
        seg_data = df_orig.loc[start:end]
        axes[2].axvspan(seg_data['timestamp_s'].min(), seg_data['timestamp_s'].max(),
                       alpha=0.2, color='red')
    
    axes[2].set_ylabel('Delta Q (Ah)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # 温度
    axes[3].plot(df_orig['timestamp_s'], df_orig['t_cell_degC'], 
                'lightgray', alpha=0.4, linewidth=0.8)
    axes[3].plot(df_filt['timestamp_s'], df_filt['t_cell_degC'], 
                'green', linewidth=1.8, zorder=5)
    
    for seg in segments:
        start, end, labels = seg
        seg_data = df_orig.loc[start:end]
        axes[3].axvspan(seg_data['timestamp_s'].min(), seg_data['timestamp_s'].max(),
                       alpha=0.2, color='red')
    
    axes[3].set_xlabel('时间戳 (s)', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('温度 (°C)', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_local_segments(df_orig, df_filt, segments, output_dir, file_name, num_plots, rated_capacity):
    """
    生成等分局部对比图
    将整个数据时间轴等分成num_plots份，每份生成一张局部对比图
    显示：SOC、delta_q、EFC
    """
    print(f"\n  生成{num_plots}张等分局部对比图...")
    
    # 计算EFC
    df_orig_efc = calculate_efc(df_orig, rated_capacity)
    df_filt_efc = calculate_efc(df_filt, rated_capacity)
    
    # 获取时间范围
    time_min = df_orig['timestamp_s'].min()
    time_max = df_orig['timestamp_s'].max()
    time_span = time_max - time_min
    
    # 等分时间段
    time_step = time_span / num_plots
    
    for i in range(num_plots):
        # 当前段的时间范围
        seg_time_start = time_min + i * time_step
        seg_time_end = time_min + (i + 1) * time_step
        
        # 提取当前段的数据
        orig_mask = (df_orig['timestamp_s'] >= seg_time_start) & (df_orig['timestamp_s'] <= seg_time_end)
        plot_orig = df_orig[orig_mask].copy()
        plot_orig_efc = df_orig_efc[orig_mask]
        
        filt_mask = (df_filt['timestamp_s'] >= seg_time_start) & (df_filt['timestamp_s'] <= seg_time_end)
        plot_filt = df_filt[filt_mask].copy()
        plot_filt_efc = df_filt_efc[filt_mask]
        
        if len(plot_orig) == 0:
            continue
        
        # 创建图形 - 3个子图
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        # 1. SOC
        axes[0].plot(plot_orig['timestamp_s'], plot_orig['soc_est'], 
                    'gray', linewidth=1, alpha=0.6, label='原始数据')
        if len(plot_filt) > 0:
            axes[0].plot(plot_filt['timestamp_s'], plot_filt['soc_est'], 
                        'blue', linewidth=2, label='保留数据', zorder=5)
        
        # 标记删除区域
        for seg in segments:
            start, end, labels = seg
            seg_data = df_orig.loc[start:end]
            seg_time_min = seg_data['timestamp_s'].min()
            seg_time_max = seg_data['timestamp_s'].max()
            
            # 如果删除段与当前时间段有交集
            if seg_time_max >= seg_time_start and seg_time_min <= seg_time_end:
                axes[0].axvspan(max(seg_time_min, seg_time_start), 
                               min(seg_time_max, seg_time_end),
                               alpha=0.3, color='red')
        
        axes[0].set_ylabel('SOC (%)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'局部对比 {i+1}/{num_plots} - 时间段: {seg_time_start/3600:.1f}h - {seg_time_end/3600:.1f}h', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Delta Q
        axes[1].plot(plot_orig['timestamp_s'], plot_orig['delta_q_Ah'], 
                    'gray', linewidth=1, alpha=0.6)
        if len(plot_filt) > 0:
            axes[1].plot(plot_filt['timestamp_s'], plot_filt['delta_q_Ah'], 
                        'purple', linewidth=2, zorder=5)
        
        for seg in segments:
            start, end, labels = seg
            seg_data = df_orig.loc[start:end]
            seg_time_min = seg_data['timestamp_s'].min()
            seg_time_max = seg_data['timestamp_s'].max()
            
            if seg_time_max >= seg_time_start and seg_time_min <= seg_time_end:
                axes[1].axvspan(max(seg_time_min, seg_time_start), 
                               min(seg_time_max, seg_time_end),
                               alpha=0.3, color='red')
        
        axes[1].set_ylabel('Delta Q (Ah)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. EFC
        axes[2].plot(plot_orig['timestamp_s'], plot_orig_efc, 
                    'gray', linewidth=1, alpha=0.6)
        if len(plot_filt) > 0:
            axes[2].plot(plot_filt['timestamp_s'], plot_filt_efc, 
                        'green', linewidth=2, zorder=5)
        
        for seg in segments:
            start, end, labels = seg
            seg_data = df_orig.loc[start:end]
            seg_time_min = seg_data['timestamp_s'].min()
            seg_time_max = seg_data['timestamp_s'].max()
            
            if seg_time_max >= seg_time_start and seg_time_min <= seg_time_end:
                axes[2].axvspan(max(seg_time_min, seg_time_start), 
                               min(seg_time_max, seg_time_end),
                               alpha=0.3, color='red')
        
        axes[2].set_xlabel('时间戳 (s)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('等效循环 (EFC)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        base_name = file_name.replace('.csv', '')
        local_path = os.path.join(output_dir, f'local_{base_name}_part{i+1:02d}.png')
        plt.savefig(local_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if (i + 1) % 5 == 0:
            print(f"    ✓ 已生成 {i+1}/{num_plots} 张局部图")
    
    print(f"    ✓ 全部局部图生成完成")


def process_file(fpath, outdir, expand_before, expand_after):
    """处理单个文件"""
    print(f"\n{'='*80}")
    print(f"处理文件: {os.path.basename(fpath)}")
    print(f"{'='*80}")
    
    # 1. 读取数据
    print("\n[1/9] 读取数据...")
    df = pd.read_csv(fpath, delimiter=';')
    print(f"  数据形状: {df.shape}")
    print(f"  时间范围: {df['timestamp_s'].min():.0f}s - {df['timestamp_s'].max():.0f}s")
    print(f"  总时长: {(df['timestamp_s'].max() - df['timestamp_s'].min())/3600:.1f} 小时")
    
    # 2. 找标记点
    print("\n[2/9] 识别标记点...")
    markers = find_all_markers(df)
    
    all_segments = []
    
    if markers:
        # 3. 扩展标记点
        print(f"\n[3/9] 扩展标记点（前{expand_before}h，后{expand_after}h）...")
        for i, marker_idx in enumerate(markers):
            start_idx, end_idx = expand_marker_to_segment(df, marker_idx, 
                                                          expand_before, expand_after)
            all_segments.append((start_idx, end_idx, '标记点扩展'))
            print(f"  标记{i+1}: [{start_idx}, {end_idx}]")
    
    # 4. SOC异常检测
    print(f"\n[4/9] SOC异常检测...")
    soc_anomalies = detect_soc_anomalies(df, DETECTION_WINDOW, 
                                         SOC_STD_THRESHOLD, SOC_JUMP_THRESHOLD)
    print(f"  检测到SOC异常段: {len(soc_anomalies)}")
    all_segments.extend(soc_anomalies)
    
    # 5. delta_q异常检测
    print(f"\n[5/9] delta_q异常检测...")
    dq_anomalies = detect_delta_q_anomalies(df, DETECTION_WINDOW,
                                            DELTA_Q_STD_THRESHOLD, DELTA_Q_JUMP_THRESHOLD)
    print(f"  检测到delta_q异常段: {len(dq_anomalies)}")
    all_segments.extend(dq_anomalies)
    
    # 6. 合并重叠段
    print(f"\n[6/9] 合并重叠段...")
    print(f"  合并前: {len(all_segments)} 个段")
    merged_segments = merge_segments_with_labels(all_segments)
    print(f"  合并后: {len(merged_segments)} 个段")
    
    # 7. 短间隔检测
    all_removal_segments = merged_segments
    
    if ENABLE_SHORT_GAP_REMOVAL and len(merged_segments) > 1:
        print(f"\n[7/9] 检测删除段之间的短间隔...")
        
        # 检测短间隔（在删除段之间）
        short_gap_segments = detect_short_gaps_between_removals(df, merged_segments, RATED_CAPACITY_AH)
        
        if short_gap_segments:
            print(f"\n  合并短间隔段到删除列表...")
            # 合并所有剔除段（原删除段 + 短间隔段）
            all_removal_segments = merged_segments + short_gap_segments
            print(f"    原删除段: {len(merged_segments)} 个")
            print(f"    短间隔段: {len(short_gap_segments)} 个")
            
            # 重新合并（处理可能的重叠）
            all_removal_segments = merge_segments_with_labels(all_removal_segments)
            print(f"    合并后总段数: {len(all_removal_segments)} 个")
        else:
            print(f"  未检测到短间隔")
    else:
        print(f"\n[7/9] 短间隔检测已禁用或删除段数量不足")
    
    # 8. 剔除所有标记的段
    print(f"\n[8/9] 剔除所有标记段...")
    df_filtered_final = remove_segments(df, all_removal_segments)
    
    total_removed = sum(seg[1] - seg[0] + 1 for seg in all_removal_segments)
    print(f"  原始数据: {len(df)} 行")
    print(f"  剔除数据: {total_removed} 行 ({total_removed/len(df)*100:.2f}%)")
    print(f"  保留数据: {len(df_filtered_final)} 行 ({len(df_filtered_final)/len(df)*100:.2f}%)")
    
    # 打印详细报告
    print_removal_report(df, all_removal_segments)
    
    # 验证
    print(f"\n  验证剔除效果:")
    print(f"    剩余cap_aged: {df_filtered_final['cap_aged_est_Ah'].notna().sum()}")
    print(f"    剩余R0: {df_filtered_final['R0_mOhm'].notna().sum()}")
    print(f"    剩余R1: {df_filtered_final['R1_mOhm'].notna().sum()}")
    
    # 9. 保存
    print(f"\n[9/9] 保存结果...")
    
    fname = f"no_checkup_{os.path.basename(fpath)}"
    
    # 保存CSV（最终结果）
    csv_path = os.path.join(outdir, fname)
    df_filtered_final.to_csv(csv_path, sep=';', index=False)
    print(f"  ✓ CSV: {csv_path}")
    
    # 全局对比图（原始数据 vs 最终结果 + 所有剔除段）
    global_path = os.path.join(outdir, f"global_{fname.replace('.csv', '.png')}")
    visualize_global(df, df_filtered_final, all_removal_segments, global_path)
    print(f"  ✓ 全局图: {global_path}")
    
    # 等分局部对比图（原始数据 vs 最终结果 + 所有剔除段）
    visualize_local_segments(df, df_filtered_final, all_removal_segments, outdir, fname, 
                             NUM_LOCAL_PLOTS, RATED_CAPACITY_AH)
    
    return df_filtered_final


def main():
    """主程序"""
    print("="*80)
    print("Check-up剔除工具 - 增强版 v2".center(80))
    print("="*80)
    
    print(f"\n策略:")
    print(f"  1. 标记点识别（cap_aged + R0 + R1）")
    print(f"  2. 时间窗口扩展（前{EXPAND_BEFORE_HOURS}h，后{EXPAND_AFTER_HOURS}h）")
    print(f"  3. SOC异常检测")
    print(f"     - 标准差 < {SOC_STD_THRESHOLD}% (持续不变)")
    print(f"     - 单次变化 > {SOC_JUMP_THRESHOLD}% (突变)")
    print(f"  4. delta_q异常检测")
    print(f"     - 标准差 < {DELTA_Q_STD_THRESHOLD} Ah (持续不变)")
    print(f"     - 单次变化 > {DELTA_Q_JUMP_THRESHOLD} Ah (跳变)")
    print(f"  5. 合并重叠段")
    print(f"  6. 短间隔检测（统计阈值法）")
    print(f"     - 检测删除段之间的短间隔")
    print(f"     - 自动识别并标记短间隔段（< 中位数30%）")
    print(f"  7. 一次性剔除所有标记段")
    print(f"  8. 生成详细报告和{NUM_LOCAL_PLOTS}张等分局部对比图")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    for fpath in FILE_PATHS:
        if os.path.exists(fpath):
            try:
                process_file(fpath, OUTPUT_DIR, EXPAND_BEFORE_HOURS, EXPAND_AFTER_HOURS)
            except Exception as e:
                print(f"\n✗ 处理失败: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("完成！".center(80))
    print(f"{'='*80}")
    print(f"\n输出文件:")
    print(f"  - no_checkup_*.csv: 剔除后的数据")
    print(f"  - global_*.png: 全局对比图")
    print(f"  - local_*_part01~20.png: {NUM_LOCAL_PLOTS}张等分局部对比图（SOC、delta_q、EFC）")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
