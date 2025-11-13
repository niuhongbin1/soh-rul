"""
反向策略：识别并保留安全的循环段，剔除所有可疑区域
思路：找出所有标记点，标记点之间间隔很大的才是安全循环段
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

OUTPUT_DIR = './data_parse/no_checkup_reverse'

# 安全间隔：两个标记点之间至少间隔多少小时才认为是安全的循环段
MIN_SAFE_GAP_HOURS = 24  # 24小时
EXPAND_MARGIN_HOURS = 8  # 额外安全边界：8小时
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
    
    print(f"  发现标记点: {len(markers)} 个")
    print(f"    - 容量测试: {len(cap_indices)} 个")
    print(f"    - R0测试: {len(r0_indices)} 个")
    print(f"    - R1测试: {len(r1_indices)} 个")
    
    # 显示标记点的时间分布
    if markers:
        marker_times = df.loc[markers, 'timestamp_s'].values
        print(f"  标记点时间分布:")
        for i, (idx, time) in enumerate(zip(markers, marker_times)):
            print(f"    标记 {i+1}: idx={idx}, time={time:.0f}s ({time/3600:.1f}h)")
    
    return markers


def identify_safe_segments(df, markers, min_gap_hours, margin_hours):
    """
    识别安全的循环段
    策略：标记点之间间隔足够大的区域才是安全区域
    """
    if not markers:
        print("  无标记点，保留全部数据")
        return [(df.index.min(), df.index.max())]
    
    marker_times = df.loc[markers, 'timestamp_s'].values
    safe_segments = []
    
    print(f"\n  识别安全循环段（最小间隔>{min_gap_hours}h，边界{margin_hours}h）:")
    
    # 检查第一个标记点之前
    first_marker_time = marker_times[0]
    first_marker_idx = markers[0]
    
    # 数据起点
    data_start_time = df['timestamp_s'].min()
    data_start_idx = df.index.min()
    
    gap_before = (first_marker_time - data_start_time) / 3600
    print(f"\n  数据起点到第一个标记: {gap_before:.1f}h")
    
    if gap_before > min_gap_hours:
        # 足够安全，保留（但要留边界）
        safe_end_time = first_marker_time - margin_hours * 3600
        safe_end_candidates = df[df['timestamp_s'] <= safe_end_time].index
        if len(safe_end_candidates) > 0:
            safe_end_idx = safe_end_candidates[-1]
            safe_segments.append((data_start_idx, safe_end_idx))
            print(f"    → 安全段1: [{data_start_idx}, {safe_end_idx}] ({gap_before:.1f}h)")
    else:
        print(f"    → 不安全，剔除")
    
    # 检查标记点之间的间隔
    for i in range(len(markers) - 1):
        current_marker_idx = markers[i]
        next_marker_idx = markers[i + 1]
        
        current_time = marker_times[i]
        next_time = marker_times[i + 1]
        
        gap = (next_time - current_time) / 3600
        print(f"\n  标记{i+1}到标记{i+2}: {gap:.1f}h")
        
        if gap > min_gap_hours:
            # 足够安全，保留中间部分（留边界）
            safe_start_time = current_time + margin_hours * 3600
            safe_end_time = next_time - margin_hours * 3600
            
            safe_start_candidates = df[df['timestamp_s'] >= safe_start_time].index
            safe_end_candidates = df[df['timestamp_s'] <= safe_end_time].index
            
            if len(safe_start_candidates) > 0 and len(safe_end_candidates) > 0:
                safe_start_idx = safe_start_candidates[0]
                safe_end_idx = safe_end_candidates[-1]
                
                if safe_start_idx < safe_end_idx:
                    safe_segments.append((safe_start_idx, safe_end_idx))
                    actual_duration = (df.loc[safe_end_idx, 'timestamp_s'] - 
                                     df.loc[safe_start_idx, 'timestamp_s']) / 3600
                    print(f"    → 安全段{len(safe_segments)}: [{safe_start_idx}, {safe_end_idx}] ({actual_duration:.1f}h)")
                else:
                    print(f"    → 边界重叠，不安全")
        else:
            print(f"    → 间隔太小，不安全")
    
    # 检查最后一个标记点之后
    last_marker_time = marker_times[-1]
    last_marker_idx = markers[-1]
    
    data_end_time = df['timestamp_s'].max()
    data_end_idx = df.index.max()
    
    gap_after = (data_end_time - last_marker_time) / 3600
    print(f"\n  最后标记到数据终点: {gap_after:.1f}h")
    
    if gap_after > min_gap_hours:
        safe_start_time = last_marker_time + margin_hours * 3600
        safe_start_candidates = df[df['timestamp_s'] >= safe_start_time].index
        if len(safe_start_candidates) > 0:
            safe_start_idx = safe_start_candidates[0]
            safe_segments.append((safe_start_idx, data_end_idx))
            print(f"    → 安全段{len(safe_segments)}: [{safe_start_idx}, {data_end_idx}] ({gap_after:.1f}h)")
    else:
        print(f"    → 不安全，剔除")
    
    return safe_segments


def extract_safe_segments(df, safe_segments):
    """提取安全段的数据"""
    if not safe_segments:
        return pd.DataFrame()
    
    dfs = []
    for start_idx, end_idx in safe_segments:
        dfs.append(df.loc[start_idx:end_idx].copy())
    
    result = pd.concat(dfs, ignore_index=False)
    return result


def visualize_segments(df_orig, df_safe, safe_segments, markers, path):
    """可视化安全段"""
    fig, axes = plt.subplots(4, 1, figsize=(22, 14))
    
    # SOC
    axes[0].plot(df_orig['timestamp_s'], df_orig['soc_est'], 
                'lightgray', alpha=0.5, linewidth=0.5, label='原始（全部）')
    
    # 标记check-up段（红色）
    if safe_segments:
        # 找出不安全的段（反向）
        all_idx = set(df_orig.index)
        safe_idx = set()
        for start, end in safe_segments:
            safe_idx.update(range(start, end + 1))
        unsafe_idx = sorted(all_idx - safe_idx)
        
        if unsafe_idx:
            unsafe_df = df_orig.loc[unsafe_idx]
            axes[0].scatter(unsafe_df['timestamp_s'], unsafe_df['soc_est'],
                          c='red', s=1, alpha=0.3, label='Check-up段（剔除）')
    
    # 安全段（蓝色）
    axes[0].plot(df_safe['timestamp_s'], df_safe['soc_est'], 
                'blue', linewidth=2, label='安全循环段（保留）', zorder=5)
    
    # 标记标记点
    if markers:
        marker_df = df_orig.loc[markers]
        axes[0].scatter(marker_df['timestamp_s'], marker_df['soc_est'],
                       c='orange', s=200, marker='*', edgecolors='black',
                       linewidths=2, label='标记点', zorder=10)
    
    axes[0].set_ylabel('SOC (%)', fontsize=14, fontweight='bold')
    axes[0].set_title('反向策略：识别并保留安全循环段', fontsize=16, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 电流
    axes[1].plot(df_orig['timestamp_s'], df_orig['i_raw_A'], 
                'lightgray', alpha=0.5, linewidth=0.5)
    axes[1].plot(df_safe['timestamp_s'], df_safe['i_raw_A'], 
                'red', linewidth=2, zorder=5)
    if markers:
        axes[1].scatter(marker_df['timestamp_s'], marker_df['i_raw_A'],
                       c='orange', s=200, marker='*', edgecolors='black',
                       linewidths=2, zorder=10)
    axes[1].set_ylabel('电流 (A)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    
    # Delta Q
    axes[2].plot(df_orig['timestamp_s'], df_orig['delta_q_Ah'], 
                'lightgray', alpha=0.5, linewidth=0.5)
    axes[2].plot(df_safe['timestamp_s'], df_safe['delta_q_Ah'], 
                'purple', linewidth=2, zorder=5)
    if markers:
        axes[2].scatter(marker_df['timestamp_s'], marker_df['delta_q_Ah'],
                       c='orange', s=200, marker='*', edgecolors='black',
                       linewidths=2, zorder=10)
    axes[2].set_ylabel('Delta Q (Ah)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # 温度
    axes[3].plot(df_orig['timestamp_s'], df_orig['t_cell_degC'], 
                'lightgray', alpha=0.5, linewidth=0.5)
    axes[3].plot(df_safe['timestamp_s'], df_safe['t_cell_degC'], 
                'green', linewidth=2, zorder=5)
    if markers:
        axes[3].scatter(marker_df['timestamp_s'], marker_df['t_cell_degC'],
                       c='orange', s=200, marker='*', edgecolors='black',
                       linewidths=2, zorder=10)
    axes[3].set_xlabel('时间戳 (s)', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('温度 (°C)', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def process_file(fpath, outdir, min_gap, margin):
    """处理文件"""
    print(f"\n{'='*80}")
    print(f"处理: {os.path.basename(fpath)}")
    print(f"{'='*80}")
    
    df = pd.read_csv(fpath, delimiter=';')
    print(f"\n[1] 原始数据: {len(df)}行")
    print(f"    时间范围: {df['timestamp_s'].min():.0f}s - {df['timestamp_s'].max():.0f}s")
    print(f"    总时长: {(df['timestamp_s'].max() - df['timestamp_s'].min())/3600:.1f}小时")
    
    print(f"\n[2] 识别标记点:")
    markers = find_all_markers(df)
    
    print(f"\n[3] 识别安全循环段:")
    safe_segments = identify_safe_segments(df, markers, min_gap, margin)
    
    print(f"\n[4] 提取安全段数据:")
    print(f"    安全段数量: {len(safe_segments)}")
    
    if safe_segments:
        df_safe = extract_safe_segments(df, safe_segments)
        print(f"    保留数据: {len(df_safe)}行 ({len(df_safe)/len(df)*100:.1f}%)")
        print(f"    剔除数据: {len(df) - len(df_safe)}行 ({(len(df)-len(df_safe))/len(df)*100:.1f}%)")
    else:
        print("    警告：没有找到安全段！")
        df_safe = pd.DataFrame()
    
    print(f"\n[5] 验证:")
    if len(df_safe) > 0:
        print(f"    剩余cap_aged: {df_safe['cap_aged_est_Ah'].notna().sum()}")
        print(f"    剩余R0: {df_safe['R0_mOhm'].notna().sum()}")
        print(f"    剩余R1: {df_safe['R1_mOhm'].notna().sum()}")
    
    print(f"\n[6] 保存结果:")
    fname = f"no_checkup_{os.path.basename(fpath)}"
    csv_path = os.path.join(outdir, fname)
    df_safe.to_csv(csv_path, sep=';', index=False)
    print(f"    ✓ CSV: {csv_path}")
    
    png_path = os.path.join(outdir, fname.replace('.csv', '.png'))
    visualize_segments(df, df_safe, safe_segments, markers, png_path)
    print(f"    ✓ 图片: {png_path}")
    
    return df_safe


def main():
    print("="*80)
    print("反向策略：识别并保留安全循环段".center(80))
    print("="*80)
    print(f"\n策略说明:")
    print(f"  传统方法：找check-up → 扩展 → 删除")
    print(f"  反向方法：找check-up → 找安全间隔 → 只保留安全段")
    print(f"\n参数:")
    print(f"  最小安全间隔: {MIN_SAFE_GAP_HOURS}小时")
    print(f"  安全边界: {EXPAND_MARGIN_HOURS}小时")
    print(f"\n说明:")
    print(f"  只有标记点之间间隔>{MIN_SAFE_GAP_HOURS}h的区域才被认为是安全的")
    print(f"  从安全区域的边界再向内收缩{EXPAND_MARGIN_HOURS}h作为最终保留区域")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出: {OUTPUT_DIR}")
    
    for fpath in FILE_PATHS:
        if os.path.exists(fpath):
            process_file(fpath, OUTPUT_DIR, MIN_SAFE_GAP_HOURS, EXPAND_MARGIN_HOURS)
    
    print(f"\n{'='*80}")
    print("完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
