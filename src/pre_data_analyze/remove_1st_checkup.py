"""
剔除第1次check-up及以前的数据，EFC重新从0计数
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

OUTPUT_DIR = './data_parse/after_1st_checkup'
NOMINAL_CAPACITY = 3.0
EFC_INTERVAL = 1.0
# ===============================================


def extract_soh(file_path, nominal_capacity, efc_interval=1.0, remove_n=1):
    """提取SOH，剔除前N次check-up"""
    
    print(f"\n处理: {os.path.basename(file_path)}")
    
    # 读取数据
    df = pd.read_csv(file_path, delimiter=';')
    print(f"  原始数据: {len(df)}行")
    
    # 找到所有check-up点
    checkup_mask = df['cap_aged_est_Ah'].notna()
    checkup_indices = df[checkup_mask].index.tolist()
    print(f"  Check-up总数: {len(checkup_indices)}")
    
    # 剔除前N次check-up及之前的数据
    if len(checkup_indices) >= remove_n:
        cut_idx = checkup_indices[remove_n - 1]
        df = df.loc[cut_idx + 1:].copy().reset_index(drop=True)
        print(f"  剔除前{remove_n}次check-up")
        print(f"  剩余数据: {len(df)}行")
        
        # EFC重新计数
        start_efc = df['EFC'].iloc[0]
        df['EFC'] = df['EFC'] - start_efc
        print(f"  EFC重新计数: {df['EFC'].min():.2f} - {df['EFC'].max():.2f}")
    
    # 更新check-up点
    checkup_mask = df['cap_aged_est_Ah'].notna()
    checkup_data = df[checkup_mask]
    print(f"  剩余Check-up: {len(checkup_data)}个")
    
    if len(checkup_data) == 0:
        print("  警告: 没有check-up点!")
        return None
    
    # 线性插值计算所有点的容量
    checkup_efc = checkup_data['EFC'].values
    checkup_capacity = checkup_data['cap_aged_est_Ah'].values
    df['capacity'] = np.interp(df['EFC'], checkup_efc, checkup_capacity)
    df['soh'] = (df['capacity'] / nominal_capacity) * 100
    df['type'] = 'estimated'
    df.loc[checkup_mask, 'type'] = 'checkup'
    
    # EFC采样
    efc_min = np.floor(df['EFC'].min())
    efc_max = np.ceil(df['EFC'].max())
    sample_efcs = np.arange(efc_min, efc_max + efc_interval, efc_interval)
    
    results = []
    for target_efc in sample_efcs:
        idx = (df['EFC'] - target_efc).abs().idxmin()
        row = df.loc[idx]
        results.append({
            'efc': target_efc,
            'soh': row['soh'],
            'capacity_ah': row['capacity'],
            'type': row['type'],
            'temperature_c': row['t_cell_degC'],
        })
    
    result_df = pd.DataFrame(results)
    print(f"  初始SOH: {result_df['soh'].iloc[0]:.2f}%")
    print(f"  最终SOH: {result_df['soh'].iloc[-1]:.2f}%")
    print(f"  总衰减: {result_df['soh'].iloc[0] - result_df['soh'].iloc[-1]:.2f}%")
    
    return result_df


def plot_results(all_results, output_dir):
    """绘图"""
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (name, df) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        
        # SOH曲线
        plt.plot(df['efc'], df['soh'], color=color, linewidth=2.5, 
                label=name, alpha=0.8)
        
        # 标记check-up点
        checkup_mask = df['type'] == 'checkup'
        if checkup_mask.any():
            plt.scatter(df.loc[checkup_mask, 'efc'], 
                       df.loc[checkup_mask, 'soh'],
                       color=color, s=150, marker='*', 
                       edgecolors='black', linewidths=2, zorder=5)
    
    plt.xlabel('EFC (从0重新计数)', fontsize=13, fontweight='bold')
    plt.ylabel('SOH (%)', fontsize=13, fontweight='bold')
    plt.title('SOH vs EFC (剔除第1次Check-up后)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'soh_after_1st.png'), dpi=300)
    plt.show()


def main():
    print("="*60)
    print("剔除第1次Check-up版本 - EFC重新计数")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    
    for file_path in FILE_PATHS:
        if not os.path.exists(file_path):
            continue
        
        df = extract_soh(file_path, NOMINAL_CAPACITY, EFC_INTERVAL, remove_n=1)
        
        if df is not None:
            # 保存CSV
            name = os.path.basename(file_path)
            output_csv = os.path.join(OUTPUT_DIR, f"soh_{name}")
            df.to_csv(output_csv, index=False)
            print(f"  ✓ 已保存: {output_csv}")
            results[name] = df
    
    # 绘图
    if results:
        plot_results(results, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print(f"完成! 输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
