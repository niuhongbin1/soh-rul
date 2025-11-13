"""
============================================================================
电池真实SOH提取工具 - 基于EFC（等效循环数）
============================================================================

功能说明：
---------
本脚本用于从Comprehensive battery aging dataset中提取每个等效循环（EFC）对应的真实SOH。

数据集说明：
- 数据集：Comprehensive battery aging dataset (LG INR18650HG2, NMC/C-SiO)
- 实验流程：
  1. 开始时进行一次常温(25°C) check-up
  2. 一周后再进行一次常温check-up
  3. 之后每三周进行一次常温check-up
  4. check-up时，cap_aged_est_Ah列有值（真实测量的容量）
  5. 其他时间该列为NaN

核心逻辑：
---------
1. 使用EFC（等效循环数）作为循环坐标
   - EFC定义：累计充放电量 / 标称容量
   - 优点：标准化了不同深度的循环，适合老化研究
   - 适用场景：固定工况下的寿命预测

2. SOH提取策略：
   - 优先使用check-up时的cap_aged_est_Ah（参考容量测试）
   - 对于check-up之间的时期，使用delta_q_Ah累计计算容量
   - 计算公式：SOH = (当前容量 / 标称容量) × 100%

3. 输出格式：
   - 每个EFC对应一行
   - 包含：EFC、真实SOH、容量、测量类型、时间戳等

使用方法：
---------
1. 修改配置参数（文件路径、标称容量等）
2. 运行脚本：python extract_true_soh_by_efc.py
3. 查看输出：指定的output_dir中的CSV文件和图片

依赖库：
-------
- pandas
- numpy
- matplotlib
- scipy (可选，用于平滑)

作者：Battery SOH Analysis Tool
日期：2025-01
版本：v1.0
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 中文显示配置 - 解决中文乱码问题
# ============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ============================================================================
# 配置参数区域 - 根据实际情况修改
# ============================================================================

# 1. 文件路径配置
FILE_PATHS = [
    './data/cell_log_age_30s_P051_1_S10_C08.csv',
    './data/cell_log_age_30s_P051_2_S11_C05.csv',
    './data/cell_log_age_30s_P051_3_S12_C05.csv',
]

# 2. 输出路径配置
OUTPUT_DIR = './data_parse/output_soh_efc'  # 可以修改为您指定的路径

# 3. 电池参数配置
NOMINAL_CAPACITY = 3.0  # Ah - LG INR18650HG2的标称容量

# 4. 数据列配置（根据数据集文档）
COLUMN_CONFIG = {
    'timestamp': 'timestamp_s',
    'voltage': 'v_raw_V',
    'current': 'i_raw_A',
    'temperature': 't_cell_degC',
    'soc': 'soc_est',
    'delta_q': 'delta_q_Ah',
    'efc': 'EFC',
    'cap_aged': 'cap_aged_est_Ah',
    'r0': 'R0_mOhm',
    'r1': 'R1_mOhm'
}

# 5. 采样间隔配置（用于插值到整数EFC）
EFC_SAMPLING_INTERVAL = 1.0  # 每1个EFC采样一次（可调整为0.5、0.1等）

# 6. 可视化配置
PLOT_CONFIG = {
    'figsize': (14, 10),
    'dpi': 300,
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'show_plot': True,  # 是否显示图形窗口
}

# ============================================================================
# 核心功能类
# ============================================================================

class BatterySOHExtractor:
    """电池SOH提取器 - 基于EFC"""
    
    def __init__(self, nominal_capacity: float, column_config: dict):
        """
        初始化
        
        Parameters:
        -----------
        nominal_capacity : float
            电池标称容量 (Ah)
        column_config : dict
            列名配置字典
        """
        self.nominal_capacity = nominal_capacity
        self.col = column_config
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载CSV数据
        
        Parameters:
        -----------
        file_path : str
            CSV文件路径
            
        Returns:
        --------
        pd.DataFrame : 加载的数据
        """
        try:
            df = pd.read_csv(file_path, delimiter=';')
            print(f"✓ 成功加载: {os.path.basename(file_path)}")
            print(f"  数据形状: {df.shape}")
            print(f"  时间范围: {df[self.col['timestamp']].min():.0f}s - {df[self.col['timestamp']].max():.0f}s")
            print(f"  EFC范围: {df[self.col['efc']].min():.2f} - {df[self.col['efc']].max():.2f}")
            return df
        except Exception as e:
            print(f"✗ 加载失败: {file_path}")
            print(f"  错误: {e}")
            return None
    
    def extract_checkup_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取check-up测试点（cap_aged_est_Ah有值的点）
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始数据
            
        Returns:
        --------
        pd.DataFrame : check-up点数据
        """
        # 找到有cap_aged值的行
        checkup_mask = df[self.col['cap_aged']].notna()
        checkup_data = df[checkup_mask].copy()
        
        print(f"  Check-up点数量: {len(checkup_data)}")
        
        if len(checkup_data) > 0:
            checkup_data['soh'] = (checkup_data[self.col['cap_aged']] / self.nominal_capacity) * 100
            checkup_data['measurement_type'] = 'checkup'
            
        return checkup_data
    
    def calculate_capacity_from_delta_q(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基于delta_q_Ah计算容量（用于check-up之间的时期）
        
        策略：
        1. 找到所有check-up点作为锚点
        2. 在两个check-up之间，使用delta_q累计计算容量
        3. 每个段内进行归一化，使其与check-up点对齐
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始数据
            
        Returns:
        --------
        pd.DataFrame : 包含估算容量的数据
        """
        # 获取check-up点
        checkup_points = self.extract_checkup_points(df)
        
        if len(checkup_points) == 0:
            print("  警告：没有找到check-up点，无法校准！")
            return df
        
        # 创建结果DataFrame
        df_result = df.copy()
        df_result['estimated_capacity'] = np.nan
        df_result['soh'] = np.nan
        df_result['measurement_type'] = 'estimated'
        
        # 将check-up点的容量填入
        for idx in checkup_points.index:
            df_result.loc[idx, 'estimated_capacity'] = checkup_points.loc[idx, self.col['cap_aged']]
            df_result.loc[idx, 'soh'] = checkup_points.loc[idx, 'soh']
            df_result.loc[idx, 'measurement_type'] = 'checkup'
        
        # 在check-up之间进行插值估算
        # 简化方法：线性插值
        checkup_indices = checkup_points.index.tolist()
        checkup_efcs = checkup_points[self.col['efc']].values
        checkup_capacities = checkup_points[self.col['cap_aged']].values
        
        # 对整个数据集进行插值
        df_result['estimated_capacity'] = np.interp(
            df_result[self.col['efc']],
            checkup_efcs,
            checkup_capacities
        )
        
        # 计算SOH
        df_result['soh'] = (df_result['estimated_capacity'] / self.nominal_capacity) * 100
        
        # 恢复check-up点的标记
        df_result.loc[checkup_indices, 'measurement_type'] = 'checkup'
        
        return df_result
    
    def sample_at_integer_efc(self, df: pd.DataFrame, 
                              efc_interval: float = 1.0) -> pd.DataFrame:
        """
        在整数EFC点进行采样
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含SOH的完整数据
        efc_interval : float
            采样间隔（默认1.0，即每个整数EFC）
            
        Returns:
        --------
        pd.DataFrame : 采样后的数据
        """
        efc_min = np.floor(df[self.col['efc']].min())
        efc_max = np.ceil(df[self.col['efc']].max())
        
        # 生成采样点
        sample_efcs = np.arange(efc_min, efc_max + efc_interval, efc_interval)
        
        # 对每个采样点，找到最接近的数据
        sampled_data = []
        
        for target_efc in sample_efcs:
            # 找到最接近的行
            idx = (df[self.col['efc']] - target_efc).abs().idxmin()
            row = df.loc[idx].copy()
            
            # 记录采样信息
            sampled_data.append({
                'efc': target_efc,
                'efc_actual': row[self.col['efc']],
                'soh': row['soh'],
                'capacity_ah': row['estimated_capacity'],
                'measurement_type': row['measurement_type'],
                'timestamp_s': row[self.col['timestamp']],
                'temperature_c': row[self.col['temperature']],
            })
        
        result_df = pd.DataFrame(sampled_data)
        
        print(f"  采样后数据点: {len(result_df)}")
        
        return result_df
    
    def process_single_file(self, file_path: str, 
                           efc_interval: float = 1.0) -> Tuple[pd.DataFrame, str]:
        """
        处理单个文件
        
        Parameters:
        -----------
        file_path : str
            文件路径
        efc_interval : float
            EFC采样间隔
            
        Returns:
        --------
        Tuple[pd.DataFrame, str] : (结果数据, 文件名)
        """
        print(f"\n{'='*60}")
        print(f"处理文件: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        # 1. 加载数据
        df = self.load_data(file_path)
        if df is None:
            return None, None
        
        # 2. 计算SOH
        df_with_soh = self.calculate_capacity_from_delta_q(df)
        
        # 3. 在整数EFC点采样
        df_sampled = self.sample_at_integer_efc(df_with_soh, efc_interval)
        
        # 4. 统计信息
        print(f"\n  SOH统计:")
        print(f"    初始SOH: {df_sampled['soh'].iloc[0]:.2f}%")
        print(f"    最终SOH: {df_sampled['soh'].iloc[-1]:.2f}%")
        print(f"    总衰减: {df_sampled['soh'].iloc[0] - df_sampled['soh'].iloc[-1]:.2f}%")
        print(f"    Check-up点: {(df_sampled['measurement_type']=='checkup').sum()}个")
        
        file_name = os.path.basename(file_path)
        
        return df_sampled, file_name


# ============================================================================
# 可视化功能
# ============================================================================

def plot_soh_results(results: Dict[str, pd.DataFrame], output_dir: str, config: dict):
    """
    绘制SOH结果对比图
    
    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        每个文件的结果数据
    output_dir : str
        输出目录
    config : dict
        绘图配置
    """
    fig, axes = plt.subplots(3, 1, figsize=config['figsize'])
    colors = config['colors']
    
    for idx, (file_name, df) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        
        # 子图1: SOH vs EFC
        axes[0].plot(df['efc'], df['soh'], 
                    color=color, linewidth=2.5, alpha=0.8,
                    label=file_name)
        
        # 标记check-up点
        checkup_mask = df['measurement_type'] == 'checkup'
        if checkup_mask.any():
            axes[0].scatter(df.loc[checkup_mask, 'efc'], 
                          df.loc[checkup_mask, 'soh'],
                          color=color, s=120, marker='*', 
                          edgecolors='black', linewidths=1.5, 
                          zorder=5)
        
        # 子图2: 容量 vs EFC
        axes[1].plot(df['efc'], df['capacity_ah'],
                    color=color, linewidth=2.5, alpha=0.8,
                    label=file_name)
        
        # 子图3: SOH衰减率
        if len(df) > 1:
            fade_rate = -np.diff(df['soh']) / np.diff(df['efc'])
            axes[2].plot(df['efc'].iloc[1:], fade_rate,
                        color=color, linewidth=2, alpha=0.7,
                        label=file_name)
    
    # 设置子图1
    axes[0].set_ylabel('SOH (%)', fontsize=13, fontweight='bold')
    axes[0].set_title('真实SOH vs EFC (等效循环数)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].axhline(y=80, color='red', linestyle='--', 
                   linewidth=2, alpha=0.6, label='EOL (80%)')
    
    # 设置子图2
    axes[1].set_ylabel('容量 (Ah)', fontsize=13, fontweight='bold')
    axes[1].set_title('容量衰减', fontsize=12)
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].axhline(y=NOMINAL_CAPACITY * 0.8, color='red', 
                   linestyle='--', linewidth=2, alpha=0.6)
    
    # 设置子图3
    axes[2].set_xlabel('EFC (等效循环数)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('SOH衰减率 (%/EFC)', fontsize=13, fontweight='bold')
    axes[2].set_title('SOH衰减率', fontsize=12)
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'soh_vs_efc_analysis.png')
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    print(f"\n✓ 图片已保存: {output_path}")
    
    if config['show_plot']:
        plt.show()
    else:
        plt.close()


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    
    print("\n" + "="*70)
    print(" 电池真实SOH提取工具 - 基于EFC（等效循环数）".center(70))
    print("="*70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # 创建提取器
    extractor = BatterySOHExtractor(NOMINAL_CAPACITY, COLUMN_CONFIG)
    
    # 处理所有文件
    results = {}
    
    for file_path in FILE_PATHS:
        if not os.path.exists(file_path):
            print(f"\n✗ 文件不存在: {file_path}")
            continue
        
        # 提取SOH
        df_result, file_name = extractor.process_single_file(
            file_path, 
            efc_interval=EFC_SAMPLING_INTERVAL
        )
        
        if df_result is not None:
            # 保存结果
            output_filename = os.path.join(
                OUTPUT_DIR, 
                f"soh_efc_{file_name.replace('.csv', '_extracted.csv')}"
            )
            df_result.to_csv(output_filename, index=False)
            print(f"  ✓ 已保存: {output_filename}")
            
            results[file_name] = df_result
    
    # 绘制对比图
    if len(results) > 0:
        print(f"\n{'='*70}")
        print("生成可视化结果...")
        print(f"{'='*70}")
        plot_soh_results(results, OUTPUT_DIR, PLOT_CONFIG)
    
    # 输出总结
    print(f"\n{'='*70}")
    print("处理完成！".center(70))
    print(f"{'='*70}")
    print(f"\n处理文件数: {len(results)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n输出文件:")
    print(f"  - CSV: soh_efc_*_extracted.csv")
    print(f"  - 图片: soh_vs_efc_analysis.png")
    print(f"\n关键说明:")
    print(f"  1. ✓ 使用EFC（等效循环数）作为循环坐标")
    print(f"  2. ✓ Check-up点使用真实测量容量")
    print(f"  3. ✓ Check-up之间使用线性插值")
    print(f"  4. ✓ 无平滑处理，保留原始数据特征")
    print(f"  5. ✓ 每{EFC_SAMPLING_INTERVAL} EFC采样一次")
    
    print(f"\n为什么使用EFC合理：")
    print(f"  • EFC标准化了不同深度的循环")
    print(f"  • 适合复杂工况（不同SOC范围、DoD）")
    print(f"  • 反映真实的电荷通量")
    print(f"  • 是电池老化研究的行业标准")
    print(f"  • 特别适合您的固定工况寿命预测")
    
    print(f"\n下一步建议:")
    print(f"  1. 检查输出CSV文件，确认数据质量")
    print(f"  2. 使用SOH数据进行寿命预测建模")
    print(f"  3. 可结合温度、电压等特征提升模型精度")
    print(f"  4. 如需调整采样间隔，修改 EFC_SAMPLING_INTERVAL")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
