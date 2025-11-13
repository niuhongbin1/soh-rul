import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 创建必要的文件夹
# os.makedirs('data_parse', exist_ok=True)
# os.makedirs('images', exist_ok=True)

# 文件路径列表 - 替换为你需要处理的多个文件路径
file_paths = [
    './data/cell_log_age_30s_P051_1_S10_C08.csv',
    # 添加更多文件路径，例如：
    './data/cell_log_age_30s_P051_2_S11_C05.csv',
    './data/cell_log_age_30s_P051_3_S12_C05.csv',
]


# 创建图形
plt.figure(figsize=(12, 8))

# 定义颜色列表，用于区分不同文件的数据
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

# 遍历所有文件
for file_index, file_path in enumerate(file_paths):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        continue
    
    # 1. 读取分号分隔的CSV文件数据
    df = pd.read_csv(file_path, delimiter=';')
    
    # 获取文件名用于保存和标识
    file_name = os.path.basename(file_path)
    
    # 2. 观测倒数第4列的数值，每增加5就划分为一部分
    col_minus_4 = df.iloc[:, -4]  # 倒数第4列
    
    # 确定划分区间 - 基于实际存在的值范围
    min_val = int(col_minus_4.min())
    max_val = int(col_minus_4.max())
    
    # 创建划分区间，每5为一个部分
    # 确保区间覆盖所有可能的值
    bins = list(range(min_val, max_val + 6, 5))
    
    # 为每个数据点分配部分序号
    section_labels = pd.cut(col_minus_4, bins, right=False, include_lowest=True)
    section_labels_str = section_labels.astype(str)
    # 获取所有可能的部分（包括没有数据的部分）
    all_sections = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]
    
    # 初始化存储结果的列表
    section_numbers = []  # 部分序号
    representative_values = []  # 代表值
    
    # 3. 对每个部分进行处理
    for section_idx, section_range in enumerate(all_sections):
        # 获取该部分的所有行
        section_mask = section_labels_str  == section_range
        section_data = df[section_mask]
        
        # 如果该部分没有数据，跳过
        if len(section_data) == 0:
            continue
            
        # 获取该部分的倒数第3列
        col_minus_3_section = section_data.iloc[:, -3]
        
        # 检查该部分的倒数第3列是否全为NaN ///////////////////////////////
        if not col_minus_3_section.isna().all():
            # 如果不全为NaN，取第一个非NaN值作为代表值
            non_nan_values = col_minus_3_section.dropna()
            if len(non_nan_values) > 0:
                representative_value = non_nan_values.iloc[0]
                if len(representative_values) > 0:
                    # representative_value = representative_values[-1]
                    representative_value =  non_nan_values.iloc[0]
                else:
                    representative_value = 3
            else:
                # 如果没有非NaN值，使用备用方法
                col_minus_5_section = section_data.iloc[:, -5]
                max_minus_5 = col_minus_5_section.abs().max()
                representative_value = max_minus_5 / 0.8
        else:
            # 如果全为NaN，取倒数第5列的最大值除以0.8作为代表值
            col_minus_5_section = section_data.iloc[:, -5]
            max_minus_5 = col_minus_5_section.abs().max()
            representative_value = max_minus_5 / 0.8
        

        # 存储结果
        section_numbers.append(section_idx * 5)  # 部分序号乘以5，如0,5,10...
        representative_values.append(representative_value)
    
    # 4. 将处理好的数据存储为逗号分隔的CSV文件
    result_df = pd.DataFrame({
        'section': section_numbers,
        'representative_value': representative_values
    })
    
    # 保存到data_parse文件夹
    output_filename = f"src/p051/data_parse/5_max_{file_index+1}.csv"
    result_df.to_csv(output_filename, index=False)
    print(f"已保存处理后的数据到: {output_filename}")
    
    # 5. 绘制折线图
    # 5. 绘制折线图
    color = colors[file_index % len(colors)]
    plt.plot(section_numbers, representative_values, 
         label=f"File {file_index+1}: {file_name}", 
         color=color, linewidth=2)
# 设置图表属性
plt.xlabel('部分序号 (每5为一个部分)')
plt.ylabel('代表值')
plt.title('多个文件的处理结果对比')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# 保存图片到images文件夹
plt.savefig('src/p051/images/5_max.png', dpi=300, bbox_inches='tight')
plt.show()

print("所有文件处理完成!")