import pandas as pd
import matplotlib.pyplot as plt
import os

# 文件路径列表 - 替换为你需要处理的多个文件路径
file_paths = [
    'D:\\数据集\\cell_log_age_ultracompr\\cell_log_age_30s_P038_1_S06_C07.csv',
    # 添加更多文件路径，例如：
    'D:\\数据集\\cell_log_age_ultracompr\\cell_log_age_30s_P038_2_S07_C07.csv',
    'D:\\数据集\\cell_log_age_ultracompr\\cell_log_age_30s_P038_3_S10_C03.csv',
]

# 创建图形
plt.figure(figsize=(12, 8))

# 定义颜色列表，用于区分不同文件的数据
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

# 遍历所有文件
for i, file_path in enumerate(file_paths):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        continue
    
    # 读取CSV文件
    df = pd.read_csv(file_path, delimiter=';')
    
    # 获取文件名用于图例
    file_name = os.path.basename(file_path)
    
    # 指定目标列（倒数第三列）
    target_column = df.columns[-3]
    
    # 过滤出指定列非NaN的行
    non_nan_rows = df[df[target_column].notna()]
    
    print(f"文件: {file_name}")
    print(f"非NaN行数: {len(non_nan_rows)}")
    
    # 如果非NaN行数大于0，则绘制数据
    if len(non_nan_rows) > 0:
        # 选择横轴列（这里使用索引作为横轴，你也可以选择其他列）
        # 例如，如果你想使用第一列作为横轴，可以使用：x_col = non_nan_rows.iloc[:, 0]
        x_data = non_nan_rows.iloc[:,-4]  # 使用行索引作为横轴
        
        # 获取目标列数据作为纵轴
        y_data = non_nan_rows.iloc[:, -3]
        
        # 绘制折线图
        color = colors[i % len(colors)]  # 循环使用颜色
        plt.plot(x_data, y_data, label=file_name, color=color, linewidth=1.5)
        
        # 可选：显示前几行数据
        if i == 0:  # 只显示第一个文件的前几行作为示例
            print("前5行非NaN数据:")
            print(non_nan_rows.head())
    else:
        print(f"警告: 文件 {file_name} 中没有非NaN数据")
    
    print("-" * 50)

# 设置图表属性
plt.xlabel('数据点索引')
plt.ylabel('目标列值')
plt.title('多个文件的目标列数据对比')
plt.legend(loc='best')  # 图例位置自动选择最佳
plt.grid(True, alpha=0.3)

# 显示图表
plt.tight_layout()
plt.show()