import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件（分号分隔）
df = pd.read_csv('./data/cell_log_age_30s_P051_1_S10_C08.csv', delimiter=';')

# 计算要截取的数据量（前1/10）
total_rows = len(df)
sample_size = total_rows // 10
print(f"数据总行数: {total_rows}")
print(f"截取前 {sample_size} 行数据进行绘图")

# 截取前1/10的数据
df_sampled = df.tail(sample_size)
# df_sampled = df.head(sample_size)   

# 获取列名列表
columns = df_sampled.columns.tolist()

# 计算要绘制的列的索引
x_col = columns[0]  # 第一列作为横轴
y1_col = columns[-6]  # 倒数第五列
y2_col = columns[-5]  # 倒数第四列  
y3_col = columns[-4]  # 倒数第三列

print(f"横轴列: {x_col}")
print(f"Y轴列1: {y1_col}")
print(f"Y轴列2: {y2_col}")
print(f"Y轴列3: {y3_col}")

# 创建图形和子图
plt.figure(figsize=(12, 10))

# 绘制第一个Y轴的数据
plt.subplot(3, 1, 1)
plt.plot(df_sampled[x_col], df_sampled[y1_col], 'b-', linewidth=1.5)
plt.ylabel(y1_col)
plt.title(f'{y1_col} vs {x_col} (前1/10数据)')
plt.grid(True, alpha=0.3)

# 绘制第二个Y轴的数据
plt.subplot(3, 1, 2)
plt.plot(df_sampled[x_col], df_sampled[y2_col], 'r-', linewidth=1.5)
plt.ylabel(y2_col)
plt.title(f'{y2_col} vs {x_col} (前1/10数据)')
plt.grid(True, alpha=0.3)

# 绘制第三个Y轴的数据
plt.subplot(3, 1, 3)
plt.plot(df_sampled[x_col], df_sampled[y3_col], 'g-', linewidth=1.5)
plt.ylabel(y3_col)
plt.xlabel(x_col)
plt.title(f'{y3_col} vs {x_col} (前1/10数据)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 可选：在同一张图上绘制三条曲线
plt.figure(figsize=(12, 6))
plt.plot(df_sampled[x_col], df_sampled[y1_col], 'b-', label=y1_col, linewidth=1.5)
plt.plot(df_sampled[x_col], df_sampled[y2_col], 'r-', label=y2_col, linewidth=1.5)
plt.plot(df_sampled[x_col], df_sampled[y3_col], 'g-', label=y3_col, linewidth=1.5)

plt.xlabel(x_col)
plt.ylabel('Y Values')
plt.title(f'Multiple Columns vs {x_col} (前1/10数据)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 显示采样数据的基本信息
print("\n采样数据基本信息:")
print(df_sampled[[x_col, y1_col, y2_col, y3_col]].describe())