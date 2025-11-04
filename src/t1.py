import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('D:\\数据集\\cell_log_age_ultracompr\\cell_log_age_30s_P001_3_S05_C06.csv', delimiter=';')

# 获取列名
columns = df.columns.tolist()
x_col = columns[0]
y1_col = columns[-6]
y2_col = columns[-5]  
y3_col = columns[-7]

# 简单绘图 - 三个Y轴在同一图中
plt.figure(figsize=(12, 6))
plt.plot(df[x_col], df[y1_col], 'b-', label=y1_col, linewidth=2)
# plt.plot(df[x_col], df[y2_col], 'r-', label=y2_col, linewidth=2)
# plt.plot(df[x_col], df[y3_col], 'g-', label=y3_col, linewidth=2)

plt.xlabel(x_col)
plt.ylabel('Y Values')
plt.title('Multiple Columns vs First Column')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()