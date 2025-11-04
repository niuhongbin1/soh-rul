import pandas as pd

# 读取CSV文件
df = pd.read_csv('D:\\数据集\\cell_log_age_ultracompr\\cell_log_age_30s_P038_1_S06_C07.csv', delimiter=';')

# 指定列（可以是列名或列索引）
# target_column = 'column_name'  # 替换为你的列名
target_column = df.columns[-3]  # 例如第3列

# 过滤出指定列非NaN的行
non_nan_rows = df[df[target_column].notna()]

# 显示结果
print(f"非NaN行数: {len(non_nan_rows)}")
print(non_nan_rows)