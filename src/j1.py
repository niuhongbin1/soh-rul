import pandas as pd

df = pd.read_csv('D:\\数据集\\cell_log_age_ultracompr\\cell_log_age_30s_P061_1_S16_C09.csv', delimiter=';')
col = df.columns[-3]  # 要检查的列

if df[col].isna().all():
    print(f"列 '{col}' 全部是NaN")
else:
    non_nan = df[col].dropna()
    print(f"列 '{col}' 有 {len(non_nan)} 个非NaN值")
    print(f"示例: {non_nan.head(3).tolist()}")