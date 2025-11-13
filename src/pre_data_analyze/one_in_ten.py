import pandas as pd

df = pd.read_csv("./data/cell_log_age_30s_P051_2_S11_C05.csv")
cut = int(len(df) /20)
df_head = df.head(cut)
df_head.to_csv("./data_parse/oneinten/cell_log_age_30s_P051_2_S11_C05.csv", index=False)
