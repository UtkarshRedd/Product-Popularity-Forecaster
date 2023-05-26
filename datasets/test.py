import pandas as pd
df = pd.read_parquet("C:/Users/utkar/Desktop/JIO/Work/forecasting/datasets/cleaned_data_2020-10-01_2021-10-18.parquet")
if df:
    print('Yes')
else:
    print('No')