import pandas as pd
import gc
import os


src_dir = './Data/JPX/csv/'
vol_thresh = 1000_0000
cols = ['date', 'issue code', 'time', 'price', 'trading volume']
types = {'date': int, 'issue code': int, 'time': int, 'price': int, 'trading volume': int}
for file in os.listdir(src_dir):
    df = pd.read_csv(src_dir + file)
    df.drop(df.columns.difference(cols), axis=1, inplace=True)
    gc.collect()
    
    # issue_codeごとにグループ分けして，それぞれのグループでのtrading_volumeの合計を計算
    vol_sum = df.groupby('issue code')['trading volume'].sum()
    codes = vol_sum[vol_sum >= vol_thresh].index.to_list()

    df = df[df['issue code'].isin(codes)]
    df.to_parquet('./Data/Filtered/' + file[len('stock_tick_'):-4] + '.parquet')
    
    del df, vol_sum
    gc.collect()

