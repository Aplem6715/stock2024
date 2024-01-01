import pandas as pd
import gc
import os


# ここがNoneの場合はディレクトリ内の全ファイルを処理
src_file = None
# src_file = './Data/JPX/csv/stock_tick_202311.csv'
src_dir = './Data/JPX/csv/'
out_dir = './Data/Filtered/'
vol_thresh = 1000_0000
cols = ['date', 'issue code', 'time', 'price', 'trading volume']
types = {'date': int, 'issue code': int, 'time': int, 'price': int, 'trading volume': int}

def filtering(file):
    df = pd.read_csv(file)
    df.drop(df.columns.difference(cols), axis=1, inplace=True)
    gc.collect()
    
    # issue_codeごとにグループ分けして，それぞれのグループでのtrading_volumeの合計を計算
    vol_sum = df.groupby('issue code')['trading volume'].sum()
    codes = vol_sum[vol_sum >= vol_thresh].index.to_list()

    df = df[df['issue code'].isin(codes)]
    df.to_parquet(out_dir + os.path.basename(file)[len('stock_tick_'):-4] + '.parquet')
    
    del df, vol_sum
    gc.collect()


if src_file is None:
    for file in os.listdir(src_dir):
        filtering(src_dir + file)
else:
    filtering(src_file)