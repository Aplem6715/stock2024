import gc
import shutil
import pandas as pd
import pickle
import os

in_dir = './Data/Split/'
out_dir = './Data/Bar/'
# in_dir = './DataTest/Split/'
# out_dir = './DataTest/Bar/'
bar_thresh = 30000


# 'trading volume'列を累積していき，vol_threshを超えたら足を作成する
def make_bar(code: int, df: pd.DataFrame, vol_thresh: int):
    # date列とtime列を結合してdatetime列を作成(dateはyyyymmdd形式，timeはhhmmsstttttt形式)
    df['datetime'] = pd.to_datetime(df['date'].astype(
        str) + df['time'].astype(str), format='%Y%m%d%H%M%S%f')

    # datetime列をインデックスに設定
    df = df.sort_values('datetime')

    df['cumsum'] = df['Volume'].cumsum()
    df['cumsum'] = df['cumsum'] // vol_thresh
    # price列からohlc列を作成
    df['Open'] = df['High'] = df['Low'] = df['Close'] = df['price']
    df = df.groupby('cumsum').agg(
        {'datetime': 'first', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    df['Volume'] = df['Volume'].astype(int)
    # datetimeの差分を計算して平均と分散を計算
    df['diff'] = df['datetime'].diff()
    df['diff'] = df['diff'].dt.total_seconds()
    # 6時間以上の差分は欠損値とする
    df.loc[df['diff'] > 6 * 60 * 60, 'diff'] = None
    mean = df['diff'].mean()
    var = df['diff'].std()
    # datetimeとdiff列を削除
    df = df.drop(columns=['datetime', 'diff'])
    print(f'{code}: Time  mean: {mean/60:4.1f}[min], var: {var/60:4.1f}[min]')
    return df


shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir, exist_ok=True)

# Data/Split/ディレクトリ内の全ディレクトリをループ
for dir in os.listdir(in_dir):
    code_dir = in_dir + dir + '/'
    df_all = pd.DataFrame()

    # Data/Split/ディレクトリ内の全ファイルをループして1つのdfにマージする
    for file in os.listdir(in_dir + dir):
        df = pd.read_parquet(code_dir + file)
        # trading volume列をvolume列に変更
        df = df.rename(columns={'trading volume': 'Volume'})
        df_all = pd.concat([df_all, df])
        del df

    bar = make_bar(dir, df_all, bar_thresh)
    # indexを1分ずつ増加するdatetime形式に変更
    bar.index = pd.date_range(
        start='2000-01-01 00:00:00', periods=bar.shape[0], freq='T')

    # 出力
    bar.to_csv(out_dir + dir + '.csv')
    
    del df_all, bar
    gc.collect()
