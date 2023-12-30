import os
import numpy as np
import pandas as pd
import finplot as fplt
from feature import add_multi_time_indicators
from utilc.zigzag import calc_zigzag

time_frames = ['1T', '1H']
zig_period = 10
out_file = './Data/train.parquet'


def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    out = arr[idx]
    return out

# Simple solution for bfill provided by financial_physician in comment below


def bfill(arr):
    return ffill(arr[::-1])[::-1]


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    df['target'] = 0
    df.loc[df['Close'].shift(-1) > df['Close'], 'target'] = 1
    df.loc[df['Close'].shift(-1) < df['Close'], 'target'] = -1
    return df


def make_zig_target(df: pd.DataFrame) -> pd.DataFrame:
    h, l = df['High'].values, df['Low'].values
    h = h.astype(np.float32)
    l = l.astype(np.float32)
    # h_pivots, l_pivotsはピボットのインデックス番号
    _, _, h_pivots, l_pivots = calc_zigzag(
        h, l, zz_period=zig_period, num_shifts=0)
    res: np.ndarray = np.full_like(h, fill_value=0)
    res[h_pivots] = 1
    res[l_pivots] = -1
    df['zig_target'] = res
    df['zig_target'].fillna(method='bfill', inplace=True)

    # fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])

    # ts_pivots_h = pd.Series(df['High'], index=df.index)
    # ts_pivots_l = pd.Series(df['Low'], index=df.index)
    # ts_pivots_h = ts_pivots_h[df['zig_target'] == 1]
    # ts_pivots_l = ts_pivots_l[df['zig_target'] == -1]

    # # Combine and sort the pivots
    # ts_pivots = pd.concat([ts_pivots_h, ts_pivots_l]).sort_index()

    # prev_idx = ts_pivots.index[0]
    # prev_val = ts_pivots.values[0]
    # for idx, val in ts_pivots.items():
    #     open_pos = (prev_idx, prev_val)
    #     close_pos = (idx, val)
    #     color = 'blue' if df.loc[idx, 'High'] == val else 'red'
    #     fplt.add_line(open_pos, close_pos, color, width=3)
    #     prev_idx = idx
    #     prev_val = val

    # fplt.show()
    return df


df_all = pd.DataFrame()

# 'Bar'ディレクトリ内の全ファイルをループ
for file in os.listdir('./Data/Bar/'):
    # csvファイルを読み込む
    df = pd.read_csv('./Data/Bar/' + file, index_col=0)
    # indexをdatetime形式に変更
    df.index = pd.to_datetime(df.index)

    # ターゲットデータを追加
    df = make_zig_target(df)

    # 1分足，1時間足のインジケーターを追加
    df = add_multi_time_indicators(df, time_frames, check_empty=False)

    # 1分足，1時間足のインジケーターを追加したdfをdf_allに追加
    df_all = pd.concat([df_all, df])

    del df

# df_allを出力
df_all.to_parquet(out_file, compression='zstd', index=True)
print(df_all.shape)
