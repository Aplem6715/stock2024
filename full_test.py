import pandas as pd
import numpy as np
import lightgbm as lgb
import finplot as fplt
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

from scripts.features.feature import add_multi_time_indicators

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

bat_path = './DataTest/Bar/72110.csv'
model_path = './Model/last_model.lgbm'
time_frames = ['1T', '1H']
feature_calc_len = 60*52  # 一目均衡のwindow3=52足分
init_balance = 100_0000
balance = init_balance
size = 1_0000


class Trade:
    def __init__(self, open_index, close_index, position):
        self.open_index = open_index
        self.close_index = close_index
        self.position = position
        self.profit = 0


df = pd.read_csv(bat_path, index_col=0, parse_dates=True)
model = lgb.Booster(model_file=model_path)
print(len(df))

# デバッグ用
# df = df.iloc[:5000, :]

position = 0
next_position = 0
start_price = 0
act = 0
close = df['Close'].values
open = df['Open'].values
trades: list[Trade] = []
for i in tqdm(range(feature_calc_len, len(df) - 1)):
    start_index = max(0, i-feature_calc_len)
    in_df = df.iloc[start_index:i, :]
    features = add_multi_time_indicators(in_df, time_frames, False)
    features = features.drop(
        columns=['Open', 'High', 'Low', 'Close']).iloc[-1, :]
    pred = model.predict(features)
    new_act = round(pred[0])

    gap_window = abs(close[i] - open[i+1])
    if act != new_act or gap_window > int(close[i]*0.01):
    # if act != new_act:

        profit = (close[i] - start_price) * position
        balance += profit

        act = new_act
        start_price = open[i+1]
        position = size * (1 if new_act == 1 else -1)
        if position != 0:
            if len(trades) > 0:
                trades[-1].close_index = i
                trades[-1].profit = profit
            trades.append(Trade(i+1, 0, position))

    i += 1

fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])

for trade in trades:
    if trade.close_index == 0:
        continue
    open_pos = (trade.open_index, open[trade.open_index])
    close_pos = (trade.close_index, close[trade.close_index])
    color = 'blue' if trade.position > 0 else 'red'
    fplt.add_line(open_pos, close_pos, color, width=3)
    color = 'green' if trade.profit > 0 else 'red'
    fplt.add_text(close_pos, f'{trade.profit}', color=color)

print(f'balance: {balance} (profit:{balance-init_balance})')
fplt.show()
