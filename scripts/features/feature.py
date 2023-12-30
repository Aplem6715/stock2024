from typing import List
import pandas as pd
import numpy as np
from util import conv_chart

from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator


def lower(a, b):
    return np.where(a < b, a, b)


def upper(a, b):
    return np.where(a > b, a, b)

def conv_chart(df: pd.DataFrame, bar_freq: str) -> pd.DataFrame:
    '''
    入力されたOHLCデータを持つ[df]DataFrameの時間足を[bar_freq]に変更する
    '''
    ret = df.resample(bar_freq,
                      label='right',
                      closed='right').agg({'Open': 'first',
                                           'High': 'max',
                                           'Low': 'min',
                                           'Close': 'last',
                                           'Volume': 'sum'})
    return ret

def add_multi_time_indicators(origin: pd.DataFrame, freqs: List[str], check_empty: bool = True) -> pd.DataFrame:
    is_first_freq = True
    for freq in freqs:
        df = conv_chart(origin, freq)
        if check_empty:
            df = df.loc[(df['Open'] != df['High']) | (df['Open'] != df['Low']) | (df['Open'] != df['Close'])]
        add_indicators(df)
        if is_first_freq:
            ret = df
        else:
            df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            df = df.add_suffix(f'_{freq}')
            ret = ret.join(df)
        is_first_freq = False
    ret = ret.fillna(method='ffill')

    print(np.isnan(ret.values).sum(axis=0))
    print(ret.shape)

    ret = ret.dropna()

    # print(np.isnan(ret.values).sum(axis=0))
    # print(ret.shape)

    return ret

# dfにインジケーター列を追加
def add_indicators(df: pd.DataFrame):
    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']

    # RSI (Relative Strength Index) インジケーターを追加
    rsi = RSIIndicator(close=c)
    df['RSI'] = rsi.rsi()

    # ストキャスティクス オシレーターを追加
    sto = StochasticOscillator(high=h, low=l, close=c)
    df['ST_K'] = sto.stoch()
    df['ST_D'] = sto.stoch_signal()

    # ROC (Rate of Change) インジケーターを追加
    roc = ROCIndicator(close=c)
    df['ROC'] = roc.roc()

    # Williams %R インジケーターを追加
    wr = WilliamsRIndicator(high=h, low=l, close=c)
    df['Wil_R'] = wr.williams_r()

    # 単純移動平均 (SMA) インジケーターを追加
    sma5 = SMAIndicator(close=c, window=5).sma_indicator()
    sma25 = SMAIndicator(close=c, window=25).sma_indicator()
    df['SMA_S'] = sma25 - sma5
    df['SMA_C'] = c - sma25

    # 指数移動平均 (EMA) インジケーターを追加
    ema5 = EMAIndicator(close=c, window=5).ema_indicator()
    ema25 = EMAIndicator(close=c, window=25).ema_indicator()
    df['EMA_S'] = ema25 - ema5
    df['EMA_C'] = c - ema25

    # MACD インジケーターを追加
    macd = MACD(close=c)
    # df['MACD'] = macd.macd()
    # df['MACD Signal'] = macd.macd_signal()
    df['MACD_H'] = macd.macd_diff()

    # On Balance Volume (OBV) インジケーターを追加
    obv = OnBalanceVolumeIndicator(close=c, volume=df['Volume']).on_balance_volume()
    obv_short = SMAIndicator(close=obv, window=5).sma_indicator()
    obv_long = SMAIndicator(close=obv, window=25).sma_indicator()
    df['OBV'] = obv_long - obv_short

    # Accumulation/Distribution (Acc/Dist) インジケーターを追加
    adi = AccDistIndexIndicator(high=h, low=l, close=c, volume=df['Volume'])
    ad = adi.acc_dist_index()
    ad_short = SMAIndicator(close=ad, window=5).sma_indicator()
    ad_long = SMAIndicator(close=ad, window=25).sma_indicator()
    df['ADI'] = ad_long - ad_short

    # 一目均衡表 インジケーターを追加
    ichimoku = IchimokuIndicator(high=h, low=l)
    ichimoku_base = ichimoku.ichimoku_base_line()
    ichimoku_conv = ichimoku.ichimoku_conversion_line()
    ichimoku1 = ichimoku.ichimoku_a()
    ichimoku2 = ichimoku.ichimoku_b()
    df['ICH_CLD'] = ichimoku2 - ichimoku1
    df['ICH_BASE'] = ichimoku_base - c
    df['ICH_CONV'] = ichimoku_conv - ichimoku_base
    df['ICH_REG_HI'] = lower(ichimoku1, ichimoku2) - h
    df['ICH_REG_LO'] = upper(ichimoku1, ichimoku2) - l
    
    return df


if __name__ == "__main__":
    df = pd.read_parquet("Data/Dukascopy/USDJPY_2012.zstd")
    # df = df.iloc[500000:500010, :]
    df = add_multi_time_indicators(df, ['1T', '5T', '1H', '4H'])
    print(df)