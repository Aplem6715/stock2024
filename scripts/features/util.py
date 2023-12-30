import pandas as pd
import numpy as np


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
