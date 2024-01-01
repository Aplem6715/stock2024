import pandas as pd
import numpy as np
import lightgbm as lgb
import gc


def balance_data(X: pd.DataFrame,
                 y: pd.Series):
    vc = y.value_counts()
    minimum_count = vc.values[-1]
    labels = vc.index
    sampled_data_X = []
    sampled_data_y = []
    for label in labels:
        sample_index = y[y == label].sample(
            minimum_count, random_state=42).index
        sampled_data_X.append(X.loc[sample_index])
        sampled_data_y.append(y.loc[sample_index])
    y = pd.concat(sampled_data_y, axis=0)
    X = pd.concat(sampled_data_X, axis=0)
    return X, y


def make_dataset(X: pd.DataFrame,
                 Y: pd.Series,
                 balance_label: bool):
    '''
    dfからLightGBM用のデータセットを作成する。
    '''

    y = Y

    non_null_index = ~(X.isnull().any(axis=1) & Y.isnull())
    X = X[non_null_index]
    y = y[non_null_index]

    if balance_label:
        print('before balance')
        print(y.value_counts())
        X, y = balance_data(X, y)
        # sampler = RandomUnderSampler(random_state=42)
        # X, y = sampler.fit_resample(X, y)
        print('after balance')
        print(y.value_counts())

    gc.collect()
    # シャッフル
    X = X.sample(frac=1, random_state=42)
    gc.collect()
    y = y.sample(frac=1, random_state=42)
    gc.collect()

    return lgb.Dataset(X, y)
