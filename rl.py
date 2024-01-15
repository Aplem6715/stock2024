from fx_env import TradeEnv
import os
from .feature import add_multi_time_indicators
import pandas as pd
import pickle
import numpy as np
import scripts.features.feature as feature
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

n_envs = 8
policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[128, 128, 64])
freqs = ['10T', '1H']
# 特徴量計算に必要なOHLCデータの長さ: 4H*SMA50基準で余裕を持って1.5倍（計算期間中はトレーニングに使われない）
FEATURE_CALC_DELAY = 60*52 # 一目均衡のwindow3=52足分

in_dir = './Data/Bar/'


def make_env(df_list: list[pd.DataFrame]):
    return TradeEnv(df_list, 10000+FEATURE_CALC_DELAY)


if __name__ == "__main__":

    df_list: list[pd.DataFrame] = [] 

    # 'Bar'ディレクトリ内の全ファイルをループ
    for file in os.listdir(in_dir):
        # csvファイルを読み込む
        df = pd.read_csv(in_dir + file, index_col=0)
        # indexをdatetime形式に変更
        df.index = pd.to_datetime(df.index)

        # 1分足，1時間足のインジケーターを追加
        df = feature.add_multi_time_indicators(df, freqs, check_empty=False)

        # 1分足，1時間足のインジケーターを追加したdfをdf_allに追加
        df_list.append(df)

    del df

    env_functions = [lambda df_list=df_list: make_env(df_list) for _ in range(n_envs)]
    env = DummyVecEnv(env_functions)

    dummy = make_env()
    with open('scaler2.pkl', 'wb') as f:
        pickle.dump(dummy.close_std, f)

    eval_df = pd.read_parquet('DataTest/test.parquet')
    eval_df = feature.add_multi_time_indicators(eval_df, freqs)
    eval_env = TradeEnv(eval_df, 10000+FEATURE_CALC_DELAY)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./rl_models/",
                                 log_path="./rl_models/", eval_freq=1_0000,
                                 deterministic=True, render=False)

    model = PPO("MlpPolicy", env, gamma=0.5, verbose=1,
                policy_kwargs=policy_kwargs, tensorboard_log='./log/test/', device="cuda")
    model.learn(total_timesteps=100_0000, callback=eval_callback)

    # import matplotlib.pyplot as plt
    # plt.hist(env.rewards, bins=50, range=(-1.1, 1.1))
    # plt.show()

    model.save('rl_models')
