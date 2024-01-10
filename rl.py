from fx_env import TradeEnv
import pandas as pd
import pickle
import numpy as np
import scripts.features.feature as feature
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[128, 128, 64])
freqs = ['10T', '1H']
# 特徴量計算に必要なOHLCデータの長さ: 4H*SMA50基準で余裕を持って1.5倍（計算期間中はトレーニングに使われない）
FEATURE_CALC_DELAY = 60*52 # 一目均衡のwindow3=52足分


def make_env():
    df_list = [pd.read_parquet('Data/train.parquet')]
    df = pd.concat(df_list)
    df = feature.add_multi_time_indicators(df, freqs)

    del df_list

    return TradeEnv(df, 10000+FEATURE_CALC_DELAY)


if __name__ == "__main__":
    n_envs = 8

    env = DummyVecEnv([make_env for _ in range(n_envs)])

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
