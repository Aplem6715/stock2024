import gymnasium as gym
import numpy as np
import random
import pandas as pd

import matplotlib.axes
import matplotlib.pyplot as plt
import mplfinance as mpf

from sklearn.preprocessing import StandardScaler


def to_pips(price):
    return price * 100


def pips2price(pips):
    return pips/100


def action_value(act):
    # sellの場合は-1
    if act == 2:
        return -1
    # それ以外はそのまま
    return act

# 未来の単純移動平均を計算


def calc_future_sma(arr, length):
    ret = np.zeros_like(arr)
    for i in range(len(arr) - length):
        crop = arr[i: i + length]
        ret[i] = np.median(crop)
    return ret


class TradeEnv(gym.Env):

    # 初期残高（円
    INIT_BALANCE = 100000

    COMMISSION = 0.0  # 取引手数料0.3pips(Rewardとは無関係)
    COMMISSION_PCT = 0.01  # 報酬手数料(reward) 何足分か＝close差分stdに対する比率. PCT * close.diff().std()
    NO_MOVE_PENALTY = COMMISSION_PCT/100
    AVOID_REWARD = 0.3  # 損失回避ボーナス 取引していないときに逆方向の動きがあった場合，close差分*AVOID_REWARDが報酬として与えられる
    CHANCE_LOSS = 0.3 # 機会損失 close差分*CHANCE_LOSSがマイナス報酬として付与される

    # ACTION_SIGN = [-1, 0 ,1]
    ACTION_SIGN = [0, 1]

    # 特徴量計算に必要なOHLCデータの長さ（計算期間中はトレーニングに使われない）
    FEATURE_CALC_DELAY = 60*52  # 一目均衡のwindow3=52足分

    # 過去何データ分を観測するか（TODO:テクニカルを含める場合は1に設定：テクニカルには過去データの情報が含まれている）
    OBSERVE_LEN = 1

    render_data_length = 48

    def __init__(self, df: pd.DataFrame, duration, position_size=1000, render_mode=None, scaler=None):

        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.ohlc = df.iloc[:, :5].copy()

        self.is_long_act_env = (self.ACTION_SIGN[1] == 1)
        self.index = 0
        self.balance = self.INIT_BALANCE
        self.position = 0
        self.position_size = position_size

        self.closes = df['Close'].values
        self.obs_arr = df.iloc[:, 5:].values
        # self.obs_arr = np.insert(self.obs_arr, -1, 0, axis=1)

        diffs = df['Close'].diff()
        std = diffs.std()
        self.std_diff2 = std * 2
        self.commission_reward = std * self.COMMISSION_PCT

        print(f'obs data: {self.obs_arr.shape}')

        if scaler is None:
            self.close_std = StandardScaler()
            self.obs_arr = self.close_std.fit_transform(self.obs_arr)
        else:
            self.close_std: StandardScaler = scaler
            self.obs_arr = self.close_std.transform(self.obs_arr)

        if duration:
            self.duration = duration
        else:
            self.duration = len(df)

        self.action_space = gym.spaces.Discrete(len(self.ACTION_SIGN))
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.obs_arr.shape[1],))
        # shape=(df.shape[1] - 5 + 1,))  # OHLCVデータ分-5
        self.reward_range = (-1.0, 1.0)
        self.is_first_render = True
        self.reward_sum = 0
        self.rewards = []
        random.seed(42)

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        if self.duration < len(self.closes):
            # ランダムな初期ディレイを与える
            self.index = random.randint(
                max(self.OBSERVE_LEN, self.FEATURE_CALC_DELAY),
                len(self.closes) - self.duration)
            self.end_idx = self.index + self.duration - self.FEATURE_CALC_DELAY
        else:
            self.index = max(self.OBSERVE_LEN, self.FEATURE_CALC_DELAY)
            self.end_idx = len(self.closes) - self.FEATURE_CALC_DELAY

        self.balance = self.INIT_BALANCE
        self.position = 0
        self.reward_sum = 0
        self.prev_act = None

        # r = pd.Series(self.rewards)
        # print(f'1% : {r.quantile(0.01)}')
        # print(f'5% : {r.quantile(0.05)}')
        # print(f'10%: {r.quantile(0.1)}')
        # print(f'25%: {r.quantile(0.25)}')
        # print(f'50%: {r.quantile(0.5)}')
        # print(f'75%: {r.quantile(0.75)}')
        # print(f'95%: {r.quantile(0.95)}')
        # print(f'99%: {r.quantile(0.99)}')
        self.rewards = []

        return self.observe(), {}

    def step(self, act: int):
        done = False
        truncated = False
        close = self.closes[self.index]
        prev_close = self.closes[self.index-1]
        close_diff = (close - prev_close)

        # update position
        if self.index + 1 >= self.end_idx:
            next_pos = 0
        else:
            pos_sign = self.ACTION_SIGN[act]
            if self.position_size > 0:
                next_pos = int(pos_sign * self.position_size)
            else:
                next_pos = int(pos_sign * self.position_size * self.balance)
        pos_diff = next_pos - self.position

        # 残高更新
        self.balance += self.position * close_diff

        if (pos_diff != 0) and (self.prev_act != act):
            self.balance -= abs(pos_diff) * self.COMMISSION

        # 報酬計算
        if self.position != 0:
            reward = np.sign(self.position) * close_diff
        elif len(self.ACTION_SIGN) == 2:
            if (close_diff < 0 and self.is_long_act_env) or (close_diff > 0 and (not self.is_long_act_env)):
                reward = abs(close_diff) * self.AVOID_REWARD
            else:
                reward = -abs(close_diff) * self.CHANCE_LOSS
        else:
            reward = 0

        if (pos_diff != 0) and (self.prev_act != act):
            reward -= self.commission_reward

        # close差分の標準偏差*2で標準化，手数料ペナルティの加算分も含める
        reward = reward / (self.std_diff2+self.commission_reward)
        reward = max(min(reward, 1), -1)
        # if reward != 0:
        #     self.rewards.append(reward)

        self.position = next_pos

        # インデックス更新
        self.update()

        # =======================================
        #        ここから次ステップの時間軸
        # =======================================

        if self.index >= self.end_idx:
            done = True

        info = {
            'balance': round(self.balance, 1),
        }

        self.reward_sum += reward
        self.reward = reward
        self.prev_act = act

        return self.observe(), reward, done, truncated, info

    def update(self):
        self.index += 1

    def render(self, mode):
        if mode == 'human':
            if self.is_first_render:
                self.init_human_render()
            self.human_render()

        print(
            '{:>4.1f}% ({:>4}/{:>4})  balance:{:>7.1f}  position:{:>8.1f}  R_Sum:{:>5.3f} Rew:{:>5.3f}                    '.format(
                (1 - (self.end_idx - self.index) / self.duration) * 100,
                self.end_idx - self.index, self.duration,
                self.balance,
                self.position,
                self.reward_sum,
                self.reward),
            end='\r')

    def init_human_render(self):
        self.fig = mpf.figure(style='charles', figsize=(7, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)  # type: plt.Axes
        self.is_first_render = False

    def human_render(self):
        if self.index < self.render_data_length:
            return
        s_time = self.index - self.render_data_length // 2
        e_time = self.index + self.render_data_length // 2
        data = self.ohlc.iloc[s_time:e_time]
        self.ax.clear()
        cur_time = self.ohlc.index[self.index]
        vline_color = 'k'
        if self.position > 0:
            vline_color = 'c'
        elif self.position < 0:
            vline_color = 'm'

        mpf.plot(data, ax=self.ax, type='candle', vlines=dict(
            vlines=[cur_time], colors=(vline_color), linewidths=5))
        # plt.draw()
        plt.pause(0.01)

    def observe(self):
        # obs = self.scaled_close[self.index - self.OBSERVE_LEN:self.index]
        obs = self.obs_arr[self.index]
        # obs[-1] = np.sign(self.position)
        return obs
