import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
from data import make_dataset

log_dir = './log/'
model_save_path = './Model/last_model.lgbm'
BINARY_PARAM = {
    'task': 'train',                # 学習、トレーニング ⇔　予測predict
    'boosting_type': 'gbdt',        # 勾配ブースティング
    'objective': 'binary',      	# 目的関数：二値分類
    'metric': 'auc',                # 分類モデルの性能を測る指標
    'learning_rate': 0.01,          # 学習率（初期値0.1）
    # 'min_data_in_leaf': 25,          # データの最小数（初期値20）
    'max_depth': -1,
    'num_leaves': 256,               # 決定木の複雑度を調整（初期値31）
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.5,
    'device': 'cpu',
    'seed': 42
}
param = BINARY_PARAM
num_boost_round = 10000


# 'Data/train.parquet'を読み込む
df = pd.read_parquet('Data/train.parquet', engine='pyarrow')
df.drop(columns=['Open', 'High', 'Low', 'Close'], inplace=True)
print(df.columns)

# zig_targetの-1を0に変換
df['zig_target'] = df['zig_target'].replace(-1, 0)

# dfを学習用と検証用に分割
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size, :]
valid_df = df.iloc[train_size:, :]

lgb_train = make_dataset(train_df.drop(
    columns=['zig_target']), train_df['zig_target'], True)
lgb_valid = make_dataset(valid_df.drop(
    columns=['zig_target']), valid_df['zig_target'], True)

del df
del train_df
del valid_df
gc.collect()

evals_result = {}
model = lgb.train(param,
                  lgb_train,
                  num_boost_round=num_boost_round,
                  valid_names=['train', 'valid'],
                  valid_sets=[lgb_train, lgb_valid],
                  callbacks=[lgb.early_stopping(100),
                             lgb.record_evaluation(evals_result),
                             lgb.log_evaluation(period=10)])
best_iteration = model.best_iteration
model.save_model(
    filename=model_save_path,
    num_iteration=best_iteration
)

lgb.plot_metric(evals_result, metric=param['metric'])
plt.savefig(log_dir+'metric_graph.png')
print(f'best_iteration: {best_iteration}')
