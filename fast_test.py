import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from data import make_dataset

model_save_path = './Model/last_model.lgbm'
data_path = './DataTest/test.parquet'
out_dir = './log/'

model = lgb.Booster(model_file=model_save_path)
df = pd.read_parquet(data_path, engine='pyarrow')

# zig_targetの-1を0に変換
df['zig_target'] = df['zig_target'].replace(-1, 0)

test_X = df.drop(columns=['Open', 'High', 'Low', 'Close', 'zig_target'])
test_y = df['zig_target']

pred_test_X = test_X.sample(100000, random_state=42)
pred_test_Y = test_y.sample(100000, random_state=42)

y_pred_prob = model.predict(pred_test_X)

sns.displot(y_pred_prob, bins=100)
plt.savefig(os.path.join(out_dir, f'pred_hist.png'))

y_pred = np.round(y_pred_prob)

confusion = confusion_matrix(pred_test_Y, y_pred).tolist()
accuracy = accuracy_score(pred_test_Y, y_pred).tolist()
importance = pd.DataFrame(model.feature_importance(
    importance_type='split'), index=pred_test_X.columns, columns=['importance'])
importance = importance.sort_values('importance', ascending=False)
importance = importance.to_dict()

print(f'acc:{accuracy}')
print(confusion)
sns.heatmap(confusion)
plt.show()
print(importance)
# lgb.plot_importance(model, importance_type='split', max_num_features=20)
lgb.plot_importance(model)
plt.show()