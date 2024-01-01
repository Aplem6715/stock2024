import pandas as pd
import os
import pickle
import shutil

# filtered_dir = './DataTest/Filtered/'
# out_dir = './DataTest/Split/'
# codes_out_file = './DataTest/split_codes.csv'
filtered_dir = './Data/Filtered/'
out_dir = './Data/Split/'
codes_out_file = './Data/split_codes.csv'

is_first = True
next_codes = []

shutil.rmtree(out_dir, ignore_errors=True)

# Data/Filtered/ディレクトリ内の全ファイルをループ
for file in os.listdir(filtered_dir):
    df = pd.read_parquet(filtered_dir + file)

    if is_first:
        codes: list[int] = df['issue code'].unique().tolist()

    ok_count = 0
    for code in codes:
        dir = out_dir + str(code)
        split_df = df[df['issue code'] == code]
        # issue code列を削除
        split_df = split_df.drop(columns='issue code')

        num_trade = split_df.shape[0]
        trade_per_day = num_trade / 20
        trade_per_min = int(num_trade / 20 / 5 / 60)

        if trade_per_min > 10 or not is_first:
            print('\033[32mOK:  ', end='')
            ok_count += 1
            os.makedirs(dir, exist_ok=True)
            split_df.to_parquet(dir + f'/split_{code}_' + file)
            if is_first:
                next_codes.append(code)
        else:
            if is_first:
                codes.remove(code)
            print('\033[31mNG:  ', end='')

        print(
            f'code:{code} /\t trade:{num_trade}[t] /\t {trade_per_day}[t/day] /\t {trade_per_min}[t/min]\033[0m')
    is_first = False
    codes = next_codes
    print(
        f'\033[32mOK\033[0m: {ok_count} / \033[31mNG\033[0m: {len(codes) - ok_count}')

# codesをcsvで保存
pd.Series(codes).to_csv(codes_out_file, index=False, header=False)
