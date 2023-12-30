# 生データから学習用データへの加工

volumeが一定以上の銘柄を見つけてlist化し，Data/JPX/issue_codes_5000_0000_over.pickleに保存
月別Tickから不要な列を削除，volumeが一定以上の銘柄のみを抽出してファイルに保存(月ごと)

ファイルをロードして銘柄ごとにファイルを分割
* Data/Split/{code}/{month}.parquet

複数月のファイルをマージ
* Data/Split/{code}/ディレクトリ内のファイルをロードしてdateとtime順にソート
* Data/Merged/{code}.parquetに出力