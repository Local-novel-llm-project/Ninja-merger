# Ninja Merger ドキュメント

[README.md](../README.md) / [README_ja.md](../README_ja.md)

このディレクトリには、現在の実装に合わせた詳しい仕様をまとめています。ルートの README は入口として短く保ち、設定、演算、CLI、出力物の詳しい説明はこのディレクトリへ分けています。

## 読む順番

1. [configuration.md](configuration.md): YAML レシピの形、既定値、設定時の注意点。
2. [operations.md](operations.md): `operation`、`post_operation`、前処理、正規化の挙動。
3. [output-artifacts.md](output-artifacts.md): 対応する入力、出力先の構成、`.difftensors` アーティファクト。
4. [cli.md](cli.md): コマンドラインオプションと、CLI 指定と設定ファイル指定の優先順位。
5. [examples-and-tools.md](examples-and-tools.md): 現在の例と補助スクリプトの注意点。

## 重要な注意点

- `models` は必須です。単一の mapping も受け付けますが、複数ステップのレシピでは list 形式にすると流れを追いやすくなります。
- `post_operation` の既定値は `add` です。`operation` の直接の結果で対象レイヤーを置き換えたい場合は、`post_operation: none` を指定します。
- `target: null` / `target: none` はフルモデルを保存しません。最初の `left` モデルをメモリ上のレイヤー元として使い、非ゼロのテンソルだけを含む疎な `<out_dir>/vector/<name>.difftensors` アーティファクトを書き出します。
- `*.difftensors` ファイルはマージ入力として読み込めます。マージ入力として扱う場合、存在しないキーは疎なゼロ値として扱われます。
- `--include-layers` と `--exclude-layers` は、各 YAML エントリのレイヤーフィルターより優先されます。
- `--dump-layers` はレイヤー名を書き出すための指定で、`--dry-run` とは併用できません。

## 実装との対応

- CLI: `main.py`
- 設定読み込み: `modules/Utils/loaders.py`
- 実行フロー: `modules/Services/merge_execution.py`
- 保存処理: `modules/Services/merge_output.py`
- 演算レジストリ: `modules/Utils/operation_dicts.py`
- レイヤー処理: `modules/Utils/layers.py`
- アーティファクト処理: `modules/Utils/diff_artifact.py`

## ルート README との関係

ルートの README は、インストール、最短のフルモデル例と疎ベクトル例、よく使う CLI オプション、詳細ページへのリンクを置く入口です。設定の既定値、演算の対応状況、アーティファクトの配置、注意が必要なケースは、このディレクトリ側で説明します。
