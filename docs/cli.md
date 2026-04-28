# CLI 仕様

[ドキュメント目次](README.md) / [README_ja.md に戻る](../README_ja.md)

CLI は `main.py` の `build_parser()` で定義されています。

```bash
python main.py [-h] [-c CONFIG] [-o OUT_DIR] [-n] [-dm MERGE_MODELS_DEVICE] [-dt TARGET_MODEL_DEVICE] [-t {float16,bfloat16,float32,float64}] [-r | --recurrent-mode | --no-recurrent-mode] [-d] [-l] [--dump-layers] [--include-layers INCLUDE_LAYERS] [--exclude-layers EXCLUDE_LAYERS]
```

注意: `main.py` は、パーサー構築前に `modules.Services.merge_runner` を読み込み、その先で `torch` / `rich` / `yaml` なども読み込みます。そのため、依存関係が不足している環境では `python main.py --help` でもインポートエラーで終了し、ヘルプは表示されません。

## 入力と出力

| オプション | 既定値 | 内容 |
| --- | --- | --- |
| `-c`, `--config` | `model_config.yaml` | YAML マージ設定。`load_config()` は文字列中の `.json` を `.yaml` に置換したパスだけを開きます。存在しない場合は元のパスと置換後パスを含む `FileNotFoundError` になります。 |
| `-o`, `--out-dir`, `--out_dir` | `./merged_models` | 出力ルート。実際の保存先は `target` に応じて下位ディレクトリが切られます。 |

## 実行

| オプション | 既定値 | 内容 |
| --- | --- | --- |
| `-n`, `--skip-layernorm`, `--skip_layernorm` | `False` | レイヤー名に `layernorm` を含むレイヤーをスキップ。 |
| `-dm`, `--merge-models-device`, `--merge_models_device` | `cpu` | `left` / `right` モデルをロードするデバイス指定。 |
| `-dt`, `--target-model-device`, `--target_model_device` | `cpu` | 明示的な `target` モデルをロードするデバイス指定。 |
| `-t`, `--torch-dtype`, `--torch_dtype` | `bfloat16` | `float16` / `bfloat16` / `float32` / `float64`。 |
| `-r`, `--recurrent-mode`, `--recurrent_mode` | `True` | 前ステップのマージ結果を次ステップに渡す。`argparse.BooleanOptionalAction` により `--no-recurrent-mode` / `--no-recurrent_mode` も使えます。 |
| `--no-recurrent-mode`, `--no-recurrent_mode` | - | `recurrent_mode` を無効化し、各ステップの `current_target_model` を `None` に戻します。 |
| `-d`, `--dry-run`, `--dry_run` | `False` | マージは実行するが保存しません。既存アーティファクトもバイパスされます。 |
| `-l`, `--save-only-last-model`, `--save_only_last_model` | `False` | 中間ステップはメモリ上だけに保持し、最後だけ保存。 |

`--dump-layers` と `--dry-run` は併用できません。`parse_args()` が `parser.error()` で終了します。

`--save-only-last-model` と `--dry-run` はパーサー上は併用できます。この場合、各ステップはマージまで実行されますが保存されず、ステップは `dry_run` になります。

## レイヤーフィルター

| オプション | 内容 |
| --- | --- |
| `--include-layers`, `--include_layers` | カンマ区切りの取り込みフィルター。空の項目は削除されます。 |
| `--exclude-layers`, `--exclude_layers` | カンマ区切りの除外フィルター。空の項目は削除されます。 |

CLI のフィルターが指定された場合、そのステップの YAML `include_layers` / `exclude_layers` より優先されます。CLI フィルターは全ステップに同じリストとして適用されます。未指定の場合だけ、各ステップの YAML フィルターが使われます。

CLI フィルターの解析は `parse_layers()` による単純なカンマ区切りです。`"model.layers.0-8,lm_head"` は `["model.layers.0-8", "lm_head"]` になりますが、CLI 側では空白区切りや Python リストのリテラルは特別扱いされません。

## 診断

| オプション | 内容 |
| --- | --- |
| `--dump-layers`, `--dump_layers` | ターゲットまたは最初のベースモデルからレイヤー名を `<stem>_layers.txt` に書き出し、マージはしません。 |

`target` が `null` の場合、ダンプ対象は最初の `left` モデルです。

ダンプモードでもコンフィグの読み込み、対象モデルの解決、`define_savename()`、レシピ書き込み、レイヤーダンプ用のモデル読み込みは行われます。マージと保存アーティファクト作成だけを避けるモードです。

## CLI と設定ファイルの優先順位

CLI オプションは `MergeRunnerOptions` に入り、実行全体に適用されます。YAML 設定は `load_config()` で `AppConfig` / `MergeRequest` に正規化されます。

| 項目 | 優先順位 / 挙動 |
| --- | --- |
| `include_layers` / `exclude_layers` | CLI 指定があれば全ステップで CLI が優先。CLI 未指定ならステップごとの YAML 値。 |
| `use_scaling` | ステップの `use_scaling` が `None` でなければ ステップ値。なければルートの `use_scaling`、さらに `use_scale`、最後は `False`。 |
| `target: recurrent` | `--recurrent-mode` が有効な場合のみ、前ステップの結果を次ステップに渡せます。最初のステップで `recurrent` を指定すると `MergeExecutionError`。 |
| 保存タイミング | `--save-only-last-model` なしなら各ステップ保存。ありなら最後のステップだけ保存。`--dry-run` は保存を常に止めます。 |
| 既存アーティファクト | 通常の衝突は `define_savename()` が既存アーティファクトを検出し、`_<timestamp>`、さらに必要なら `_<timestamp>_NN` 接尾辞付きのステムにリネームして回避します。`skipped_existing` は、その後の競合や外部から渡されたステムが既に存在する場合の防御的な状態です。 |

## 実行サマリー

実行時は `rich` による全体表示、ステップ表示、レイヤー進捗、実行サマリーが表示されます。ステップ状態は主に次の値です。

| 状態 | 意味 |
| --- | --- |
| `saved` | アーティファクト保存完了。 |
| `computed_only` | `--save-only-last-model` により中間結果をメモリ保持。 |
| `dry_run` | 保存なし。 |
| `dumped_layers` | レイヤーダンプ完了。 |
| `skipped_existing` | `define_savename()` 後の保存予定ステムに既存アーティファクトが見つかったため、上書きを避けてスキップ。通常の名前衝突はタイムスタンプ / 接尾辞付きステムにリネームされます。 |
| `load_failed` | 入力モデルの読み込み失敗。 |
| `merge_failed` | マージ処理が結果を返さなかった。 |

`main()` は `run_merge()` から上がった `MergeExecutionError` を `SystemExit(1)` に変換します。ただし、ステップ内の読み込み / マージ失敗は `MergeStepResult` として記録され、処理自体は次ステップに進みます。

## 例

CUDA 上でフルモデルを出力:

```bash
python main.py -c config.yaml -o merged_models -dm cuda:0 -dt cuda:0 -t bfloat16
```

最後のステップだけ保存:

```bash
python main.py -c chain.yaml -o merged_models --save-only-last-model
```

設定ファイルを変えずにレイヤーを絞る:

```bash
python main.py -c config.yaml --include-layers "model.layers.0-8,lm_head"
```

レイヤー名だけ確認:

```bash
python main.py -c config.yaml --dump-layers -o layer_dumps
```
