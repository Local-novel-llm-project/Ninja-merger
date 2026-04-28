# Ninja Merger

[English](README.md) | [日本語](README_ja.md)

Ninja Merger は、YAML レシピをもとに PyTorch / Hugging Face 系モデルをマージするツールです。明示的な `target` を指定したフルモデル出力、`target: null` による疎なマージベクトル、複数ステップの `recurrent` チェーン、レイヤー単位のフィルタリング、各種マージ演算を扱えます。

この README は入口用です。実装に合わせた詳細仕様は [docs/README.md](docs/README.md) から参照できます。

## 現在の主な機能

- 必須の `models` セクションによる YAML 駆動の複数ステップマージ。
- Hugging Face のモデル ID / ディレクトリ、`*.safetensors`、`*.pth`、`*.bin`、Ninja Merger の `*.difftensors` 入力。
- `target` に既存モデルを指定したフルモデル出力。
- `target: null` / `target: none` による `*.difftensors` 疎ベクトル出力。
- `left`、`right`、`target` で前ステップ結果を参照する `recurrent`。
- レイヤーの include / exclude / drop、レイヤー一覧 dump、サイズ不一致時の処理。
- prefix / regex によるレイヤー別 `velocity` / `post_velocity`。
- 四則演算、`passthrough`、`none`、Widen、angle / complex、QEIC 系演算。`complex_mix` はレジストリにありますが未実装です。
- `rich` による進捗表示、ステップ別サマリー、実行レシピの保存。

## インストール

`pyproject.toml` では Python `>=3.13` を要求しています。

```bash
git clone https://github.com/Local-novel-llm-project/Ninja-merger.git
cd Ninja-merger
uv sync
```

`uv` を使わない場合は、依存関係を直接入れます。

```bash
pip install -r requirements.txt
```

## 最短例

フルモデルとして保存したい場合は、`target` に既存モデルを指定します。`post_operation` の既定値は `add` なので、マージ結果そのものを対象モデルの各レイヤーへ入れたい場合は `post_operation: none` を明示します。

```yaml
models:
  - name: mix-full-model
    target: path/to/target-model
    left: path/to/base-model
    right: path/to/tuned-model
    operation: mix
    velocity: 0.35
    post_operation: none
```

実行例:

```bash
python main.py -c config.yaml -o merged_models --merge-models-device cuda:0 --target-model-device cuda:0 --torch-dtype bfloat16
```

フルモデルではなく疎なベクトルを作る場合は `target: null` を使います。Ninja Merger は最初の `left` モデルをメモリ上のレイヤー元として使い、非ゼロのテンソルだけを `merged_models/vector/*.difftensors` に保存します。トークナイザーやフルモデルディレクトリは保存しません。`post_operation` はこの場合も既定で `add` なので、`left - right` のような素直な差分を作る例では `post_operation: none` を明示します。

```yaml
models:
  - name: delta-vector
    target: null
    left: path/to/base-model
    right: path/to/tuned-model
    operation: sub
    post_operation: none
```

## CLI の基本

```bash
python main.py -c model_config.yaml -o merged_models
```

よく使うオプション:

| オプション | 内容 |
| --- | --- |
| `-c`, `--config` | YAML レシピ。既定値は `model_config.yaml`。 |
| `-o`, `--out-dir` | 出力先ディレクトリ。既定値は `./merged_models`。 |
| `-dm`, `--merge-models-device` | `left` / `right` モデルのロード先。既定値は `cpu`。 |
| `-dt`, `--target-model-device` | 明示的な `target` モデルのロード先。既定値は `cpu`。 |
| `-t`, `--torch-dtype` | `float16`、`bfloat16`、`float32`、`float64`。 |
| `--no-recurrent-mode` | 前ステップ結果の再利用を無効化。 |
| `-l`, `--save-only-last-model` | 中間結果をメモリ上に保持し、最後のステップだけ保存。 |
| `-d`, `--dry-run` | 出力物を書かずにマージ経路だけ実行。 |
| `--dump-layers` | マージせずレイヤー名を出力。`--dry-run` とは併用不可。 |
| `--include-layers`, `--exclude-layers` | CLI 側のレイヤーフィルタ。設定ファイル側より優先。 |

全体は [docs/cli.md](docs/cli.md) にまとめています。

## ドキュメント

- [docs/README.md](docs/README.md): ドキュメントの入口。
- [docs/configuration.md](docs/configuration.md): YAML スキーマ、既定値、例。
- [docs/operations.md](docs/operations.md): マージ演算と補助処理。
- [docs/output-artifacts.md](docs/output-artifacts.md): 入力形式、出力形式、保存先。
- [docs/cli.md](docs/cli.md): コマンドライン仕様。
- [docs/examples-and-tools.md](docs/examples-and-tools.md): 例と補助ツールの位置づけ。

## 開発

```bash
pytest
```

テストではパーサー、設定正規化、演算レジストリ、レイヤー処理、疎ベクトル、主要演算の参照結果を確認しています。

## ライセンス

Apache License 2.0
