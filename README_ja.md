# 🥷 Ninja Merger

[English](README.md) | [日本語](README_ja.md)

## 概要

Ninja Merger は、PyTorch ベースの深層学習モデル (特に Transformer モデル) をマージするためのツールです。複数のモデルを組み合わせて、新しいモデルを作成したり、既存のモデルを微調整したりすることができます。さまざまなマージ手法 (加算、減算、混合、QEIC など) をサポートしており、柔軟なモデルマージが可能です。

## 特徴

*   **多様なマージ手法:**
    *   基本的な四則演算 (加算、減算、乗算、除算)
    *   モデルの混合 (Mix, Average)
    *   レイヤー絞り込みやエクスポート向けの Passthrough
    *   テンソルの連結 (Concatenation)
    *   最大/最小プーリング (MaxPool, MinPool)
    *   幾何平均 (GeometricMean)
    *   標準偏差を考慮した減算 (StdSub)
    *   モデルの幅を広げる WidenMerge
    *   複素数を用いたマージ (ComplexAdd, ComplexAngleMerge)
    *   量子もつれに触発された計算に基づくマージ (QEICAdd, QeicMix, QeicSub)
*   **柔軟な設定:**
    *   YAML 形式の設定ファイルを使用して、マージプロセスを詳細に制御できます。
    *   複数のモデルマージ設定を単一のファイルで管理できます。
    *   モデル固有の設定 (キー変換、レイヤー挿入) を定義できます。
    *   レイヤーごとの velocity を設定できます。
    *   マージするレイヤーを範囲や名前で指定できます。
    *   サイズの異なるレイヤーの処理方法を選択できます (スキップまたは共通部分のみ使用)。
*   **LoRA サポート:**
    *   LoRA (Low-Rank Adaptation) モデルを自動的にマージできます。
*   **レイヤー削除:**
    *   `drop_layers` で指定したレイヤーを出力モデルの `state_dict` から除去できます。
*   **詳細なログ出力:**
    *   `rich` ライブラリを使用した、視覚的にわかりやすいログ出力を提供します。
*   **拡張性:**
    *   モジュール構造により、新しいマージ手法や前処理/後処理オプションを簡単に追加できます。

## インストール

```bash
git clone https://github.com/Local-novel-llm-project/Ninja-merger.git
cd ninja-merger
pip install -r requirements.txt
```

## 使い方

1.  **設定ファイルの作成:** `config.yaml` という名前の YAML ファイルを作成し、マージするモデル、マージ手法、その他のオプションを指定します。
2.  **コマンドの実行:** 以下のコマンドを実行して、モデルをマージします。

```bash
python main.py -c config.yaml -o merged_models
```
## 対応フォーマット
- `*.safetensors`
- AutomodelForCausalLM (HuggingFace)

## 設定ファイルの例
```yaml
models:
- name: "model_add"
  left: "path/to/model1"
  right: "path/to/model2"
  operation: "add"
- name: "model_mix_recurrent"
  left: "model_add"  # result of the previous merge operation
  right: "path/to/model3"
  operation: "mix"
  velocity: 0.5
- name: "model_passthrough"
  left: "path/to/model4"
  operation: "passthrough"
  drop_layers:
    - "model.layers.24"
    - "lm_head"

```

## コマンドライン引数
```
usage: main.py [-h] [-c CONFIG] [-o OUT_DIR] [-n] [-dm MERGE_MODELS_DEVICE] [-dt TARGET_MODEL_DEVICE] [-t TORCH_DTYPE] [-r RECURRENT_MODE] [-d] [-l]
               [--dump_layers] [--include_layers INCLUDE_LAYERS] [--exclude_layers EXCLUDE_LAYERS]

Merge models

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to the JSON configuration file
  -o OUT_DIR, --out_dir OUT_DIR
                        Directory to save the merged model
  -n, --skip_layernorm  Skip layernorm during merging
  -dm MERGE_MODELS_DEVICE, --merge_models_device MERGE_MODELS_DEVICE
                        Device for merging models
  -dt TARGET_MODEL_DEVICE, --target_model_device TARGET_MODEL_DEVICE
                        Device for the target model
  -t TORCH_DTYPE, --torch_dtype TORCH_DTYPE
                        Torch data type
  -r RECURRENT_MODE, --recurrent_mode RECURRENT_MODE
                        use target recurrent mode
  -d, --dry_run         Dump processed layer infos without merging
  -l, --save_only_last_model
                        Only last model saved
  --dump_layers         Dump model layers to a file instead of merging
  --include_layers INCLUDE_LAYERS
                        Comma-separated list of layers to include
  --exclude_layers EXCLUDE_LAYERS
                        Comma-separated list of layers to exclude
```

## 貢献

バグ報告、機能リクエスト、プルリクエストは大歓迎です。

## ライセンス

Apache License 2.0
