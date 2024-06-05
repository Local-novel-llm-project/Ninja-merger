# 🥷ninja-merger
[English](README.md) | *日本語*


ninja-merger はLLMをベクトルマージするための小さなツールです。


## インストール

1. リポジトリのクローン
   ```bash
   git clone https://github.com/Local-novel-llm-project/ninja-merger
   cd ninja-merger
   ```

1. (オプション、推奨) Python仮想環境の作成とアクティベート
   ```bash
   # for example, we use venv
   python -m venv venv
   ```

1. pipを使って依存関係のインストール
   ```bash
   pip install -r requirements.txt
   ```


## 使い方

```bash
python ninja-merger.py -c <your yaml config>.yml
```

## 設定

ninja-merger はマージ方法の設定にYAMLフォーマットを使用しています。
設定ファイルの例は `examples` フォルダ以下にあります。

各設定の詳細は設定ファイル例の中にコメントで書いています。


## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)