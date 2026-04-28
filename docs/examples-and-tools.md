# 例と補助ツール

[ドキュメント目次](README.md) / [README_ja.md に戻る](../README_ja.md)

このページは `examples/` と `Tools/` の解説です。現行仕様の解説については [configuration.md](configuration.md)、[operations.md](operations.md)、[cli.md](cli.md)、[output-artifacts.md](output-artifacts.md) を確認してください。

## 例

| ファイル | 位置づけ |
| --- | --- |
| `examples/examples.yaml` | YAML の最小例。フルモデル出力、疎ベクトル作成、保存済みベクトル適用を示します。 |
| `examples/cheat_sheet.yaml` | 設定項目を広めに並べたリファレンスです。CLI 前提の `torch_dtype` / `skip_layernorm` は YAML には書かず、CLI で渡す前提です。 |
| `examples/lora_merge.yaml` | LoRA をマージする場合の YAML 例です。 |

`target: null` / `target: none` はフルモデルではなく `<out_dir>/vector/*.difftensors` を保存します。フルモデルを保存したい例では、`target` に既存モデルのパスまたは ID を指定してください。

## 補助ツール

### `Tools/cpt_merge_search.py`

CPT チェックポイントの組み合わせ候補 YAML を生成し、必要なら `main.py` を呼び出して候補を実行する補助スクリプトです。

注意点:

- 生成 YAML には由来情報として `torch_dtype` が入りますが、`main.py` の実行時 dtype は CLI の `-t` / `--torch-dtype` で決まります。
- 単発候補は `target: null` を使うため、通常は `<merged_root>/vector/<name>.difftensors` を生成します。
- 連鎖候補の最終ステップは `target: recurrent` を使うため、出力先は `<merged_root>/recurrent/...` 系になります。
- `find_final_artifact()` は `vector/<final_model_name>_*` だけを探す狭いヘルパーです。完全一致の `.difftensors` や `recurrent/` 側の出力物を拾えない場合があります。評価まで自動実行する用途ではこのヘルパーは向きません。

### `Tools/extract_layer_info.py`

`.safetensors` / `.pth` のキーを `<model>.txt` に書き出す旧式のヘルパーです。メインのローダーと同じ入力範囲ではありません。

現行仕様に沿ってレイヤー名を確認する場合は、できるだけ CLI の `--dump-layers` を使ってください。

```bash
python main.py -c config.yaml --dump-layers -o layer_dumps
```
