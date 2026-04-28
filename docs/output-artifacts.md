# 入力形式と出力アーティファクト

[ドキュメント目次](README.md) / [README_ja.md に戻る](../README_ja.md)

このページは `modules/Utils/loaders.py`、`modules/Services/merge_output.py`、`modules/Utils/diff_artifact.py`、`modules/Utils/utility.py`、`modules/Utils/models.py` の現在仕様をまとめたものです。

## 入力形式

| 入力 | ロード方法 |
| --- | --- |
| `*.safetensors` | `safetensors.torch.load_file` で `state_dict` として読み込み、`DummyModel` に包みます。 |
| `*.pth` / `*.bin` | `torch.load`。辞書に `model` があればそれを、なければ `state_dict`、それもなければロード結果そのものを `state_dict` として扱います。その結果に `weight` があれば `weight` を `state_dict` として使います。`model` / `state_dict` 形式では `config` があれば保持します。 |
| Hugging Face モデル ID / ディレクトリ | `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`。 |
| `*.difftensors` | Ninja Merger の疎アーティファクト。`sparse_vector` は保存済みテンソルだけを持つ疎モデルとして読み込み、`sparse_delta` の通常ロードは基準参照元から復元します。マージ入力側の差分ロードは加算用の疎モデルとして扱います。 |

`left` / `right` は文字列なら 1 要素リストへ正規化され、`"none"` / `"null"` は空リストとして扱われます。`target` も未指定、`"none"`、`"null"` は `None` に正規化されます。`velocity` の既定値は `1.0`、`post_velocity` の既定値も `1.0` です。

`.difftensors` を `left` / `right` として使う場合、`load_and_prepare_models()` は `load_model(..., diff_mode="delta")` で読み込みます。アーティファクトに存在しないキーは `_ninja_sparse_zero_missing` により 0 とみなせる疎な参照元として扱われます。

## 出力先

保存先ステムは `define_savename()` で決まります。

| `target` | 出力先 |
| --- | --- |
| `null` / `none` | `<out_dir>/vector/<name>.difftensors` |
| `recurrent` | `<out_dir>/recurrent/<name>` または `<name>.pth` |
| その他のモデルパスまたは ID | `<out_dir>/<target_basename>/<name>` または `<name>.pth` |

`define_savename()` には、`target == "lora"` なら `<out_dir>/lora/<name>` を返す旧式の保存名分岐があります。ただし `_resolve_target_model()` は `lora` を実行時の特別な対象として扱いません。設定の `target: lora` はモデル ID / パス `lora` としてロードされるため、現在の実行時対象としては推奨しません。

`name` がない場合は、`left` / `right` の短縮名と演算の短縮名から生成されます。短縮名は拡張子 `.safetensors` / `.pth` / `.bin` を落とし、20 文字を超える場合は先頭 8 文字と末尾 8 文字を `...` でつなぎます。`right` が空なら `none` は名前に入りません。

生成された基底名が 100 文字を超える場合は、SHA-256 の先頭 8 桁に短縮されます。保存ステムが既存アーティファクトと衝突する場合は `YYYYMMDDHHMMSS` のタイムスタンプを付け、それでも衝突する場合は `_02` 以降の接尾辞を付けます。

既存チェックはディレクトリ、`<stem>.pth`、`<stem>.difftensors`、`<stem>_layers.txt` を対象にします。通常の衝突は `define_savename()` が空きステムへタイムスタンプ / 接尾辞付きでリネームするため、同じ設定を再実行しても既存アーティファクトを上書きせず別名保存になります。保存直前にも同じ既存チェックがありますが、通常の再実行で使う分岐ではなく、`define_savename()` 後に外部プロセスなどが同じステムを作った場合に備えた防御的な競合チェックです。この再確認で存在した場合だけ `skipped_existing` になります。

## 保存されるファイル

| 条件 | 保存物 |
| --- | --- |
| `target: null` / `none` | `.difftensors` 疎ベクトルアーティファクト。トークナイザーは保存しません。 |
| 出力モデルが `DummyModel` | `<stem>.pth`。`{"config": ..., "model": state_dict}` を保存します。 |
| Hugging Face モデル | `<stem>/` ディレクトリ。`save_pretrained()` とトークナイザー保存を実行します。 |
| レシピ情報がある保存対象 | `<basename>_recipe.yaml`。そのステップの正規化済みリクエストを保存します。 |
| `--dump-layers` | `<stem>_layers.txt` とレシピ。マージは行いません。 |

`target: null` はフルモデル保存ではなく、疎ベクトル保存です。フルモデルが必要なら `target` に既存のモデルパスまたは ID を指定してください。

## `.difftensors`

Ninja Merger の `.difftensors` は safetensors メタデータ付きの疎アーティファクトです。拡張子は常に `.difftensors` です。

現行 CLI が `target: null` / `none` で通常保存するのは `artifact_type: sparse_vector` です。非ゼロテンソルだけを保存し、ゼロテンソルは省略します。全テンソルがゼロなら `__ninja_empty__` センチネルを保存します。メタデータには `artifact_type`、`format_version`、`source_reference`、存在するローカルパスの場合は `source_reference_resolved`、設定があれば `merged_config_json` が入ります。

ローダー側は `artifact_type: sparse_delta` も読めます。これは基準モデルとの差分や削除キーを持つ形式で、メタデータには `base_reference`、存在するローカルパスの場合は `base_reference_resolved`、`tensor_modes_json`、`deleted_keys_json`、設定があれば `merged_config_json` が入ります。`tensor_modes_json` の各キーは `delta` または `replace` です。

通常ロードは `diff_mode="reconstruct"` です。`sparse_vector` は保存済みテンソルだけを持つ `DummyModel` として読み込まれ、欠けているキーは疎な参照元として 0 扱いできます。`sparse_delta` は `base_reference` をロードして、`delta` は基準値に加算、`replace` は置換、`deleted_keys_json` は除外して復元します。

差分ロードは `diff_mode="delta"` です。`sparse_vector` はそのまま加算用の疎モデルになります。`sparse_delta` は `replace` または削除キーを含む場合、加算用差分として曖昧なため `ValueError` になります。

## トークナイザーの保存元

| 条件 | トークナイザー参照元 |
| --- | --- |
| 疎ベクトルアーティファクト (`target: null` / `none`) | 保存しません。 |
| `target: recurrent` | 前の対象モデル設定にある `name_or_path` / `_name_or_path`。 |
| 明示 target | `target` の値。 |

`.difftensors` をトークナイザー参照元に指定した場合、`load_tokenizer()` はアーティファクトメタデータの `base_reference_resolved` / `source_reference_resolved` / `base_reference` / `source_reference` の順で参照元を解決します。

トークナイザーの保存に失敗した場合は警告を出し、モデル保存は継続を試みます。`recurrent` では前の対象モデル設定にトークナイザー参照元情報がない場合も警告になります。

## スケーリング

`use_scaling` が有効な場合、保存前に絶対値が `1e-7` 未満の非ゼロ浮動小数テンソルを含むテンソルを検出し、`scale_tensor_inplace()` で絶対値が `1e-7` 未満の要素を `10000.0` 倍します。対象は浮動小数テンソルだけです。

ただし `.difftensors` 疎ベクトルでは、値を厳密に保つためスケーリングはスキップされます。

## レイヤー削除と設定整理

`drop_layers` 適用後、マージ確定時に `prune_config_for_dropped_layers()` が対象設定と現在の `state_dict` を見て設定を整理します。

`state_dict` に `model.audio_tower.` / `model.embed_audio.` がなければ、`audio_config`、`audio_token_id`、`boa_token_id`、`eoa_token_id`、`eoa_token_index` を除去します。

`state_dict` に `model.vision_tower.` / `model.embed_vision.` がなければ、`vision_config`、`image_token_id`、`video_token_id`、`boi_token_id`、`eoi_token_id`、`vision_soft_tokens_per_image` を除去します。

`state_dict` に `model.language_model.` があり、audio / vision がどちらもなく、設定に `text_config` がある場合は、`text_config` の値を上位へ展開して `text_config` を除去します。このとき `architectures` は `["Gemma4ForCausalLM"]` になり、`model_type` は `text_config.model_type`、なければ既存値、さらにそれもなければ `gemma4_text` になります。
