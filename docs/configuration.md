# 設定ファイル仕様

[ドキュメント目次](README.md) / [README_ja.md に戻る](../README_ja.md)

Ninja Merger の設定ファイルは YAML です。CLI の `-c` / `--config` で指定し、既定値は `model_config.yaml` です。`models` が必須で、単一のマッピングまたはマッピングのリストを受け付けます。

`load_config()` は指定パスの拡張子が `.json` でも同名の `.yaml` を読みます。設定ファイルのルートはマッピングである必要があり、`models` がない場合、または `models` がマッピング / リスト以外の場合は読み込み時にエラーになります。

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

## トップレベル

| キー | 型 | 既定値 | 内容 |
| --- | --- | --- | --- |
| `models` | mapping or list[mapping] | 必須 | マージ手順。1 件だけならリストにしなくても読み込めます。 |
| `use_scaling` | bool | `False` | 保存前に極小値をスケーリングするか。`target: null` の `.difftensors` では値保持のためスキップされます。 |
| `use_scale` | bool | `False` | `use_scaling` の互換エイリアス。 |
| `key_transformations` | mapping | `{}` | 旧形式のモデル別 key 変換。各 `model_config` にマージされます。 |
| `insert_layers` | mapping | `{}` | 旧形式のモデル別レイヤー挿入。各 `model_config` にマージされます。 |

`torch_dtype`、`skip_layernorm`、出力先、デバイス、recurrent モード、dry-run、dump-layers は YAML ではなく CLI オプションです。トップレベルに未知のキーを書いても、このローダーでは基本的に参照されません。

## `models` エントリ

| キー | 型 | 既定値 | 内容 |
| --- | --- | --- | --- |
| `name` | str | 自動生成 | 出力名。空文字、`.`、`..`、パス区切り文字は不可。同一設定内で重複不可。 |
| `target` | str/list/null | `null` | 明示的な対象。`null` / `none` は疎ベクトル出力、`recurrent` は前ステップ結果、その他はモデルパスまたは ID としてロード。 |
| `left` | str/list | `[]` | 左側モデル。通常 1 件以上。`recurrent` 参照可。`none` / `null` は空扱い。 |
| `right` | str/list | `[]` | 右側モデル。`passthrough` / `none` 以外では 1 件以上必要。`none` / `null` は空扱い。 |
| `operation` | str | `sub` | マージ演算。詳細は [operations.md](operations.md)。 |
| `velocity` | number/object | `1.0` | 全レイヤー共通の演算係数。複素数は `{real: 0.2, imag: 0.3}` 形式で指定できます。 |
| `velocities` | mapping | 未指定 | レイヤー別 `velocity`。指定時は `velocity` より優先。 |
| `post_operation` | str | `add` | 対象モデルの元値と演算結果をどう合成するか。単純置換したい場合は `none` を明示します。 |
| `post_velocity` | number | `1.0` | 全レイヤー共通の後段係数。 |
| `post_velocities` | mapping | 未指定 | レイヤー別 `post_velocity`。指定時は `post_velocity` より優先。 |
| `preprocess` | str | `none` | 前処理。現行レジストリは `none` / `pcd`。検証上は Basic 系演算のみ対応しますが、`pcd` は下記の注意点を確認してください。 |
| `post_preprocess` | str | `none` | 予約項目。現行実装では `none` のまま使います。 |
| `normalization` | str | `none` | 正規化。現行レジストリは `none` / `norm_std_mean` / `match_std_mean` / `proc_std_mean` / `angle_merge` / `quantile_match`。Basic 系演算のみ対応。 |
| `include_layers` | str/list/null | `null` | マージ対象に含めるレイヤー。 |
| `exclude_layers` | str/list/null | `null` | マージ対象から除外するレイヤー。 |
| `drop_layers` | str/list/null | `null` | 出力 `state_dict` から削除するレイヤー。 |
| `unmatch_size_layer_op` | str | `skip` | サイズ不一致時の扱い。`skip` / `only_common_range`。 |
| `use_scaling` | bool/null | トップレベル値 | エントリ単位のスケーリング上書き。 |
| `force_merge_single` | bool | `False` | 互換用の予約的な補助指定。現行の標準経路では実質的に挙動変更へ反映されません。 |
| `v2s_empty_default` | str | `v1` | 互換用の予約的な補助指定。現行の標準経路では実質的に挙動変更へ反映されません。 |
| `v2s_single_default` | str | `auto` | 互換用の予約的な補助指定。現行の標準経路では実質的に挙動変更へ反映されません。 |
| `model_config` | mapping | `{}` | モデル別の key 変換 / レイヤー挿入。 |

`qeic_*` 演算では、必要に応じて次のキーも生の設定から参照されます。

| キー | 既定値 | 内容 |
| --- | --- | --- |
| `qeic_corr_method` | `pearson` | 相関計算。`pearson` / `spearman`。 |
| `qeic_merge_method` | `average` | 相関行列の合成。`average` / `geometric_mean` / `quantum_inspired`。 |
| `qeic_alpha_mode` | `correlation` | `qeic_add` / `qeic_mix` の係数。`correlation` / `fixed`。 |
| `qeic_beta_mode` | `abs` | `qeic_sub` の係数。`abs` / `threshold`。 |
| `qeic_sub_threshold` | `-0.1` | `threshold` モードの閾値。 |

## 読み込み時の検証

読み込み時には、設定構造と演算の組み合わせが検証されます。

- `models` の各エントリはマッピングである必要があります。
- `name` は文字列のみです。空文字、空白だけ、`.`、`..`、パス区切り文字を含む値、同一設定内の重複はエラーです。
- `operation` はレジストリにある実装済み演算だけを受け付けます。`complex_mix` はレジストリにありますが、未実装扱いなのでエラーになります。
- `preprocess != none` は Basic 系演算だけで使えます。
- `normalization != none` は Basic 系演算だけで使えます。
- `post_preprocess != none` は現行実装ではどの演算でもエラーです。
- `post_operation != none` は `add` / `sub` / `mul` / `div` / `mix` / `avg` / `passthrough` / `none` / `concat` / `maxpool` / `minpool` / `geometric_mean` / `std_sub` / `angle_merge` で使えます。
- `passthrough` と `none` 以外の演算は、正規化後に 1 件以上の `right` が必要です。

`left` / `right` にマッピングなど文字列 / リスト以外の型を渡すと、現行ローダーでは空リストに正規化されます。これは便利な入力形式ではなく、実行時エラーにつながりやすいので避けてください。

`preprocess: pcd` は検証上は Basic 系演算で許可されますが、[operations.md](operations.md) にある通り、現行 `BasicMerger` の呼び出し経路では処理済みテンソルだけを渡すため、通常の `preprocess: pcd` 経路は実装と呼び出し形が噛み合っていません。

`angle_merge` でも `post_operation` は使えますが、現行実装では `post_velocity` ではなく `velocity` が `post_operation` 側へ渡ります。詳細は [operations.md](operations.md) を参照してください。

## `target` の意味

`target` は保存形式にも影響します。

| `target` | 実行時の意味 | 主な出力先 |
| --- | --- | --- |
| `null` / `none` | 対象モデルを別途ロードせず、最初の `left` を処理対象にする | `<out_dir>/vector/<name>.difftensors` |
| `recurrent` | 同一実行内の直前ステップのメモリ上のマージ結果を対象にする | `<out_dir>/recurrent/<name>` |
| モデルパスまたは ID | そのモデルを対象としてロードして更新する | `<out_dir>/<target_basename>/<name>` |

`target: null` は「フルモデル保存」ではなく疎ベクトル保存です。フルモデルを保存したい場合は、明示的な対象モデルを指定してください。

`target` は `none` / `null` を大文字小文字なしで受け付けます。リストで 1 件だけ指定した場合、`target_value_scalar` ではその 1 件が単一値として扱われます。リストで複数指定する形式は保存先の決定など一部では先頭要素を使いますが、通常の対象指定としては単一値を推奨します。

`target` の値が正確に `llava` / `vlm` / `llava-next` の場合は、`is_llava_next` フラグが立ち、レイヤー判定や skip 判定で先頭の `language_model.` が正規化されます。これはレイヤーキー正規化用のフラグであり、実行時対象の特別なエイリアスではありません。対象モデル自体は通常の `target` 指定と同じロードパスで読み込まれます。

## `left` / `right`

`left` と `right` は文字列またはリストで書けます。

```yaml
left: path/to/base-model
right:
  - path/to/tuned-a
  - path/to/tuned-b
```

複数の `left` と `right` を指定した Basic 系演算では、全ペアの処理結果を平均してから対象レイヤーに反映します。

`passthrough` と `none` は `right` なしで動作します。その他の演算では `right` が必要です。

`left` / `right` の文字列 `none` / `null` は空扱いです。リスト内に `none` / `null` が混ざっている場合も、その要素だけ除外されます。`recurrent` または `recurrent...` で始まる値は同一実行内の直前マージ結果をメモリ上で参照しますが、前ステップがない状態で使うとエラーになります。

## `post_operation` の注意

実装上の既定値は `post_operation: add` です。つまり、演算結果を対象モデルの元レイヤーに加算します。

単純に `operation` の結果を出力したい場合は、次のように `post_operation: none` を明示してください。

```yaml
models:
  - name: direct-mix
    target: path/to/target-model
    left: path/to/base-model
    right: path/to/tuned-model
    operation: mix
    velocity: 0.5
    post_operation: none
```

疎ベクトルを対象モデルに足し戻す場合は、`operation: none` と `post_operation: add` の組み合わせが使えます。

```yaml
models:
  - name: apply-vector
    target: path/to/target-model
    left: merged_models/vector/delta-vector.difftensors
    right: none
    operation: none
    post_operation: add
```

## レイヤーフィルタ

`include_layers`、`exclude_layers`、`drop_layers` は、文字列またはリストを受け付けます。

```yaml
include_layers: "model.layers.0-8,model.norm,lm_head"
exclude_layers:
  - model.layers.3.mlp
  - model.layers.4.self_attn
drop_layers: "model.audio_tower. model.vision_tower."
```

指定方法:

- `model.layers.2-5` または `2-5` はレイヤー番号の範囲です。
- `5-2` のように逆順で書いた範囲は `2-5` として扱われます。
- それ以外は部分文字列としてレイヤー名に照合。
- 文字列はカンマ区切りを優先します。カンマがない場合、`model.audio_tower. model.vision_tower.` のような空白区切りの複数キーも受け付けます。
- `include_layers` がある場合、範囲や部分文字列に合わないレイヤーは除外。
- 特定文字列の取り込み指定は特定文字列の除外指定より先に判定されるため、同じレイヤーが両方に一致すると取り込み指定が優先されます。
- 範囲指定の取り込み指定と除外指定の両方に一致する `model.layers.N` は除外されます。
- `drop_layers` はマージ対象から外すだけでなく、保存される `state_dict` からも削除。
- `drop_layers` は取り込み / 除外 / スキップより先に判定されます。
- 同じストレージを共有する対象テンソルは、最初のキーだけをマージ対象にし、後続エイリアスはスキップされます。

CLI の `--include-layers` / `--exclude-layers` を指定した場合は、設定ファイル側の値より優先されます。

`drop_layers` が適用された場合、マルチモーダル用の audio/vision キーが残っていなければ、保存前に設定から対応する audio/vision 設定も削除されます。テキスト部分だけが残る Gemma 系では `text_config` の値を上位へ移し、`architectures` / `model_type` もテキストモデル向けに更新されます。

## レイヤー別 velocity

`velocities` と `post_velocities` は同じ形式です。数値を置くと接頭辞マッチ、`type: regex` を置くと正規表現マッチになります。

```yaml
velocities:
  "model.embed_tokens.": 0.1
  "model.layers.(\\d+).":
    type: regex
    value: "0.3 + 0.01 * int(match.group(1))"
  "DEFAULT": 0.5
```

正規表現の `value` が文字列の場合、制限付きの式として評価されます。利用できるものは、数値、四則演算、`int`、`float`、`complex`、`abs`、`min`、`max`、`round`、`match.group(...)`、`layer_name` です。

補足:

- `velocities` が指定されている場合は、最初の `left` モデルの `state_dict` キーを使ってレイヤーごとの辞書に展開され、`velocity` より優先されます。
- `post_velocities` が指定されている場合も同様に `post_velocity` より優先されます。
- `DEFAULT` がない場合のフォールバックは `1.0` です。
- パターンは YAML の順序で評価され、最初に一致したものが使われます。
- 数値の値を持つパターンは `layer_name.startswith(pattern)` で判定されます。
- 正規表現の辞書に `regex` キーがない場合は、マッピングのキー自体を正規表現として使います。
- 正規表現式の評価に失敗した場合、そのレイヤーは `DEFAULT` または `1.0` になります。
- 全体の `velocity` だけは `{real: 0.2, imag: 0.3}` 形式を複素数テンソルに正規化します。`velocities` / `post_velocities` の各レイヤー値ではこの `{real, imag}` 形式は正規化されません。

## `model_config`

`model_config` は、ロード後の `state_dict` に対してモデル別の変換を行います。

```yaml
models:
  - name: transformed-merge
    target: path/to/target-model
    left: path/to/base-model
    right: path/to/tuned-model
    operation: add
    post_operation: none
    model_config:
      path/to/base-model:
        key_transformations:
          - type: replace
            replace:
              "transformer.": "model."
          - type: regex
            regex: "^blocks\\.(\\d+)\\.(.*)"
            replace: "model.layers.\\1.\\2"
        insert_layers:
          - source_model: path/to/extra.safetensors
            source_key: "extra.weight"
            target_key: "model.layers.10.extra.weight"
```

`key_transformations`:

| `type` | 必須キー | 内容 |
| --- | --- | --- |
| `replace` | `replace` mapping | 文字列置換を順に適用。 |
| `regex` | `regex`, `replace` | 正規表現置換。`\1` などのキャプチャ参照可。 |

`insert_layers`:

| キー | 内容 |
| --- | --- |
| `source_model` | 挿入元モデル。 |
| `source_key` | 挿入元 `state_dict` key。 |
| `target_key` | 挿入先 key。既存 key は上書き。 |
| `device` | 任意。挿入元モデルのロード先デバイス。省略時は `cpu`。 |
| `torch_dtype` | 任意。挿入元モデルのロード dtype。省略時は `torch.float32`。 |

トップレベル互換形式を使う場合は、最終的に `model_config[model_name]` に入る形で書きます。

```yaml
key_transformations:
  path/to/base-model:
    key_transformations:
      - type: replace
        replace:
          "old.": "new."
```

互換形式のマージは `left` / `right` / `target` に実際に現れるモデル名ごとに行われます。`recurrent` は対象外です。トップレベルの `key_transformations` と `insert_layers` は浅いマージになり、同じモデルの既存 `model_config` に対して後から `update()` されるため、同じキーがある場合はトップレベル互換形式の値が優先されます。

## 例: recurrent チェーン

```yaml
models:
  - name: stage1-vector
    target: null
    left: path/to/base-model
    right: path/to/tuned-mid
    operation: sub
    post_operation: none

  - name: stage2-full
    target: path/to/base-model
    left: recurrent
    right: path/to/tuned-late
    operation: mix
    velocity: 0.4
    post_operation: add
```

`left: recurrent` は、上の `stage1-vector` が保存した `.difftensors` を読み直す指定ではなく、同じ実行中に保持されている前ステップのメモリ上の結果を使う指定です。別実行で再開する場合は、`left: merged_models/vector/stage1-vector.difftensors` のように保存済みアーティファクトのパスを明示してください。

`--no-recurrent-mode` を使うと、前ステップ結果は次ステップへ渡されません。
