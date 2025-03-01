
# `config.yaml` ドキュメント

このドキュメントでは、Ninja-merger の設定ファイルである `config.yaml` の書き方について詳細に説明します。

**注意:** このドキュメントは、バージョン2.0 (最新版) に基づいています。古いバージョンでは、一部の設定が異なる場合があります。

## 概要

`config.yaml` ファイルは、モデルマージツールの動作を制御するための設定を記述するファイルです。YAML 形式で記述します。

YAML は、人間にとって読みやすく、書きやすいように設計されたデータシリアル化形式です。基本的な文法は以下の通りです。

*   **キーと値のペア:**  `key: value` の形式でデータを記述します。
*   **インデント:**  インデント (スペース2つまたは4つ) を使用して、データの階層構造を表します。 **重要:** インデントが正しくないと、設定ファイルが正しく読み込まれません。
*   **リスト:**  `-` の後にスペースを入れ、要素を記述します。
*   **コメント:**  `#` の後にコメントを記述します。

例：

```yaml
# これはコメントです
key1: value1  # キーと値のペア
key2:
  - item1  # リスト
  - item2
  - key3: value3 # ネストされたキーと値
```

`config.yaml` ファイルは、主に以下の3つのセクションで構成されます。

1.  **`models` セクション:** マージするモデル、マージ方法、`velocity` などを指定します。
2.  **`model_config` セクション:**  キー変換 (key_transformations) やレイヤー挿入 (insert_layers) など、個々のモデルに適用する詳細設定
3.  **グローバル設定:**  `torch_dtype` など、ツール全体の動作に影響する設定を指定します。

## `models` セクション

`models` セクションは、リスト形式で、複数のマージ設定を記述できます。各マージ設定は、以下のキーを持つ辞書形式で記述します。

| キー               | 説明                                                                                                                                                                                                                                                                                                             | 型                                                                   | 必須 | デフォルト値 | 例                                                                                                                                                                                                                                                             |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------- | :--: | :----------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`             | (オプション) マージ設定の名前。ファイル名の一部として使用されます。指定しない場合は、モデル名と操作名から自動生成されます。                                                                                                                                                                                                    | str                                                                  |  ×  |              | `my_merge_config`                                                                                                                                                                                                                                               |
| `left`             | マージするモデルのリスト (左側)。少なくとも1つのモデルを指定する必要があります。                                                                                                                                                                                                                             | list[str]                                                            |  ○  |              | `["model1.safetensors", "model2.safetensors"]`                                                                                                                                                                                                               |
| `right`            | マージするモデルのリスト (右側)。少なくとも1つのモデルを指定する必要があります。                                                                                                                                                                                                                             | list[str]                                                            |  ○  |              | `["model3.safetensors"]`                                                                                                                                                                                                                                  |
| `target`           | マージ結果の出力先。`null` (新しいモデルを作成)、`"recurrent"` (前回のマージ結果を上書き)、または既存のモデルのパスを指定できます。                                                                                                                                                                                 | str or null                                                          |  ○  |              | `null`, `"recurrent"`, `"model4.safetensors"`                                                                                                                                                                                                            |
| `operation`        | マージ方法を指定します。利用可能な値は以下の表を参照してください。                                                                                                                                                                                                                                         | str                                                                  |  ○  |              | `add`, `sub`, `complex_angle_merge`                                                                                                                                                                                                                   |
| `velocities`       | (オプション) レイヤーごとの `velocity` を指定します。辞書形式で、キーにレイヤー名 (またはパターン)、値に `velocity` を指定します。`DEFAULT` キーでデフォルト値を指定できます。                                                                                                                                     | dict[str, float or complex or dict]                                |  ×  | `{}`         | `{"model.layers.0.": 0.1, "model.layers.1.": 0.2, "DEFAULT": 0.5, "model.layers.(\d+).": {"type": "regex", "value": "0.05 * int(match.group(1))"}}`                                                                                                    |
| `post_velocities`  | (オプション) レイヤーごとの `post_velocity` を指定します。`velocities` と同様の形式で指定します。                                                                                                                                                                                                                       | dict[str, float or complex or dict]                                |  ×  | `{}`         | `{"model.layers.0.": 1.1, "model.layers.1.": 1.2, "DEFAULT": 1.0, "model.layers.(\d+).": {"type": "regex", "value": "1.0 + 0.01 * int(match.group(1))"}}`                                                                                                |
| `velocity`         | (オプション, `velocities` が指定されていない場合) モデル全体に適用する `velocity` を指定します。                                                                                                                                                                                                             | float or complex                                                     | ×   | `1.0`        | `0.5`, `0.2+0.3j`                                                                                                                                                                                                                                            |
| `post_velocity`    | (オプション, `post_velocities` が指定されていない場合) モデル全体に適用する `post_velocity` を指定します。                                                                                                                                          | float                                                              | ×   |  `1.0`       |  `2.0`                                                                                                                                   |
| `post_operation`   | (オプション) マージ後に適用する演算 (後処理) を指定します。利用可能な値は以下の表を参照。                                                                                                                                              | str                                              | ×   | `"add"`      | `"add"`, `"subfrom"`, `"mul"`                                                                                                                                              |
| `model_config`     | (オプション) 各モデルに適用する詳細設定 (キー変換、レイヤー挿入) を指定します。辞書形式で、キーにモデル名 (パス)、値に設定を記述します。詳細は後述の `model_config` セクションを参照してください。                                                                                                         | dict[str, dict]                                                       |  ×  | `{}`         | (後述の `model_config` セクションを参照)                                                                                                                                                                                                                       |
| `preprocess`    | 適用する前処理                                     | str                                                                                               | ×   |  `"none"`    | `"none"`, `"pcd"`                                                                                                                                       |
| `post_preprocess`    | 適用する後処理                                     | str                                                                                               | ×   |  `"none"`    | `"none"`                                                                                                                                       |
| `normalization`   | 正規化                                      | str                                                                                           | ×    | `"none"`     | `"none"`, `"norm_std_mean"`                                                                                                                                    |
| `include_layers` | (オプション) マージに含めるレイヤーを指定します。カンマ区切りの文字列、またはリストで指定します。                                                                          | str or list[str]                                  | ×   |  `None`     | `"model.layers.0,model.layers.1"`, `["model.layers.0", "model.layers.1"]`                                                                |
| `exclude_layers` | (オプション) マージから除外するレイヤーを指定します。`include_layers` と同様の形式で指定します。                                                                      | str or list[str]                                   | ×   | `None`      | `"model.layers.2,model.layers.3"`                                                                                                                       |
| `drop_layers`    | (オプション) マージから削除するレイヤーを指定します。`include_layers` と同様の形式で指定します。                                                                    | str or list[str]                                  | ×    | `None`      | `"model.layers.4"`                                                                                                                                  |
| `unmatch_size_layer_op` | レイヤーサイズが一致しない場合の挙動を指定します。                                                                                 | str                                              | ×    | `skip`      |  `skip`, `only_common_range`                                                                                                                            |
| `force_merge_single`   |  `angle_merge`でvelocityが設定されていない時,１つのsub_modelのみとマージを行う際に、エラーを出すかどうか                                   | bool                                              | ×     | `False`     | `True`, `False`                                                                                                                                |
| `v2s_empty_default`    | `angle_merge`でvelocityが設定されていない時、sub_modelsが空の場合に使用するデフォルト値                                         | str                                                | ×     |  `v1`      | `"v1"`, `"zero"`                                                                                                                     |
| `v2s_single_default`   | `angle_merge`でvelocityが設定されていない時、sub_modelsが１つの場合に使用するデフォルト値                                         | str                                                | ×    | `"auto"`     | `"auto"`, `"v1"`, `"zero"`                                                                                                                 |

**利用可能な `operation`:**

| `operation`           | 説明                                                                                   |
| :---------------------- | :------------------------------------------------------------------------------------- |
| `add`                  | 加算 (`left` + `right`)                                                                 |
| `sub`                  | 減算 (`left` - `right`)                                                                 |
| `mul`                  | 乗算 (`left` * `right`)                                                                 |
| `div`                  | 除算 (`left` / `right`)                                                                 |
| `mix`                  | 線形補間 (`left` * (1 - `velocity`) + `right` * `velocity`)                                |
| `avg`                  | 平均 (`(left` + `right`) / 2)                                                           |
| `concat`               | 結合 (次元を増やす)                                                                   |
| `maxpool`              | 最大値プーリング                                                                       |
| `minpool`              | 最小値プーリング                                                                       |
| `geometric_mean`       | 幾何平均                                                                             |
| `std_sub`        | `left`から`right`の標準偏差を引く                                 |
| `widen`                | (特殊) モデルの幅を広げる                                                                 |
| `complexadd`           | 複素数加算                                                                             |
| `angle_merge`           | 角度に基づくマージ                                                                      |
| `complex_angle_merge` | 複素数角度に基づくマージ                                                                  |
| `qeic_add`             | QEIC (Quantized Embedding Information Correlation) を使用した加算                       |
| `qeic_mix`             | QEIC を使用した線形補間                                                                 |
| `qeic_sub`             | QEIC を使用した減算                                                                   |

**利用可能な `post_operation`:**

| `post_operation`    | 説明                                          |
| :------------------ | :-------------------------------------------- |
| `add`              | 加算 (`tensor` + `post_velocity` * `original`)  |
| `sub`              | 減算 (`tensor` - `post_velocity` * `original`)   |
| `subfrom`          | 減算 (`post_velocity` * `original` - `tensor`) |
| `mul`              | 乗算 (`tensor` * `post_velocity`)             |
| `div`              | 除算 (`tensor` / `post_velocity`)             |
| `divby`            | 除算 (`post_velocity` / `tensor`)             |
| `mix`              | 線形補間                                  |
|`concat`             | 結合                                          |
|`maxpool`            | 最大プーリング                               |
|`minpool`            | 最小プーリング                               |
|`geometric_mean`     | 幾何平均                                      |
|`angle`     | 角度                                      |

## `model_config` セクション

`model_config` セクションでは、個々のモデルに適用する詳細設定 (キー変換、レイヤー挿入) を指定します。

`model_config` は辞書形式で、キーにモデル名 (パス)、値にそのモデルに適用する設定を記述します。

```yaml
model_config:
  model1.safetensors:  # モデル名 (パス)
    key_transformations:  # キー変換
      # ...
    insert_layers:  # レイヤー挿入
      # ...
  model2.safetensors:
    # ...
```

### `key_transformations`

`key_transformations` は、モデルの `state_dict` のキーを変換するためのルールを指定します。辞書形式で、キーにモデル名、値に変換ルールのリストを指定します。

各変換ルールは辞書形式で、以下のキーを持ちます。

| キー      | 説明                                                                                                               | 型      | 必須 |
| :-------- | :----------------------------------------------------------------------------------------------------------------- | :------ | :--: |
| `type`    | 変換の種類。`"replace"` (単純な文字列置換) または `"regex"` (正規表現による置換) を指定します。                           | str     |  ○  |
| `replace` | (`type` が `"replace"` の場合) 置換前後の文字列をキーと値のペアで指定します。                                                    | dict    |  ○  |
| `regex`   | (`type` が `"regex"` の場合) 置換前のキーにマッチする正規表現パターンを指定します。                                            | str     |  ○  |
| `replace` | (`type` が `"regex"` の場合) 置換後のキーを指定します。`\1`, `\2` などで正規表現のキャプチャグループを参照できます。 | str     |  ○  |

例:

```yaml
key_transformations:
  model1.safetensors:
    - type: replace
      replace:
        "old_key_part": "new_key_part"
        "another_old_part": "another_new_part"
    - type: regex
      regex: "^old_prefix\.(.*)"
      replace: "new_prefix.\\1"
```

### `insert_layers`

`insert_layers` は、他のモデルからレイヤーを挿入するためのルールを指定します。辞書形式で、キーにモデル名(挿入先)、値に挿入ルールのリストを指定します。

各挿入ルールは辞書形式で、以下のキーを持ちます。

| キー           | 説明                                                                                                           | 型      | 必須 |
| :------------- | :------------------------------------------------------------------------------------------------------------- | :------ | :--: |
| `source_model` | 挿入元のモデルのパス。                                                                                          | str     |  ○  |
| `source_key`   | 挿入元のモデルから取得するテンソルのキー。                                                                         | str     |  ○  |
| `target_key`   | 挿入先のモデルでのテンソルのキー (このキーにテンソルが上書きされる)。                                                   | str     |  ○  |

例:

```yaml
insert_layers:
  model1.safetensors:
    - source_model: model2.safetensors
      source_key: "model.layers.2.mlp.gate_proj.weight"
      target_key: "model.layers.3.mlp.gate_proj.weight"
```

## グローバル設定

`config.yaml` ファイルのトップレベルには、以下のグローバル設定を記述できます (これらは `models` リストの外側に記述します)。

| キー              | 説明                                                                        | 型          | デフォルト値 | 例            |
| :---------------- | :-------------------------------------------------------------------------- | :---------- | :----------- | :------------ |
| `torch_dtype`     | モデルの読み込み/保存に使用するデータ型を指定します。                           | str         | `bfloat16`   | `float32`, `bfloat16` |
| `skip_layernorm`  | `True` に設定すると、レイヤー正規化層をマージから除外します。               | bool        | `False`      | `True`, `False` |

## 例

```yaml
# 例1: 単純なモデルのマージ
models:
  - left: [model1.safetensors]
    right: [model2.safetensors]
    operation: add

# 例2: velocity を指定した線形補間
models:
  - left: [model1.safetensors]
    right: [model2.safetensors]
    operation: mix
    velocity: 0.3

# 例3: レイヤーごとの velocity を指定
models:
  - left: [model1.safetensors]
    right: [model2.safetensors]
    operation: add
    velocities:
      "model.layers.0.": 0.1
      "model.layers.1.": 0.2
      "model.layers.(\d+).":  # 正規表現
        type: regex
        value: "0.05 * int(match.group(1))"  # レイヤー番号に応じて増加
      "DEFAULT": 0.5  # デフォルト値

# 例4: キー変換とレイヤー挿入
models:
  - left: [model1.safetensors]
    right: [model2.safetensors]
    operation: add
    model_config:
      model1.safetensors:
        key_transformations:
          - type: replace
            replace:
              "transformer.": "model."
          - type: regex
            regex: "^blocks\.(\d+)\.res\.(.*)"
            replace: "layers.\\1.residual.\\2"
        insert_layers:
          - source_model: model3.safetensors
            source_key: "extra_layer.weight"
            target_key: "model.layers.10.extra_layer.weight"

# 例5: 複数のマージ設定
models:
  - name: merge1
    left: [model1.safetensors]
    right: [model2.safetensors]
    operation: add
  - name: merge2
    left: [model3.safetensors]
    right: [model4.safetensors]
    operation: sub
    velocities:
      "model.layers.0.": 0.8
      "DEFAULT": 0.2
```

```yaml
# グローバル設定
torch_dtype: float32
skip_layernorm: true

models:
  - name: complex_merge_with_everything # 名前を設定
    left:
      - very_long_model_name_left_1.safetensors
      - very_long_model_name_left_2.safetensors
    right:
      - very_long_model_name_right_1.safetensors
    target: null  # 新しいモデルを作成
    operation: complex_angle_merge # 複雑なマージ
    velocities:  # レイヤーごとの velocity
      "model.embed_tokens.": 0.1     # embedding 層
      "model.layers.0.": 0.2          # 最初の layer
      "model.layers.1.": 0.3
      "model.layers.(\d+).":         # 正規表現で layer を指定
        type: regex
        value: "0.4 + 0.01 * int(match.group(1))"  # layer 番号に応じて増加
      "model.layers.(1[0-9]|2[0-3]).": # 10-23層
         type: regex
         value: 0.9
      "model.norm.": 0.7             # 最後の normalization
      "lm_head.": 0.8                # 言語モデルの head
      "DEFAULT": 0.5                 # デフォルト
    post_velocities:  # レイヤーごとの post_velocity
      "model.embed_tokens.": 1.5
      "model.layers.0.": 1.2
      "model.layers.1.": 1.0
      "model.layers.(\d+).":
        type: regex
        value: "1.0 - 0.005 * int(match.group(1))"
      "model.norm.": 0.9
      "lm_head.": 1.1
      "DEFAULT": 1.0  # デフォルト
    post_operation: subfrom       # 後処理
    preprocess: none            # 前処理
    post_preprocess: none       # 後処理
    normalization: norm_std_mean   # 正規化
    include_layers: model.layers.0,model.layers.1,model.layers.2-5,model.norm # 含めるレイヤー
    exclude_layers: model.layers.3.mlp,model.layers.4.attn # 除外
    drop_layers: model.layers.24 # ドロップ
    unmatch_size_layer_op: only_common_range # サイズ不一致の場合は共通部分
    force_merge_single: True
    v2s_empty_default: zero
    v2s_single_default: auto
    model_config:  # 各モデルに適用する設定
      very_long_model_name_left_1.safetensors:
        key_transformations: # キー変換
          - type: replace
            replace:
              "old_prefix.": "new_prefix."
              "another_old.": "another_new."
          - type: regex
            regex: "^transformer\.(.*)"
            replace: "model.\\1"
        insert_layers:
          - source_model: extra_layers_model.safetensors
            source_key: "extra_layer.0.weight"
            target_key: "model.layers.10.extra_layer.weight"
      very_long_model_name_left_2.safetensors:
        key_transformations: # キー変換
          - type: regex
            regex: "^left2\.(.*)"
            replace: "left_two.\\1"
      very_long_model_name_right_1.safetensors:
        insert_layers:
          - source_model: another_extra_layers_model.safetensors
            source_key: "another_extra_layer.0.weight"
            target_key: "model.layers.11.another_extra.weight"

```

**解説:**

*   **グローバル設定:**
    *   `torch_dtype`: `float32` (デフォルトの `bfloat16` から変更)
    *   `skip_layernorm`: `true` (レイヤー正規化層を除外)

*   **`models` セクション:**
    *   `name`: `complex_merge_with_everything` (このマージ設定に名前を付ける)
    *   `left`: 2つのモデルを指定 (長いモデル名)
    *   `right`: 1つのモデルを指定 (長いモデル名)
    *   `target`: `null` (新しいモデルを作成)
    *   `operation`: `complex_angle_merge` (複雑なマージ方法)
    *   `velocities`: レイヤーごとに細かく `velocity` を設定
        *   単純な文字列マッチ (`model.embed_tokens.`, `model.layers.0.`, ...)
        *   正規表現 (`model.layers.(\d+).`) と計算 (`0.4 + 0.01 * int(match.group(1))`)
        *   特定の範囲(`model.layers.(1[0-9]|2[0-3]).`)
        *   デフォルト値 (`DEFAULT`)
    *   `post_velocities`: レイヤーごとに細かく `post_velocity` を設定 (velocitiesとほぼ同様)
    *   `post_operation`: `subfrom`
    *   `preprocess`, `post_preprocess`: `none`
    *   `normalization`: `norm_std_mean`
    *  `include_layers`: 一部のレイヤーを含める
    *  `exclude_layers`: 一部のレイヤーを除外
    *  `drop_layers`: 特定のレイヤーを削除
    * `unmatch_size_layer_op`: `only_common_range` (サイズが一致しない場合は、共通部分のみを使用)
    *   `model_config`:
        *   `very_long_model_name_left_1.safetensors`:
            *   `key_transformations`:
                *   単純な置換 (`replace`) と正規表現 (`regex`) を両方使用
            *   `insert_layers`:
                *   `extra_layers_model.safetensors` から `extra_layer.0.weight` を挿入
        *   `very_long_model_name_left_2.safetensors`:
            * `key_transformations`:
                * 正規表現を使用
        * `very_long_model_name_right_1.safetensors`:
           *   `insert_layers`:
                *   `another_extra_layers_model.safetensors` から `another_extra_layer.0.weight` を挿入

