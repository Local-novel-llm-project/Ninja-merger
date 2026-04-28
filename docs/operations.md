# 演算仕様

[ドキュメント目次](README.md) / [README_ja.md に戻る](../README_ja.md)

演算は `modules/Utils/operation_dicts.py` のレジストリで管理されています。設定ファイルの `operation` によって `BasicMerger`、`ComplexMerger`、`CustomMerger`、`QeicMerger` のいずれかが選ばれます。

レジストリは演算ごとに次の対応フラグを持ちます。

| フラグ | 内容 |
| --- | --- |
| `implemented` | `False` の演算はレジストリに名前だけ残っていますが、検証で拒否されます。 |
| `supports_preprocess` | `preprocess != none` を許可するか。 |
| `supports_post_preprocess` | `post_preprocess != none` を許可するか。現行レジストリでは全演算 `False` です。 |
| `supports_normalization` | `normalization != none` を許可するか。 |
| `supports_post_operation` | `post_operation != none` を許可するか。 |
| `requires_sub_models` | `right` / サブモデルが 1 つ以上必要か。`passthrough` と `none` だけ `False` です。 |

現行レジストリの演算対応状況は次の通りです。

| `operation` | マージャー | 実装済み | 前処理 | 正規化 | 後段演算 | サブモデル |
| --- | --- | --- | --- | --- | --- | --- |
| `add` / `sub` / `mul` / `div` / `mix` / `avg` / `concat` / `maxpool` / `minpool` / `geometric_mean` / `std_sub` | `BasicMerger` | yes | yes | yes | yes | required |
| `passthrough` / `none` | `BasicMerger` | yes | yes | yes | yes | optional |
| `widen` | `CustomMerger` | yes | no | no | no | required |
| `complexadd` / `complex_angle_merge` | `ComplexMerger` | yes | no | no | no | required |
| `angle_merge` | `ComplexMerger` | yes | no | no | yes | required |
| `qeic_add` / `qeic_mix` / `qeic_sub` | `QeicMerger` | yes | no | no | no | required |
| `complex_mix` | `ComplexMerger` | no | no | no | no | required |

`post_preprocess` は全演算で非対応です。

## 基本系演算

基本系は全て `preprocess`、`normalization`、`post_operation` に対応します。`passthrough` と `none` 以外はサブモデルが必要です。

| `operation` | 式の概要 | 備考 |
| --- | --- | --- |
| `add` | `(left + right) * velocity` |  |
| `sub` | `(left - right) * velocity` |  |
| `mul` | `(left * right) * velocity` |  |
| `div` | `(left / right) * velocity` | 0 除算には注意。 |
| `mix` | `left * (1 - velocity) + right * velocity` | 補間用途。 |
| `avg` | `(left + right) * 0.5` | `velocity` は互換のため受け取るだけです。 |
| `passthrough` | `left.detach().clone()` | `right` 不要。フィルター / 削除だけしたい場合に使えます。 |
| `none` | `left.detach().clone()` | `passthrough` と同じ実装。`right` 不要。疎ベクトル適用などで使います。 |
| `concat` | `torch.cat((left, right), dim=0)` | 形状が変わるため通常のコピー経路では扱いに注意。 |
| `maxpool` | `max(left, right)` |  |
| `minpool` | `min(left, right)` |  |
| `geometric_mean` | `sqrt(left * right)` | 負値を含むテンソルでは結果に注意。 |
| `std_sub` | `(((left - mean(left)) / max(std(left), 1e-7)) - ((right - mean(right)) / max(std(right), 1e-7))) * velocity` | `torch.std_mean` を使います。 |

複数の `left` / `right` がある場合、基本系は全ペアを処理し、結果を平均してから次段へ渡します。

## 複素数 / 角度系演算

| `operation` | マージャー | 対応フラグ | 備考 |
| --- | --- | --- | --- |
| `complexadd` | `ComplexMerger` | 補助指定非対応、サブモデル必須 | `target + (avg(subs) - target) * 0.1 * velocity`。 |
| `angle_merge` | `ComplexMerger` | `post_operation` のみ対応、サブモデル必須 | サブモデル間の角度から係数を作り、`post_operation` も利用できます。 |
| `complex_angle_merge` | `ComplexMerger` | 補助指定非対応、サブモデル必須 | 複素数係数を使った角度マージ。 |

`complex_mix` はレジストリ上に残っていますが、`implemented=False` なので設定すると検証で拒否されます。

`angle_merge` は `post_operation` を受け付けますが、内部実装では `post_velocity` ではなく `velocity` を `post_operation` 関数へ渡します。基本系演算では通常どおり `post_velocity` を使います。`post_operation: none` は辞書未ヒット時の何もしない処理として扱われ、角度係数で重み付けされたサブモデル平均側だけがコピーされます。

`angle_merge` の実数テンソルでの係数は概ね次の形です。

```text
cos_ij = dot(sub_i, sub_j) / clamp(norm(sub_i) * norm(sub_j), min=1e-7)
theta = mean(cos_ij).unsqueeze(-1)
t = n * cos(theta) / (1 + (n - 1) * cos(theta))
avg = sum(subs) / n
processed = post_operation(target * (1 - t), avg * t, velocity)
```

`complex_angle_merge` は複素数化した係数を使います。`complex_mix` 側の式は現状テストの安定参照値からは除外されています。

## カスタム系演算

| `operation` | マージャー | 備考 |
| --- | --- | --- |
| `widen` | `CustomMerger` | Widen 風の重み統合。全補助指定が非対応です。`velocity` は WIDEN の閾値 `t` として渡され、`s` は実装上 `1.0` 固定です。 |

## QEIC 系演算

| `operation` | 概要 |
| --- | --- |
| `qeic_add` | 相関行列に基づいて `(base + sub)` 系の重み付け加算を行います。 |
| `qeic_mix` | 相関行列に基づいて `base` / `sub` を混合します。 |
| `qeic_sub` | 負の相関を考慮して `base - sub` 方向を反映します。 |

QEIC 系はレジストリ上、`preprocess`、`post_preprocess`、`normalization`、`post_operation` の全てに非対応です。サブモデルは必須です。

QEIC は対象レイヤーの forward hook から相関行列を作ります。モデルの forward が成立する必要があり、`config.vocab_size` を使った `(1, 64)` のダミー入力を投げる実装です。対象レイヤー名が `.weight` / `.bias` で終わる場合は、その直前までをモジュール名として hook します。

QEIC 追加キー:

| キー | 既定値 | 値 |
| --- | --- | --- |
| `qeic_corr_method` | `pearson` | `pearson` / `spearman` |
| `qeic_merge_method` | `average` | `average` / `geometric_mean` / `quantum_inspired` |
| `qeic_alpha_mode` | `correlation` | `correlation` / `fixed` |
| `qeic_beta_mode` | `abs` | `abs` / `threshold` |
| `qeic_sub_threshold` | `-0.1` | 数値 |

QEIC の内部式は、最初に base/sub の相関行列を `qeic_merge_method` で統合し、最初の統合済み相関行列を使います。相関行列の先頭次元が対象テンソルの先頭次元と一致しない場合、そのレイヤーは実質的に何もしない処理になります。

```text
add:
  alpha = clamp(corr, 0, 1)                  # alpha_mode=correlation
  alpha = 0.5                                # alpha_mode=fixed
  merged = (1 - alpha) * target + alpha * (base + sub)

mix:
  alpha = clamp(corr, 0, 1) * velocity       # alpha_mode=correlation
  alpha = 0.5                                # alpha_mode=fixed
  merged = (1 - alpha) * base + alpha * sub

sub:
  alpha = -abs(corr)                         # beta_mode=abs
  alpha = corr < threshold ? -abs(corr) : 0  # beta_mode=threshold
  merged = target + alpha * (base - sub)
```

複数の base/sub モデルがある場合、相関行列は全ペア分を集めますが、重み本体の式には `base_slices[0]` と `sub_slices[0]` だけが渡されます。

## `post_operation`

`post_operation` は「対象の元テンソル」と「`operation` の処理結果」を合成します。`none` は `POST_OPERATION_REGISTRY` には登録されていませんが、辞書未ヒット時の何もしない処理として扱われ、処理結果をそのまま対象へコピーします。

| `post_operation` | 式の概要 |
| --- | --- |
| `none` | `processed` |
| `add` | `target + processed * post_velocity` |
| `sub` | `target - processed * post_velocity` |
| `subfrom` | `processed - target * post_velocity` |
| `mul` | `target * processed * post_velocity` |
| `div` | `target / processed * post_velocity` |
| `divby` | `processed * post_velocity / target` |
| `mix` | `target * (1 - post_velocity) + processed * post_velocity` |
| `concat` | `torch.cat((target, processed * post_velocity), dim=0)` |
| `maxpool` | `max(target, processed * post_velocity)` |
| `minpool` | `min(target, processed * post_velocity)` |
| `geometric_mean` | `sqrt(target * processed * post_velocity)` |
| `angle` | `acos(clamp(dot(target, processed) / (norm(target) * norm(processed)), -1, 1))` |

既定値は `add` です。単純なマージ結果にしたい場合は `post_operation: none` を明示してください。`concat` は shape を変え、`angle` は最終次元を reduce するため、通常のコピー経路では扱いに注意が必要です。

## 前処理と正規化

| 種類 | 値 | 内容 |
| --- | --- | --- |
| `preprocess` | `none` | 何もしない。 |
| `preprocess` | `pcd` | Git Re-Basin 風にサブテンソルの並びを合わせる関数です。ただし現行 `BasicMerger` は処理済みテンソルだけを渡すため、通常の `preprocess: pcd` 経路は実装と呼び出し形が噛み合っていません。 |
| `normalization` | `none` | 何もしない。 |
| `normalization` | `norm_std_mean` | 処理結果を平均 0 / 標準偏差 1 に正規化。 |
| `normalization` | `match_std_mean` | `(processed - (mean(processed) - mean(target))) * (std(processed) / std(target))`。名前から想像する「target の標準偏差へ合わせる」式とは逆比率なので注意。 |
| `normalization` | `proc_std_mean` | 標準化空間で処理して対象分布へ戻します。ラッパー経由の通常パスでは既に処理済みテンソルを受け取るため、テスト上は `processed * max(std(target), 1e-7) + mean(target)` と同等です。 |
| `normalization` | `angle_merge` | サブモデル間の角度から混合係数を作ります。 |
| `normalization` | `quantile_match` | 分位点マッチングで分布を寄せます。 |

`post_preprocess` は現行実装では予約項目で、`none` 以外を指定すると検証で拒否されます。

`normalization` はレジストリ上、基本系演算にだけ許可されています。`quantile_match` は順位 / 分位点補間を行うため、生成参照テストでは専用の参照値が未実装の例外として扱われています。

## 形状不一致

`unmatch_size_layer_op` は次の 2 種類です。

| 値 | 内容 |
| --- | --- |
| `skip` | 形状が合わないレイヤーをスキップ。 |
| `only_common_range` | テンソルの共通範囲だけをスライスして処理。次元数が違うテンソルは除外。 |

同じストレージを共有する tied weight は二重加算を避けるため、同一対象テンソルの別名レイヤーがスキップされる場合があります。
