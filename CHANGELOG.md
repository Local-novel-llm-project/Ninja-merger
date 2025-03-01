
### 変更点

*   **モジュール構造の改善:**
    *   `modules` ディレクトリを導入し、コードを機能ごとに整理されたモジュールに分割しました。これにより、コードの可読性、保守性、再利用性が向上しました。
    *   `Merger` 抽象基底クラスと、`BasicMerger`, `ComplexMerger`, `CustomMerger`, `QeicMerger` などの具象クラスを導入し、様々なマージ戦略を柔軟に実装・選択できるようになりました。
    *   `MergerFactory` クラスを導入し、設定ファイルに基づいて適切な `Merger` インスタンスを生成するようにしました。
*   **設定ファイルの改善:**
    *   設定ファイルを YAML 形式に変更し、可読性と保守性を向上させました。
    *   `models` リストを導入し、複数のモデルマージ設定を単一のファイルで管理できるようになりました。
    *   `model_config` セクションを追加し、モデル固有の設定 (キー変換、レイヤー挿入) を定義できるようになりました。
    *   `velocities` および `post_velocities` セクションを追加し、レイヤーごとの velocity を設定できるようになりました。
    *   `unmatch_size_layer_op` オプションを追加し、サイズの異なるレイヤーの処理方法 (スキップまたは共通部分のみ使用) を指定できるようになりました。
*   **機能追加:**
    *   **複素数マージ (ComplexMerge):** 複素数を用いた新しいマージ手法を追加しました。これにより、モデルの位相情報を考慮したマージが可能になり、特定の設定で性能が向上する可能性があります。
        *   `ComplexAdd`, `ComplexMix`, `ComplexAngleMerge`, `norm_angle_t_calc` などの関数を追加しました。
    *   **WidenMerge:** モデルの幅を広げるマージ手法を追加しました。
    *   **QEIC (Quantum-Entanglement-Inspired Calculation):** 量子もつれに触発された計算に基づくマージ手法を追加しました。
        *   `calculate_correlation_matrices`, `merge_correlation_matrices`, `QeicAdd`, `QeicMix`, `QeicSub` などの関数を追加しました。
    *   **QuantileMatch:** 分位点マッチングによる前処理を追加し、異なる分布を持つモデルをマージする際の安定性を向上させました。
    *   **複数のベースモデルのマージ:** 複数のベースモデルを同時にマージできるようになりました。
    *   **LoRA マージの自動化:** `load_and_prepare_models` 関数内で、LoRA モデルが指定された場合に自動的に LoRA をマージするようにしました。
    *   **設定ファイルでのLoRAマージ:** config に lora の項目を追加できるようになりました。
*   **リファクタリング:**
    *   `prepare_models_for_merging` 関数を `prepare_model_metadata` と `load_and_prepare_models` に分割し、コードの可読性と効率を向上させました。
    *   `main.py` 内のデバッグ用 `print` 文をコメントアウトし、`rich` ライブラリを使用したより洗練されたロギングに置き換えました。
    *   `get_skip_layers` 関数を修正し、`unmatch_size_layer_op` 設定を考慮するようにしました。
    *   `Passthrough` 関数にエラーハンドリングを追加しました。
    *   `define_savename` 関数を改善し、より柔軟なファイル名生成とレシピ (モデル設定) の保存に対応しました。
    *   `scale_tensor_inplace` 関数を追加し、モデル保存時のスケーリング処理を効率化しました。
    *   `operation_dicts.py` を追加して、各種演算、前処理、正規化、後処理をより整理された形で管理するようにしました.
*   **その他:**
    *   `rich` ライブラリを使用した、より詳細で視覚的にわかりやすいログ出力を実装しました。
    *   `torch.bfloat16` テンソルの直接保存をサポート (scale factor をメタデータとして保存)。
    *   未使用のコードやコメントを削除しました。
    *   型ヒントを広範囲に追加し、コードの可読性と保守性を向上させました。
    *   `README.md` のひな形をコメントで示しました。

### バグ修正

*   `get_skip_layers` 関数で、`target_model` が `None` の場合に、`base_state_dicts` の最初の要素を `target` として使用するように修正しました。
*   `dump_layers` 関数で、`target_model` が `None` の場合に、最初のベースモデルのレイヤーをダンプするように修正しました。
*   `main.py` で `prepare_models_for_merging` が2回呼び出されていた問題を修正しました。
*   `Passthrough` 関数で `v2s` がリストでない場合にエラーが発生しない問題を修正しました.

### 既知の問題点

*   `GitReBasin` 関数は計算量が O(n^3) であり、非常に大きなテンソルを扱う場合にパフォーマンスの問題が発生する可能性があります。
*   `QuantileMatch` 関数は、入力テンソルが大きい場合にメモリ不足エラーを引き起こす可能性があります。
*   `calculate_average_angle` 関数は、すべてのレイヤーの角度の平均を計算するため、異なるレイヤーが異なる方向に変化している場合に不適切な結果をもたらす可能性があります。

