import os
import re

from rich import print


def parse_layer_specifications(layers_str):
    if layers_str is None:
        return [], []
    ranges = []
    specific_layers = []
    for layer in layers_str:
        parts = layer.split(",")
        if len(parts) > 1 and "-" in parts[1]:
            start, end = map(int, parts[1].split("-"))
            ranges.append(range(start, end + 1))
        if parts[0].strip():
            specific_layers.append(parts[0].strip())
    return ranges, specific_layers


def is_layer_included(
    layer_name, include_ranges, include_specific, exclude_ranges, exclude_specific
):
    # 特定のレイヤー名のチェック
    for incl in include_specific:
        if incl in layer_name:
            return True
    for excl in exclude_specific:
        if excl in layer_name:
            return False

    # レンジに基づくチェック
    match = re.search(r"model\.layers\.(\d+)", layer_name)
    if match:
        index = int(match.group(1))
        included = (
            any(index in r for r in include_ranges) if len(include_ranges) > 0 else True
        )
        excluded = (
            any(index in r for r in exclude_ranges)
            if len(exclude_ranges) > 0
            else False
        )
        return included and not excluded

    return len(include_specific) <= 0  # レイヤー名が特定の形式に合致しない場合は含める


def is_layer_dropped(layer_name, drop_ranges, drop_specific):
    # 特定のレイヤー名のチェック
    if layer_name in drop_specific:
        return True

    # レンジに基づくチェック
    match = re.search(r"model\.layers\.(\d+)", layer_name)
    if match:
        index = int(match.group(1))
        dropped = any(index in r for r in drop_ranges) if drop_ranges else False
        return dropped

    return False  # レイヤー名が特定の形式に合致しない場合はドロップしない


def parse_layers(layers_str):
    if layers_str is None:
        return None
    return layers_str.split(",")


def get_skip_layers(
    target_model,
    base_models,
    sub_models,
    unmatch_size_layer_op,
    is_llava_next=False,
):
    skip_layers = []

    if target_model is None:
        base_state_dicts = [b.state_dict() for b in base_models]
        target = base_state_dicts[0]  # 便宜上、最初の base model を使う
    else:
        target = target_model.state_dict()
        base_state_dicts = [b.state_dict() for b in base_models]

    if not isinstance(sub_models, list):
        sub_models = [sub_models]
    sub_state_dicts = [s.state_dict() for s in sub_models]

    # --- すべてのキーを収集 ---
    all_keys = set(target.keys())
    for base_state_dict in base_state_dicts:
        all_keys.update(base_state_dict.keys())
    for sub_state_dict in sub_state_dicts:
        all_keys.update(sub_state_dict.keys())

    # 修正: unmatch_size_layer_op が only_common_range の場合、
    # サイズチェックを行わず、マージ時に共通部分のみ使用するように変更
    if unmatch_size_layer_op == "only_common_range":
        print(
            "[cyan]Using only_common_range mode - Size mismatches will be handled during merging[/cyan]"
        )
        # 重要なレイヤーでサイズの違いがあれば、ログに出力
        for target_key in all_keys:
            k = target_key
            if is_llava_next:
                k = target_key.replace("language_model.", "", 1)

            # キーの存在チェックのみ行い、サイズチェックはスキップ
            key_missing = False
            for sub_state_dict in sub_state_dicts:
                if k not in sub_state_dict:
                    print(f"[yellow] Right key not found: {k}, skip...[/yellow]")
                    skip_layers.append(k)
                    key_missing = True
                    break
            if key_missing:
                continue

            for base_state_dict in base_state_dicts:
                if k not in base_state_dict:
                    print(f"[yellow] Base key not found: {k}, skip...[/yellow]")
                    skip_layers.append(k)
                    key_missing = True
                    break
            if key_missing:
                continue

            # サイズが異なる場合は警告を出すだけ
            if target_model is not None:
                if (
                    target_model.state_dict()[target_key].size()
                    != base_state_dicts[0][target_key].size()
                ):
                    print(
                        f"[cyan] Base key size mismatch: {target_key}, will use common range[/cyan]"
                    )
                    # skip_layers には追加しない

            for sub_state_dict in sub_state_dicts:
                if target_model is None:
                    if (
                        base_state_dicts[0][target_key].size()
                        != sub_state_dict[target_key].size()
                    ):
                        print(
                            f"[cyan] Sub key size mismatch: {target_key}, will use common range[/cyan]"
                        )
                        # skip_layers には追加しない
                else:
                    if (
                        target_model.state_dict()[target_key].size()
                        != sub_state_dict[target_key].size()
                    ):
                        print(
                            f"[cyan] Sub key size mismatch: {target_key}, will use common range[/cyan]"
                        )
                        # skip_layers には追加しない
    else:
        # 元の実装（サイズチェックを含む）
        for target_key in all_keys:
            k = target_key
            if is_llava_next:
                k = target_key.replace("language_model.", "", 1)

            key_missing = False
            for sub_state_dict in sub_state_dicts:
                if k not in sub_state_dict:
                    print(f"[yellow] Right key not found: {k}, skip...[/yellow]")
                    skip_layers.append(k)
                    key_missing = True
                    break
            if key_missing:
                continue

            for base_state_dict in base_state_dicts:
                if k not in base_state_dict:
                    print(f"[yellow] Base key not found: {k}, skip...[/yellow]")
                    skip_layers.append(k)
                    key_missing = True
                    break
            if key_missing:
                continue
            # サイズチェック
            if target_model is not None:
                if (
                    target_model.state_dict()[target_key].size()
                    != base_state_dicts[0][target_key].size()
                ):
                    print(
                        f"[yellow] Base key size mismatch: {target_key}, skip...[/yellow]"
                    )
                    skip_layers.append(target_key)
                    continue
            for sub_state_dict in sub_state_dicts:
                if target_model is None:
                    if (
                        base_state_dicts[0][target_key].size()
                        != sub_state_dict[target_key].size()
                    ):
                        print(
                            f"[yellow] Sub key size mismatch: {target_key}, skip...[/yellow]"
                        )
                        skip_layers.append(target_key)
                        break
                else:
                    if (
                        target_model.state_dict()[target_key].size()
                        != sub_state_dict[target_key].size()
                    ):
                        print(
                            f"[yellow] Sub key size mismatch: {target_key}, skip...[/yellow]"
                        )
                        skip_layers.append(target_key)
                        break

    print(f"Skip layers before return: {skip_layers}")
    return skip_layers


def dump_layers(model_state_dict, savename):
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    layer_names = list(model_state_dict.keys())
    with open(f"{savename}_layers.txt", "w") as f:
        f.write("\n".join(layer_names))
    print(f"Layer names dumped to [cyan]{savename}_layers.txt[/cyan]")
    print("[bold]Layer names:[/bold]")
    print("\n".join(layer_names))


def is_qeic_target_layer(
    layer_name, include_ranges, include_specific, exclude_ranges, exclude_specific
):
    """
    指定されたレイヤーがQEICの対象レイヤーかどうかを判定する。

    Args:
        layer_name: チェックするレイヤー名。
        include_ranges: 含めるレイヤーの範囲のリスト。
        include_specific: 含める特定のレイヤー名のリスト。
        exclude_ranges: 除外するレイヤーの範囲のリスト。
        exclude_specific: 除外する特定のレイヤー名のリスト。

    Returns:
        bool: QEICの対象レイヤーであればTrue、そうでなければFalse。
    """

    # is_layer_included 関数のロジックをコピーし、QEIC用に調整
    for incl in include_specific:
        if incl in layer_name:
            return True
    for excl in exclude_specific:
        if excl in layer_name:
            return False

    match = re.search(r"model\.layers\.(\d+)", layer_name)
    if match:
        index = int(match.group(1))
        included = (
            any(index in r for r in include_ranges) if len(include_ranges) > 0 else True
        )
        excluded = (
            any(index in r for r in exclude_ranges)
            if len(exclude_ranges) > 0
            else False
        )
        return included and not excluded

    return len(include_specific) <= 0


def prepare_velocities(velocities_config, state_dict):
    """
    レイヤーごとの velocity を準備する。

    Args:
        velocities_config (dict): 設定ファイル内の velocities の設定。
        state_dict (dict): モデルの state_dict。

    Returns:
        dict: レイヤー名をキー、velocity 値を値とする辞書。
    """
    per_layer_velocities = {}
    default_velocity = velocities_config.get("DEFAULT", 1.0)  # デフォルト値

    for layer_name in state_dict.keys():
        matched = False
        for pattern, velocity_info in velocities_config.items():
            if pattern == "DEFAULT":
                continue

            if isinstance(
                velocity_info, (int, float, complex)
            ):  # typeが指定されていない場合は、直接代入
                # 単純な文字列マッチ
                if layer_name.startswith(pattern):
                    per_layer_velocities[layer_name] = velocity_info
                    matched = True
                    break
            elif (
                isinstance(velocity_info, dict) and velocity_info.get("type") == "regex"
            ):
                # 正規表現マッチ
                match = re.match(velocity_info["regex"], layer_name)
                if match:
                    value = velocity_info["value"]
                    if isinstance(value, (int, float, complex)):
                        # そのままの値を設定
                        per_layer_velocities[layer_name] = value
                    elif isinstance(value, str):
                        # 文字列の場合は、eval で評価 (キャプチャグループを使用可能)
                        # 例: "0.1 * int(match.group(1))"
                        try:
                            per_layer_velocities[layer_name] = eval(
                                value, {"match": match, "layer_name": layer_name}
                            )
                        except Exception as e:
                            print(
                                f"Error evaluating velocity expression: {value}, error: {e}"
                            )
                            per_layer_velocities[layer_name] = default_velocity
                    matched = True
                    break

            else:
                print(
                    f"[yellow]Warning: Invalid velocity config for pattern '{pattern}'. Skipping.[/yellow]"
                )

        if not matched:
            # どのパターンにもマッチしなかった場合は、デフォルト値を使用
            per_layer_velocities[layer_name] = default_velocity

    return per_layer_velocities


def prepare_post_velocities(post_velocities_config, state_dict):
    """
    レイヤーごとの post_velocity を準備する。

    Args:
        post_velocities_config (dict): 設定ファイルの post_velocities の設定。
        state_dict (dict): モデルの state_dict。

    Returns:
        dict: レイヤー名をキー、post_velocity 値を値とする辞書。
    """
    per_layer_post_velocities = {}
    default_post_velocity = post_velocities_config.get("DEFAULT", 1.0)

    for layer_name in state_dict.keys():
        matched = False
        for pattern, post_velocity_info in post_velocities_config.items():
            if pattern == "DEFAULT":
                continue

            if isinstance(post_velocity_info, (int, float, complex)):
                # 単純な文字列マッチ
                if layer_name.startswith(pattern):
                    per_layer_post_velocities[layer_name] = post_velocity_info
                    matched = True
                    break

            elif (
                isinstance(post_velocity_info, dict)
                and post_velocity_info.get("type") == "regex"
            ):
                # 正規表現マッチ
                match = re.match(post_velocity_info["regex"], layer_name)
                if match:
                    value = post_velocity_info["value"]
                    if isinstance(value, (int, float, complex)):
                        # そのままの値を設定
                        per_layer_post_velocities[layer_name] = value
                    elif isinstance(value, str):
                        # 文字列の場合は、eval で評価 (キャプチャグループを使用可能)
                        try:
                            per_layer_post_velocities[layer_name] = eval(
                                value, {"match": match, "layer_name": layer_name}
                            )
                        except Exception as e:
                            print(
                                f"Error evaluating post_velocity expression: {value}, error: {e}"
                            )
                            per_layer_post_velocities[layer_name] = (
                                default_post_velocity
                            )
                    matched = True
                    break
            else:
                print(
                    f"[yellow]Warning: Invalid post_velocity config for pattern '{pattern}'. Skipping.[/yellow]"
                )

        if not matched:
            per_layer_post_velocities[layer_name] = default_post_velocity

    return per_layer_post_velocities
