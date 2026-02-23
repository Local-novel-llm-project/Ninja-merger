import ast
import os
import re
from typing import Any, Dict, Iterable, Match

from rich import print


def _to_layer_tokens(layers_spec) -> list[str]:
    if layers_spec is None:
        return []

    if isinstance(layers_spec, str):
        raw_items = layers_spec.split(",")
    elif isinstance(layers_spec, Iterable):
        raw_items = []
        for item in layers_spec:
            if isinstance(item, str):
                raw_items.extend(item.split(","))
            else:
                raw_items.append(str(item))
    else:
        raw_items = [str(layers_spec)]

    return [item.strip() for item in raw_items if str(item).strip()]


def parse_layer_specifications(layers_spec):
    ranges = []
    specific_layers = []

    for token in _to_layer_tokens(layers_spec):
        # Supported examples:
        # - "model.layers.2-5"
        # - "2-5"
        range_match = re.search(r"model\.layers\.(\d+)\s*-\s*(\d+)", token)
        if range_match is None:
            range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)

        if range_match is not None:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                start, end = end, start
            ranges.append(range(start, end + 1))
            continue

        specific_layers.append(token)

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

    # include 指定がある場合、形式に合致しないレイヤーは除外する。
    return len(include_specific) <= 0 and len(include_ranges) <= 0


def is_layer_dropped(layer_name, drop_ranges, drop_specific):
    # 特定のレイヤー名のチェック
    for dropped in drop_specific:
        if dropped in layer_name:
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
    if isinstance(layers_str, str):
        return [layer.strip() for layer in layers_str.split(",") if layer.strip()]
    if isinstance(layers_str, Iterable):
        return [str(layer).strip() for layer in layers_str if str(layer).strip()]
    return [str(layers_str)]


def _get_tensor_by_key(state_dict: Dict[str, Any], key: str, fallback_key: str):
    if key in state_dict:
        return state_dict[key]
    return state_dict.get(fallback_key)


def _shape_str(tensor: Any) -> str:
    if hasattr(tensor, "shape"):
        return str(tuple(tensor.shape))
    return "N/A"


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
        target_state_dict = base_state_dicts[0]  # 便宜上、最初の base model を使う
    else:
        target_state_dict = target_model.state_dict()
        base_state_dicts = [b.state_dict() for b in base_models]

    if not isinstance(sub_models, list):
        sub_models = [sub_models]
    sub_state_dicts = [s.state_dict() for s in sub_models]

    # --- すべてのキーを収集 ---
    all_keys = set(target_state_dict.keys())
    for base_state_dict in base_state_dicts:
        all_keys.update(base_state_dict.keys())
    for sub_state_dict in sub_state_dicts:
        all_keys.update(sub_state_dict.keys())

    if unmatch_size_layer_op == "only_common_range":
        print(
            "[cyan]Using only_common_range mode - Size mismatches will be handled during merging[/cyan]"
        )

    for target_key in all_keys:
        lookup_key = (
            target_key.replace("language_model.", "", 1)
            if is_llava_next
            else target_key
        )

        missing = False
        for sub_state_dict in sub_state_dicts:
            if _get_tensor_by_key(sub_state_dict, lookup_key, target_key) is None:
                print(f"[yellow] Right key not found: {lookup_key}, skip...[/yellow]")
                skip_layers.append(lookup_key)
                missing = True
                break
        if missing:
            continue

        for base_state_dict in base_state_dicts:
            if _get_tensor_by_key(base_state_dict, lookup_key, target_key) is None:
                print(f"[yellow] Base key not found: {lookup_key}, skip...[/yellow]")
                skip_layers.append(lookup_key)
                missing = True
                break
        if missing:
            continue

        reference = _get_tensor_by_key(target_state_dict, lookup_key, target_key)
        if reference is None:
            reference = _get_tensor_by_key(base_state_dicts[0], lookup_key, target_key)

        if reference is None:
            print(
                f"[yellow]Warning: Invalid reference tensor for key '{target_key}', skip...[/yellow]"
            )
            skip_layers.append(target_key)
            continue

        # only_common_range ではサイズ不一致をスキップせず、警告のみ出す
        if unmatch_size_layer_op == "only_common_range":
            for base_state_dict in base_state_dicts:
                base_tensor = _get_tensor_by_key(base_state_dict, lookup_key, target_key)
                if (
                    base_tensor is not None
                    and hasattr(base_tensor, "shape")
                    and hasattr(reference, "shape")
                    and tuple(base_tensor.shape) != tuple(reference.shape)
                ):
                    print(
                        f"[cyan] Base key size mismatch: {target_key} ({_shape_str(base_tensor)} vs {_shape_str(reference)}), will use common range[/cyan]"
                    )
            for sub_state_dict in sub_state_dicts:
                sub_tensor = _get_tensor_by_key(sub_state_dict, lookup_key, target_key)
                if (
                    sub_tensor is not None
                    and hasattr(sub_tensor, "shape")
                    and hasattr(reference, "shape")
                    and tuple(sub_tensor.shape) != tuple(reference.shape)
                ):
                    print(
                        f"[cyan] Sub key size mismatch: {target_key} ({_shape_str(sub_tensor)} vs {_shape_str(reference)}), will use common range[/cyan]"
                    )
            continue

        # skip モードではサイズ不一致を除外
        if not hasattr(reference, "shape"):
            print(
                f"[yellow]Warning: Invalid tensor for key '{target_key}', skip...[/yellow]"
            )
            skip_layers.append(target_key)
            continue

        mismatch = False
        for base_state_dict in base_state_dicts:
            base_tensor = _get_tensor_by_key(base_state_dict, lookup_key, target_key)
            if (
                base_tensor is None
                or not hasattr(base_tensor, "shape")
                or tuple(base_tensor.shape) != tuple(reference.shape)
            ):
                print(
                    f"[yellow] Base key size mismatch: {target_key}, skip...[/yellow]"
                )
                skip_layers.append(target_key)
                mismatch = True
                break
        if mismatch:
            continue

        for sub_state_dict in sub_state_dicts:
            sub_tensor = _get_tensor_by_key(sub_state_dict, lookup_key, target_key)
            if (
                sub_tensor is None
                or not hasattr(sub_tensor, "shape")
                or tuple(sub_tensor.shape) != tuple(reference.shape)
            ):
                print(
                    f"[yellow] Sub key size mismatch: {target_key}, skip...[/yellow]"
                )
                skip_layers.append(target_key)
                mismatch = True
                break
        if mismatch:
            continue

    skip_layers = list(dict.fromkeys(skip_layers))
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

    return len(include_specific) <= 0 and len(include_ranges) <= 0


def prepare_velocities(velocities_config, state_dict):
    """
    レイヤーごとの velocity を準備する。

    Args:
        velocities_config (dict): 設定ファイル内の velocities の設定。
        state_dict (dict): モデルの state_dict。

    Returns:
        dict: レイヤー名をキー、velocity 値を値とする辞書。
    """
    per_layer_velocities: Dict[str, Any] = {}
    default_velocity = velocities_config.get("DEFAULT", 1.0)  # デフォルト値

    for layer_name in state_dict.keys():
        matched = False
        for pattern, velocity_info in velocities_config.items():
            if pattern == "DEFAULT":
                continue

            if isinstance(velocity_info, (int, float, complex)):
                # 単純な文字列マッチ
                if layer_name.startswith(pattern):
                    per_layer_velocities[layer_name] = velocity_info
                    matched = True
                    break

            elif isinstance(velocity_info, dict) and velocity_info.get("type") == "regex":
                # 正規表現マッチ。regex が未指定の場合は key 側を正規表現として扱う。
                regex_pattern = velocity_info.get("regex", pattern)
                match = re.match(regex_pattern, layer_name)
                if match:
                    value = velocity_info.get("value", default_velocity)
                    if isinstance(value, (int, float, complex)):
                        per_layer_velocities[layer_name] = value
                    elif isinstance(value, str):
                        try:
                            per_layer_velocities[layer_name] = _safe_eval_expression(
                                value, match, layer_name
                            )
                        except Exception as e:
                            print(
                                f"Error evaluating velocity expression: {value}, error: {e}"
                            )
                            per_layer_velocities[layer_name] = default_velocity
                    else:
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
    per_layer_post_velocities: Dict[str, Any] = {}
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

            elif isinstance(post_velocity_info, dict) and post_velocity_info.get("type") == "regex":
                # 正規表現マッチ。regex が未指定の場合は key 側を正規表現として扱う。
                regex_pattern = post_velocity_info.get("regex", pattern)
                match = re.match(regex_pattern, layer_name)
                if match:
                    value = post_velocity_info.get("value", default_post_velocity)
                    if isinstance(value, (int, float, complex)):
                        # そのままの値を設定
                        per_layer_post_velocities[layer_name] = value
                    elif isinstance(value, str):
                        try:
                            per_layer_post_velocities[layer_name] = _safe_eval_expression(
                                value, match, layer_name
                            )
                        except Exception as e:
                            print(
                                f"Error evaluating post_velocity expression: {value}, error: {e}"
                            )
                            per_layer_post_velocities[layer_name] = (
                                default_post_velocity
                            )
                    else:
                        per_layer_post_velocities[layer_name] = default_post_velocity
                    matched = True
                    break
            else:
                print(
                    f"[yellow]Warning: Invalid post_velocity config for pattern '{pattern}'. Skipping.[/yellow]"
                )

        if not matched:
            per_layer_post_velocities[layer_name] = default_post_velocity

    return per_layer_post_velocities


_ALLOWED_FUNCS = {
    "int": int,
    "float": float,
    "complex": complex,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
}


def _safe_eval_expression(expr: str, match: Match[str], layer_name: str):
    """Safely evaluates a restricted arithmetic expression for velocity configs."""
    parsed = ast.parse(expr, mode="eval")

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex, str)):
                return node.value
            raise ValueError("Unsupported constant value.")

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise ValueError("Unsupported binary operator.")

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")

        if isinstance(node, ast.Name):
            if node.id == "layer_name":
                return layer_name
            raise ValueError(f"Unsupported identifier: {node.id}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
                func = _ALLOWED_FUNCS[node.func.id]
                args = [eval_node(arg) for arg in node.args]
                return func(*args)

            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "match"
                and node.func.attr == "group"
            ):
                args = [eval_node(arg) for arg in node.args]
                return match.group(*args)

            raise ValueError("Unsupported function call.")

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    return eval_node(parsed)
