import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import yaml

from .operation_registry import get_operation_short_name


def _config_get(config_or_request, key, default=None):
    if config_or_request is None:
        return default
    if hasattr(config_or_request, "get"):
        return config_or_request.get(key, default)
    return default


def _recipe_payload(config_or_request):
    if config_or_request is None:
        return {}
    if hasattr(config_or_request, "to_dict"):
        return config_or_request.to_dict()
    if isinstance(config_or_request, dict):
        return config_or_request
    return {}


def get_savename(path):
    path_path = Path(path)
    path_parts = path_path.parts
    if path_parts[0] == "/" or ":" in path_parts[0]:
        path_parts = path_parts[1:]

    return "-".join(path_parts)


def _shorten_model_name(name):
    short_name = (
        os.path.basename(name)
        .replace(".safetensors", "")
        .replace(".pth", "")
        .replace(".bin", "")
    )
    if len(short_name) > 20:
        return short_name[:8] + "..." + short_name[-8:]
    return short_name


def _default_merge_label(base, sub, model_dict):
    left_model_name = _shorten_model_name(base[0])
    right_model_name = _shorten_model_name(sub[0]) if sub else "none"
    model_names = (
        f"{left_model_name}_and_{right_model_name}" if sub else left_model_name
    )
    operation = get_operation_short_name(_config_get(model_dict, "operation", "merge"))
    return f"{model_names}_{operation}"


def _build_base_name(base, sub, model_dict):
    config_name = _config_get(model_dict, "name")
    base_label = config_name or _default_merge_label(base, sub, model_dict)
    if len(base_label) <= 100:
        return base_label

    hash_value = hashlib.sha256(base_label.encode()).hexdigest()[:8]
    return hash_value


def _resolve_output_base_dir(target, out_dir):
    if target == "lora":
        return os.path.join(out_dir, "lora")
    if target == "recurrent":
        return os.path.join(out_dir, "recurrent")
    if target is None or target == "null":
        return os.path.join(out_dir, "vector")

    target_name = os.path.basename(target[0] if isinstance(target, list) else target)
    return os.path.join(out_dir, target_name)


def _resolve_parent_dir(target, out_dir):
    base_dir_path = _resolve_output_base_dir(target, out_dir)
    os.makedirs(base_dir_path, exist_ok=True)
    return base_dir_path


def _resolve_candidate_save_stem(base, sub, target, out_dir, model_dict):
    parent_dir = _resolve_parent_dir(target, out_dir)
    base_name = _build_base_name(base, sub, model_dict)
    return os.path.join(parent_dir, base_name)


def _resolve_unique_save_stem(candidate_stem):
    if not output_artifact_exists(candidate_stem):
        return candidate_stem

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    candidate_with_timestamp = f"{candidate_stem}_{timestamp}"
    if not output_artifact_exists(candidate_with_timestamp):
        return candidate_with_timestamp

    suffix = 2
    while True:
        candidate_with_suffix = f"{candidate_stem}_{timestamp}_{suffix:02}"
        if not output_artifact_exists(candidate_with_suffix):
            return candidate_with_suffix
        suffix += 1


def _write_recipe(save_name, basename, model_dict):
    recipe_payload = _recipe_payload(model_dict)
    if not recipe_payload:
        return

    recipe_dir = os.path.dirname(save_name)
    os.makedirs(recipe_dir, exist_ok=True)
    recipe_path = os.path.join(recipe_dir, basename + "_recipe.yaml")

    try:
        with open(recipe_path, "w", encoding="utf-8") as f:
            yaml.dump(recipe_payload, f, default_flow_style=False, sort_keys=False)
    except Exception as error:
        print(f"Warning: Failed to save recipe: {error}")


def write_recipe_file(save_stem, model_dict):
    basename = os.path.basename(save_stem)
    _write_recipe(save_stem, basename, model_dict)


def output_artifact_exists(save_stem):
    if os.path.isdir(save_stem):
        return True
    if os.path.exists(f"{save_stem}.pth"):
        return True
    if os.path.exists(f"{save_stem}.difftensors"):
        return True
    return os.path.exists(f"{save_stem}_layers.txt")


def define_savename(base, sub, target, out_dir, index, model_dict=None):
    """
    モデルの保存名を定義する関数。

    Args:
        base (list): ベースモデルのパスのリスト。
        sub (list): サブモデルのパスのリスト。
        target (str): ターゲットモデルのパス、"lora"、"recurrent"、または None。
        out_dir (str): 出力ディレクトリ。
        index (int): モデル設定のインデックス (未使用)。
        model_dict (dict): モデル設定の辞書。

    Returns:
        str: 保存先のパス。
    """

    model_dict = model_dict or {}
    candidate_stem = _resolve_candidate_save_stem(base, sub, target, out_dir, model_dict)
    return _resolve_unique_save_stem(candidate_stem)


def get_slice_to(t, indice):
    if len(indice) == 1:
        return t[: indice[0]]
    elif len(indice) == 2:
        return t[: indice[0], : indice[1]]
    elif len(indice) == 3:
        return t[: indice[0], : indice[1], : indice[2]]
    # 2次元以下の場合も対応
    elif t.ndim <= 2:
        return t[: indice[0]] if t.ndim == 1 else t[: indice[0], : indice[1]]
    else:  # 4次元以上の場合
        # TODO: support N dimentional, 現在はとりあえずそのまま返す
        return t


def slice_tensor(t, indices):  # get_slice_toから変更
    """テンソルを指定されたインデックスでスライスする。"""
    return t[tuple([slice(None, i) for i in indices] + [Ellipsis])]


def scale_tensor_inplace(tensor, threshold, scale_factor):
    """
    テンソル内の閾値以下の値をスケーリングする (inplace)。

    Args:
        tensor (torch.Tensor): スケーリングするテンソル。
        threshold (float): スケーリングの閾値。
        scale_factor (float): スケーリング係数。

    Returns:
        torch.Tensor: スケーリングされたテンソル (元のテンソルが変更される)
    """
    mask = torch.abs(tensor) < threshold
    tensor[mask] *= scale_factor
    return tensor


def filter_state_dict_keys(state_dict, dropped_layers: Iterable[str] | None):
    """指定されたキーを state_dict から取り除いたコピーを返す。"""
    dropped = set(dropped_layers or [])
    if not dropped:
        return state_dict

    filtered = type(state_dict)()
    for key, value in state_dict.items():
        if key not in dropped:
            filtered[key] = value

    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        filtered._metadata = dict(metadata)

    return filtered


def prepare_tensor_slices(
    target_state_dict, key, base_models, sub_models, unmatch_size_layer_op, console
):
    """テンソルのスライスを準備するヘルパー関数。"""
    v = target_state_dict[key]

    def _collect_model_tensors(models):
        tensors = []
        for model in models:
            state_dict = model.state_dict()
            if key in state_dict:
                tensors.append(state_dict[key])
                continue
            if getattr(model, "_ninja_sparse_zero_missing", False):
                tensors.append(torch.zeros_like(v))
        return tensors

    base_tensors = _collect_model_tensors(base_models)
    sub_tensors = _collect_model_tensors(sub_models)

    if unmatch_size_layer_op == "only_common_range":
        base_same_ndim = [t for t in base_tensors if t.ndim == v.ndim]
        sub_same_ndim = [t for t in sub_tensors if t.ndim == v.ndim]

        dropped_base = len(base_tensors) - len(base_same_ndim)
        dropped_sub = len(sub_tensors) - len(sub_same_ndim)
        if dropped_base > 0 or dropped_sub > 0:
            console.print(
                f"  [yellow]Layer {key}: skipped {dropped_base} base and {dropped_sub} sub tensors due to ndim mismatch in only_common_range mode.[/yellow]"
            )

        common_tensors = [v] + base_same_ndim + sub_same_ndim
        min_size = tuple(
            min(t.shape[dim] for t in common_tensors) for dim in range(v.ndim)
        )
        console.print(
            f"  [cyan]Merging only common range for layer {key}. Common size: {min_size}[/cyan]"
        )

        v_slice = slice_tensor(v, min_size)
        base_slices = [slice_tensor(t, min_size) for t in base_same_ndim]
        sub_slices = [slice_tensor(t, min_size) for t in sub_same_ndim]
        return v_slice, base_slices, sub_slices, min_size

    min_size = tuple(v.shape)
    return v, base_tensors, sub_tensors, min_size


def _complex_real_dtype(dtype):
    if dtype in (torch.float16, torch.float32, torch.float64):
        return dtype
    return torch.float32


def to_complex_tensor(val, device, dtype):
    """
    数値または複素数を複素数テンソルに変換するヘルパー関数
    """
    real_dtype = _complex_real_dtype(dtype)
    if isinstance(val, complex):
        return torch.complex(
            torch.tensor(val.real, device=device, dtype=real_dtype),
            torch.tensor(val.imag, device=device, dtype=real_dtype),
        )
    else:
        return torch.complex(
            torch.tensor(float(val), device=device, dtype=real_dtype),
            torch.tensor(0.0, device=device, dtype=real_dtype),
        )
