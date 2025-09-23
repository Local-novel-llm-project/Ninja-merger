import glob
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

import torch
import yaml


def get_savename(path):
    path_path = Path(path)
    path_parts = path_path.parts
    if path_parts[0] == "/" or ":" in path_parts[0]:
        path_parts = path_parts[1:]

    return "-".join(path_parts)


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

    # 1. 設定ファイルから name キーの値を取得 (存在する場合)
    config_name = model_dict.get("name")

    # 2. name キーが存在しない場合は、model_names と operation を生成
    if not config_name:

        def shorten_model_name(name):
            name = (
                os.path.basename(name)
                .replace(".safetensors", "")
                .replace(".pth", "")
                .replace(".bin", "")
            )
            if len(name) > 20:  # 例: 20 文字を超える場合は短縮
                return name[:8] + "..." + name[-8:]
            return name

        left_model_name = shorten_model_name(base[0])  # 最初のモデル名
        right_model_name = shorten_model_name(sub[0])  # 最初のモデル名

        model_names = f"{left_model_name}_and_{right_model_name}"

        operation_map = {  # 操作名と短い名前の対応
            "add": "add",
            "sub": "sub",
            "mul": "mul",
            "div": "div",
            "mix": "mix",
            "avg": "avg",
            "concat": "concat",
            "maxpool": "maxpool",
            "minpool": "minpool",
            "geometric_mean": "gmean",
            "std_sub": "stdsub",
            "widen": "widen",
            "complexadd": "cadd",
            "angle_merge": "angm",
            "complex_angle_merge": "cam",
            "qeic_add": "qadd",
            "qeic_mix": "qmix",
            "qeic_sub": "qsub",
        }
        operation = operation_map.get(
            model_dict.get("operation", "merge"), "unk"
        )  # 未知の操作は "unk"

    # 3. タイムスタンプを生成
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 4. ファイル名を生成
    if config_name:
        basename = f"{config_name}_{timestamp}"
    else:
        basename = f"{model_names}_{operation}_{timestamp}"

    # 5. ファイル名が長すぎる場合は、ハッシュ化 (モデル名と操作部分のみ)
    if len(basename) > 100:
        if config_name:  # config_nameがある場合は、config_nameをハッシュ化
            hash_value = hashlib.sha256(config_name.encode()).hexdigest()[:8]
            basename = f"{hash_value}_{timestamp}"
        else:  # config_name がない場合は、model_names と operationをハッシュ化
            hash_value = hashlib.sha256((model_names + operation).encode()).hexdigest()[
                :8
            ]
            basename = f"{hash_value}_{timestamp}"

    # 6. 連番を追加してファイル名の衝突を回避
    if target == "lora":
        dir_path = os.path.join(out_dir, "lora")
    elif target == "recurrent":
        dir_path = os.path.join(out_dir, "recurrent")
    elif target is None or target == "null":
        dir_path = os.path.join(out_dir, "vector")
    else:
        target_name = os.path.basename(target[0] if isinstance(target, list) else target)
        dir_path = os.path.join(out_dir, target_name)

    os.makedirs(dir_path, exist_ok=True)  # ディレクトリが存在しない場合は作成

    # 既存のファイル名とパターンマッチング
    pattern = os.path.join(dir_path, basename + "*")
    existing_files = glob.glob(pattern)
    serial_number = 1

    if existing_files:  # 同じファイル名が存在する場合は、連番をインクリメント
        # 最大の連番 + 1 を取得
        existing_numbers = [
            int(re.search(r"_(\d+)(?:\.safetensors|\.yaml)$", f).group(1))
            for f in existing_files
            if re.search(r"_(\d+)(?:\.safetensors|\.yaml)$", f)
        ]
        if existing_numbers:
            serial_number = max(existing_numbers) + 1

    basename = f"{basename}_{serial_number:03}"  # 連番を付与

    # 7. ディレクトリとファイル名を結合
    if target == "lora":
        save_name = os.path.join(dir_path, basename + ".safetensors")
    elif target == "recurrent":
        save_name = os.path.join(dir_path, basename + ".safetensors")
    elif target is None or target == "null":
        save_name = os.path.join(dir_path, basename + ".safetensors")
    else:
        target_name = os.path.basename(target[0] if isinstance(target, list) else target)
        save_name = os.path.join(dir_path, basename + ".safetensors")

    # 8. レシピを保存 (モデル設定を YAML 形式で保存)
    if model_dict:
        recipe_dir = os.path.dirname(save_name)
        os.makedirs(recipe_dir, exist_ok=True)
        recipe_path = os.path.join(recipe_dir, basename + "_recipe.yaml")  # 連番付き

        try:
            with open(recipe_path, "w", encoding="utf-8") as f:
                yaml.dump(model_dict, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Warning: Failed to save recipe: {e}")

    return save_name


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


def prepare_tensor_slices(
    target_state_dict, key, base_models, sub_models, unmatch_size_layer_op, console
):
    """テンソルのスライスを準備するヘルパー関数。"""
    v = target_state_dict[key]

    if unmatch_size_layer_op == "only_common_range":
        min_size = min(
            v.shape,
            *[b.state_dict()[key].shape for b in base_models if key in b.state_dict()],
            *[s.state_dict()[key].shape for s in sub_models if key in s.state_dict()],
        )
        console.print(
            f"  [cyan]Merging only common range for layer {key}. Common size: {min_size}[/cyan]"
        )
    else:
        min_size = v.shape

    v_slice = (
        slice_tensor(v, min_size) if unmatch_size_layer_op == "only_common_range" else v
    )
    base_slices = [
        slice_tensor(b.state_dict()[key], min_size)
        if unmatch_size_layer_op == "only_common_range"
        else b.state_dict()[key]
        for b in base_models
        if key in b.state_dict()
    ]
    sub_slices = [
        slice_tensor(s.state_dict()[key], min_size)
        if unmatch_size_layer_op == "only_common_range"
        else s.state_dict()[key]
        for s in sub_models
        if key in s.state_dict()
    ]
    return v_slice, base_slices, sub_slices, min_size


def to_complex_tensor(val, device, dtype):
    """
    数値または複素数を複素数テンソルに変換するヘルパー関数
    """
    if isinstance(val, complex):
        return torch.complex(
            torch.tensor(val.real, device=device, dtype=dtype),
            torch.tensor(val.imag, device=device, dtype=dtype),
        )
    else:
        return torch.complex(
            torch.tensor(float(val), device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype),
        )
