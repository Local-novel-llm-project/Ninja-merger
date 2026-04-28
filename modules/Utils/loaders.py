import copy
import os

import torch
import yaml
from safetensors.torch import load_file

from modules.Utils.diff_artifact import (
    is_diff_artifact_path,
    load_sparse_diff_artifact,
    load_sparse_diff_as_delta_model,
    resolve_diff_base_reference,
)
from modules.Utils.models import DummyModel
from modules.Utils.merge_types import AppConfig, MergeRequest
from modules.Utils.operation_dicts import validate_operation_request


def _validate_model_entry_names(models_list, yaml_path):
    seen_names = {}

    for index, model_entry in enumerate(models_list):
        if not isinstance(model_entry, dict):
            raise TypeError(
                f"Each entry in 'models' must be a mapping, got "
                f"{type(model_entry).__name__} at index {index} in {yaml_path}"
            )

        config_name = model_entry.get("name")
        if config_name is None:
            continue

        if not isinstance(config_name, str):
            raise TypeError(
                f"'name' must be a string at index {index} in {yaml_path}, "
                f"got {type(config_name).__name__}"
            )

        if not config_name.strip():
            raise ValueError(
                f"'name' must not be empty at index {index} in {yaml_path}"
            )

        invalid_separators = [sep for sep in (os.path.sep, os.path.altsep) if sep]
        if config_name in {".", ".."} or any(
            sep in config_name for sep in invalid_separators
        ):
            raise ValueError(
                f"'name' must be a plain directory name without path separators "
                f"at index {index} in {yaml_path}: {config_name!r}"
            )

        previous_index = seen_names.get(config_name)
        if previous_index is not None:
            raise ValueError(
                f"Duplicate model name '{config_name}' found in {yaml_path} "
                f"at indices {previous_index} and {index}"
            )

        seen_names[config_name] = index


def load_model(model_path, device, torch_dtype, *, diff_mode="reconstruct"):
    print(f"load_model: model input: {model_path}")

    if isinstance(model_path, list):
        model_path = model_path[0]
    model_path = os.fspath(model_path)

    if is_diff_artifact_path(model_path):
        if diff_mode == "delta":
            model = load_sparse_diff_as_delta_model(model_path)
        else:
            model = load_sparse_diff_artifact(model_path, device, torch_dtype, load_model)
    elif model_path.endswith(".safetensors"):
        state_dict = load_file(model_path, device=device)
        model = DummyModel(state_dict)
    elif model_path.endswith(".pth") or model_path.endswith(".bin"):
        data = torch.load(model_path, map_location=device)
        if isinstance(data, dict) and "model" in data:
            state_dict = data["model"]
            config_dict = data.get("config")
            model = DummyModel(state_dict, config_dict=config_dict)
        elif isinstance(data, dict) and "state_dict" in data:
            state_dict = data["state_dict"]
            config_dict = data.get("config")
            model = DummyModel(state_dict, config_dict=config_dict)
        else:
            state_dict = data
            if "weight" in state_dict:
                state_dict = state_dict["weight"]
            model = DummyModel(state_dict)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    return model


def _normalize_model_names(model_value):
    if model_value is None:
        return []
    if isinstance(model_value, str):
        if model_value.lower() in {"none", "null"}:
            return []
        return [model_value]
    if isinstance(model_value, list):
        return [
            item
            for item in model_value
            if not (isinstance(item, str) and item.lower() in {"none", "null"})
        ]
    return []


def _normalize_velocity_value(velocity):
    if isinstance(velocity, dict) and "real" in velocity and "imag" in velocity:
        return torch.complex(
            torch.tensor(velocity["real"]), torch.tensor(velocity["imag"])
        )
    return velocity


def _target_value_scalar(target_value):
    if isinstance(target_value, str) and target_value.lower() in {"none", "null"}:
        return None
    if isinstance(target_value, list) and len(target_value) == 1:
        item = target_value[0]
        if isinstance(item, str) and item.lower() in {"none", "null"}:
            return None
        return item
    return target_value


def _merge_model_config_overrides(model_entry, key_transformations, insert_layers):
    merged_entry = copy.deepcopy(model_entry)

    for model_type in ["left", "right", "target"]:
        if model_type not in merged_entry:
            continue

        if model_type in ["left", "right"] and isinstance(merged_entry[model_type], str):
            merged_entry[model_type] = [merged_entry[model_type]]

        model_names = _normalize_model_names(merged_entry[model_type])
        for model_name in model_names:
            if model_name == "recurrent":
                continue

            if "model_config" not in merged_entry:
                merged_entry["model_config"] = {}

            merged_config = {}
            merged_config.update(key_transformations.get(model_name, {}))
            merged_config.update(insert_layers.get(model_name, {}))

            if model_name not in merged_entry["model_config"]:
                merged_entry["model_config"][model_name] = {}
            merged_entry["model_config"][model_name].update(merged_config)

    return merged_entry


def _build_merge_request(model_entry):
    normalized_entry = copy.deepcopy(model_entry)

    normalized_entry["left"] = _normalize_model_names(normalized_entry.get("left"))
    normalized_entry["right"] = _normalize_model_names(normalized_entry.get("right"))

    if "velocity" not in normalized_entry and "velocities" not in normalized_entry:
        normalized_entry["velocity"] = 1.0
    else:
        normalized_entry["velocity"] = _normalize_velocity_value(
            normalized_entry.get("velocity")
        )

    if (
        "post_velocity" not in normalized_entry
        and "post_velocities" not in normalized_entry
    ):
        normalized_entry["post_velocity"] = 1.0

    if "target" not in normalized_entry:
        normalized_entry["target"] = None
    elif (
        isinstance(normalized_entry["target"], str)
        and normalized_entry["target"].lower() in {"none", "null"}
    ):
        normalized_entry["target"] = None

    target_value = normalized_entry.get("target")
    target_value_scalar = _target_value_scalar(target_value)
    is_llava_next = (
        isinstance(target_value_scalar, str)
        and target_value_scalar.lower() in ["llava", "vlm", "llava-next"]
    )

    request = MergeRequest(
        raw_config=normalized_entry,
        name=normalized_entry.get("name"),
        base_model_names=normalized_entry["left"],
        sub_model_names=normalized_entry.get("right", []),
        target_value=target_value,
        target_value_scalar=target_value_scalar,
        operation=normalized_entry.get("operation", "sub"),
        post_operation=normalized_entry.get("post_operation", "add"),
        preprocess=normalized_entry.get("preprocess", "none"),
        post_preprocess=normalized_entry.get("post_preprocess", "none"),
        normalization=normalized_entry.get("normalization", "none"),
        velocity=normalized_entry.get("velocity"),
        velocities=normalized_entry.get("velocities"),
        post_velocity=normalized_entry.get("post_velocity", 1.0),
        post_velocities=normalized_entry.get("post_velocities"),
        unmatch_size_layer_op=normalized_entry.get("unmatch_size_layer_op", "skip"),
        include_layers=normalized_entry.get("include_layers"),
        exclude_layers=normalized_entry.get("exclude_layers"),
        drop_layers=normalized_entry.get("drop_layers"),
        use_scaling=normalized_entry.get("use_scaling"),
        force_merge_single=normalized_entry.get("force_merge_single", False),
        v2s_empty_default=normalized_entry.get("v2s_empty_default", "v1"),
        v2s_single_default=normalized_entry.get("v2s_single_default", "auto"),
        is_llava_next=is_llava_next,
        model_config=normalized_entry.get("model_config", {}),
    )
    validate_operation_request(request)
    return request


def load_config(config_path):
    yaml_path = config_path.replace(".json", ".yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path} or {yaml_path}")

    if not isinstance(config, dict):
        raise TypeError(
            f"Config root must be a mapping, got {type(config).__name__} in {yaml_path}"
        )

    if "models" not in config:
        raise KeyError(f"Missing required 'models' section in {yaml_path}")

    models_list = config["models"]
    if isinstance(models_list, dict):
        models_list = [models_list]
    elif not isinstance(models_list, list):
        raise TypeError(
            "'models' must be a list of merge entries or a single mapping, "
            f"got {type(models_list).__name__} in {yaml_path}"
        )

    _validate_model_entry_names(models_list, yaml_path)

    use_scaling = config.get("use_scaling", config.get("use_scale", False))

    # key_transformations と insert_layers を models 内の各 model_config にマージ
    key_transformations = config.get("key_transformations", {})
    insert_layers = config.get("insert_layers", {})

    normalized_models = [
        _build_merge_request(
            _merge_model_config_overrides(
                model_entry,
                key_transformations,
                insert_layers,
            )
        )
        for model_entry in models_list
    ]

    return AppConfig(models=normalized_models, use_scaling=use_scaling)


def load_vlm_model(model, lora_name, device, torch_dtype):
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model, torch_dtype=torch_dtype, device_map=device, low_cpu_mem_usage=True
    )
    if lora_name is not None:
        from peft import PeftModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[green]start merging LoRA[/green]")
        model = PeftModel.from_pretrained(model, lora_name, device=device)
        model = model.merge_and_unload()
        print("[green]LoRA merged[/green]")
    state_dict = model.state_dict()
    return model, state_dict


def load_llava_model(model, lora_name, device, torch_dtype):
    from llava.mm_utils import get_model_name_from_path  # type: ignore
    from llava.model.builder import load_pretrained_model  # type: ignore

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model,
        model_base=None,
        model_name=get_model_name_from_path(model),
        torch_dtype=torch_dtype,
        device_map=device,
    )
    if lora_name is not None:
        from peft import PeftModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[green]start merging LoRA[/green]")
        model = PeftModel.from_pretrained(model, lora_name, device=device)
        model = model.merge_and_unload()
        print("[green]LoRA merged[/green]")
    state_dict = model.state_dict()
    return model, state_dict, tokenizer, image_processor


def load_tokenizer(model_name):
    if isinstance(model_name, list):
        model_name = model_name[0]
    model_name = os.fspath(model_name)

    if is_diff_artifact_path(model_name):
        model_name = resolve_diff_base_reference(model_name)

    try:
        from transformers import AutoTokenizer

        if os.path.exists(model_name):
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        return tokenizer
    except Exception as e:
        raise Exception(f"Failed to load tokenizer for {model_name}: {str(e)}")


def load_and_unscale_diff(filepath, metadata, key):
    """
    保存されたテンソルとメタデータを読み込み、逆スケーリングして元のテンソルを復元する。

    Args:
        filepath (str): テンソルとメタデータが保存されているファイルパス(ディレクトリ)
        metadata (dict): メタデータ
        key (str): テンソルのキー

    Returns:
        torch.Tensor: 復元されたテンソル。
    """
    # Noneの場合はそのまま返す
    if metadata is None:
        return None

    # bfloat16テンソルのロード
    tensor_filepath = os.path.join(filepath, f"{key}.bfloat16")
    tensor = torch.load(tensor_filepath)

    # メタデータを使って逆スケーリング
    mask = torch.tensor(metadata["scaled_indices"], dtype=torch.bool)
    scale_factor = metadata["scale_factor"]
    original_dtype = metadata["original_dtype"]
    tensor[mask] /= scale_factor

    # 元のデータ型に戻す
    if original_dtype == "torch.float64":
        tensor = tensor.double()
    elif original_dtype == "torch.float32":
        tensor = tensor.float()
    # 必要に応じて他の型にも対応

    return tensor
