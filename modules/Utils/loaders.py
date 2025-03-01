import os

import torch
import yaml
from peft import PeftModel
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
)

from modules.Utils.models import DummyModel


def load_model(model_path, device, torch_dtype):
    print(f"load_model: model input: {model_path}")

    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path, device=device)
        model = DummyModel(state_dict)
    elif model_path.endswith(".pth") or model_path.endswith(".bin"):
        state_dict = torch.load(model_path, map_location=device)
        model = DummyModel(state_dict)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    return model


def load_config(config_path):
    yaml_path = config_path.replace(".json", ".yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path} or {yaml_path}")

    models_list = config["models"]
    use_scaling = config.get("use_scale", False)

    # key_transformations と insert_layers を models 内の各 model_config にマージ
    key_transformations = config.get("key_transformations", {})
    insert_layers = config.get("insert_layers", {})

    for model_entry in models_list:
        for model_type in ["left", "right", "target"]:
            if model_type not in model_entry:
                continue
            if isinstance(model_entry[model_type], str):
                model_entry[model_type] = [model_entry[model_type]]

            if not isinstance(model_entry[model_type], list):
                continue

            for model_name in model_entry[model_type]:
                if model_name == "recurrent":
                    continue

                # model_config が存在しない場合は、空の辞書を作成
                if "model_config" not in model_entry:
                    model_entry["model_config"] = {}

                # キー変換とレイヤー挿入の設定をマージ
                merged_config = {}
                merged_config.update(key_transformations.get(model_name, {}))
                merged_config.update(insert_layers.get(model_name, {}))

                # モデル固有の設定を追加
                if model_name not in model_entry["model_config"]:
                    model_entry["model_config"][model_name] = {}
                model_entry["model_config"][model_name].update(merged_config)

        # velocity の処理 (既存のコード)
        velocity = model_entry.get("velocity")
        if isinstance(velocity, dict) and "real" in velocity and "imag" in velocity:
            model_entry["velocity"] = torch.complex(
                torch.tensor(velocity["real"]), torch.tensor(velocity["imag"])
            )
        # ... (その他の velocity の処理) ...
        elif isinstance(velocity, (int, float)):
            # 修正: int, float の場合はそのままの値を使用
            model_entry["velocity"] = velocity

        # target が存在しない場合は、null を設定
        if "target" not in model_entry:
            model_entry["target"] = None

    return (models_list, use_scaling)


def load_vlm_model(model, lora_name, device, torch_dtype):
    model = AutoModelForVision2Seq.from_pretrained(
        model, torch_dtype=torch_dtype, device_map=device, low_cpu_mem_usage=True
    )
    if lora_name is not None:
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

    try:
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
