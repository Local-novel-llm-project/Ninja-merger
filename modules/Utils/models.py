# modules/Utils/models.py
import copy
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .merge_types import MergeRequest


@dataclass
class PreparedModels:
    base_models: list[nn.Module]
    sub_models: list[nn.Module]
    velocity: Any
    post_velocity: Any


class DummyModel(nn.Module):
    def __init__(self, state_dict, models=None, config_dict=None):
        super().__init__()
        self.state_dict_data = state_dict
        if config_dict:
            self._config = DummyConfig(config_dict=config_dict)
        else:
            configs = [m.config for m in models] if models else []
            self._config = DummyConfig(configs=configs)

    def state_dict(self):
        return self.state_dict_data

    @property
    def config(self):
        return self._config  # 読み取り専用の config 属性


class DummyConfig:
    def __init__(self, configs=None, config_dict=None):
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)
        elif configs:
            # 最初の config をベースに共通の属性をコピー
            base_config = configs[0]
            for key, value in base_config.__dict__.items():
                # RVC固有のメタデータは後で特別扱いするため、ここではスキップ
                if key not in ["info", "version", "sr", "f0", "name"]:
                    setattr(self, key, value)

            # マージされたことを示すために名前を変更
            self._name_or_path = "merged_model"
            # アーキテクチャ情報も更新
            self.architectures = ["DummyModel"]

            # RVCモデルのメタデータを全モデルから集約
            self.info = [getattr(c, "info", "N/A") for c in configs]
            self.version = [getattr(c, "version", "N/A") for c in configs]
            self.sr = [getattr(c, "sr", "N/A") for c in configs]
            self.f0 = [getattr(c, "f0", False) for c in configs]
            self.name = [getattr(c, "name", "N/A") for c in configs]

        else:
            self._name_or_path = "dummy_model"
            self.vocab_size = 0
            self.model_type = "dummy"
            self.architectures = ["DummyModel"]

    def to_dict(self):
        return self.__dict__


def _config_to_plain_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return copy.deepcopy(config)
    if hasattr(config, "to_dict"):
        return copy.deepcopy(config.to_dict())
    return copy.deepcopy(getattr(config, "__dict__", {}))


def _set_config_attr(config, key, value):
    setattr(config, key, value)
    if hasattr(config, "__dict__"):
        config.__dict__[key] = value


def _remove_config_attr(config, key):
    if hasattr(config, "__dict__"):
        config.__dict__.pop(key, None)
    if hasattr(config, key):
        try:
            delattr(config, key)
        except (AttributeError, TypeError):
            pass


def prune_config_for_dropped_layers(config, state_dict):
    """drop_layers の結果に応じて config から不要な multimodal 設定を除去する。"""
    if config is None or state_dict is None:
        return

    state_keys = list(state_dict.keys())
    has_text = any(k.startswith("model.language_model.") for k in state_keys)
    has_audio = any(
        k.startswith(("model.audio_tower.", "model.embed_audio.")) for k in state_keys
    )
    has_vision = any(
        k.startswith(("model.vision_tower.", "model.embed_vision.")) for k in state_keys
    )

    if not has_audio:
        for key in [
            "audio_config",
            "audio_token_id",
            "boa_token_id",
            "eoa_token_id",
            "eoa_token_index",
        ]:
            _remove_config_attr(config, key)

    if not has_vision:
        for key in [
            "vision_config",
            "image_token_id",
            "video_token_id",
            "boi_token_id",
            "eoi_token_id",
            "vision_soft_tokens_per_image",
        ]:
            _remove_config_attr(config, key)

    if has_text and not has_audio and not has_vision and hasattr(config, "text_config"):
        text_dict = _config_to_plain_dict(getattr(config, "text_config"))
        for key, value in text_dict.items():
            _set_config_attr(config, key, value)

        _remove_config_attr(config, "text_config")
        _set_config_attr(config, "architectures", ["Gemma4ForCausalLM"])
        _set_config_attr(
            config,
            "model_type",
            text_dict.get("model_type", getattr(config, "model_type", "gemma4_text")),
        )


def merge_lora(model, lora_name, device):
    if lora_name is not None:
        from peft import PeftModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[green]start merging LoRA[/green]")
        model = PeftModel.from_pretrained(model, lora_name, device=device)
        model = model.merge_and_unload()
        print("[green]LoRA merged[/green]")
    return model


def apply_transformations(model, model_config):
    if not model_config:
        return model

    state_dict = model.state_dict()

    # キー変換
    if "key_transformations" in model_config:
        transformations = model_config["key_transformations"]
        for transform in transformations:
            if transform["type"] == "replace":
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k
                    for old, new in transform["replace"].items():
                        new_key = new_key.replace(old, new)
                    new_state_dict[new_key] = v
                state_dict = new_state_dict
            elif transform["type"] == "regex":
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = re.sub(transform["regex"], transform["replace"], k)
                    new_state_dict[new_key] = v
                state_dict = new_state_dict

    # レイヤー挿入
    if "insert_layers" in model_config:
        insertions = model_config["insert_layers"]
        for insertion in insertions:
            # 挿入元のモデルを読み込む
            from ..Utils.loaders import load_model

            source_model = load_model(
                insertion["source_model"],
                insertion.get("device", "cpu"),
                insertion.get("torch_dtype", torch.float32),
            )
            source_state_dict = source_model.state_dict()

            # 挿入元のテンソルを取得
            if insertion["source_key"] not in source_state_dict:
                print(
                    f"[yellow]Warning: Source key '{insertion['source_key']}' not found in '{insertion['source_model']}'. Skipping insertion.[/yellow]"
                )
                continue
            source_tensor = source_state_dict[insertion["source_key"]]

            # 挿入先のキーが存在するかチェック (存在する場合は常に上書き)
            state_dict[insertion["target_key"]] = source_tensor

    # 更新したstate_dictをモデルに設定
    if hasattr(model, "state_dict_data"):
        model.state_dict_data.clear()
        model.state_dict_data.update(state_dict)
    else:  # Hugging Face モデルの場合
        model.load_state_dict(state_dict)
    return model


def prepare_model_metadata(model_dict):
    """
    モデルのメタデータ (名前、設定など) を準備する。

    Args:
        model_dict (dict): モデルの設定 (config.yaml から読み込まれたもの)。

    Returns:
        MergeRequest: 必要なメタデータを含む構造体。
    """
    if isinstance(model_dict, MergeRequest):
        return model_dict
    raise TypeError(
        "prepare_model_metadata expects a MergeRequest. "
        "Config normalization should happen in load_config()."
    )


def _load_configured_model(model_name, request, merge_models_device, torch_dtype):
    from ..Utils.loaders import load_model

    model = load_model(
        model_name,
        merge_models_device,
        torch_dtype,
        diff_mode="delta",
    )
    model = merge_lora(model, None, merge_models_device)
    model_config = request.model_config.get(model_name, {})
    return apply_transformations(model, model_config)


def _resolve_model_sequence(
    model_names, request, merge_models_device, torch_dtype, recurrent_model=None
):
    models = []
    for model_name in model_names:
        if model_name.startswith("recurrent"):
            if recurrent_model is None:
                raise ValueError(
                    "Config references 'recurrent', but no previous merge result is available"
                )
            models.append(recurrent_model)
            continue
        models.append(
            _load_configured_model(
                model_name, request, merge_models_device, torch_dtype
            )
        )
    return models


def load_and_prepare_models(request, merge_models_device, torch_dtype, recurrent_model=None):
    """
    モデルをロードし、マージのための準備を行う。

    Args:
        request (MergeRequest): 正規化済みのマージ設定。
        merge_models_device (str): モデルをロードするデバイス。
        torch_dtype (torch.dtype): モデルのデータ型。
        recurrent_model (torch.nn.Module, optional): 前のマージ結果のモデル. Defaults to None.

    Returns:
        PreparedModels: ベース/サブモデルと速度設定を保持する構造体。
    """
    from ..Utils.layers import prepare_post_velocities, prepare_velocities

    request = prepare_model_metadata(request)
    right_model_names = request.sub_model_names
    if request.operation not in {"passthrough", "none"} and not right_model_names:
        raise ValueError(
            "right models are required unless operation is 'passthrough' or 'none'"
        )

    base_models = _resolve_model_sequence(
        request.base_model_names,
        request,
        merge_models_device,
        torch_dtype,
        recurrent_model=recurrent_model,
    )
    sub_models = _resolve_model_sequence(
        right_model_names,
        request,
        merge_models_device,
        torch_dtype,
        recurrent_model=recurrent_model,
    )

    velocities_config = request.velocities
    if velocities_config:
        if base_models:
            velocity = prepare_velocities(velocities_config, base_models[0].state_dict())
        else:
            velocity = None
    else:
        velocity = request.velocity

    post_velocities_config = request.post_velocities
    if post_velocities_config:
        post_velocity = prepare_post_velocities(
            post_velocities_config, base_models[0].state_dict()
        )
    else:
        post_velocity = request.post_velocity

    return PreparedModels(
        base_models=base_models,
        sub_models=sub_models,
        velocity=velocity,
        post_velocity=post_velocity,
    )
