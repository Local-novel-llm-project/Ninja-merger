# modules/Utils/models.py
import re

import torch
from peft import PeftModel
from torch import nn


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

def merge_lora(model, lora_name, device):
    if lora_name is not None:
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
        dict: 必要なメタデータを含む辞書。
    """
    metadata = {}

    # モデル名 (左右)
    metadata["base_model_names"] = model_dict["left"]
    metadata["sub_model_names"] = model_dict["right"]

    # target 関連
    metadata["target_value"] = model_dict.get("target")
    metadata["target_value_scalar"] = (
        metadata["target_value"][0]
        if isinstance(metadata["target_value"], list)
        and len(metadata["target_value"]) == 1
        else metadata["target_value"]
    )
    metadata["is_llava_next"] = False  # デフォルト値
    if isinstance(metadata["target_value_scalar"], str) and metadata[
        "target_value_scalar"
    ].lower() in ["llava", "vlm", "llava-next"]:
        metadata["is_llava_next"] = True

    # マージ設定
    metadata["unmatch_size_layer_op"] = model_dict.get("unmatch_size_layer_op", "skip")
    metadata["include_layers"] = model_dict.get("include_layers", None)
    metadata["exclude_layers"] = model_dict.get("exclude_layers", None)
    metadata["drop_layers"] = model_dict.get("drop_layers", None)
    metadata["operation"] = model_dict.get("operation", "sub")
    metadata["post_operation"] = model_dict.get("post_operation", "add")
    metadata["preprocess"] = model_dict.get("preprocess", "none")
    metadata["post_preprocess"] = model_dict.get("post_preprocess", "none")
    metadata["post_velocity"] = model_dict.get("post_velocity", 1.0)
    metadata["normalization"] = model_dict.get("normalization", "none")
    metadata["force_merge_single"] = model_dict.get("force_merge_single", False)
    metadata["v2s_empty_default"] = model_dict.get("v2s_empty_default", "v1")
    metadata["v2s_single_default"] = model_dict.get("v2s_single_default", "auto")
    metadata["use_scaling"] = model_dict.get("use_scaling")

    # velocities, post_velocities は load_and_prepare_models で処理
    # model_config も同様

    return metadata


def load_and_prepare_models(model_dict, merge_models_device, torch_dtype, recurrent_model=None):
    """
    モデルをロードし、マージのための準備を行う。

    Args:
        model_dict (dict): モデルの設定 (config.yaml から読み込まれたもの)。
        merge_models_device (str): モデルをロードするデバイス。
        torch_dtype (torch.dtype): モデルのデータ型。
        recurrent_model (torch.nn.Module, optional): 前のマージ結果のモデル. Defaults to None.

    Returns:
        tuple: (base_models, sub_models, velocity, post_velocity) のタプル。
            base_models (list): ベースモデルのリスト。
            sub_models (list): サブモデルのリスト。
            velocity (dict or float or complex): レイヤーごとのvelocity(dict)または、モデル全体に適用されるvelocity(float, complex)。
            post_velocity (dict or float): レイヤーごとのpost_velocity(dict) または、モデル全体に適用されるpost_velocity(float)。
    """
    from ..Utils.layers import prepare_post_velocities, prepare_velocities
    from ..Utils.loaders import load_model
    from ..Utils.models import apply_transformations, merge_lora

    base_models = []
    for model_name in model_dict["left"]:
        if model_name.startswith("recurrent"):
            if recurrent_model is not None:
                base_models.append(recurrent_model)
        else:
            model = load_model(model_name, merge_models_device, torch_dtype)
            model = merge_lora(model, None, merge_models_device)
            model_config = model_dict.get("model_config", {}).get(model_name, {})
            model = apply_transformations(model, model_config)
            base_models.append(model)

    sub_models = []
    for model_name in model_dict["right"]:
        if model_name.startswith("recurrent"):
            if recurrent_model is not None:
                sub_models.append(recurrent_model)
        else:
            model = load_model(model_name, merge_models_device, torch_dtype)
            model = merge_lora(model, None, merge_models_device)
            model_config = model_dict.get("model_config", {}).get(model_name, {})
            model = apply_transformations(model, model_config)
            sub_models.append(model)

    velocities_config = model_dict.get("velocities", None)
    if velocities_config:
        if base_models:
            velocity = prepare_velocities(velocities_config, base_models[0].state_dict())
        else:
            velocity = None
    else:
        velocity = model_dict.get("velocity")

    post_velocities_config = model_dict.get("post_velocities", None)
    if post_velocities_config:
        post_velocity = prepare_post_velocities(
            post_velocities_config, base_models[0].state_dict()
        )
    else:
        post_velocity = model_dict.get("post_velocity", 1.0)

    return base_models, sub_models, velocity, post_velocity
