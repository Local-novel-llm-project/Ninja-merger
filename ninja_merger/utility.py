# Copyright (c) 2024 TylorShine
#                    UmiseTokikaze
#                    Exveria
# license: Apache-2.0

import json
import os
from pathlib import Path
import yaml


def get_savename(path):
    path_path = Path(path)
    path_parts = path_path.parts
    if path_parts[0] == "/" or ":" in path_parts[0]:
        path_parts = path_parts[1:]
        
    return "-".join(path_parts)


def define_savename(base, sub, name_target_model, out_dir):
    base = get_savename(base)
    sub = get_savename(sub)
    model_name =  f"{sub}_{base}"
    if name_target_model == "lora":
        save_name = os.path.join(out_dir, "lora", model_name)
        return save_name

    if name_target_model is not None:
        target_name = get_savename(name_target_model)
        save_name = os.path.join(out_dir, target_name, model_name)
    else:
        save_name = os.path.join(out_dir, "vector", model_name)

    return save_name


def load_config(config_path):
    # YAMLファイルを優先する
    yaml_path = config_path.replace('.json', '.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # JSONファイルを読み込む
        with open(config_path, "r") as f:
            config = json.load(f)
    models_list = config["models"]
    name_target_model, name_target_model_type, name_lora_model = config["target_model"], config.get("target_model_type", "llm"), config.get("lora_model", None)
    
    return models_list, (name_target_model, name_target_model_type, name_lora_model)
