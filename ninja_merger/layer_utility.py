# Copyright (c) 2024 TylorShine
#                    UmiseTokikaze
#                    Exveria
# license: Apache-2.0

import os
import re
from rich import print


def parse_layer_specifications(layers_str):
    if layers_str is None:
        return [], []
    ranges = []
    specific_layers = []
    # parts = layers_str.split(',')
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


# include_layersとexclude_layersを解析する関数
def parse_layers(layers_str):
    if layers_str is None:
        return None
    return layers_str.split(",")


def get_skip_layers(
    target_model,
    base_models,
    sub_model,
    unmatch_size_layer_op,
    is_llava_next=False,
):
    # それぞれのstate_dictからキーとテンソルのサイズを取得し、比較する。もしも合致しないものがあればそれだけを表示する。
    skip_layers = []
    if target_model is None:
        base_state_dicts = [b.state_dict() for b in base_models]
        target = base_state_dicts[0]
    else:
        target = target_model.state_dict()
        base_state_dicts = [b.state_dict() for b in base_models]
    sub_state_dict = sub_model.state_dict()
        
    for target_key in target.keys():
        # orig_target_key = target_key
        if is_llava_next:
            target_key = target_key.replace("language_model.", "", 1)

        if target_key not in sub_state_dict.keys():
            print(f"[yellow] left key not found: {target_key}, skip...[/yellow]")
            skip_layers.append(target_key)
            continue
        for base_state_dict in base_state_dicts:
            if target_key not in base_state_dict.keys():
                print(f"[yellow] base key not found: {target_key}, skip...[/yellow]")
                skip_layers.append(target_key)
                continue
            if base_state_dict[target_key].size() != sub_state_dict[target_key].size():
                print("\n ------------------------------------")
                print(f"[bold red] mismatch size: {target_key}[/bold red]")
                print("[bold]tensor size[/bold]")
                print(f"  base: {base_state_dict[target_key].size()}")
                print(f"  right   : {sub_state_dict[target_key].size()}")
                if unmatch_size_layer_op == "skip":
                    skip_layers.append(target_key)
                    break

    if len(skip_layers) == 0:
        print()
        print("[green]match all layer tensor size[/green]")
    else:
        print(skip_layers)
    return skip_layers


def dump_layers(model_state_dict, savename):
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    layer_names = list(model_state_dict.keys())
    with open(f"{savename}_layers.txt", "w") as f:
        f.write("\n".join(layer_names))
    print(f"Layer names dumped to [cyan]{savename}_layers.txt[/cyan]")
    print("[bold]Layer names:[/bold]")
    print("\n".join(layer_names))
