# Copyright (c) 2024 TylorShine
#                    UmiseTokikaze
#                    Exveria
# license: Apache-2.0

import gc

import argparse
import os
import sys

import torch
from tqdm import tqdm

from src.layer_utility import dump_layers, get_skip_layers, parse_layers
from src.merger import merge
from src.utility import define_savename, load_config
from src.model_loader import load_model, load_vlm_model, load_llava_model, load_left_right_models, load_processor, load_tokenizer

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='Merge models')
parser.add_argument('-c', '--config', type=str, default='model_config.yaml', help='Path to the JSON configuration file')
parser.add_argument('-o', '--out_dir', type=str, default='./merged_models', help='Directory to save the merged model')
parser.add_argument('-n', '--skip_layernorm', action='store_true', help='Skip layernorm during merging')
parser.add_argument('-dm', '--merge_models_device', type=str, default='cpu', help='Device for merging models')
parser.add_argument('-dt', '--target_model_device', type=str, default='cpu', help='Device for the target model')
parser.add_argument('-t', '--torch_dtype', type=str, default='bfloat16', help='Torch data type')
parser.add_argument('-r', '--recurrent_mode', type=bool, default=True , help='use target recurrent mode')
parser.add_argument('-d', '--dry_run', action='store_true', help='Dump processed layer infos without merging')
parser.add_argument('-l', '--save_only_last_model', action='store_true', help='Only last model saved')
parser.add_argument('--dump_layers', action='store_true', help='Dump model layers to a file instead of merging')
parser.add_argument('--include_layers', type=str, default=None, help='Comma-separated list of layers to include')
parser.add_argument('--exclude_layers', type=str, default=None, help='Comma-separated list of layers to exclude')


args = parser.parse_args()

# layernormをスキップするか
skip_layernorm = args.skip_layernorm

# デバイスの設定
merge_models_device = args.merge_models_device
target_model_device = args.target_model_device

# 精度の設定
torch_dtype = getattr(torch, args.torch_dtype)


include_layers = parse_layers(args.include_layers)
exclude_layers = parse_layers(args.exclude_layers)

def main():
    models_list, (name_target_model, name_target_model_type, lora_name) = load_config(args.config)
    target_model, target_state_dict = None, None
    is_llava_next = False
    if name_target_model is not None:
        if name_target_model_type == "vlm":
            target_model, target_state_dict = load_vlm_model(name_target_model, lora_name, target_model_device,torch_dtype)
        elif name_target_model_type == "llava-next":
            is_llava_next = True
            target_model, target_state_dict = load_vlm_model(name_target_model, lora_name, target_model_device,torch_dtype)
        elif name_target_model_type == "llava":
            target_model, target_state_dict, tokenizer, _ = load_llava_model(name_target_model, lora_name, target_model_device,torch_dtype)
        else:
            target_model, target_state_dict = load_model(name_target_model, lora_name, target_model_device,torch_dtype)

    if models_list is None and lora_name is not None:
        print(f"saving tokenizer from {name_target_model}...")
        if name_target_model_type == "llava":
            pass
        elif name_target_model_type == "vlm":
            tokenizer = load_processor(name_target_model)
        else:
            tokenizer = load_tokenizer(name_target_model)
        tokenizer.save_pretrained(define_savename(name_target_model, lora_name, "lora"))
        print("tokenizer save done")
        print("saving model...")
        target_model.save_pretrained(define_savename(name_target_model, lora_name, "lora"))
        print("model save done")
        sys.exit(0)


    for i, model_dict in enumerate(models_list):
        savename = define_savename(model_dict["left"], model_dict["right"], name_target_model)

        (base_model, base_state_dict), (sub_model, sub_state_dict), velocity = load_left_right_models(model_dict,merge_models_device)
        if i > 0 and target_model is not None:
            if model_dict["left"] == "recurrent":
                base_state_dict = target_model.state_dict()
                print("recurrent left")
            elif model_dict["right"] == "recurrent":
                sub_state_dict = target_model.state_dict()
                print("recurrent right")
            # elif args.recurrent_mode:
            #     target_state_dict = target_model.state_dict()
            #     print("recurrent mode")

        if args.dump_layers:
            dump_layers(base_state_dict, savename)
            continue

        if os.path.exists(savename) and not args.dry_run:
            print(f"skip: {savename}")
            continue
        else:
            print("\n ------------------------------------")

            include_layers = model_dict.get("include_layers", None)
            exclude_layers = model_dict.get("exclude_layers", None)
            drop_layers = model_dict.get("drop_layers", None)
            post_operation = model_dict.get("post_operation", "add")
            post_velocity = model_dict.get("post_velocity", 1.0)
            normalization = model_dict.get("normalization", "none")
            unmatch_size_layer_op = model_dict.get("unmatch_size_layer_op", "skip")

            skip_layers = get_skip_layers(target_state_dict, base_state_dict, sub_state_dict, unmatch_size_layer_op, is_llava_next=is_llava_next)

            print("start merge")
            if name_target_model is not None:
                target_state_dict = merge(skip_layernorm,target_state_dict, base_state_dict, sub_state_dict, velocity, post_velocity, skip_layers, model_dict["operation"], post_operation, normalization, include_layers, exclude_layers, drop_layers, unmatch_size_layer_op, args.dry_run, is_llava_next=is_llava_next)
                del base_state_dict, sub_state_dict
            else:
                base_state_dict = merge(skip_layernorm,target_state_dict, base_state_dict, sub_state_dict, velocity, post_velocity, skip_layers, model_dict["operation"], post_operation, normalization, include_layers, exclude_layers, drop_layers, unmatch_size_layer_op, args.dry_run, is_llava_next=is_llava_next)
                del sub_state_dict

            if args.dry_run:
                print("dry running: no save models")
                continue

            print(" === ")

            if i >= len(models_list) - 1 or not args.save_only_last_model:
                print("saving model...")
                if name_target_model is not None:
                    print(f"saving tokenizer from {name_target_model}...")
                    if name_target_model_type == "llava":
                        pass
                    elif name_target_model_type == "vlm":
                        tokenizer = load_processor(name_target_model)
                    else:
                        tokenizer = load_tokenizer(name_target_model)
                else:
                    print(f"saving tokenizer from {model_dict['left']}...")
                    if name_target_model_type == "llava":
                        pass
                    elif name_target_model_type == "vlm":
                        tokenizer = load_processor(model_dict['left'])
                    else:
                        tokenizer = load_tokenizer(model_dict['left'])
                tokenizer.save_pretrained(savename)
                print("tokenizer save done")

                del sub_model
                if name_target_model is not None:
                    del base_model
                    target_model.save_pretrained(savename)
                else:
                    base_model.save_pretrained(savename)
                    del base_model

                print("model save done")
            else:
                del base_model, sub_model

        gc.collect()


if __name__ == '__main__':
    main()