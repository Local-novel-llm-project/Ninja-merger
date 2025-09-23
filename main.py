# main.py

# Copyright (c) 2024 TylorShine
#                    UmiseTokikaze
#                    Exveria
# license: Apache-2.0

import argparse
import gc
import os
import sys

import torch
from rich.console import Console
from transformers import AutoTokenizer

from modules.Merger.factory import MergerFactory
from modules.Utils.layers import dump_layers, get_skip_layers, parse_layers
from modules.Utils.loaders import (
    load_config,
    load_model,
    load_tokenizer,
)
from modules.Utils.models import (
    load_and_prepare_models,
    prepare_model_metadata,
)
from modules.Utils.models import DummyModel
from modules.Utils.utility import define_savename, scale_tensor_inplace

console = Console()


def main(args):
    skip_layernorm = args.skip_layernorm
    merge_models_device = args.merge_models_device
    target_model_device = args.target_model_device
    torch_dtype = getattr(torch, args.torch_dtype)
    include_layers = parse_layers(args.include_layers)
    exclude_layers = parse_layers(args.exclude_layers)

    models_list, use_scaling = load_config(args.config)
    target_model = None
    # is_llava_next = False # prepare_model_metadata で取得

    for i, model_dict in enumerate(models_list):
        console.rule(
            f"[bold blue]Processing Merge Configuration {i + 1}/{len(models_list)}[/bold blue]"
        )

        # 1. メタデータの準備
        metadata = prepare_model_metadata(model_dict)

        target_value = metadata["target_value"]  # model_dict.get("target")

        if target_value == "recurrent":
            if target_model is None:
                console.print(
                    "[red]Error: 'recurrent' target specified, but no previous merge has been performed.[/red]"
                )
                sys.exit(1)
            console.print("  Using 'recurrent' target (result of previous merge).")

        elif target_value is None or target_value == "null":
            target_model = None
            console.print("  Using 'null' target (creating a new model).")

        else:
            try:
                console.print(f"  Loading target model: {target_value}")
                target_model = load_model(
                    target_value, target_model_device, torch_dtype
                )
                # if target_value.lower() in ["llava", "vlm", "llava-next"]:
                #     is_llava_next = True
            except Exception as e:
                console.print(
                    f"[red]Error loading target model '{target_value}': {e}[/red]"
                )
                sys.exit(1)

        savename = define_savename(
            metadata["base_model_names"],
            metadata["sub_model_names"],
            metadata["target_value"],
            args.out_dir,
            i,
            model_dict,
        )
        console.print(f"  Saving to: [cyan]{savename}[/cyan]")

        # try:
        #     base_models, sub_models, velocity, post_velocity = prepare_models_for_merging( # 削除
        #         model_dict, merge_models_device, torch_dtype # 削除
        #     ) # 削除

        #     console.print(
        #         f"Base models: {[model.config.name_or_path for model in base_models]}"
        #     )
        #     if base_models:
        #         console.print(
        #             f"Base models[0] config: {base_models[0].config}"
        #         )  # 最初の base_model の config を表示
        #         console.print(
        #             f"Base models[0] first layer keys: {list(base_models[0].state_dict().keys())[:5]}"
        #         )  # 最初の数レイヤーのキーを表示

        #     console.print(f"Sub models: {sub_models}")  # sub_models の内容を表示
        #     if sub_models:
        #         console.print(
        #             f"Sub models[0] config: {sub_models[0].config}"
        #         )  # 最初の sub_model の config を表示
        #         console.print(
        #             f"Sub models[0] first layer keys: {list(sub_models[0].state_dict().keys())[:5]}"
        #         )  # 最初の数レイヤーのキーを表示

        #     console.print(
        #         f"  Base models: {[model.config.name_or_path for model in base_models]}"
        #     )
        #     console.print(
        #         f"  Sub models: {[model.config.name_or_path for model in sub_models]}"
        #     )
        # except Exception as e:
        #     console.print(f"[red]Error loading base/sub models: {e}[/red]")
        #     continue

        if args.dump_layers:
            if target_model is None:
                # base_models, sub_models, _, _ = load_and_prepare_models(model_dict, merge_models_device, torch_dtype) # コメントアウト
                console.print(
                    "[yellow]Warning: Target model is None. Dumping layers from the first base model.[/yellow]"
                )
                try:
                    # dump_layers(base_models[0].state_dict(), savename) # コメントアウト
                    # モデルをロードする
                    base_models, _, _, _ = load_and_prepare_models(
                        model_dict, merge_models_device, torch_dtype
                    )
                    dump_layers(base_models[0].state_dict(), savename)
                except Exception as e:
                    console.print(f"[red]Error dumping layers: {e}[/red]")
            else:
                try:
                    dump_layers(target_model.state_dict(), savename)
                except Exception as e:
                    console.print(f"[red]Error dumping layers: {e}[/red]")
            continue

        if os.path.exists(savename) and not args.dry_run:
            console.print(f"  [yellow]Skipping: {savename} (already exists)[/yellow]")
            continue

        # try:
        #     # unmatch_size_layer_op を取得
        #     unmatch_size_layer_op = model_dict.get("unmatch_size_layer_op", "skip")
        #     console.print(f"  Unmatch size layer operation: {unmatch_size_layer_op}")

        #     # load_left_right_models に unmatch_size_layer_op を渡す
        #     base_models, sub_models, velocity = prepare_models_for_merging(
        #         model_dict, merge_models_device, torch_dtype
        #     )

        # except Exception as e:
        #     console.print(f"[red]Error loading base/sub models: {e}[/red]")
        #     # 語彙サイズの不一致エラーの場合、ヒントを表示
        #     if "The size of tensor a" in str(
        #         e
        #     ) and "must match the size of tensor b" in str(e):
        #         console.print(
        #             "[yellow]This error may be related to vocabulary size mismatch.[/yellow]"
        #         )
        #         console.print("[yellow]Try using the following options:[/yellow]")
        #         console.print(
        #             "1. Set unmatch_size_layer_op: 'only_common_range' in your config"
        #         )
        #         console.print(
        #             "2. Or use --exclude_layers 'embed_tokens,lm_head' to exclude embedding layers"
        #         )

        #     continue

        # console.print("\n ------------------------------------") # コメントアウト

        # 2. モデルのロードと準備
        try:
            base_models, sub_models, velocity, post_velocity = load_and_prepare_models(
                model_dict, merge_models_device, torch_dtype, target_model
            )
        except Exception as e:
            console.print(f"[red]Error loading base/sub models: {e}[/red]")
            # 語彙サイズの不一致エラーの場合、ヒントを表示
            if "The size of tensor a" in str(
                e
            ) and "must match the size of tensor b" in str(e):
                console.print(
                    "[yellow]This error may be related to vocabulary size mismatch.[/yellow]"
                )
                console.print("[yellow]Try using the following options:[/yellow]")
                console.print(
                    "1. Set unmatch_size_layer_op: 'only_common_range' in your config"
                )
                console.print(
                    "2. Or use --exclude_layers 'embed_tokens,lm_head' to exclude embedding layers"
                )
            continue

        # include_layers = model_dict.get("include_layers", None) # metadata に移動
        # exclude_layers = model_dict.get("exclude_layers", None) # metadata に移動
        # drop_layers = model_dict.get("drop_layers", None) # metadata に移動
        # operation = model_dict.get("operation", "sub") # metadata に移動
        # post_operation = model_dict.get("post_operation", "add") # metadata に移動
        # preprocess = model_dict.get("preprocess", "none") # metadata に移動
        # post_preprocess = model_dict.get("post_preprocess", "none") # metadata に移動
        # post_velocity = model_dict.get("post_velocity", 1.0) # metadata と load_and_prepare_models
        # normalization = model_dict.get("normalization", "none") # metadata に移動
        # unmatch_size_layer_op = model_dict.get("unmatch_size_layer_op", "skip") # metadataに移動
        # force_merge_single = model_dict.get("force_merge_single", False) # metadataに移動
        # v2s_empty_default = model_dict.get("v2s_empty_default", "v1") # metadataに移動
        # v2s_single_default = model_dict.get("v2s_single_default", "auto") # metadataに移動
        # use_scaling = model_dict.get("use_scaling", True) # metadataに移動

        # 3. skip_layers の取得
        try:
            skip_layers = get_skip_layers(
                target_model,
                base_models,
                sub_models,
                metadata["unmatch_size_layer_op"],
                is_llava_next=metadata["is_llava_next"],
            )
        except Exception as e:
            console.print(f"[red]Error getting skip layers: {e}[/red]")
            continue

        console.print("  [green]Starting merge...[/green]")
        console.print(f"  Operation: {metadata['operation']}")
        console.print(f"  Velocity: {velocity}")

        if target_model is None:
            console.print("  Post-operation: N/A (target is None)")
        else:
            console.print(f"  Post-operation: {metadata['post_operation']}")

        try:
            # MergerFactory を使用して Merger インスタンスを作成
            merger_factory = MergerFactory()
            merger = merger_factory.create_merger(
                metadata["operation"],
                skip_layernorm,
                target_model,
                base_models,
                sub_models,
                velocity,
                post_velocity,  # post_velocity は load_and_prepare_models から返される
                skip_layers,
                metadata["post_operation"],
                metadata["preprocess"],
                metadata["post_preprocess"],
                metadata["normalization"],
                metadata["include_layers"],
                metadata["exclude_layers"],
                metadata["drop_layers"],
                metadata["unmatch_size_layer_op"],
                is_llava_next=metadata["is_llava_next"],
                force_merge_single=metadata["force_merge_single"],
                v2s_empty_default=metadata["v2s_empty_default"],
                v2s_single_default=metadata["v2s_single_default"],
                model_dict=model_dict,  # model_dict も渡すことを忘れない
            )
            # 作成した Merger インスタンスの merge メソッドを呼び出す
            target_model = merger.merge()

        except Exception as e:
            console.print(f"[red]Error during merge: {e}[/red]")
            continue

        if args.dry_run:
            console.print("  [yellow]Dry run: No models saved.[/yellow]")
            continue

        console.print(" === ")
        console.print("  [green]Saving model...[/green]")

        try:
            try:
                if target_value == "recurrent":
                    console.print("    Saving tokenizer from previous target model...")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            target_model.config.name_or_path, trust_remote_code=True
                        )
                        tokenizer.save_pretrained(savename)
                        console.print("    [green]Tokenizer saved.[/green]")
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Could not save tokenizer for recurrent target: {e}[/yellow]"
                        )

                elif target_value is None or target_value == "null":
                    console.print(
                        f"    Saving tokenizer from {metadata['base_model_names'][0]}..."
                    )
                    tokenizer = load_tokenizer(metadata["base_model_names"][0])
                    tokenizer.save_pretrained(savename)
                    console.print("    [green]Tokenizer saved.[/green]")
                else:
                    console.print(f"    Saving tokenizer from {target_value}...")
                    tokenizer = load_tokenizer(target_value)
                    tokenizer.save_pretrained(savename)
                    console.print("    [green]Tokenizer saved.[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to save tokenizer: {e}[/yellow]")

            if use_scaling:
                console.print(
                    "    Checking for small values in the merged model and scaling if necessary..."
                )
                threshold = 1e-7
                scale_factor = 10000.0
                if target_model is None:
                    state_dict = base_models[
                        0
                    ].state_dict()  # targetがないので、Leftから取得
                else:
                    state_dict = target_model.state_dict()

                for key, value in state_dict.items():
                    if value.dtype != torch.bfloat16:
                        if torch.any(torch.abs(value) < threshold):
                            console.print(
                                f"      Warning: Tensor '{key}' contains values smaller than {threshold}. Scaling..."
                            )
                            value = scale_tensor_inplace(
                                value.float(), threshold, scale_factor
                            )
                            # state_dict[key] = value.bfloat16()
                            # targetがある場合は、target_modelに値を戻す。
                            if target_model is not None:
                                target_model.state_dict()[key] = value.bfloat16()
                            else:  # targetがない場合は、base_modelに直接戻す
                                base_models[0].state_dict()[key] = value.bfloat16()

            if target_model is None:
                base_models[0].save_pretrained(savename)
            elif isinstance(target_model, DummyModel):
                console.print(f"    Saving model and config as a .pth file to {savename}.pth")
                save_data = {
                    "config": target_model.config.to_dict(),
                    "model": target_model.state_dict(),
                }
                torch.save(save_data, f"{savename}.pth")
            else:
                target_model.save_pretrained(savename)
            console.print("    [green]Model saved.[/green]")

        except Exception as e:
            console.print(f"[red]Error saving model/tokenizer: {e}[/red]")
            continue

        del base_models
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge models")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="model_config.yaml",
        help="Path to the JSON configuration file",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="./merged_models",
        help="Directory to save the merged model",
    )
    parser.add_argument(
        "-n",
        "--skip_layernorm",
        action="store_true",
        help="Skip layernorm during merging",
    )
    parser.add_argument(
        "-dm",
        "--merge_models_device",
        type=str,
        default="cpu",
        help="Device for merging models",
    )
    parser.add_argument(
        "-dt",
        "--target_model_device",
        type=str,
        default="cpu",
        help="Device for the target model",
    )
    parser.add_argument(
        "-t", "--torch_dtype", type=str, default="bfloat16", help="Torch data type"
    )
    parser.add_argument(
        "-r",
        "--recurrent_mode",
        type=bool,
        default=True,
        help="use target recurrent mode",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        action="store_true",
        help="Dump processed layer infos without merging",
    )
    parser.add_argument(
        "-l",
        "--save_only_last_model",
        action="store_true",
        help="Only last model saved",
    )
    parser.add_argument(
        "--dump_layers",
        action="store_true",
        help="Dump model layers to a file instead of merging",
    )
    parser.add_argument(
        "--include_layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to include",
    )
    parser.add_argument(
        "--exclude_layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to exclude",
    )

    args = parser.parse_args()

    main(args)
