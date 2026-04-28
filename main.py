# main.py

# Copyright (c) 2024 TylorShine
#                    UmiseTokikaze
#                    Exveria
# license: Apache-2.0

import argparse
from typing import Sequence

from modules.Services.merge_runner import (
    MergeExecutionError,
    MergeRunnerOptions,
    run_merge,
)

TORCH_DTYPE_CHOICES = ("float16", "bfloat16", "float32", "float64")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge model checkpoints from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python main.py -c g4novel_vector.yaml --merge-models-device cuda:0"
        ),
    )

    input_group = parser.add_argument_group("Input / Output")
    execution_group = parser.add_argument_group("Execution")
    filter_group = parser.add_argument_group("Layer Filters")
    debug_group = parser.add_argument_group("Diagnostics")

    input_group.add_argument(
        "-c",
        "--config",
        type=str,
        default="model_config.yaml",
        help="Path to the YAML merge configuration file",
    )
    input_group.add_argument(
        "-o",
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default="./merged_models",
        help="Directory for merged model outputs",
    )
    execution_group.add_argument(
        "-n",
        "--skip-layernorm",
        "--skip_layernorm",
        dest="skip_layernorm",
        action="store_true",
        help="Skip LayerNorm-like layers during merging",
    )
    execution_group.add_argument(
        "-dm",
        "--merge-models-device",
        "--merge_models_device",
        dest="merge_models_device",
        type=str,
        default="cpu",
        help="Device used to load base/sub models",
    )
    execution_group.add_argument(
        "-dt",
        "--target-model-device",
        "--target_model_device",
        dest="target_model_device",
        type=str,
        default="cpu",
        help="Device used to load explicit target models",
    )
    execution_group.add_argument(
        "-t",
        "--torch-dtype",
        "--torch_dtype",
        dest="torch_dtype",
        type=str,
        choices=TORCH_DTYPE_CHOICES,
        default="bfloat16",
        help="Torch dtype used when loading transformer models",
    )
    execution_group.add_argument(
        "-r",
        "--recurrent-mode",
        "--recurrent_mode",
        dest="recurrent_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow later config entries to reference previous merge results via 'recurrent'",
    )
    execution_group.add_argument(
        "-d",
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Run the merge pipeline without writing output artifacts",
    )
    execution_group.add_argument(
        "-l",
        "--save-only-last-model",
        "--save_only_last_model",
        dest="save_only_last_model",
        action="store_true",
        help="Keep intermediate merge results in memory and save only the final step",
    )
    debug_group.add_argument(
        "--dump-layers",
        "--dump_layers",
        dest="dump_layers",
        action="store_true",
        help="Write layer names for the selected target/base model instead of merging",
    )
    filter_group.add_argument(
        "--include-layers",
        "--include_layers",
        dest="include_layers",
        type=str,
        default=None,
        help="Comma-separated layer filters to include",
    )
    filter_group.add_argument(
        "--exclude-layers",
        "--exclude_layers",
        dest="exclude_layers",
        type=str,
        default=None,
        help="Comma-separated layer filters to exclude",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> MergeRunnerOptions:
    parser = build_parser()
    namespace = parser.parse_args(argv)

    if namespace.dump_layers and namespace.dry_run:
        parser.error("--dump-layers already avoids saving merged artifacts, so it cannot be combined with --dry-run")

    return MergeRunnerOptions(
        config=namespace.config,
        out_dir=namespace.out_dir,
        skip_layernorm=namespace.skip_layernorm,
        merge_models_device=namespace.merge_models_device,
        target_model_device=namespace.target_model_device,
        torch_dtype=namespace.torch_dtype,
        recurrent_mode=namespace.recurrent_mode,
        dry_run=namespace.dry_run,
        save_only_last_model=namespace.save_only_last_model,
        dump_layers=namespace.dump_layers,
        include_layers=namespace.include_layers,
        exclude_layers=namespace.exclude_layers,
    )


def main(options: MergeRunnerOptions | None = None) -> None:
    runner_options = options or parse_args()
    try:
        run_merge(runner_options)
    except MergeExecutionError as error:
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
