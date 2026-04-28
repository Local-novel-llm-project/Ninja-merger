import gc

import torch

from ._common import console
from modules.Utils.diff_artifact import is_sparse_diff_request, save_sparse_vector_artifact
from modules.Utils.layers import dump_layers
from modules.Utils.loaders import load_tokenizer
from modules.Utils.models import DummyModel, load_and_prepare_models
from modules.Utils.utility import scale_tensor_inplace, write_recipe_file


def dump_layers_for_configuration(
    request, target_model, merge_models_device, torch_dtype, save_name
):
    if target_model is not None:
        try:
            write_recipe_file(save_name, request)
            dump_layers(target_model.state_dict(), save_name)
        except Exception as error:
            console.print(f"[red]Error dumping layers: {error}[/red]")
        return

    console.print(
        "[yellow]Warning: Target model is None. Dumping layers from the first base model.[/yellow]"
    )
    prepared_models = None
    try:
        prepared_models = load_and_prepare_models(
            request, merge_models_device, torch_dtype
        )
        write_recipe_file(save_name, request)
        dump_layers(prepared_models.base_models[0].state_dict(), save_name)
    except Exception as error:
        console.print(f"[red]Error dumping layers: {error}[/red]")
    finally:
        if prepared_models is not None:
            del prepared_models
            gc.collect()


def _resolve_tokenizer_source(merged_model, request):
    if request.target_value_scalar == "recurrent":
        console.print("    Saving tokenizer from previous target model...")
        tokenizer_source = getattr(merged_model.config, "name_or_path", None)
        if tokenizer_source is None:
            tokenizer_source = getattr(merged_model.config, "_name_or_path", None)
        if tokenizer_source is None:
            raise ValueError(
                "Previous target model config has no tokenizer source information"
            )
        return tokenizer_source

    if request.target_value_scalar is None or request.target_value_scalar == "null":
        tokenizer_source = request.base_model_names[0]
        console.print(f"    Saving tokenizer from {tokenizer_source}...")
        return tokenizer_source

    console.print(f"    Saving tokenizer from {request.target_value_scalar}...")
    return request.target_value_scalar


def _save_tokenizer(save_name, merged_model, request):
    try:
        tokenizer_source = _resolve_tokenizer_source(merged_model, request)
        tokenizer = load_tokenizer(tokenizer_source)
        tokenizer.save_pretrained(save_name)
        console.print("    [green]Tokenizer saved.[/green]")
    except Exception as error:
        if request.target_value_scalar == "recurrent":
            console.print(
                f"[yellow]Warning: Could not save tokenizer for recurrent target: {error}[/yellow]"
            )
        else:
            console.print(f"[yellow]Warning: Failed to save tokenizer: {error}[/yellow]")


def _resolve_sparse_vector_source_reference(request):
    if not request.base_model_names:
        return None
    return request.base_model_names[0]


def _maybe_scale_small_tensors(target_model, current_use_scaling):
    if not current_use_scaling:
        return

    console.print(
        "    Checking for small values in the merged model and scaling if necessary..."
    )
    threshold = 1e-7
    scale_factor = 10000.0
    state_dict = target_model.state_dict()

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            continue
        if torch.any((torch.abs(value) < threshold) & (value != 0)):
            console.print(
                f"      Warning: Tensor '{key}' contains values smaller than {threshold}. Scaling..."
            )
            scaled = scale_tensor_inplace(
                value.detach().clone().float(), threshold, scale_factor
            ).to(device=value.device, dtype=value.dtype)
            value.copy_(scaled)


def save_merged_output(
    save_name,
    merged_model,
    current_use_scaling,
    request,
    *,
    base_state_snapshot=None,
):
    console.print(" === ")
    console.print("  [green]Saving model...[/green]")

    try:
        write_recipe_file(save_name, request)

        if is_sparse_diff_request(request):
            if current_use_scaling:
                console.print(
                    "[yellow]Skipping scaling for sparse vector artifacts to preserve exact tensor values.[/yellow]"
                )

            artifact_path, stored_count, skipped_zero_count = save_sparse_vector_artifact(
                save_name,
                merged_model,
                _resolve_sparse_vector_source_reference(request),
            )
            console.print(
                f"    Saved sparse vector artifact to {artifact_path} "
                f"({stored_count} stored tensors, {skipped_zero_count} zero tensors skipped)."
            )
            console.print("    [green]Model saved.[/green]")
            return

        _save_tokenizer(save_name, merged_model, request)
        _maybe_scale_small_tensors(merged_model, current_use_scaling)

        if isinstance(merged_model, DummyModel):
            console.print(f"    Saving model and config as a .pth file to {save_name}.pth")
            save_data = {
                "config": merged_model.config.to_dict(),
                "model": merged_model.state_dict(),
            }
            torch.save(save_data, f"{save_name}.pth")
        else:
            merged_model.save_pretrained(save_name)
        console.print("    [green]Model saved.[/green]")
    except Exception as error:
        console.print(f"[red]Error saving model/tokenizer: {error}[/red]")
