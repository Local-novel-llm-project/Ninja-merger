import gc

import torch
from rich.panel import Panel
from rich.table import Table

from ._common import console
from .merge_output import dump_layers_for_configuration, save_merged_output
from modules.Merger.factory import MergerFactory
from modules.Utils.layers import get_skip_layers, parse_layers
from modules.Utils.loaders import load_config, load_model
from modules.Utils.merge_types import (
    MergeContext,
    MergeExecutionError,
    MergeStepResult,
)
from modules.Utils.models import load_and_prepare_models, prepare_model_metadata
from modules.Utils.utility import define_savename, output_artifact_exists


def _request_display_name(index, request):
    return request.name or f"config_{index + 1}"


def _summarize_model_refs(model_names):
    if not model_names:
        return "-"
    if len(model_names) == 1:
        return model_names[0]
    return f"{model_names[0]} (+{len(model_names) - 1})"


def _summarize_target(target_value_scalar):
    if target_value_scalar is None or target_value_scalar == "null":
        return "new model"
    return str(target_value_scalar)


def _print_run_overview(app_config, options):
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Config", options.config)
    table.add_row("Entries", str(len(app_config.models)))
    table.add_row("Output", options.out_dir)
    table.add_row("Merge device", options.merge_models_device)
    table.add_row("Target device", options.target_model_device)
    table.add_row("Dtype", options.torch_dtype)
    table.add_row("Recurrent", "enabled" if options.recurrent_mode else "disabled")
    table.add_row(
        "Save mode",
        "final only" if options.save_only_last_model else "every step",
    )
    if options.dump_layers:
        table.add_row("Mode", "dump layers")
    elif options.dry_run:
        table.add_row("Mode", "dry run")
    console.set_run_overview(
        Panel(table, title="[bold]Run Overview[/bold]", border_style="blue")
    )


def _print_request_overview(
    index,
    total_configs,
    request,
    output_display,
    current_use_scaling,
    effective_include_layers,
    effective_exclude_layers,
):
    details = Table(show_header=False, box=None, padding=(0, 1))
    details.add_column(style="bold magenta")
    details.add_column()
    details.add_row("Name", _request_display_name(index, request))
    details.add_row("Operation", request.operation)
    details.add_row("Left", _summarize_model_refs(request.base_model_names))
    details.add_row("Right", _summarize_model_refs(request.sub_model_names))
    details.add_row("Target", _summarize_target(request.target_value_scalar))
    details.add_row("Output", output_display)
    details.add_row("Scaling", "on" if current_use_scaling else "off")
    if effective_include_layers:
        details.add_row("Include", str(effective_include_layers))
    if effective_exclude_layers:
        details.add_row("Exclude", str(effective_exclude_layers))

    console.set_request_overview(
        Panel(
            details,
            title=f"[bold]Configuration {index + 1}/{total_configs}[/bold]",
            border_style="cyan",
        )
    )


def _status_label(status):
    labels = {
        "saved": "[green]saved[/green]",
        "computed_only": "[cyan]in memory[/cyan]",
        "dry_run": "[yellow]dry run[/yellow]",
        "dumped_layers": "[blue]dumped layers[/blue]",
        "skipped_existing": "[yellow]skipped existing[/yellow]",
        "load_failed": "[red]load failed[/red]",
        "merge_failed": "[red]merge failed[/red]",
    }
    return labels.get(status, status)


def _print_run_summary(step_results):
    counts = {}
    for step_result in step_results:
        counts[step_result.status] = counts.get(step_result.status, 0) + 1

    count_table = Table(show_header=False, box=None, padding=(0, 1))
    count_table.add_column(style="bold green")
    count_table.add_column()
    for status, count in sorted(counts.items()):
        count_table.add_row(status, str(count))

    console.print(
        Panel(count_table, title="[bold]Run Summary[/bold]", border_style="green")
    )

    detail_table = Table(title="Per-step Results", header_style="bold magenta")
    detail_table.add_column("Step", justify="right")
    detail_table.add_column("Name")
    detail_table.add_column("Operation")
    detail_table.add_column("Status")
    detail_table.add_column("Output", overflow="fold")

    for step_index, step_result in enumerate(step_results, start=1):
        detail_table.add_row(
            str(step_index),
            step_result.request_name or "-",
            step_result.operation or "-",
            _status_label(step_result.status),
            step_result.save_stem or "-",
        )

    console.print(detail_table)


def _resolve_effective_layer_filters(request, cli_include_layers, cli_exclude_layers):
    effective_include_layers = (
        cli_include_layers
        if cli_include_layers is not None
        else request.include_layers
    )
    effective_exclude_layers = (
        cli_exclude_layers
        if cli_exclude_layers is not None
        else request.exclude_layers
    )
    return effective_include_layers, effective_exclude_layers


def _resolve_target_model(
    current_target_model, request, target_model_device, torch_dtype
):
    if request.target_value_scalar == "recurrent":
        if current_target_model is None:
            console.print(
                "[red]Error: 'recurrent' target specified, but no previous merge has been performed.[/red]"
            )
            raise MergeExecutionError("recurrent target requested before any merge")
        console.print("  Using 'recurrent' target (result of previous merge).")
        return current_target_model

    if request.target_value_scalar is None or request.target_value_scalar == "null":
        console.print("  Using 'null' target (creating a new model).")
        return None

    try:
        console.print(f"  Loading target model: {request.target_value_scalar}")
        return load_model(request.target_value_scalar, target_model_device, torch_dtype)
    except Exception as error:
        console.print(
            f"[red]Error loading target model '{request.target_value_scalar}': {error}[/red]"
        )
        raise MergeExecutionError(str(error)) from error


def _print_vocab_size_mismatch_hint(error):
    error_text = str(error)
    if (
        "The size of tensor a" in error_text
        and "must match the size of tensor b" in error_text
    ):
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


def _load_merge_inputs(request, merge_models_device, torch_dtype, target_model):
    try:
        return load_and_prepare_models(
            request, merge_models_device, torch_dtype, target_model
        )
    except Exception as error:
        console.print(f"[red]Error loading base/sub models: {error}[/red]")
        _print_vocab_size_mismatch_hint(error)
        return None


def _build_merge_context(
    request,
    options,
    merge_inputs,
    resolved_target_model,
    effective_include_layers,
    effective_exclude_layers,
):
    skip_layers = get_skip_layers(
        resolved_target_model,
        merge_inputs.base_models,
        merge_inputs.sub_models,
        request.unmatch_size_layer_op,
        is_llava_next=request.is_llava_next,
    )

    return MergeContext(
        request=request,
        skip_layernorm=options.skip_layernorm,
        target_model=resolved_target_model,
        base_models=merge_inputs.base_models,
        sub_models=merge_inputs.sub_models,
        velocity=merge_inputs.velocity,
        post_velocity=merge_inputs.post_velocity,
        skip_layers=skip_layers,
        include_layers=effective_include_layers,
        exclude_layers=effective_exclude_layers,
    )


def _run_merge_operation(merger_factory, context):
    console.print("  [green]Starting merge...[/green]")
    console.print(f"  Operation: {context.request.operation}")
    console.print(f"  Velocity: {context.velocity}")

    if context.target_model is None:
        console.print("  Post-operation: N/A (target is None)")
    else:
        console.print(f"  Post-operation: {context.request.post_operation}")

    try:
        merger = merger_factory.create_merger(context)
        return merger.merge()
    except Exception as error:
        console.print(f"[red]Error during merge: {error}[/red]")
        return None


def _process_merge_request(
    index,
    total_configs,
    request,
    options,
    should_save_output,
    default_use_scaling,
    cli_include_layers,
    cli_exclude_layers,
    merger_factory,
    current_target_model,
    torch_dtype,
):
    console.rule(
        f"[bold blue]Processing Merge Configuration {index + 1}/{total_configs}[/bold blue]"
    )
    console.start_request(
        index,
        total_configs,
        _request_display_name(index, request),
        request.operation,
    )

    request = prepare_model_metadata(request)
    current_use_scaling = (
        request.use_scaling
        if request.use_scaling is not None
        else default_use_scaling
    )
    effective_include_layers, effective_exclude_layers = (
        _resolve_effective_layer_filters(
            request, cli_include_layers, cli_exclude_layers
        )
    )
    output_display = "(in memory only)"

    resolved_target_model = _resolve_target_model(
        current_target_model,
        request,
        options.target_model_device,
        torch_dtype,
    )

    save_stem = None
    if options.dump_layers or should_save_output:
        save_stem = define_savename(
            request.base_model_names,
            request.sub_model_names,
            request.target_value,
            options.out_dir,
            index,
            request,
        )
        output_display = save_stem

    _print_request_overview(
        index,
        total_configs,
        request,
        output_display,
        current_use_scaling,
        effective_include_layers,
        effective_exclude_layers,
    )

    if options.dump_layers:
        dump_layers_for_configuration(
            request,
            resolved_target_model,
            options.merge_models_device,
            torch_dtype,
            save_stem,
        )
        return MergeStepResult(
            status="dumped_layers",
            target_model=resolved_target_model,
            save_stem=save_stem,
            request_name=_request_display_name(index, request),
            operation=request.operation,
        )

    if (
        should_save_output
        and save_stem is not None
        and output_artifact_exists(save_stem)
        and not options.dry_run
    ):
        console.print(f"  [yellow]Skipping: {save_stem} (already exists)[/yellow]")
        return MergeStepResult(
            status="skipped_existing",
            target_model=resolved_target_model,
            save_stem=save_stem,
            request_name=_request_display_name(index, request),
            operation=request.operation,
        )

    merge_inputs = _load_merge_inputs(
        request,
        options.merge_models_device,
        torch_dtype,
        resolved_target_model,
    )
    if merge_inputs is None:
        return MergeStepResult(
            status="load_failed",
            target_model=resolved_target_model,
            save_stem=save_stem,
            message="failed to load merge inputs",
            request_name=_request_display_name(index, request),
            operation=request.operation,
        )

    try:
        context = _build_merge_context(
            request,
            options,
            merge_inputs,
            resolved_target_model,
            effective_include_layers,
            effective_exclude_layers,
        )
        merged_model = _run_merge_operation(merger_factory, context)
        if merged_model is None:
            return MergeStepResult(
                status="merge_failed",
                target_model=resolved_target_model,
                save_stem=save_stem,
                message="merge returned no model",
                request_name=_request_display_name(index, request),
                operation=request.operation,
            )

        if options.dry_run:
            console.print("  [yellow]Dry run: No models saved.[/yellow]")
            return MergeStepResult(
                status="dry_run",
                target_model=merged_model,
                save_stem=save_stem,
                merged_model=merged_model,
                request_name=_request_display_name(index, request),
                operation=request.operation,
            )

        if not should_save_output:
            console.print(
                "  [cyan]Intermediate result kept in memory only (--save-only-last-model).[/cyan]"
            )
            return MergeStepResult(
                status="computed_only",
                target_model=merged_model,
                save_stem=None,
                merged_model=merged_model,
                request_name=_request_display_name(index, request),
                operation=request.operation,
            )

        save_merged_output(
            save_stem,
            merged_model,
            current_use_scaling,
            request,
        )
        return MergeStepResult(
            status="saved",
            target_model=merged_model,
            save_stem=save_stem,
            merged_model=merged_model,
            request_name=_request_display_name(index, request),
            operation=request.operation,
        )
    finally:
        del merge_inputs
        gc.collect()


def run_merge(options):
    torch_dtype = getattr(torch, options.torch_dtype)
    cli_include_layers = parse_layers(options.include_layers)
    cli_exclude_layers = parse_layers(options.exclude_layers)

    app_config = load_config(options.config)
    target_model = None
    merger_factory = MergerFactory()
    step_results = []

    console.start_live()
    try:
        _print_run_overview(app_config, options)
        console.configure_run_progress(len(app_config.models))

        for index, request in enumerate(app_config.models):
            step_result = _process_merge_request(
                index,
                len(app_config.models),
                request,
                options,
                not options.save_only_last_model or index == len(app_config.models) - 1,
                app_config.use_scaling,
                cli_include_layers,
                cli_exclude_layers,
                merger_factory,
                target_model,
                torch_dtype,
            )
            step_results.append(step_result)
            console.advance_request(step_result.status)
            target_model = step_result.target_model if options.recurrent_mode else None
    finally:
        console.stop_live()

    if step_results:
        _print_run_summary(step_results)
