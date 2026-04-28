# Merger/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import types
from typing import Any, Dict, List, Tuple, Union

import torch
from rich.panel import Panel
from rich.table import Table

from ..common_console import console
from ..Utils.layers import (
    is_layer_dropped,
    is_layer_included,
    parse_layer_specifications,
)
from ..Utils.merge_types import MergeContext, MergeLayerContext
from ..Utils.models import prune_config_for_dropped_layers
from ..Utils.utility import (
    filter_state_dict_keys,
    prepare_tensor_slices,
    to_complex_tensor,
)


@dataclass(frozen=True)
class LayerDecision:
    target_key: str
    status: str
    message: str


class Merger(ABC):
    """Abstract base class for all model mergers."""

    def __init__(self, context: MergeContext):
        self.context = context
        self.request = context.request
        self.skip_layernorm = context.skip_layernorm
        self.target_model = context.target_model
        self.base_models = context.base_models
        self.sub_models = context.sub_models
        self.velocity = context.velocity
        self.post_velocity = context.post_velocity
        self.skip_layers = context.skip_layers
        self.operation = self.request.operation
        self.post_operation = self.request.post_operation
        self.preprocess = self.request.preprocess
        self.post_preprocess = self.request.post_preprocess
        self.normalization = self.request.normalization
        self.include_layers = context.include_layers
        self.exclude_layers = context.exclude_layers
        self.drop_layers = self.request.drop_layers
        self.unmatch_size_layer_op = self.request.unmatch_size_layer_op
        self.is_llava_next = self.request.is_llava_next
        self.force_merge_single = self.request.force_merge_single
        self.v2s_empty_default = self.request.v2s_empty_default
        self.v2s_single_default = self.request.v2s_single_default
        self.model_dict = self.request.raw_config
        self.console = console

        self.target = (
            self.target_model if self.target_model is not None else self.base_models[0]
        )
        self.target_state_dict = self.target.state_dict()

        self.include_ranges, self.include_specific = parse_layer_specifications(
            self.include_layers
        )
        self.exclude_ranges, self.exclude_specific = parse_layer_specifications(
            self.exclude_layers
        )
        self.drop_ranges, self.drop_specific = parse_layer_specifications(
            self.drop_layers
        )
        self.included_layers: List[str] = []
        self.excluded_layers: List[str] = []
        self.dropped_layers: List[str] = []
        self._included_target_tensor_aliases: dict[tuple[Any, ...], str] = {}

    @abstractmethod
    def merge(self) -> torch.nn.Module:
        pass

    def _check_layer_compatibility(self, k: str) -> bool:
        v = self.target_state_dict[k]

        def _compatible_sparse_model(model):
            state_dict = model.state_dict()
            if k in state_dict:
                return v.shape == state_dict[k].shape
            return getattr(model, "_ninja_sparse_zero_missing", False)

        if self.unmatch_size_layer_op == "skip":
            for b in self.base_models:
                if not _compatible_sparse_model(b):
                    self.console.print(
                        f"[yellow]  Skipping layer {k} due to size mismatch or missing key in base model.[/yellow]"
                    )
                    return False
            for s in self.sub_models:
                if not _compatible_sparse_model(s):
                    self.console.print(
                        f"[yellow]  Skipping layer {k} due to size mismatch or missing key in sub model.[/yellow]"
                    )
                    return False
        return True

    def _prepare_tensor_slices(
        self, k: str
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        v_slice, base_slices, sub_slices, _ = prepare_tensor_slices(
            self.target_state_dict,
            k,
            self.base_models,
            self.sub_models,
            self.unmatch_size_layer_op,
            self.console,
        )
        v2s_slices = []
        return v_slice, base_slices, sub_slices, v2s_slices

    def _display_tensor_info(
        self, title: str, tensor: torch.Tensor, prefix: str = ""
    ):
        if getattr(self.console, "is_live", False):
            return
        if tensor is not None:
            self.console.print(
                f"  [{prefix}]{title} - First 5 elements: {tensor.flatten()[:5]}[/{prefix}]"
            )

    def _resolve_layer_value(self, value, layer_key: str, *, default: float = 1.0):
        if isinstance(value, dict):
            return value.get(layer_key, default)
        if value is None:
            return default
        return value

    def _resolve_velocity(self, layer_key: str):
        return self._resolve_layer_value(self.velocity, layer_key, default=1.0)

    def _resolve_post_velocity(self, layer_key: str):
        return self._resolve_layer_value(self.post_velocity, layer_key, default=1.0)

    def _format_velocity_display(self, value, layer_key: str):
        resolved_value = self._resolve_layer_value(value, layer_key)
        if resolved_value is None:
            return "None"

        tensor_value = to_complex_tensor(
            resolved_value,
            self.target_state_dict[layer_key].device,
            self.target_state_dict[layer_key].dtype,
        )
        if tensor_value.is_complex():
            return f"{tensor_value.real.item():.4f}+{tensor_value.imag.item():.4f}j"
        return str(tensor_value.item())

    def _build_operation_details(self, layer_key: str) -> str:
        return (
            f"Operation: {self.operation}\n"
            f"Preprocess: {self.preprocess}\n"
            f"Velocity: {self._format_velocity_display(self.velocity, layer_key)}\n"
            f"Normalization: {self.normalization}\n"
            f"Post-operation: {self.post_operation}\n"
            f"Post-preprocess: {self.post_preprocess}\n"
            f"Post-velocity: {self._format_velocity_display(self.post_velocity, layer_key)}"
        )

    def _log_operation_details(self, k: str):
        self.console.set_merge_info(
            Panel(
                self._build_operation_details(k),
                title=f"[bold]Merge Info[/bold] [dim]{k}[/dim]",
                style="cyan",
            )
        )

    def _normalized_layer_key(self, layer_key: str) -> str:
        if self.is_llava_next:
            return layer_key.replace("language_model.", "", 1)
        return layer_key

    def _target_tensor_alias_id(self, layer_key: str) -> tuple[Any, ...] | None:
        tensor = self.target_state_dict.get(layer_key)
        if not isinstance(tensor, torch.Tensor):
            return None
        return (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.device),
            str(tensor.dtype),
        )

    def _classify_layer(self, layer_key: str) -> LayerDecision:
        normalized_key = self._normalized_layer_key(layer_key)

        if is_layer_dropped(normalized_key, self.drop_ranges, self.drop_specific):
            return LayerDecision(
                target_key=layer_key,
                status="dropped",
                message=f"  [red]Dropping layer: {layer_key}[/red]",
            )

        if (
            normalized_key in self.skip_layers
            or layer_key in self.skip_layers
            or (self.skip_layernorm and "layernorm" in normalized_key.lower())
        ):
            return LayerDecision(
                target_key=layer_key,
                status="excluded",
                message=f"  [yellow]Skipping layer: {layer_key}[/yellow]",
            )

        if not is_layer_included(
            normalized_key,
            self.include_ranges,
            self.include_specific,
            self.exclude_ranges,
            self.exclude_specific,
        ):
            return LayerDecision(
                target_key=layer_key,
                status="excluded",
                message=f"  [yellow]Excluding layer: {layer_key}[/yellow]",
            )

        return LayerDecision(
            target_key=layer_key,
            status="included",
            message=f"  [green]Including layer: {layer_key}[/green]",
        )

    def _record_layer_decision(self, decision: LayerDecision):
        if decision.status == "included":
            self.included_layers.append(decision.target_key)
        elif decision.status == "excluded":
            self.excluded_layers.append(decision.target_key)
        elif decision.status == "dropped":
            self.dropped_layers.append(decision.target_key)
        self.console.print(decision.message)

    def _filter_layers(self):
        self._included_target_tensor_aliases = {}
        for k in self.target_state_dict.keys():
            decision = self._classify_layer(k)
            if decision.status == "included":
                alias_id = self._target_tensor_alias_id(k)
                if alias_id is not None:
                    canonical_key = self._included_target_tensor_aliases.get(alias_id)
                    if canonical_key is not None:
                        decision = LayerDecision(
                            target_key=k,
                            status="excluded",
                            message=(
                                f"  [yellow]Skipping aliased layer: {k} "
                                f"(same storage as {canonical_key})[/yellow]"
                            ),
                        )
                    else:
                        self._included_target_tensor_aliases[alias_id] = k
            self._record_layer_decision(decision)
        merge_name = self.request.name or self.operation
        self.console.reset_layer_progress(
            len(self.included_layers),
            f"Layers: {merge_name}",
        )

    def _build_layer_context(self, layer_key: str) -> MergeLayerContext | None:
        if layer_key not in self.included_layers:
            return None

        if not self._check_layer_compatibility(layer_key):
            self.excluded_layers.append(layer_key)
            return None

        v_slice, base_slices, sub_slices, _ = self._prepare_tensor_slices(layer_key)
        self._log_operation_details(layer_key)
        self.console.advance_layer(layer_key)
        return MergeLayerContext(
            key=layer_key,
            target_slice=v_slice,
            base_slices=base_slices,
            sub_slices=sub_slices,
            velocity=self._resolve_velocity(layer_key),
            post_velocity=self._resolve_post_velocity(layer_key),
        )

    def _iter_merge_layer_contexts(self):
        for layer_key in self.target_state_dict.keys():
            layer_context = self._build_layer_context(layer_key)
            if layer_context is not None:
                yield layer_context

    def _print_summary(self):
        table = Table(title="Layer Summary", show_header=True, header_style="bold magenta")
        table.add_column("Layer Name", style="dim", width=60)
        table.add_column("Status", justify="right")

        for layer in self.included_layers:
            table.add_row(layer, "[green]Included[/green]")
        for layer in self.excluded_layers:
            table.add_row(layer, "[yellow]Excluded[/yellow]")
        for layer in self.dropped_layers:
            table.add_row(layer, "[red]Dropped[/red]")

        self.console.print(table)

    def _apply_dropped_layers(self):
        if not self.dropped_layers:
            return

        if hasattr(self.target, "state_dict_data") and isinstance(
            self.target.state_dict_data, dict
        ):
            for layer in self.dropped_layers:
                self.target.state_dict_data.pop(layer, None)
        else:
            existing_dropped = set(getattr(self.target, "_ninja_dropped_layers", set()))
            existing_dropped.update(self.dropped_layers)
            self.target._ninja_dropped_layers = existing_dropped

            if not hasattr(self.target, "_ninja_original_state_dict"):
                self.target._ninja_original_state_dict = self.target.state_dict

                def _filtered_state_dict(model, *args, **kwargs):
                    state_dict = model._ninja_original_state_dict(*args, **kwargs)
                    return filter_state_dict_keys(
                        state_dict, getattr(model, "_ninja_dropped_layers", set())
                    )

                self.target.state_dict = types.MethodType(
                    _filtered_state_dict, self.target
                )

        self.target_state_dict = self.target.state_dict()
        if hasattr(self.target, "config"):
            prune_config_for_dropped_layers(self.target.config, self.target_state_dict)

    def _post_merge_finalize(self):
        return None

    def _finalize_merge(self) -> torch.nn.Module:
        self._apply_dropped_layers()
        self.console.complete_layer_progress()
        self._print_summary()
        self._post_merge_finalize()
        return self.target
