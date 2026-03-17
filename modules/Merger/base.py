# Merger/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..Utils.layers import (
    is_layer_dropped,
    is_layer_included,
    parse_layer_specifications,
)
from ..Utils.utility import prepare_tensor_slices, to_complex_tensor


class Merger(ABC):
    """Abstract base class for all model mergers.

    This class defines the common interface and shared functionalities for different
    merging strategies. Subclasses must implement the `merge` method.

    Attributes:
        skip_layernorm (bool): If True, layernorm layers are skipped during merge.
        target_model (torch.nn.Module): The model to which the merge results are applied.
            If None, the first base model is used as the target.
        base_models (List[torch.nn.Module]): A list of models to be used as the base for the merge.
        sub_models (List[torch.nn.Module]): A list of models to be subtracted or otherwise
            combined with the base models.
        velocity (Union[float, complex, torch.Tensor, Dict]): The primary coefficient
            for the merge operation. Can be a single value or a per-layer dictionary.
        post_velocity (Union[float, complex, torch.Tensor, Dict]): The coefficient for the
            post-merge operation, applied after the main merge.
        skip_layers (List[str]): A list of layer names to explicitly skip.
        operation (str): The primary merge operation to perform (e.g., 'add', 'sub').
        post_operation (str): The operation to perform after the primary merge.
        preprocess (str): The preprocessing to apply to tensors before the merge.
        post_preprocess (str): The preprocessing to apply after the merge.
        normalization (str): The normalization method to apply.
        include_layers (List[str]): Specifications for layers to include in the merge.
        exclude_layers (List[str]): Specifications for layers to exclude from the merge.
        drop_layers (List[str]): Specifications for layers to drop from the model.
        unmatch_size_layer_op (str): How to handle layers with mismatched sizes.
        is_llava_next (bool): Special handling for Llava-NeXT models.
        force_merge_single (bool): Forces merging even with a single model.
        v2s_empty_default (str): Default behavior for empty velocity-to-slice.
        v2s_single_default (str): Default behavior for single velocity-to-slice.
        model_dict (Dict[str, Any]): The original configuration dictionary for the merge.
        console (Console): Rich console for pretty printing.
        target (torch.nn.Module): The effective target model for the merge.
        target_state_dict (Dict[str, torch.Tensor]): The state dictionary of the target model.
    """

    def __init__(
        self,
        skip_layernorm: bool,
        target_model: torch.nn.Module,
        base_models: List[torch.nn.Module],
        sub_models: List[torch.nn.Module],
        velocity: Union[float, complex, torch.Tensor, Dict[str, Union[float, complex, torch.Tensor]]],
        post_velocity: Union[float, complex, torch.Tensor, Dict[str, Union[float, complex, torch.Tensor]]],
        skip_layers: List[str],
        operation: str,
        post_operation: str,
        preprocess: str,
        post_preprocess: str,
        normalization: str,
        include_layers: List[str],
        exclude_layers: List[str],
        drop_layers: List[str],
        unmatch_size_layer_op: str,
        is_llava_next: bool = False,
        force_merge_single: bool = False,
        v2s_empty_default: str = "v1",
        v2s_single_default: str = "auto",
        model_dict: Dict[str, Any] = None,
    ):
        self.skip_layernorm = skip_layernorm
        self.target_model = target_model
        self.base_models = base_models
        self.sub_models = sub_models
        self.velocity = velocity
        self.post_velocity = post_velocity
        self.skip_layers = skip_layers
        self.operation = operation
        self.post_operation = post_operation
        self.preprocess = preprocess
        self.post_preprocess = post_preprocess
        self.normalization = normalization
        self.include_layers = include_layers
        self.exclude_layers = exclude_layers
        self.drop_layers = drop_layers
        self.unmatch_size_layer_op = unmatch_size_layer_op
        self.is_llava_next = is_llava_next
        self.force_merge_single = force_merge_single
        self.v2s_empty_default = v2s_empty_default
        self.v2s_single_default = v2s_single_default
        self.model_dict = model_dict or {}
        self.console = Console()

        # ターゲットモデルの初期化
        self.target = target_model if target_model is not None else base_models[0]
        self.target_state_dict = self.target.state_dict()

        # レイヤー指定のパース
        (
            self.include_ranges,
            self.include_specific,
        ) = parse_layer_specifications(include_layers)
        (
            self.exclude_ranges,
            self.exclude_specific,
        ) = parse_layer_specifications(exclude_layers)
        (self.drop_ranges, self.drop_specific) = parse_layer_specifications(drop_layers)
        # レイヤー処理用のリスト
        self.included_layers: List[str] = []
        self.excluded_layers: List[str] = []
        self.dropped_layers: List[str] = []

    @abstractmethod
    def merge(self) -> torch.nn.Module:
        """Performs the model merging.

        This is an abstract method that must be implemented by all concrete
        merger subclasses.

        Returns:
            torch.nn.Module: The merged model.
        """
        pass

    def _check_layer_compatibility(self, k: str) -> bool:
        """Checks if a layer is compatible for merging across all models.

        A layer is considered incompatible if its key is missing or its tensor
        shape does not match the target model's corresponding layer. This check
        is only performed if `unmatch_size_layer_op` is set to 'skip'.

        Args:
            k (str): The name of the layer to check.

        Returns:
            bool: True if the layer is compatible, False otherwise.
        """
        v = self.target_state_dict[k]

        if self.unmatch_size_layer_op == "skip":
            for b in self.base_models:
                if k not in b.state_dict() or v.shape != b.state_dict()[k].shape:
                    self.console.print(f"[yellow]  Skipping layer {k} due to size mismatch or missing key in base model.[/yellow]")
                    return False
            for s in self.sub_models:
                if k not in s.state_dict() or v.shape != s.state_dict()[k].shape:
                    self.console.print(f"[yellow]  Skipping layer {k} due to size mismatch or missing key in sub model.[/yellow]")
                    return False
        return True

    def _prepare_tensor_slices(self, k: str) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Prepares tensor slices from the target, base, and sub models for a given layer.

        Args:
            k (str): The name of the layer.

        Returns:
            A tuple containing:
            - The tensor slice from the target model.
            - A list of tensor slices from the base models.
            - A list of tensor slices from the sub models.
            - A list of tensor slices from the velocity-to-slice models (if any).
        """
        v_slice, base_slices, sub_slices, _ = prepare_tensor_slices(
            self.target_state_dict,
            k,
            self.base_models,
            self.sub_models,
            self.unmatch_size_layer_op,
            self.console,
        )
        v2s_slices = []  # Placeholder for now
        return v_slice, base_slices, sub_slices, v2s_slices

    def _display_tensor_info(self, title: str, tensor: torch.Tensor, prefix: str = ""):
        """Displays debugging information for a given tensor.

        Args:
            title (str): The title for the information display.
            tensor (torch.Tensor): The tensor to display information about.
            prefix (str): A prefix for the console output style.
        """
        if tensor is not None:
            self.console.print(f"  [{prefix}]{title} - First 5 elements: {tensor.flatten()[:5]}[/{prefix}]")

    def _log_operation_details(self, k: str):
        """Logs the detailed parameters of the merge operation for a specific layer.

        Args:
            k (str): The name of the layer for which to log details.
        """
        # 修正: to_complex_tensor を使用
        if isinstance(self.velocity, dict):
            velocity_display = to_complex_tensor(
                self.velocity.get(k, 0.0),  # デフォルト値 0.0
                self.target_state_dict[k].device,
                self.target_state_dict[k].dtype,
            )
        else:
            velocity_display = torch.tensor(self.velocity, device=self.target_state_dict[k].device, dtype=self.target_state_dict[k].dtype)
        if isinstance(self.post_velocity, dict):
            post_velocity_display = to_complex_tensor(
                self.post_velocity.get(k, 0.0),  # デフォルト値 0.0
                self.target_state_dict[k].device,
                self.target_state_dict[k].dtype,
            )
        else:
            post_velocity_display = torch.tensor(self.post_velocity, device=self.target_state_dict[k].device, dtype=self.target_state_dict[k].dtype)
            
        if isinstance(velocity_display, torch.Tensor) and velocity_display.is_complex():
            velocity_display = f"{velocity_display.real.item():.4f}+{velocity_display.imag.item():.4f}j"
        elif isinstance(velocity_display, complex):
            velocity_display = f"{velocity_display.real:.4f}+{velocity_display.imag:.4f}j"

        if isinstance(post_velocity_display, torch.Tensor) and post_velocity_display.is_complex():
            post_velocity_display = f"{post_velocity_display.real.item():.4f}+{post_velocity_display.imag.item():.4f}j"
        elif isinstance(post_velocity_display, complex):
            post_velocity_display = f"{post_velocity_display.real:.4f}+{post_velocity_display.imag:.4f}j"

        self.console.print(
            Panel(
                f"Operation: {self.operation}\n"
                f"Preprocess: {self.preprocess}\n"
                f"Velocity: {velocity_display}\n"
                f"Normalization: {self.normalization}\n"
                f"Post-operation: {self.post_operation}\n"
                f"Post-preprocess: {self.post_preprocess}\n"
                f"Post-velocity: {post_velocity_display}",
                title="[bold]Merge Info[/bold]",
                style="cyan",
            )
        )

    def _filter_layers(self):
        """Filters layers based on include, exclude, and drop specifications.

        Populates the `self.included_layers`, `self.excluded_layers`, and
        `self.dropped_layers` lists based on the user-provided configuration.
        """
        for k in self.target_state_dict.keys():
            target_k = k
            normalized_key = (
                k.replace("language_model.", "", 1) if self.is_llava_next else k
            )

            # レイヤーのドロップ、スキップ、除外判定
            if is_layer_dropped(normalized_key, self.drop_ranges, self.drop_specific):
                self.dropped_layers.append(target_k)
                self.console.print(f"  [red]Dropping layer: {target_k}[/red]")
                continue
            if (
                normalized_key in self.skip_layers
                or target_k in self.skip_layers
                or (self.skip_layernorm and "layernorm" in normalized_key.lower())
            ):
                self.excluded_layers.append(target_k)
                self.console.print(f"  [yellow]Skipping layer: {target_k}[/yellow]")
                continue
            if not is_layer_included(
                normalized_key,
                self.include_ranges,
                self.include_specific,
                self.exclude_ranges,
                self.exclude_specific,
            ):
                self.excluded_layers.append(target_k)
                self.console.print(f"  [yellow]Excluding layer: {target_k}[/yellow]")
                continue
            self.included_layers.append(target_k)
            self.console.print(f"  [green]Including layer: {target_k}[/green]")

    def _print_summary(self):
        """Prints a summary table of which layers were included, excluded, or dropped."""
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
