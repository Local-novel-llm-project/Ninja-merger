from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from Utils import operation_dicts
from Utils.layers import (
    is_layer_dropped,
    is_layer_included,
    parse_layer_specifications,
)

console = Console()


class Merger(ABC):
    """Merger の抽象基底クラス。"""

    def __init__(
        self,
        skip_layernorm: bool,
        target_model: torch.nn.Module,
        base_models: List[torch.nn.Module],
        sub_models: List[torch.nn.Module],
        velocity: Union[
            float, complex, torch.Tensor, Dict[str, Union[float, complex, torch.Tensor]]
        ],
        post_velocity: Union[
            float, complex, torch.Tensor, Dict[str, Union[float, complex, torch.Tensor]]
        ],  # 型を変更
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
        # ... (初期化処理は変更なし) ...
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
        self.model_dict = model_dict
        self.console = Console()

        # ...

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
        """モデルをマージする抽象メソッド。具象クラスで実装が必要。"""
        pass

    # ... (ヘルパーメソッド群もここに) ...
    def _prepare_tensor_slices(
        self, k: str
    ) -> tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Size]:
        """テンソルのスライスを準備するヘルパー関数。"""
        # ... 実装は変更なし ...
        v = self.target_state_dict[k]

        if self.unmatch_size_layer_op == "only_common_range":
            min_size = min(
                v.shape,
                *[
                    b.state_dict()[k].shape
                    for b in self.base_models
                    if k in b.state_dict()
                ],
                *[
                    s.state_dict()[k].shape
                    for s in self.sub_models
                    if k in s.state_dict()
                ],
            )
            self.console.print(
                f"  [cyan]Merging only common range for layer {k}. Common size: {min_size}[/cyan]"
            )
        else:
            min_size = v.shape

        v_slice = (
            operation_dicts.get_slice_to(v, min_size)
            if self.unmatch_size_layer_op == "only_common_range"
            else v
        )
        base_slices = [
            operation_dicts.get_slice_to(b.state_dict()[k], min_size)
            if self.unmatch_size_layer_op == "only_common_range"
            else b.state_dict()[k]
            for b in self.base_models
            if k in b.state_dict()
        ]
        sub_slices = [
            operation_dicts.get_slice_to(s.state_dict()[k], min_size)
            if self.unmatch_size_layer_op == "only_common_range"
            else s.state_dict()[k]
            for s in self.sub_models
            if k in s.state_dict()
        ]
        return v_slice, base_slices, sub_slices, min_size

    def _check_layer_compatibility(self, k: str) -> bool:
        """レイヤーの互換性をチェックするヘルパー関数。"""
        # ... 実装は変更なし ...
        v = self.target_state_dict[k]

        if self.unmatch_size_layer_op == "skip":
            for b in self.base_models:
                if k not in b.state_dict() or v.shape != b.state_dict()[k].shape:
                    self.console.print(
                        f"[yellow]  Skipping layer {k} due to size mismatch or missing key in base model.[/yellow]"
                    )
                    return False
            for s in self.sub_models:
                if k not in s.state_dict() or v.shape != s.state_dict()[k].shape:
                    self.console.print(
                        f"[yellow]  Skipping layer {k} due to size mismatch or missing key in sub model.[/yellow]"
                    )
                    return False
        return True

    def _display_tensor_info(self, title: str, tensor: torch.Tensor, prefix: str = ""):
        """テンソルの情報を表示するヘルパー関数。"""
        if tensor is not None:
            self.console.print(
                f"  [{prefix}]{title} - First 5 elements: {tensor.flatten()[:5]}[/{prefix}]"
            )

    def _log_operation_details(self, k):  # キーを追加
        """操作の詳細をログに出力する関数."""
        if isinstance(self.velocity, dict):
            velocity_display = self.velocity.get(
                k, "N/A"
            )  # レイヤーに対応する velocity を取得
            if (
                isinstance(velocity_display, torch.Tensor)
                and velocity_display.is_complex()
            ):
                velocity_display = f"{velocity_display.real.item():.4f}+{velocity_display.imag.item():.4f}j"
            elif isinstance(velocity_display, complex):
                velocity_display = (
                    f"{velocity_display.real:.4f}+{velocity_display.imag:.4f}j"
                )
        else:
            velocity_display = self.velocity
            if isinstance(self.velocity, torch.Tensor) and self.velocity.is_complex():
                velocity_display = (
                    f"{self.velocity.real.item():.4f}+{self.velocity.imag.item():.4f}j"
                )
            elif isinstance(self.velocity, complex):
                velocity_display = f"{self.velocity.real:.4f}+{self.velocity.imag:.4f}j"

        self.console.print(
            Panel(
                f"Operation: {self.operation}\n"
                f"Preprocess: {self.preprocess}\n"
                f"Velocity: {velocity_display}\n"
                f"Normalization: {self.normalization}\n"
                f"Post-operation: {self.post_operation}\n"
                f"Post-preprocess: {self.post_preprocess}\n"
                f"Post-velocity: {self.post_velocity}",
                title="[bold]Merge Info[/bold]",
                style="cyan",
            )
        )

    def _filter_layers(self):
        """レイヤーのフィルタリングを行う。"""
        for k in self.target_state_dict.keys():
            target_k = k
            if self.is_llava_next:
                k = k.replace("language_model.", "", 1)  # llava-next 用のキー調整

            # レイヤーのドロップ、スキップ、除外判定
            if is_layer_dropped(k, self.drop_ranges, self.drop_specific):
                self.dropped_layers.append(k)
                self.console.print(f"  [red]Dropping layer: {k}[/red]")
                continue
            if (k in self.skip_layers) or (self.skip_layernorm and "layernorm" in k):
                self.excluded_layers.append(k)
                self.console.print(f"  [yellow]Skipping layer: {k}[/yellow]")
                continue
            if not is_layer_included(
                k,
                self.include_ranges,
                self.include_specific,
                self.exclude_ranges,
                self.exclude_specific,
            ):
                self.excluded_layers.append(k)
                self.console.print(f"  [yellow]Excluding layer: {k}[/yellow]")
                continue
            self.included_layers.append(k)
            self.console.print(f"  [green]Including layer: {k}[/green]")

    def _print_summary(self):
        """マージ結果のサマリーを表示する。"""
        table = Table(
            title="Layer Summary", show_header=True, header_style="bold magenta"
        )
        table.add_column("Layer Name", style="dim", width=60)
        table.add_column("Status", justify="right")

        for layer in self.included_layers:
            table.add_row(layer, "[green]Included[/green]")
        for layer in self.excluded_layers:
            table.add_row(layer, "[yellow]Excluded[/yellow]")
        for layer in self.dropped_layers:
            table.add_row(layer, "[red]Dropped[/red]")

        self.console.print(table)
