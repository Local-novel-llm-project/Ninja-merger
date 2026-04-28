# Merger/complex_merger.py

import torch
from rich.panel import Panel

from ..Calc.complex_calc import ComplexMix, norm_angle_t_calc
from ..Merger.base import Merger
from ..Utils.operation_dicts import OPERATION_DICT, POST_OPERATION_DICT


class ComplexMerger(Merger):
    """複素数関連の演算を行う Merger クラス。"""

    def merge(self) -> torch.nn.Module:
        self.console.rule("[bold blue]Starting Complex Model Merge Process[/bold blue]")
        self._filter_layers()

        for layer_context in self._iter_merge_layer_contexts():
            target_k = layer_context.key
            v_slice = layer_context.target_slice
            sub_slices = layer_context.sub_slices
            velocity = layer_context.velocity

            if self.operation == "complexadd":
                if not getattr(self.console, "is_live", False):
                    self.console.print(
                        Panel(
                            f"Applying ComplexAdd operation to layer: {target_k}",
                            title="[bold]ComplexAdd Operation[/bold]",
                            style="yellow",
                        )
                    )
                try:
                    avg = sum(sub_slices) / len(sub_slices)

                    # velocity をテンソルに変換
                    if not isinstance(velocity, torch.Tensor):
                        device = (
                            v_slice.device
                            if hasattr(v_slice, "device")
                            else torch.device("cpu")
                        )
                        if isinstance(velocity, complex):
                            velocity = torch.complex(
                                torch.tensor(velocity.real, device=device),
                                torch.tensor(velocity.imag, device=device),
                            )
                        else:
                            velocity = torch.tensor(float(velocity), device=device)

                    t = torch.tensor(0.1).to(
                        velocity.device
                    )
                    before_tensor = v_slice

                    self._display_tensor_info(
                        "ComplexAdd - Before", before_tensor, "yellow"
                    )

                    processed_v = OPERATION_DICT[self.operation](
                        v_slice, avg, t, velocity
                    )

                    self._display_tensor_info(
                        "ComplexAdd - Diff", processed_v - before_tensor, "cyan"
                    )
                    self._display_tensor_info(
                        "ComplexAdd - After", processed_v, "green"
                    )
                    v_slice.copy_(processed_v)

                except Exception as e:
                    self.console.print(
                        f"[red]Error during ComplexAdd operation: {e}[/red]"
                    )
                    raise

            elif self.operation == "angle_merge":
                if not getattr(self.console, "is_live", False):
                    self.console.print(
                        Panel(
                            f"Applying AngleMerge operation to layer: {target_k}",
                            title="[bold]AngleMerge Operation[/bold]",
                            style="green",
                        )
                    )
                try:
                    post_operation_func = POST_OPERATION_DICT.get(
                        self.post_operation,
                        lambda original_tensor, processed_tensor, post_velocity: processed_tensor,
                    )
                    before_tensor = v_slice
                    self._display_tensor_info(
                        "AngleMerge - Before", before_tensor, "yellow"
                    )

                    processed_v = OPERATION_DICT[self.operation](
                        post_operation_func,
                        v_slice,
                        sub_slices,
                        velocity,
                        force_merge_single=self.force_merge_single,
                        v2s_empty_default=self.v2s_empty_default,
                        v2s_single_default=self.v2s_single_default,
                    )

                    if processed_v is not None:
                        self._display_tensor_info(
                            "AngleMerge - Diff (processed)",
                            processed_v - before_tensor,
                            "cyan",
                        )
                    self._display_tensor_info(
                        "AngleMerge - After (processed)", processed_v, "green"
                    )
                    if processed_v is not None:
                        v_slice.copy_(processed_v)

                except Exception as e:
                    self.console.print(
                        f"[red]Error during AngleMerge operation: {e}[/red]"
                    )
                    raise

            elif self.operation == "complex_angle_merge":
                if not isinstance(velocity, torch.Tensor):
                    device = (
                        v_slice.device
                        if hasattr(v_slice, "device")
                        else torch.device("cpu")
                    )
                    if isinstance(velocity, complex):
                        velocity = torch.complex(
                            torch.tensor(velocity.real, device=device),
                            torch.tensor(velocity.imag, device=device),
                        )
                    else:
                        velocity = torch.tensor(float(velocity), device=device)
                if not getattr(self.console, "is_live", False):
                    self.console.print(
                        Panel(
                            f"Applying ComplexAngleMerge operation to layer: {target_k}",
                            title="[bold]ComplexAngleMerge Operation[/bold]",
                            style="blue",
                        )
                    )
                try:
                    before_tensor = v_slice
                    self._display_tensor_info(
                        "ComplexAngleMerge - Before", before_tensor, "yellow"
                    )

                    processed_v = OPERATION_DICT[self.operation](
                        v_slice,
                        sub_slices,
                        velocity,
                        complex_mix_func=ComplexMix,
                        t_calc_func=norm_angle_t_calc,
                        **{"layer_key": target_k},
                    )

                    self._display_tensor_info(
                        "ComplexAngleMerge - Diff",
                        processed_v - before_tensor,
                        "cyan",
                    )
                    self._display_tensor_info(
                        "ComplexAngleMerge - After", processed_v, "green"
                    )
                    v_slice.copy_(processed_v)

                except Exception as e:
                    self.console.print(
                        f"[red]Error during ComplexAngleMerge operation: {e}[/red]"
                    )
                    raise
        return self._finalize_merge()
