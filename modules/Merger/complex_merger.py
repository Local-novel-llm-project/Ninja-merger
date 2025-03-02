# Merger/complex_merger.py

import torch
from rich.panel import Panel

from ..Calc.complex_calc import ComplexMix, norm_angle_t_calc
from ..Merger.base import Merger
from ..Utils.operation_dicts import OPERATION_DICT, POST_OPERATION_DICT
from ..Utils.utility import (
    prepare_tensor_slices,
)


class ComplexMerger(Merger):
    """複素数関連の演算を行う Merger クラス。"""

    def merge(self) -> torch.nn.Module:
        self.console.rule("[bold blue]Starting Complex Model Merge Process[/bold blue]")
        self._filter_layers()

        for k in torch.utils.data.DataLoader(
            list(self.target_state_dict.keys()), batch_size=1
        ):
            k = k[0]
            if k not in self.included_layers:
                continue

            target_k = k
            if self.is_llava_next:
                k = k.replace("language_model.", "", 1)

            if not self._check_layer_compatibility(k):
                self.excluded_layers.append(target_k)
                continue

            # 変更: ヘルパー関数を使用
            v_slice, base_slices, sub_slices, min_size = prepare_tensor_slices(
                self.target_state_dict,
                k,
                self.base_models,
                self.sub_models,
                self.unmatch_size_layer_op,
                self.console,
            )
            self._log_operation_details(k)

            # velocity を取得 (レイヤーごとに異なる可能性がある)
            velocity = (
                self.velocity[k] if isinstance(self.velocity, dict) else self.velocity
            )

            if self.operation == "complexadd":
                self.console.print(
                    Panel(
                        f"Applying ComplexAdd operation to layer: {target_k}",
                        title="[bold]ComplexAdd Operation[/bold]",
                        style="yellow",
                    )
                )
                try:
                    avg = sum(sub_slices) / len(sub_slices)
                    t = torch.tensor(0.1).to(
                        self.velocity.device
                    )  # これ、velocity が複素数の場合は？
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
                self.console.print(
                    Panel(
                        f"Applying AngleMerge operation to layer: {target_k}",
                        title="[bold]AngleMerge Operation[/bold]",
                        style="green",
                    )
                )
                try:
                    before_tensor = v_slice
                    self._display_tensor_info(
                        "AngleMerge - Before", before_tensor, "yellow"
                    )

                    # 変更: Utils からインポートした OPERATION_DICT を使用
                    processed_v = OPERATION_DICT[self.operation](
                        POST_OPERATION_DICT[self.post_operation],  # ここも
                        v_slice,
                        sub_slices,
                        velocity,  # ここ、velocityで良い？
                        self.force_merge_single,
                        self.v2s_empty_default,
                        self.v2s_single_default,
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

                    # 変更: Utils からインポートした OPERATION_DICT を使用
                    processed_v = OPERATION_DICT[self.operation](
                        v_slice,
                        sub_slices,
                        velocity,  # ここ、velocityで良い？
                        complex_mix_func=ComplexMix,
                        t_calc_func=norm_angle_t_calc,
                        **{"layer_key": k},
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
        self._print_summary()
        return self.target
