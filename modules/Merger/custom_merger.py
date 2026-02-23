# Merger/custom_merger.py
import torch
from rich.panel import Panel

from ..Merger.base import Merger
from ..Utils.operation_dicts import OPERATION_DICT


class CustomMerger(Merger):
    """Widen などの特殊なマージを行うMerger クラス。"""

    def merge(self) -> torch.nn.Module:
        self.console.rule("[bold blue]Starting Custom Model Merge Process[/bold blue]")
        self._filter_layers()

        for target_k in self.target_state_dict.keys():
            if target_k not in self.included_layers:
                continue

            if not self._check_layer_compatibility(target_k):
                self.excluded_layers.append(target_k)
                continue

            v_slice = self.target_state_dict[target_k]
            self._log_operation_details(target_k)

            if self.operation == "widen":
                self.console.print(
                    Panel(
                        f"Applying WIDEN operation to layer: {target_k}",
                        title="[bold]Widen Operation[/bold]",
                        style="magenta",
                    )
                )
                try:
                    if self.velocity is None:
                        widen_velocity = 1.0
                    elif isinstance(self.velocity, dict):
                        widen_velocity = self.velocity.get(target_k, 1.0)
                    else:
                        widen_velocity = self.velocity

                    merged_weight = OPERATION_DICT[self.operation](
                        self.target,
                        self.base_models,
                        self.sub_models,
                        target_k,
                        t=widen_velocity,
                        s=1.0,  # ここ、s は設定から取得する？
                    )
                    if merged_weight is not None:
                        v_slice.copy_(merged_weight.to(device=v_slice.device, dtype=v_slice.dtype))

                except Exception as e:
                    self.console.print(f"[red]Error during WIDEN operation: {e}[/red]")
                    raise
            else:
                self.console.print(f"[red] Unexpected operation: {self.operation} [/red]")
                raise ValueError(f"Unexpected operation: {self.operation}")

        self._print_summary()
        return self.target
