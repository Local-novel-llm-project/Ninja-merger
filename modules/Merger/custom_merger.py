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

        for layer_context in self._iter_merge_layer_contexts():
            target_k = layer_context.key
            v_slice = layer_context.target_slice

            if self.operation == "widen":
                if not getattr(self.console, "is_live", False):
                    self.console.print(
                        Panel(
                            f"Applying WIDEN operation to layer: {target_k}",
                            title="[bold]Widen Operation[/bold]",
                            style="magenta",
                        )
                    )
                try:
                    merged_weight = OPERATION_DICT[self.operation](
                        self.target,
                        self.base_models,
                        self.sub_models,
                        target_k,
                        t=layer_context.velocity,
                        s=1.0,  # ここ、s は設定から取得する？
                    )
                    if merged_weight is not None:
                        v_slice.copy_(
                            merged_weight.to(device=v_slice.device, dtype=v_slice.dtype)
                        )

                except Exception as e:
                    self.console.print(f"[red]Error during WIDEN operation: {e}[/red]")
                    raise
            else:
                self.console.print(f"[red] Unexpected operation: {self.operation} [/red]")
                raise ValueError(f"Unexpected operation: {self.operation}")

        return self._finalize_merge()
