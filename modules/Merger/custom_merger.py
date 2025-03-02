# Merger/custom_merger.py
import torch
from rich.panel import Panel

from ..Merger.base import Merger
from ..Utils.operation_dicts import OPERATION_DICT
from ..Utils.utility import prepare_tensor_slices


class CustomMerger(Merger):
    """Widen などの特殊なマージを行うMerger クラス。"""

    def merge(self) -> torch.nn.Module:
        self.console.rule("[bold blue]Starting Custom Model Merge Process[/bold blue]")
        self._filter_layers()

        for k in torch.utils.data.DataLoader(
            list(self.target_state_dict.keys()), batch_size=1
        ):
            k = k[0]  # バッチ解除
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
            self._log_operation_details()

            if self.operation == "widen":
                self.console.print(
                    Panel(
                        f"Applying WIDEN operation to layer: {target_k}",
                        title="[bold]Widen Operation[/bold]",
                        style="magenta",
                    )
                )
                try:
                    # 変更: Utils からインポートした OPERATION_DICT を使用
                    merged_weight = OPERATION_DICT[self.operation](
                        self.target,
                        self.base_models,
                        self.sub_models,
                        target_k,
                        t=self.velocity,  # ここ、self.velocity で良い？
                        s=1.0,  # ここ、s は設定から取得する？
                    )

                except Exception as e:
                    self.console.print(f"[red]Error during WIDEN operation: {e}[/red]")
                    raise
            else:
                self.console.print(
                    f"[red] Unexpected operation: {self.operation} [/red]"
                )
                raise ValueError(f"Unexpected operation: {self.operation}")

        self._print_summary()
        return self.target
