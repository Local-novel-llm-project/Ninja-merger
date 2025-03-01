import torch
from Merger.base import Merger
from rich.panel import Panel
from Utils.operation_dicts import OPERATION_DICTS


class CustomMerger(Merger):
    """Widen などの特殊なマージを行うMerger クラス。"""

    def merge(self) -> torch.nn.Module:
        # ... (CustomMerger の実装) ...
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

            v_slice, base_slices, sub_slices, min_size = self._prepare_tensor_slices(k)
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
                    merged_weight = OPERATION_DICTS[self.operation](
                        self.target,
                        self.base_models,
                        self.sub_models,
                        target_k,
                        t=self.velocity,
                        s=1.0,
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
