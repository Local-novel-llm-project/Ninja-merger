import torch

from ..Merger.base import Merger
from ..Proc.normalization import normalize_tensor
from ..Proc.post_processing import post_process_tensor
from ..Proc.pre_processing import preprocess_tensor
from ..Utils.operation_dicts import OPERATION_DICT


class BasicMerger(Merger):
    """基本的な四則演算などを行う Merger クラス。"""

    def merge(self) -> torch.nn.Module:
        self.console.rule("[bold blue]Starting Basic Model Merge Process[/bold blue]")
        self._filter_layers()

        for k in torch.utils.data.DataLoader(
            list(self.target_state_dict.keys()), batch_size=1
        ):
            k = k[0]
            if k not in self.included_layers:
                continue

            if not self._check_layer_compatibility(k):
                self.excluded_layers.append(k)
                continue

            v_slice, base_slices, sub_slices, _ = self._prepare_tensor_slices(k)
            self._log_operation_details(k)
            try:
                base_processed_values = []

                velocity = (
                    self.velocity[k]
                    if isinstance(self.velocity, dict)
                    else self.velocity
                )

                for b_idx, (b_slice, b) in enumerate(
                    zip(base_slices, self.base_models)
                ):
                    for s_idx, (s_slice, s) in enumerate(
                        zip(sub_slices, self.sub_models)
                    ):
                        if id(b_slice) != id(s_slice):
                            processed = OPERATION_DICT[self.operation](
                                v_slice,
                                b_slice,
                                s_slice,
                                velocity,
                            )
                            base_processed_values.append(processed)

                if not base_processed_values:
                    self.console.print(
                        "[yellow]  Warning: No valid tensor pairs to process for this layer[/yellow]"
                    )
                    continue

                processed_v = base_processed_values[0]

                # 前処理
                processed_v = preprocess_tensor(processed_v, self.preprocess)

                # 正規化
                if self.normalization != "none":
                    processed_v = normalize_tensor(
                        processed_v,
                        self.normalization,
                        v_slice,
                        base_processed_values,
                    )

                # 後処理
                processed_v = post_process_tensor(
                    processed_v,
                    self.post_operation,
                    self.post_velocity[k]
                    if isinstance(self.post_velocity, dict)
                    else self.post_velocity,
                    v_slice,
                )

                v_slice.copy_(processed_v)

            except Exception as e:
                self.console.print(
                    f"[red]Error during basic merge operation: {e}[/red]"
                )
                raise

        self._print_summary()
        return self.target
