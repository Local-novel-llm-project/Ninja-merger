import torch

from ..Merger.base import Merger
from ..Proc.normalization import normalize_tensor
from ..Proc.post_processing import post_process_tensor
from ..Proc.pre_processing import preprocess_tensor
from ..Utils.models import DummyConfig, DummyModel
from ..Utils.operation_dicts import OPERATION_DICT


class BasicMerger(Merger):
    """基本的な四則演算などを行う Merger クラス。"""

    def merge(self) -> torch.nn.Module:
        self.console.rule("[bold blue]Starting Basic Model Merge Process[/bold blue]")
        self._filter_layers()

        for layer_context in self._iter_merge_layer_contexts():
            k = layer_context.key
            v_slice = layer_context.target_slice
            base_slices = layer_context.base_slices
            sub_slices = layer_context.sub_slices
            try:
                base_processed_values = []
                post_velocity = layer_context.post_velocity
                velocity = layer_context.velocity

                if self.operation in {"passthrough", "none"}:
                    if len(base_slices) != 1:
                        raise ValueError(
                            f"{self.operation} operation requires exactly one base model"
                        )
                    processed = OPERATION_DICT[self.operation](
                        v_slice,
                        base_slices[0],
                        None,
                        velocity,
                    )
                    base_processed_values.append(processed)
                else:
                    for b_slice in base_slices:
                        for s_slice in sub_slices:
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

                if len(base_processed_values) == 1:
                    processed_v = base_processed_values[0]
                else:
                    # 複数ペアを処理した場合は平均化して安定化。
                    processed_v = torch.stack(base_processed_values, dim=0).mean(dim=0)

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
                    post_velocity,
                    v_slice,
                )

                v_slice.copy_(processed_v)

            except Exception as e:
                self.console.print(
                    f"[red]Error during basic merge operation: {e}[/red]"
                )
                raise

        return self._finalize_merge()

    def _post_merge_finalize(self):
        if isinstance(self.target, DummyModel):
            all_models = self.base_models + self.sub_models
            configs = [m.config for m in all_models if hasattr(m, "config")]
            self.target._config = DummyConfig(configs=configs)
