import torch
from Calc.qeic_calc import calculate_correlation_matrices
from Merger.base import Merger
from Utils.layers import is_qeic_target_layer
from Utils.operation_dicts import OPERATION_DICTS


class QeicMerger(Merger):
    """QEIC 関連の演算を行う Merger クラス。"""

    def merge(self) -> torch.nn.Module:
        self.console.rule("[bold blue]Starting QEIC Model Merge Process[/bold blue]")
        self._filter_layers()

        for k in torch.utils.data.DataLoader(
            list(self.target_state_dict.keys()), batch_size=1
        ):
            k = k[0]  # バッチ化を解除
            self.console.print(f"[blue]Processing layer: {k}[/blue]")

            # ベースとサブモデルに同じキーが存在するか詳細チェック
            base_keys = [k in b.state_dict() for b in self.base_models]
            sub_keys = [k in s.state_dict() for s in self.sub_models]

            if not all(base_keys) or not all(sub_keys):
                self.console.print(
                    f"[yellow]Warning: Layer {k} not present in all models. Base: {base_keys}, Sub: {sub_keys}[/yellow]"
                )
                continue

            # サイズ一致チェックを追加
            base_shapes = [
                b.state_dict()[k].shape for b in self.base_models if k in b.state_dict()
            ]
            sub_shapes = [
                s.state_dict()[k].shape for s in self.sub_models if k in s.state_dict()
            ]

            if len(set(str(s) for s in base_shapes + sub_shapes)) > 1:
                self.console.print(
                    f"[red]Shape mismatch for {k}: Base: {base_shapes}, Sub: {sub_shapes}[/red]"
                )
                # サイズ不一致があるレイヤーはスキップ
                self.console.print(
                    f"[yellow]Skipping layer {k} due to shape mismatch[/yellow]"
                )
                continue

            if k not in self.included_layers:
                continue

            target_k = k
            if self.is_llava_next:
                k = k.replace("language_model.", "", 1)

            if not self._check_layer_compatibility(k):
                self.excluded_layers.append(target_k)
                continue

            v_slice, base_slices, sub_slices, _ = self._prepare_tensor_slices(k)
            self._log_operation_details()

            # ベースとサブのスライスが空でないことを確認
            if not base_slices or not sub_slices:
                self.console.print(
                    f"[yellow]Warning: No valid slices for layer {k}. Skipping QEIC.[/yellow]"
                )
                continue

            # テンソルのサイズが一致することを確認
            if (
                v_slice.shape != base_slices[0].shape
                or v_slice.shape != sub_slices[0].shape
            ):
                self.console.print(
                    f"[red]Size mismatch after slicing for layer {k}: {v_slice.shape} vs {base_slices[0].shape} vs {sub_slices[0].shape}[/red]"
                )
                self.console.print(
                    "[yellow]Skipping this layer for QEIC operation[/yellow]"
                )
                continue

            if not self.sub_models:
                continue

            k_module = (
                ".".join(k.split(".")[:-1]) if k.endswith((".weight", ".bias")) else k
            )

            if not is_qeic_target_layer(
                k_module,
                self.include_ranges,
                self.include_specific,
                self.exclude_ranges,
                self.exclude_specific,
            ):
                continue

            self.console.print(f"  [magenta]Applying QEIC to layer: {k}[/magenta]")

            all_base_corr_matrices = []
            all_sub_corr_matrices = []

            for base_model in self.base_models:
                for sub_model in self.sub_models:
                    (
                        base_corr_matrices,
                        sub_corr_matrices,
                    ) = calculate_correlation_matrices(
                        base_model,
                        sub_model,
                        [k],  # 現在のレイヤーのみ
                        self.model_dict.get("qeic_corr_method", "pearson"),
                        v_slice.device,
                    )
                    all_base_corr_matrices.extend(base_corr_matrices)
                    all_sub_corr_matrices.extend(sub_corr_matrices)
            if not all_base_corr_matrices or not all_sub_corr_matrices:
                self.console.print(
                    f"[yellow]  Warning: No correlation matrices to merge for layer {k}. Skipping QEIC.[/yellow]"
                )
                continue

            kwargs = {
                "layers": [k],
                "corr_method": self.model_dict.get("qeic_corr_method", "pearson"),
                "merge_method": self.model_dict.get("qeic_merge_method", "average"),
                "alpha_mode": self.model_dict.get("qeic_alpha_mode", "correlation"),
                "beta_mode": self.model_dict.get("qeic_beta_mode", "abs"),
                "sub_threshold": self.model_dict.get("qeic_sub_threshold", -0.1),
                "base_model": None,
                "sub_model": None,
                "device": v_slice.device,
                "base_corr_matrices": all_base_corr_matrices,
                "sub_corr_matrices": all_sub_corr_matrices,
            }

            try:
                merged_weight = OPERATION_DICTS[self.operation](
                    v_slice, base_slices[0], sub_slices[0], self.velocity, **kwargs
                )
                v_slice.copy_(merged_weight)  # 結果をコピー
            except Exception as e:
                self.console.print(f"[red]Error during QEIC operation: {e}[/red]")
                self.console.print(f"[yellow]Skipping layer {k} due to error[/yellow]")
                continue

        self._print_summary()
        return self.target
