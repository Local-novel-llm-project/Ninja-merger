from typing import Any, Dict, List, Union

import torch
from Merger.base import Merger
from Merger.basic_merger import BasicMerger
from Merger.complex_merger import ComplexMerger
from Merger.custom_merger import CustomMerger
from Merger.qeic_merger import QeicMerger


class MergerFactory:
    """Merger インスタンスを生成するファクトリークラス。"""

    def __init__(self):
        self.merger_classes: Dict[str, type[Merger]] = {
            "add": BasicMerger,
            "sub": BasicMerger,
            "mul": BasicMerger,
            "div": BasicMerger,
            "mix": BasicMerger,
            "avg": BasicMerger,
            "concat": BasicMerger,
            "maxpool": BasicMerger,
            "minpool": BasicMerger,
            "geometric_mean": BasicMerger,
            "std_sub": BasicMerger,
            "widen": CustomMerger,
            "complexadd": ComplexMerger,
            "angle_merge": ComplexMerger,
            "complex_angle_merge": ComplexMerger,
            "qeic_add": QeicMerger,
            "qeic_mix": QeicMerger,
            "qeic_sub": QeicMerger,
        }

    def create_merger(
        self,
        operation: str,
        skip_layernorm: bool,
        target_model: torch.nn.Module,
        base_models: List[torch.nn.Module],
        sub_models: List[torch.nn.Module],
        velocity: Union[float, complex, torch.Tensor],
        post_velocity: float,
        skip_layers: List[str],
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
    ) -> Merger:
        """指定された operation に基づいて適切な Merger インスタンスを生成。"""

        if operation in self.merger_classes:
            merger_class = self.merger_classes[operation]
        else:
            raise ValueError(f"Unsupported merge operation: {operation}")

        return merger_class(
            skip_layernorm,
            target_model,
            base_models,
            sub_models,
            velocity,
            post_velocity,
            skip_layers,
            operation,
            post_operation,
            preprocess,
            post_preprocess,
            normalization,
            include_layers,
            exclude_layers,
            drop_layers,
            unmatch_size_layer_op,
            is_llava_next,
            force_merge_single,
            v2s_empty_default,
            v2s_single_default,
            model_dict,
        )
