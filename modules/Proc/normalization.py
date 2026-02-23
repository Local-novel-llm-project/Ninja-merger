from typing import List, Optional

import torch
from ..Utils.operation_dicts import NORMALIZATION_DICT  # NORMALIZATION_DICTをimport


def normalize_tensor(
    tensor: torch.Tensor,
    normalization_type: str,
    original_tensor: torch.Tensor,
    base_processed_values: Optional[List[torch.Tensor]] = None,
    *args,
    **kwargs,
):
    """指定された正規化をテンソルに適用する。"""
    if base_processed_values is None:
        base_processed_values = []
    normalize_func = NORMALIZATION_DICT.get(normalization_type)
    if normalize_func:
        return normalize_func(
            tensor,
            normalization_type,
            original_tensor,
            base_processed_values,
            *args,
            **kwargs,
        )
    return tensor
