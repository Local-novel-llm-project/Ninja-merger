from typing import Callable

import torch
from ..Utils.operation_dicts import POST_OPERATION_DICT  # POST_OPERATION_DICTをimport


def post_process_tensor(
    tensor: torch.Tensor,
    post_operation_type: str,
    post_velocity: float,
    original_tensor: torch.Tensor,
):
    """指定された後処理をテンソルに適用する。"""

    post_op_func: Callable = POST_OPERATION_DICT.get(post_operation_type)
    if post_op_func:
        return post_op_func(original_tensor, tensor, post_velocity)
    return tensor
