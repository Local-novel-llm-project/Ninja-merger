import torch

from ..Utils.operation_dicts import PREPROCESS_DICT  # PREPROCESS_DICTをimport


def preprocess_tensor(tensor: torch.Tensor, preprocess_type: str, *args, **kwargs):
    """指定された前処理をテンソルに適用する。"""
    preprocess_func = PREPROCESS_DICT.get(preprocess_type)
    if preprocess_func:
        return preprocess_func(tensor, *args, **kwargs)
    return tensor
