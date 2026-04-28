from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


class MergeExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class MergeRunnerOptions:
    config: str = "model_config.yaml"
    out_dir: str = "./merged_models"
    skip_layernorm: bool = False
    merge_models_device: str = "cpu"
    target_model_device: str = "cpu"
    torch_dtype: str = "bfloat16"
    recurrent_mode: bool = True
    dry_run: bool = False
    save_only_last_model: bool = False
    dump_layers: bool = False
    include_layers: str | None = None
    exclude_layers: str | None = None


@dataclass(frozen=True)
class MergeRequest:
    raw_config: dict[str, Any]
    base_model_names: list[str]
    sub_model_names: list[str]
    target_value: Any
    target_value_scalar: Any
    operation: str
    post_operation: str
    preprocess: str
    post_preprocess: str
    normalization: str
    velocity: Any
    velocities: Any
    post_velocity: Any
    post_velocities: Any
    unmatch_size_layer_op: str
    include_layers: Any
    exclude_layers: Any
    drop_layers: Any
    use_scaling: Any
    force_merge_single: bool
    v2s_empty_default: str
    v2s_single_default: str
    is_llava_next: bool
    model_config: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self.raw_config[key]

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        return self.raw_config.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw_config)


@dataclass(frozen=True)
class AppConfig:
    models: list[MergeRequest]
    use_scaling: bool


@dataclass(frozen=True)
class MergeContext:
    request: MergeRequest
    skip_layernorm: bool
    target_model: nn.Module | None
    base_models: list[nn.Module]
    sub_models: list[nn.Module]
    velocity: Any
    post_velocity: Any
    skip_layers: list[str]
    include_layers: Any
    exclude_layers: Any


@dataclass(frozen=True)
class MergeLayerContext:
    key: str
    target_slice: torch.Tensor
    base_slices: list[torch.Tensor]
    sub_slices: list[torch.Tensor]
    velocity: Any
    post_velocity: Any


@dataclass(frozen=True)
class MergeStepResult:
    status: str
    target_model: nn.Module | None
    save_stem: str | None = None
    merged_model: nn.Module | None = None
    message: str | None = None
    request_name: str | None = None
    operation: str | None = None
