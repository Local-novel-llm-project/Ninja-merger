from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..Calc.basic_calc import (
    Add,
    Avg,
    Concatenation,
    Div,
    GeometricMean,
    MaxPool,
    MinPool,
    Mix,
    Mul,
    Passthrough,
    PostAdd,
    PostAngle,
    PostConcatenation,
    PostDiv,
    PostDivBy,
    PostGeometricMean,
    PostMaxPool,
    PostMinPool,
    PostMix,
    PostMul,
    PostSub,
    PostSubFrom,
    StdSub,
    Sub,
)
from ..Calc.complex_calc import ComplexAdd, ComplexAngleMerge, ComplexMix
from ..Calc.custom_calc import WidenMerge
from ..Calc.norm_calc import MatchStdMean, NormAngleMerge, NormStdMean, ProcStdMean
from ..Calc.process_calc import GitReBasin, QuantileMatch
from ..Calc.qeic_calc import QeicAdd, QeicMix, QeicSub


@dataclass(frozen=True)
class OperationSpec:
    func: Callable[..., Any]
    merger_kind: str
    short_name: str
    supports_preprocess: bool = False
    supports_post_preprocess: bool = False
    supports_normalization: bool = False
    supports_post_operation: bool = False
    requires_sub_models: bool = True
    implemented: bool = True


@dataclass(frozen=True)
class CallableSpec:
    func: Callable[..., Any]


def _build_callable_dict(
    registry: dict[str, OperationSpec | CallableSpec],
) -> dict[str, Callable[..., Any]]:
    return {name: spec.func for name, spec in registry.items()}


def _identity_preprocess(tensor, *args, **kwargs):
    return tensor


def _identity_normalization(
    tensor,
    normalization_type,
    original_tensor,
    base_processed_values,
    *args,
    **kwargs,
):
    return tensor


def _wrap_normalizer(normalizer: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(
        tensor,
        normalization_type,
        original_tensor,
        base_processed_values,
        *args,
        **kwargs,
    ):
        return normalizer(
            lambda v1, v2s, velocity: tensor,
            original_tensor,
            base_processed_values,
            0,
            **kwargs,
        )

    return wrapped


OPERATION_REGISTRY: dict[str, OperationSpec] = {
    "add": OperationSpec(
        func=Add,
        merger_kind="basic",
        short_name="add",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "sub": OperationSpec(
        func=Sub,
        merger_kind="basic",
        short_name="sub",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "mul": OperationSpec(
        func=Mul,
        merger_kind="basic",
        short_name="mul",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "div": OperationSpec(
        func=Div,
        merger_kind="basic",
        short_name="div",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "mix": OperationSpec(
        func=Mix,
        merger_kind="basic",
        short_name="mix",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "avg": OperationSpec(
        func=Avg,
        merger_kind="basic",
        short_name="avg",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "passthrough": OperationSpec(
        func=Passthrough,
        merger_kind="basic",
        short_name="pass",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
        requires_sub_models=False,
    ),
    "none": OperationSpec(
        func=Passthrough,
        merger_kind="basic",
        short_name="none",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
        requires_sub_models=False,
    ),
    "concat": OperationSpec(
        func=Concatenation,
        merger_kind="basic",
        short_name="concat",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "maxpool": OperationSpec(
        func=MaxPool,
        merger_kind="basic",
        short_name="maxpool",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "minpool": OperationSpec(
        func=MinPool,
        merger_kind="basic",
        short_name="minpool",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "geometric_mean": OperationSpec(
        func=GeometricMean,
        merger_kind="basic",
        short_name="gmean",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "std_sub": OperationSpec(
        func=StdSub,
        merger_kind="basic",
        short_name="stdsub",
        supports_preprocess=True,
        supports_normalization=True,
        supports_post_operation=True,
    ),
    "widen": OperationSpec(
        func=WidenMerge,
        merger_kind="custom",
        short_name="widen",
    ),
    "complexadd": OperationSpec(
        func=ComplexAdd,
        merger_kind="complex",
        short_name="cadd",
    ),
    "angle_merge": OperationSpec(
        func=NormAngleMerge,
        merger_kind="complex",
        short_name="angm",
        supports_post_operation=True,
    ),
    "complex_angle_merge": OperationSpec(
        func=ComplexAngleMerge,
        merger_kind="complex",
        short_name="cam",
    ),
    "complex_mix": OperationSpec(
        func=ComplexMix,
        merger_kind="complex",
        short_name="cmix",
        implemented=False,
    ),
    "qeic_add": OperationSpec(
        func=QeicAdd,
        merger_kind="qeic",
        short_name="qadd",
    ),
    "qeic_mix": OperationSpec(
        func=QeicMix,
        merger_kind="qeic",
        short_name="qmix",
    ),
    "qeic_sub": OperationSpec(
        func=QeicSub,
        merger_kind="qeic",
        short_name="qsub",
    ),
}

OPERATION_DICT = _build_callable_dict(OPERATION_REGISTRY)


PREPROCESS_REGISTRY: dict[str, CallableSpec] = {
    "none": CallableSpec(func=_identity_preprocess),
    "pcd": CallableSpec(func=GitReBasin),
}

PREPROCESS_DICT = _build_callable_dict(PREPROCESS_REGISTRY)


NORMALIZATION_REGISTRY: dict[str, CallableSpec] = {
    "none": CallableSpec(func=_identity_normalization),
    "norm_std_mean": CallableSpec(func=_wrap_normalizer(NormStdMean)),
    "match_std_mean": CallableSpec(func=_wrap_normalizer(MatchStdMean)),
    "proc_std_mean": CallableSpec(func=_wrap_normalizer(ProcStdMean)),
    "angle_merge": CallableSpec(func=_wrap_normalizer(NormAngleMerge)),
    "quantile_match": CallableSpec(func=_wrap_normalizer(QuantileMatch)),
}

NORMALIZATION_DICT = _build_callable_dict(NORMALIZATION_REGISTRY)


POST_OPERATION_REGISTRY: dict[str, CallableSpec] = {
    "add": CallableSpec(func=PostAdd),
    "sub": CallableSpec(func=PostSub),
    "subfrom": CallableSpec(func=PostSubFrom),
    "mul": CallableSpec(func=PostMul),
    "div": CallableSpec(func=PostDiv),
    "divby": CallableSpec(func=PostDivBy),
    "mix": CallableSpec(func=PostMix),
    "concat": CallableSpec(func=PostConcatenation),
    "maxpool": CallableSpec(func=PostMaxPool),
    "minpool": CallableSpec(func=PostMinPool),
    "geometric_mean": CallableSpec(func=PostGeometricMean),
    "angle": CallableSpec(func=PostAngle),
}

POST_OPERATION_DICT = _build_callable_dict(POST_OPERATION_REGISTRY)


def get_operation_spec(operation: str) -> OperationSpec:
    try:
        return OPERATION_REGISTRY[operation]
    except KeyError as error:
        raise ValueError(f"Unsupported merge operation: {operation}") from error


def get_operation_short_name(operation: str, default: str = "unk") -> str:
    spec = OPERATION_REGISTRY.get(operation)
    return spec.short_name if spec is not None else default


def validate_operation_request(request) -> None:
    spec = get_operation_spec(request.operation)

    if not spec.implemented:
        raise ValueError(f"Operation '{request.operation}' is registered but not implemented")

    if request.preprocess != "none" and not spec.supports_preprocess:
        raise ValueError(
            f"Operation '{request.operation}' does not support preprocess='{request.preprocess}'"
        )

    if request.post_preprocess != "none" and not spec.supports_post_preprocess:
        raise ValueError(
            "post_preprocess is not implemented for "
            f"operation '{request.operation}'"
        )

    if request.normalization != "none" and not spec.supports_normalization:
        raise ValueError(
            f"Operation '{request.operation}' does not support normalization='{request.normalization}'"
        )

    if request.post_operation != "none" and not spec.supports_post_operation:
        raise ValueError(
            f"Operation '{request.operation}' does not support post_operation='{request.post_operation}'"
        )

    if spec.requires_sub_models and not request.sub_model_names:
        raise ValueError(
            f"Operation '{request.operation}' requires at least one right model"
        )


__all__ = [
    "CallableSpec",
    "NORMALIZATION_DICT",
    "NORMALIZATION_REGISTRY",
    "OPERATION_DICT",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "POST_OPERATION_DICT",
    "POST_OPERATION_REGISTRY",
    "PREPROCESS_DICT",
    "PREPROCESS_REGISTRY",
    "get_operation_short_name",
    "get_operation_spec",
    "validate_operation_request",
]
