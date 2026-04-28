import pytest
import torch
import torch.nn as nn

from modules.Merger.basic_merger import BasicMerger
from modules.Merger.complex_merger import ComplexMerger
from modules.Proc.post_processing import post_process_tensor
from modules.Utils.operation_dicts import (
    NORMALIZATION_REGISTRY,
    OPERATION_REGISTRY,
    POST_OPERATION_REGISTRY,
)
from tests.helpers import make_merge_context, make_merge_request


class StructuredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)


def _fill_model(model: nn.Module, offset: float) -> None:
    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            values = torch.linspace(
                1.0 + offset + index,
                2.0 + offset + index,
                steps=param.numel(),
                dtype=param.dtype,
            ).reshape_as(param)
            param.copy_(values)


def _make_structured_models():
    model_a = StructuredModel()
    model_b = StructuredModel()
    _fill_model(model_a, 0.0)
    _fill_model(model_b, 2.0)
    return model_a, model_b


def _make_three_structured_models():
    model_a = StructuredModel()
    model_b = StructuredModel()
    model_c = StructuredModel()
    _fill_model(model_a, 0.0)
    _fill_model(model_b, 2.0)
    _fill_model(model_c, 4.0)
    return model_a, model_b, model_c


def _ref_add(base, sub, velocity):
    return (base + sub) * velocity


def _ref_sub(base, sub, velocity):
    return (base - sub) * velocity


def _ref_mul(base, sub, velocity):
    return (base * sub) * velocity


def _ref_div(base, sub, velocity):
    return (base / sub) * velocity


def _ref_mix(base, sub, velocity):
    return base * (1.0 - velocity) + sub * velocity


def _ref_avg(base, sub, velocity):
    return (base + sub) * 0.5


def _ref_passthrough(base, sub, velocity):
    return base.clone()


def _ref_maxpool(base, sub, velocity):
    return torch.maximum(base, sub)


def _ref_minpool(base, sub, velocity):
    return torch.minimum(base, sub)


def _ref_geometric_mean(base, sub, velocity):
    return torch.sqrt(base * sub)


def _ref_std_sub(base, sub, velocity):
    eps = 1e-7
    base_std, base_mean = torch.std_mean(base)
    sub_std, sub_mean = torch.std_mean(sub)
    return (
        ((base - base_mean) / max(base_std, eps))
        - ((sub - sub_mean) / max(sub_std, eps))
    ) * velocity


def _ref_post_add(original, processed, velocity):
    return original + processed * velocity


def _ref_post_sub(original, processed, velocity):
    return original - processed * velocity


def _ref_post_subfrom(original, processed, velocity):
    return processed - original * velocity


def _ref_post_mul(original, processed, velocity):
    return original * processed * velocity


def _ref_post_div(original, processed, velocity):
    return original / processed * velocity


def _ref_post_divby(original, processed, velocity):
    return processed * velocity / original


def _ref_post_mix(original, processed, velocity):
    return original * (1.0 - velocity) + processed * velocity


def _ref_post_maxpool(original, processed, velocity):
    return torch.maximum(original, processed * velocity)


def _ref_post_minpool(original, processed, velocity):
    return torch.minimum(original, processed * velocity)


def _ref_post_geometric_mean(original, processed, velocity):
    return torch.sqrt(original * processed * velocity)


def _ref_norm_none(processed, original, base_processed_values):
    return processed


def _ref_norm_std_mean(processed, original, base_processed_values):
    eps = 1e-7
    std, mean = torch.std_mean(processed)
    return (processed - mean) / max(std, eps)


def _ref_match_std_mean(processed, original, base_processed_values):
    eps = 1e-7
    orig_std, orig_mean = torch.std_mean(original)
    std, mean = torch.std_mean(processed)
    return (processed - (mean - orig_mean)) * (max(std, eps) / max(orig_std, eps))


def _ref_proc_std_mean(processed, original, base_processed_values):
    eps = 1e-7
    orig_std, orig_mean = torch.std_mean(original)
    return processed * max(orig_std, eps) + orig_mean


def _ref_norm_angle_merge(processed, original, base_processed_values):
    return processed


BASIC_REFERENCE_CASES = {
    "add": _ref_add,
    "sub": _ref_sub,
    "mul": _ref_mul,
    "div": _ref_div,
    "mix": _ref_mix,
    "avg": _ref_avg,
    "passthrough": _ref_passthrough,
    "none": _ref_passthrough,
    "maxpool": _ref_maxpool,
    "minpool": _ref_minpool,
    "geometric_mean": _ref_geometric_mean,
    "std_sub": _ref_std_sub,
}

BASIC_REFERENCE_EXEMPTIONS = {
    "concat": "changes tensor shape and cannot be copied back into the target slice",
}

POST_REFERENCE_CASES = {
    "add": _ref_post_add,
    "sub": _ref_post_sub,
    "subfrom": _ref_post_subfrom,
    "mul": _ref_post_mul,
    "div": _ref_post_div,
    "divby": _ref_post_divby,
    "mix": _ref_post_mix,
    "maxpool": _ref_post_maxpool,
    "minpool": _ref_post_minpool,
    "geometric_mean": _ref_post_geometric_mean,
}

POST_REFERENCE_EXEMPTIONS = {
    "concat": "changes tensor shape and is not a stable in-place merger target",
    "angle": "reduces along the last dimension and is not used as a standard post-process copy-back path",
}

NORMALIZATION_REFERENCE_CASES = {
    "none": _ref_norm_none,
    "norm_std_mean": _ref_norm_std_mean,
    "match_std_mean": _ref_match_std_mean,
    "proc_std_mean": _ref_proc_std_mean,
    "angle_merge": _ref_norm_angle_merge,
}

NORMALIZATION_REFERENCE_EXEMPTIONS = {
    "quantile_match": "distribution matching needs a dedicated oracle because it performs rank-based interpolation",
}

COMPLEX_REFERENCE_CASES = {
    "complexadd": "covered",
    "angle_merge": "covered",
}

COMPLEX_REFERENCE_EXEMPTIONS = {
    "complex_angle_merge": "complex-phase interpolation needs a dedicated oracle for stable expected values",
}


def _run_basic_merge(
    operation: str,
    *,
    velocity: float = 0.25,
    post_operation: str = "none",
    post_velocity: float = 1.0,
):
    model_a, model_b = _make_structured_models()
    original_a_state = {k: v.clone() for k, v in model_a.state_dict().items()}
    original_b_state = {k: v.clone() for k, v in model_b.state_dict().items()}

    right_models = [] if operation == "passthrough" else ["model_b.safetensors"]
    sub_models = [] if operation == "passthrough" else [model_b]

    merger = BasicMerger(
        make_merge_context(
            base_models=[model_a],
            sub_models=sub_models,
            velocity=velocity,
            post_velocity=post_velocity,
            request=make_merge_request(
                operation=operation,
                right=right_models,
                post_operation=post_operation,
            ),
        )
    )

    merged_model = merger.merge()
    return merged_model.state_dict(), original_a_state, original_b_state


def _run_basic_merge_with_normalization(normalization: str, *, velocity: float = 0.5):
    model_a, model_b, model_c = _make_three_structured_models()
    original_a_state = {k: v.clone() for k, v in model_a.state_dict().items()}
    original_b_state = {k: v.clone() for k, v in model_b.state_dict().items()}
    original_c_state = {k: v.clone() for k, v in model_c.state_dict().items()}

    merger = BasicMerger(
        make_merge_context(
            base_models=[model_a],
            sub_models=[model_b, model_c],
            velocity=velocity,
            post_velocity=1.0,
            request=make_merge_request(
                operation="add",
                right=["model_b.safetensors", "model_c.safetensors"],
                post_operation="none",
                normalization=normalization,
            ),
        )
    )

    merged_model = merger.merge()
    return merged_model.state_dict(), original_a_state, original_b_state, original_c_state


def _run_complex_merge(operation: str, *, velocity: float, post_operation: str = "none"):
    model_a, model_b, model_c = _make_three_structured_models()
    original_a_state = {k: v.clone() for k, v in model_a.state_dict().items()}
    original_b_state = {k: v.clone() for k, v in model_b.state_dict().items()}
    original_c_state = {k: v.clone() for k, v in model_c.state_dict().items()}

    merger = ComplexMerger(
        make_merge_context(
            base_models=[model_a],
            sub_models=[model_b, model_c],
            velocity=velocity,
            post_velocity=1.0,
            request=make_merge_request(
                operation=operation,
                right=["model_b.safetensors", "model_c.safetensors"],
                post_operation=post_operation,
            ),
        )
    )

    merged_model = merger.merge()
    return merged_model.state_dict(), original_a_state, original_b_state, original_c_state


def _reference_angle_merge(base, sub_tensors, velocity, post_operation):
    pairwise_cosines = [
        (left * right).sum(dim=-1)
        / (torch.norm(left, dim=-1) * torch.norm(right, dim=-1)).clamp(min=1e-7)
        for left, right in [(sub_tensors[0], sub_tensors[1])]
    ]
    theta = torch.stack(pairwise_cosines, dim=-1).mean(dim=-1).unsqueeze(-1)
    t = len(sub_tensors) * torch.cos(theta) / (
        1.0 + (len(sub_tensors) - 1) * torch.cos(theta)
    )
    avg = sum(sub_tensors) / len(sub_tensors)
    blended_base = base * (1.0 - t)
    blended_sub = avg * t

    if post_operation == "none":
        return blended_sub

    return POST_REFERENCE_CASES[post_operation](blended_base, blended_sub, velocity)


def _assert_state_dict_matches_reference(merged_state, expected_by_name):
    for name, tensor in merged_state.items():
        assert torch.allclose(
            tensor, expected_by_name[name], atol=1e-5, rtol=1e-5
        ), f"Mismatch in {name}"


def test_basic_reference_cases_cover_all_implemented_basic_operations():
    implemented_basic = {
        name
        for name, spec in OPERATION_REGISTRY.items()
        if spec.merger_kind == "basic" and spec.implemented
    }
    assert implemented_basic == (
        set(BASIC_REFERENCE_CASES) | set(BASIC_REFERENCE_EXEMPTIONS)
    )


@pytest.mark.parametrize("operation", sorted(BASIC_REFERENCE_CASES))
@pytest.mark.parametrize("velocity", [0.25, 0.75])
def test_generated_basic_merger_matches_reference(operation, velocity):
    merged_state, original_a_state, original_b_state = _run_basic_merge(
        operation,
        velocity=velocity,
    )

    reference_fn = BASIC_REFERENCE_CASES[operation]
    expected = {}
    for name, original_tensor in original_a_state.items():
        sub_tensor = (
            original_b_state[name] if operation != "passthrough" else original_tensor
        )
        expected[name] = reference_fn(original_tensor, sub_tensor, velocity)

    _assert_state_dict_matches_reference(merged_state, expected)


def test_post_reference_cases_cover_all_supported_shape_stable_post_operations():
    assert set(POST_OPERATION_REGISTRY) == (
        set(POST_REFERENCE_CASES) | set(POST_REFERENCE_EXEMPTIONS)
    )


@pytest.mark.parametrize("post_operation", sorted(POST_REFERENCE_CASES))
@pytest.mark.parametrize("post_velocity", [0.25, 0.75])
def test_generated_post_processing_matches_reference(post_operation, post_velocity):
    original = torch.tensor(
        [[1.5, 2.0], [2.5, 3.0]],
        dtype=torch.float32,
    )
    processed = torch.tensor(
        [[3.0, 3.5], [4.0, 4.5]],
        dtype=torch.float32,
    )

    expected = POST_REFERENCE_CASES[post_operation](
        original,
        processed,
        post_velocity,
    )
    actual = post_process_tensor(
        processed.clone(),
        post_operation,
        post_velocity,
        original.clone(),
    )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_normalization_reference_cases_cover_all_supported_normalizations():
    assert set(NORMALIZATION_REGISTRY) == (
        set(NORMALIZATION_REFERENCE_CASES) | set(NORMALIZATION_REFERENCE_EXEMPTIONS)
    )


@pytest.mark.parametrize("normalization", sorted(NORMALIZATION_REFERENCE_CASES))
@pytest.mark.parametrize("velocity", [0.25, 0.75])
def test_generated_basic_merger_normalization_matches_reference(
    normalization, velocity
):
    merged_state, original_a_state, original_b_state, original_c_state = (
        _run_basic_merge_with_normalization(normalization, velocity=velocity)
    )

    reference_fn = NORMALIZATION_REFERENCE_CASES[normalization]
    expected = {}
    for name, original_tensor in original_a_state.items():
        base_processed_values = [
            _ref_add(original_tensor, original_b_state[name], velocity),
            _ref_add(original_tensor, original_c_state[name], velocity),
        ]
        processed = torch.stack(base_processed_values, dim=0).mean(dim=0)
        expected[name] = reference_fn(processed, original_tensor, base_processed_values)

    _assert_state_dict_matches_reference(merged_state, expected)


def test_complex_reference_cases_cover_all_implemented_complex_operations():
    implemented_complex = {
        name
        for name, spec in OPERATION_REGISTRY.items()
        if spec.merger_kind == "complex" and spec.implemented
    }
    assert implemented_complex == (
        set(COMPLEX_REFERENCE_CASES) | set(COMPLEX_REFERENCE_EXEMPTIONS)
    )


@pytest.mark.parametrize("velocity", [0.25, 0.75])
def test_generated_complexadd_matches_reference(velocity):
    merged_state, original_a_state, original_b_state, original_c_state = (
        _run_complex_merge("complexadd", velocity=velocity)
    )

    expected = {}
    for name, original_tensor in original_a_state.items():
        avg_sub = (original_b_state[name] + original_c_state[name]) / 2
        expected[name] = original_tensor + (avg_sub - original_tensor) * 0.1 * velocity

    _assert_state_dict_matches_reference(merged_state, expected)


@pytest.mark.parametrize("velocity", [0.25, 0.75])
@pytest.mark.parametrize("post_operation", ["none", "add"])
def test_generated_angle_merge_matches_reference(post_operation, velocity):
    merged_state, original_a_state, original_b_state, original_c_state = (
        _run_complex_merge(
            "angle_merge",
            velocity=velocity,
            post_operation=post_operation,
        )
    )

    expected = {}
    for name, original_tensor in original_a_state.items():
        expected[name] = _reference_angle_merge(
            original_tensor,
            [original_b_state[name], original_c_state[name]],
            velocity,
            post_operation,
        )

    _assert_state_dict_matches_reference(merged_state, expected)
