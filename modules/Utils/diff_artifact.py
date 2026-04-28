from __future__ import annotations

import copy
import json
import os
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .models import DummyModel


DIFF_TENSOR_EXTENSION = ".difftensors"
DIFF_ARTIFACT_TYPE = "sparse_delta"
VECTOR_ARTIFACT_TYPE = "sparse_vector"
DIFF_FORMAT_VERSION = "1"
DIFF_SENTINEL_KEY = "__ninja_empty__"


def is_diff_artifact_path(model_path) -> bool:
    return str(model_path).endswith(DIFF_TENSOR_EXTENSION)


def diff_artifact_path(save_stem: str) -> str:
    return f"{save_stem}{DIFF_TENSOR_EXTENSION}"


def is_sparse_diff_request(request) -> bool:
    return request.target_value_scalar is None or request.target_value_scalar == "null"


def snapshot_model_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _config_to_plain_dict(config) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return copy.deepcopy(config)
    if hasattr(config, "to_dict"):
        return copy.deepcopy(config.to_dict())
    return copy.deepcopy(getattr(config, "__dict__", {}))


def read_diff_artifact_metadata(model_path: str) -> dict[str, str]:
    with safe_open(model_path, framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata()
    return dict(metadata or {})


def parse_diff_artifact_metadata(model_path: str) -> dict[str, object]:
    metadata = read_diff_artifact_metadata(model_path)
    return {
        "artifact_type": metadata.get("artifact_type"),
        "format_version": metadata.get("format_version"),
        "base_reference": metadata.get("base_reference"),
        "base_reference_resolved": metadata.get("base_reference_resolved"),
        "source_reference": metadata.get("source_reference"),
        "source_reference_resolved": metadata.get("source_reference_resolved"),
        "tensor_modes": json.loads(metadata.get("tensor_modes_json", "{}")),
        "deleted_keys": json.loads(metadata.get("deleted_keys_json", "[]")),
        "merged_config": json.loads(metadata.get("merged_config_json", "{}")),
    }


def resolve_diff_base_reference(model_path: str) -> str:
    metadata = parse_diff_artifact_metadata(model_path)
    base_reference_resolved = metadata.get("base_reference_resolved")
    if base_reference_resolved and os.path.exists(base_reference_resolved):
        return base_reference_resolved

    source_reference_resolved = metadata.get("source_reference_resolved")
    if source_reference_resolved and os.path.exists(source_reference_resolved):
        return source_reference_resolved

    base_reference = metadata.get("base_reference")
    if base_reference:
        return base_reference

    source_reference = metadata.get("source_reference")
    if source_reference:
        return source_reference

    raise ValueError(f"Diff artifact is missing reference metadata: {model_path}")


def _build_diff_metadata(base_reference: str, merged_model, tensor_modes, deleted_keys):
    metadata = {
        "artifact_type": DIFF_ARTIFACT_TYPE,
        "format_version": DIFF_FORMAT_VERSION,
        "base_reference": base_reference,
        "tensor_modes_json": json.dumps(tensor_modes, sort_keys=True),
        "deleted_keys_json": json.dumps(sorted(deleted_keys)),
    }

    if os.path.exists(base_reference):
        metadata["base_reference_resolved"] = os.path.abspath(base_reference)

    config_payload = _config_to_plain_dict(getattr(merged_model, "config", None))
    if config_payload:
        metadata["merged_config_json"] = json.dumps(config_payload, sort_keys=True)

    return metadata


def _build_vector_metadata(source_reference: str | None, merged_model):
    metadata = {
        "artifact_type": VECTOR_ARTIFACT_TYPE,
        "format_version": DIFF_FORMAT_VERSION,
    }

    if source_reference:
        metadata["source_reference"] = source_reference
        if os.path.exists(source_reference):
            metadata["source_reference_resolved"] = os.path.abspath(source_reference)

    config_payload = _config_to_plain_dict(getattr(merged_model, "config", None))
    if config_payload:
        metadata["merged_config_json"] = json.dumps(config_payload, sort_keys=True)

    return metadata


def _can_store_as_delta(base_tensor: torch.Tensor, merged_tensor: torch.Tensor) -> bool:
    return (
        base_tensor.shape == merged_tensor.shape
        and base_tensor.dtype == merged_tensor.dtype
        and (base_tensor.is_floating_point() or base_tensor.is_complex())
    )


def _validate_base_reference(base_reference: str):
    if not base_reference:
        raise ValueError("Sparse diff artifacts require a base model reference")
    if str(base_reference).startswith("recurrent"):
        raise ValueError(
            "Sparse diff artifacts cannot be saved from a recurrent base reference"
        )


def save_sparse_diff_artifact(
    save_stem: str,
    base_reference: str,
    merged_model,
    base_state_snapshot: dict[str, torch.Tensor],
):
    _validate_base_reference(base_reference)

    merged_state = merged_model.state_dict()
    merged_keys = set(merged_state.keys())
    base_keys = set(base_state_snapshot.keys())
    deleted_keys = sorted(base_keys - merged_keys)

    tensor_modes = {}
    stored_tensors = {}

    for key, merged_tensor in merged_state.items():
        merged_tensor_cpu = merged_tensor.detach().cpu()
        base_tensor = base_state_snapshot.get(key)

        if base_tensor is not None and torch.equal(merged_tensor_cpu, base_tensor):
            continue

        if base_tensor is not None and _can_store_as_delta(base_tensor, merged_tensor_cpu):
            stored_tensors[key] = (merged_tensor_cpu - base_tensor).contiguous()
            tensor_modes[key] = "delta"
            continue

        stored_tensors[key] = merged_tensor_cpu.clone().contiguous()
        tensor_modes[key] = "replace"

    if not stored_tensors:
        stored_tensors[DIFF_SENTINEL_KEY] = torch.empty(0, dtype=torch.uint8)

    artifact_path = diff_artifact_path(save_stem)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    save_file(
        stored_tensors,
        artifact_path,
        metadata=_build_diff_metadata(
            base_reference,
            merged_model,
            tensor_modes,
            deleted_keys,
        ),
    )

    return artifact_path, len(tensor_modes), len(deleted_keys)


def save_sparse_vector_artifact(
    save_stem: str,
    merged_model,
    source_reference: str | None = None,
):
    merged_state = merged_model.state_dict()
    stored_tensors = {}

    for key, merged_tensor in merged_state.items():
        merged_tensor_cpu = merged_tensor.detach().cpu()
        if torch.count_nonzero(merged_tensor_cpu).item() == 0:
            continue
        stored_tensors[key] = merged_tensor_cpu.clone().contiguous()

    if not stored_tensors:
        stored_tensors[DIFF_SENTINEL_KEY] = torch.empty(0, dtype=torch.uint8)

    artifact_path = diff_artifact_path(save_stem)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    save_file(
        stored_tensors,
        artifact_path,
        metadata=_build_vector_metadata(source_reference, merged_model),
    )

    stored_count = len(stored_tensors) - int(DIFF_SENTINEL_KEY in stored_tensors)
    skipped_zero_count = len(merged_state) - stored_count
    return artifact_path, stored_count, skipped_zero_count


def load_sparse_diff_artifact(
    model_path: str,
    device,
    torch_dtype,
    base_loader: Callable[[str, str, torch.dtype], object],
):
    metadata = parse_diff_artifact_metadata(model_path)
    if metadata.get("artifact_type") == VECTOR_ARTIFACT_TYPE:
        return load_sparse_vector_model(model_path)
    if metadata.get("artifact_type") != DIFF_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported diff artifact type: {metadata.get('artifact_type')}")

    base_reference = resolve_diff_base_reference(model_path)
    base_model = base_loader(base_reference, device, torch_dtype)
    base_state = base_model.state_dict()
    diff_tensors = load_file(model_path, device="cpu")

    tensor_modes = metadata["tensor_modes"]
    deleted_keys = set(metadata["deleted_keys"])
    merged_state = {
        key: value.detach().clone()
        for key, value in base_state.items()
        if key not in deleted_keys
    }

    for key, stored_tensor in diff_tensors.items():
        if key == DIFF_SENTINEL_KEY:
            continue

        mode = tensor_modes.get(key, "replace")
        if mode == "delta":
            if key not in base_state:
                raise KeyError(
                    f"Diff artifact contains delta for missing base key '{key}'"
                )
            base_tensor = base_state[key]
            merged_state[key] = base_tensor + stored_tensor.to(
                device=base_tensor.device, dtype=base_tensor.dtype
            )
            continue

        if mode != "replace":
            raise ValueError(f"Unsupported sparse diff tensor mode: {mode}")

        if key in base_state:
            base_tensor = base_state[key]
            merged_state[key] = stored_tensor.to(
                device=base_tensor.device, dtype=base_tensor.dtype
            ).clone()
        else:
            merged_state[key] = stored_tensor.clone()

    merged_config = metadata["merged_config"]
    if not merged_config:
        merged_config = _config_to_plain_dict(getattr(base_model, "config", None))

    return DummyModel(merged_state, config_dict=merged_config)


def load_sparse_diff_as_delta_model(model_path: str):
    metadata = parse_diff_artifact_metadata(model_path)
    if metadata.get("artifact_type") == VECTOR_ARTIFACT_TYPE:
        return load_sparse_vector_model(model_path)
    if metadata.get("artifact_type") != DIFF_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported diff artifact type: {metadata.get('artifact_type')}")

    tensor_modes = metadata["tensor_modes"]
    deleted_keys = metadata["deleted_keys"]
    replace_keys = sorted(key for key, mode in tensor_modes.items() if mode != "delta")
    if replace_keys or deleted_keys:
        raise ValueError(
            "Diff artifact cannot be used as an additive delta source because it "
            f"contains replace/delete entries. replace={replace_keys}, deleted={deleted_keys}"
        )

    diff_tensors = load_file(model_path, device="cpu")
    delta_state = {
        key: value.clone()
        for key, value in diff_tensors.items()
        if key != DIFF_SENTINEL_KEY
    }

    delta_model = DummyModel(
        delta_state,
        config_dict=metadata["merged_config"] or {"architectures": ["DummyModel"]},
    )
    delta_model._ninja_sparse_zero_missing = True
    delta_model._ninja_diff_base_reference = resolve_diff_base_reference(model_path)
    delta_model._ninja_diff_source_path = model_path
    return delta_model


def load_sparse_vector_model(model_path: str):
    metadata = parse_diff_artifact_metadata(model_path)
    if metadata.get("artifact_type") != VECTOR_ARTIFACT_TYPE:
        raise ValueError(
            f"Unsupported vector artifact type: {metadata.get('artifact_type')}"
        )

    vector_tensors = load_file(model_path, device="cpu")
    vector_state = {
        key: value.clone()
        for key, value in vector_tensors.items()
        if key != DIFF_SENTINEL_KEY
    }
    vector_model = DummyModel(
        vector_state,
        config_dict=metadata["merged_config"] or {"architectures": ["DummyModel"]},
    )
    vector_model._ninja_sparse_zero_missing = True
    vector_model._ninja_diff_source_path = model_path
    vector_model._ninja_diff_base_reference = resolve_diff_base_reference(model_path)
    return vector_model
