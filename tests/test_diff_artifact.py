import torch
from safetensors.torch import save_file

from modules.Merger.basic_merger import BasicMerger
from modules.Services import merge_output
from modules.Services.merge_output import save_merged_output
from modules.Utils.diff_artifact import (
    diff_artifact_path,
    parse_diff_artifact_metadata,
    resolve_diff_base_reference,
    save_sparse_diff_artifact,
    save_sparse_vector_artifact,
    snapshot_model_state_dict,
)
from modules.Utils.loaders import load_model
from modules.Utils.models import DummyModel
from modules.Utils.utility import output_artifact_exists
from tests.helpers import make_merge_context, make_merge_request


def test_sparse_diff_artifact_round_trip_reconstructs_exact_state(tmp_path):
    base_state = {
        "layer1.weight": torch.ones(2, 2, dtype=torch.float32),
        "layer1.bias": torch.zeros(2, dtype=torch.float32),
        "layer2.weight": torch.full((2, 2), 3.0, dtype=torch.float32),
    }
    base_path = tmp_path / "base.safetensors"
    save_file(base_state, str(base_path))

    base_model = load_model(str(base_path), "cpu", torch.float32)
    base_snapshot = snapshot_model_state_dict(base_model)
    merged_state = {
        "layer1.weight": base_state["layer1.weight"] + 0.5,
        "layer1.bias": base_state["layer1.bias"],
    }
    merged_model = DummyModel(
        {key: value.clone() for key, value in merged_state.items()},
        config_dict={"architectures": ["DummyModel"], "model_type": "dummy"},
    )

    save_stem = str(tmp_path / "vector" / "merge_a")
    artifact_path, changed_count, deleted_count = save_sparse_diff_artifact(
        save_stem,
        str(base_path),
        merged_model,
        base_snapshot,
    )

    loaded = load_model(artifact_path, "cpu", torch.float32)

    assert changed_count == 1
    assert deleted_count == 1
    assert isinstance(loaded, DummyModel)
    assert resolve_diff_base_reference(artifact_path) == str(base_path)
    assert set(loaded.state_dict().keys()) == set(merged_state.keys())
    for key, expected in merged_state.items():
        assert torch.equal(loaded.state_dict()[key], expected)


def test_save_merged_output_uses_difftensors_for_vector_requests(tmp_path, monkeypatch):
    merged_model = DummyModel(
        {
            "layer.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "layer.bias": torch.zeros(2, dtype=torch.float32),
        },
        config_dict={"architectures": ["DummyModel"], "model_type": "dummy"},
    )
    request = make_merge_request(left=["base-model"], target=None)
    save_stem = str(tmp_path / "vector" / "merge_b")
    tokenizer_called = False

    def fail_if_tokenizer_called(_model_name):
        nonlocal tokenizer_called
        tokenizer_called = True
        raise AssertionError("tokenizer should not be loaded for sparse diff output")

    monkeypatch.setattr(merge_output, "load_tokenizer", fail_if_tokenizer_called)

    save_merged_output(
        save_stem,
        merged_model,
        current_use_scaling=True,
        request=request,
    )

    assert not tokenizer_called
    assert output_artifact_exists(save_stem)
    assert not (tmp_path / "vector" / "merge_b.pth").exists()
    assert diff_artifact_path(save_stem) == f"{save_stem}.difftensors"
    assert (tmp_path / "vector" / "merge_b.difftensors").exists()
    metadata = parse_diff_artifact_metadata(f"{save_stem}.difftensors")
    assert metadata["artifact_type"] == "sparse_vector"
    assert metadata["source_reference"] == "base-model"


def test_difftensors_can_be_applied_to_an_explicit_target_via_none_operation(tmp_path):
    vector_state = {
        "layer1.weight": torch.full((2, 2), 0.25, dtype=torch.float32),
        "layer1.bias": torch.full((2,), -0.5, dtype=torch.float32),
        "layer2.weight": torch.zeros((2, 2), dtype=torch.float32),
    }
    target_state = {
        "layer1.weight": torch.full((2, 2), 10.0, dtype=torch.float32),
        "layer1.bias": torch.full((2,), 4.0, dtype=torch.float32),
        "layer2.weight": torch.full((2, 2), -2.0, dtype=torch.float32),
    }

    diff_stem = str(tmp_path / "vector" / "delta_apply")
    diff_path, stored_count, skipped_zero_count = save_sparse_vector_artifact(
        diff_stem,
        DummyModel({key: value.clone() for key, value in vector_state.items()}),
        source_reference="base-model",
    )

    delta_model = load_model(diff_path, "cpu", torch.float32, diff_mode="delta")
    target_model = DummyModel({key: value.clone() for key, value in target_state.items()})

    merger = BasicMerger(
        make_merge_context(
            request=make_merge_request(
                left=[diff_path],
                right=[],
                target="target-model",
                operation="none",
                post_operation="add",
                post_velocity=1.0,
            ),
            target_model=target_model,
            base_models=[delta_model],
            sub_models=[],
        )
    )

    merged_model = merger.merge()
    assert stored_count == 2
    assert skipped_zero_count == 1

    for key, target_tensor in target_state.items():
        assert torch.equal(merged_model.state_dict()[key], target_tensor + vector_state[key])
