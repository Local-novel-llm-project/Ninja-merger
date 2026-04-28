import pytest
import torch
import torch.nn as nn

from modules.Merger.basic_merger import BasicMerger
from modules.Utils.utility import define_savename
from tests.helpers import make_merge_context, make_merge_request


class SimpleModel(nn.Module):
    """A simple neural network with two linear layers for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        """Forward pass."""
        return self.layer2(self.layer1(x))


@pytest.fixture
def dummy_models():
    """Provides two simple models with predictable initial weights."""
    model_a = SimpleModel()
    model_b = SimpleModel()
    with torch.no_grad():
        for param in model_a.parameters():
            param.fill_(1.0)
        for param in model_b.parameters():
            param.fill_(2.0)
    return model_a, model_b


def test_basic_merger_add(dummy_models):
    """
    Tests the 'add' operation of the BasicMerger.
    Verifies that the parameters of the merged model are the sum of the
    parameters of the two input models.
    """
    model_a, model_b = dummy_models

    # Save the original state of model_a because the merge is done in-place.
    original_a_state = {k: v.clone() for k, v in model_a.state_dict().items()}
    original_b_state = model_b.state_dict()

    base_models = [model_a]
    sub_models = [model_b]

    # Instantiate the BasicMerger.
    # target_model is None, so base_models[0] (model_a) will be used as the target.
    merger = BasicMerger(
        make_merge_context(
            base_models=base_models,
            sub_models=sub_models,
            velocity=1.0,
            post_velocity=1.0,
            request=make_merge_request(
                operation="add",
                post_operation="none",
            ),
        )
    )

    merged_model = merger.merge()

    # Verification: Check if the parameters of the merged model are correct.
    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            # The merge operation is assumed to be: base + sub * velocity
            # So, the final parameter should be original_a + original_b
            expected_param = original_a_state[name] + original_b_state[name]
            assert torch.all(
                torch.eq(param.data, expected_param)
            ), f"Parameter mismatch for layer: {name}"


def test_basic_merger_passthrough_without_sub_models(dummy_models):
    """
    Tests the 'passthrough' operation of the BasicMerger.
    Verifies that a single base model can be passed through without any sub model.
    """
    model_a, _ = dummy_models
    original_a_state = {k: v.clone() for k, v in model_a.state_dict().items()}

    merger = BasicMerger(
        make_merge_context(
            base_models=[model_a],
            sub_models=[],
            velocity=1.0,
            post_velocity=1.0,
            request=make_merge_request(
                operation="passthrough",
                right=[],
                post_operation="none",
            ),
        )
    )

    merged_model = merger.merge()

    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            assert torch.all(
                torch.eq(param.data, original_a_state[name])
            ), f"Parameter mismatch for layer: {name}"


def test_basic_merger_adds_sparse_vector_only_once_for_tied_target_weights():
    shared_target = torch.full((2, 2), 10.0, dtype=torch.float32)
    target_model = nn.Module()
    target_model.state_dict = lambda: {
        "lm_head.weight": shared_target,
        "model.language_model.embed_tokens.weight": shared_target,
    }

    delta_model = nn.Module()
    delta_model.state_dict = lambda: {
        "lm_head.weight": torch.full((2, 2), 1.0, dtype=torch.float32),
        "model.language_model.embed_tokens.weight": torch.full(
            (2, 2), 1.0, dtype=torch.float32
        ),
    }
    delta_model._ninja_sparse_zero_missing = True

    merger = BasicMerger(
        make_merge_context(
            request=make_merge_request(
                left=["delta-model"],
                right=[],
                target="target-model",
                operation="none",
                post_operation="add",
            ),
            target_model=target_model,
            base_models=[delta_model],
            sub_models=[],
        )
    )

    merged_model = merger.merge()
    merged_state = merged_model.state_dict()
    expected = torch.full((2, 2), 11.0, dtype=torch.float32)

    assert torch.equal(merged_state["lm_head.weight"], expected)
    assert torch.equal(merged_state["model.language_model.embed_tokens.weight"], expected)


def test_basic_merger_sub_vector_generation_only_once_for_tied_base_weights():
    shared_left = torch.full((2, 2), 10.0, dtype=torch.float32)
    shared_right = torch.full((2, 2), 9.0, dtype=torch.float32)

    left_model = nn.Module()
    left_model.state_dict = lambda: {
        "lm_head.weight": shared_left,
        "model.language_model.embed_tokens.weight": shared_left,
    }

    right_model = nn.Module()
    right_model.state_dict = lambda: {
        "lm_head.weight": shared_right,
        "model.language_model.embed_tokens.weight": shared_right,
    }

    merger = BasicMerger(
        make_merge_context(
            request=make_merge_request(
                left=["left-model"],
                right=["right-model"],
                target=None,
                operation="sub",
                post_operation="none",
            ),
            base_models=[left_model],
            sub_models=[right_model],
        )
    )

    merged_model = merger.merge()
    merged_state = merged_model.state_dict()
    expected = torch.full((2, 2), 1.0, dtype=torch.float32)

    assert torch.equal(merged_state["lm_head.weight"], expected)
    assert torch.equal(merged_state["model.language_model.embed_tokens.weight"], expected)


def test_basic_merger_drop_layers_removes_keys_from_state_dict(dummy_models):
    model_a, _ = dummy_models

    merger = BasicMerger(
        make_merge_context(
            base_models=[model_a],
            sub_models=[],
            velocity=1.0,
            post_velocity=1.0,
            request=make_merge_request(
                operation="passthrough",
                right=[],
                post_operation="none",
                drop_layers=["layer1"],
            ),
        )
    )

    merged_model = merger.merge()
    merged_state = merged_model.state_dict()

    assert "layer1.weight" not in merged_state
    assert "layer1.bias" not in merged_state
    assert "layer2.weight" in merged_state
    assert "layer2.bias" in merged_state


def test_prepare_model_metadata_passthrough_allows_missing_right():
    metadata = make_merge_request(
        left=["model_a.safetensors"],
        right=[],
        operation="passthrough",
        target=None,
    )

    assert metadata["base_model_names"] == ["model_a.safetensors"]
    assert metadata["sub_model_names"] == []
    assert metadata.base_model_names == ["model_a.safetensors"]
    assert metadata.sub_model_names == []


def test_define_savename_passthrough_without_right(tmp_path):
    save_name = define_savename(
        ["model_a.safetensors"],
        [],
        None,
        str(tmp_path),
        0,
        {"left": ["model_a.safetensors"], "operation": "passthrough"},
    )

    assert save_name == str(tmp_path / "vector" / "model_a_pass")
    assert not save_name.endswith(".safetensors")
