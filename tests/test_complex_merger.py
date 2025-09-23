import pytest
import torch
import torch.nn as nn

from modules.Merger.complex_merger import ComplexMerger


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
    model_c = SimpleModel()
    with torch.no_grad():
        for param in model_a.parameters():
            param.fill_(1.0)
        for param in model_b.parameters():
            param.fill_(2.0)
        for param in model_c.parameters():
            param.fill_(3.0)
    return model_a, model_b, model_c


def test_complex_merger_complexadd(dummy_models):
    """
    Tests the 'complexadd' operation of the ComplexMerger.
    """
    model_a, model_b, model_c = dummy_models

    original_a_state = {k: v.clone() for k, v in model_a.state_dict().items()}
    original_b_state = model_b.state_dict()
    original_c_state = model_c.state_dict()

    base_models = [model_a]
    sub_models = [model_b, model_c]

    merger = ComplexMerger(
        skip_layernorm=False,
        target_model=None,
        base_models=base_models,
        sub_models=sub_models,
        velocity=0.5,
        post_velocity=1.0,
        skip_layers=[],
        operation="complexadd",
        post_operation="none",
        preprocess="none",
        post_preprocess="none",
        normalization="none",
        include_layers=None,
        exclude_layers=None,
        drop_layers=None,
        unmatch_size_layer_op="skip",
        model_dict={},
    )

    merged_model = merger.merge()

    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            # complexadd: v_slice + (avg - v_slice) * t * velocity
            # avg = (2.0 + 3.0) / 2 = 2.5
            # t = 0.1 (default in complex_merger)
            # velocity = 0.5
            # expected = 1.0 + (2.5 - 1.0) * 0.1 * 0.5 = 1.0 + 1.5 * 0.05 = 1.0 + 0.075 = 1.075
            avg_sub = (original_b_state[name] + original_c_state[name]) / 2
            t = 0.1
            velocity = 0.5
            expected_param = original_a_state[name] + (avg_sub - original_a_state[name]) * t * velocity
            assert torch.allclose(
                param.data, expected_param
            ), f"Parameter mismatch for layer: {name}"
