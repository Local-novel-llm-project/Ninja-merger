import pytest
import torch
import torch.nn as nn

from modules.Merger.basic_merger import BasicMerger


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
        skip_layernorm=False,
        target_model=None,
        base_models=base_models,
        sub_models=sub_models,
        velocity=1.0,  # A velocity of 1.0 results in a simple addition.
        post_velocity=1.0,
        skip_layers=[],
        operation="add",
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

    # Verification: Check if the parameters of the merged model are correct.
    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            # The merge operation is assumed to be: base + sub * velocity
            # So, the final parameter should be original_a + original_b
            expected_param = original_a_state[name] + original_b_state[name]
            assert torch.all(
                torch.eq(param.data, expected_param)
            ), f"Parameter mismatch for layer: {name}"
