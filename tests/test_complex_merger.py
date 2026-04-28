import pytest
import torch
import torch.nn as nn

from modules.Merger.complex_merger import ComplexMerger
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
        make_merge_context(
            base_models=base_models,
            sub_models=sub_models,
            velocity=0.5,
            post_velocity=1.0,
            request=make_merge_request(
                operation="complexadd",
                post_operation="none",
            ),
        )
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


def test_complex_merger_angle_merge(dummy_models):
    """
    Tests that the 'angle_merge' operation accepts its keyword-only options and
    produces the expected angle-weighted blend when sub-model directions match.
    """
    model_a, model_b, model_c = dummy_models

    merger = ComplexMerger(
        make_merge_context(
            base_models=[model_a],
            sub_models=[model_b, model_c],
            velocity=0.5,
            post_velocity=1.0,
            request=make_merge_request(
                operation="angle_merge",
                post_operation="add",
            ),
        )
    )

    merged_model = merger.merge()

    with torch.no_grad():
        # NormAngleMerge first computes t from the pairwise cosine similarity of the
        # sub-model directions, then PostAdd combines the angle-weighted base and
        # average sub tensors as: v1 * (1 - t) + avg * t * velocity.
        cosine_similarity = torch.tensor(1.0)
        t = 2 * torch.cos(cosine_similarity) / (1.0 + torch.cos(cosine_similarity))
        expected_value = (1.0 * (1.0 - t) + 2.5 * t * 0.5).item()

        for _, param in merged_model.named_parameters():
            expected_param = torch.full_like(param.data, expected_value)
            assert torch.allclose(param.data, expected_param)
