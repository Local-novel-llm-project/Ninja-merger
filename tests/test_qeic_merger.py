from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from modules.Merger.qeic_merger import QeicMerger
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
        for name, param in model_a.named_parameters():
            param.data = torch.ones_like(param.data)
        for name, param in model_b.named_parameters():
            param.data = torch.ones_like(param.data) * 2
    return model_a, model_b


@patch("modules.Merger.qeic_merger.calculate_correlation_matrices")
@patch("modules.Merger.qeic_merger.is_qeic_target_layer")
def test_qeic_merger_basic(mock_is_target, mock_calc_corr, dummy_models):
    """
    Tests the basic flow of the QeicMerger.
    Mocks the complex parts (correlation calculation, target layer check)
    to focus on the overall merge logic.
    """
    mock_is_target.return_value = True
    # Mock the correlation calculation to return dummy tensors
    mock_calc_corr.return_value = (
        [torch.ones(10, 10) * 0.5],
        [torch.ones(10, 10) * 0.8],
    )

    model_a, model_b = dummy_models
    original_a_state = {k: v.clone() for k, v in model_a.state_dict().items()}

    base_models = [model_a]
    sub_models = [model_b]

    merger = QeicMerger(
        make_merge_context(
            base_models=base_models,
            sub_models=sub_models,
            velocity=0.5,
            post_velocity=1.0,
            request=make_merge_request(
                operation="qeic_merge",
                post_operation="none",
                qeic_corr_method="pearson",
                qeic_merge_method="average",
                qeic_alpha_mode="correlation",
                qeic_beta_mode="abs",
                qeic_sub_threshold=-0.1,
            ),
        )
    )

    # Mock the operation dict to avoid running the actual complex calculation
    with patch("modules.Merger.qeic_merger.OPERATION_DICT") as mock_op_dict:
        # Let's define a simple mock function for 'qeic_merge'
        def simple_qeic_merge(v_slice, base_slice, sub_slice, velocity, **kwargs):
            # A simple linear interpolation for testing purposes
            return base_slice + (sub_slice - base_slice) * velocity

        mock_op_dict.__getitem__.return_value = simple_qeic_merge

        merged_model = merger.merge()

        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                # Expected: 1.0 + (2.0 - 1.0) * 0.5 = 1.5
                expected_param = torch.ones_like(param.data) * 1.5
                assert torch.allclose(param.data, expected_param), f"Parameter mismatch for layer: {name}"
