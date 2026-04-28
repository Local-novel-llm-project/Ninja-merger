import torch

from modules.Utils.layers import get_skip_layers, is_layer_dropped, parse_layer_specifications


def test_parse_layer_specifications_accepts_whitespace_separated_keys():
    ranges, specific_layers = parse_layer_specifications(
        "model.audio_tower. model.vision_tower."
    )

    assert ranges == []
    assert specific_layers == ["model.audio_tower.", "model.vision_tower."]


def test_is_layer_dropped_matches_whitespace_separated_keys():
    drop_ranges, drop_specific = parse_layer_specifications(
        "model.audio_tower. model.vision_tower."
    )

    assert is_layer_dropped(
        "model.audio_tower.layers.2.lconv1d.linear_end.input_min",
        drop_ranges,
        drop_specific,
    )


def test_get_skip_layers_treats_sparse_diff_sources_as_zero_for_missing_keys():
    class TinyModel:
        def __init__(self, state_dict, *, sparse_zero_missing=False):
            self._state_dict = state_dict
            self._ninja_sparse_zero_missing = sparse_zero_missing

        def state_dict(self):
            return self._state_dict

    target_model = TinyModel(
        {
            "layer.weight": torch.ones(2, 2),
            "layer.bias": torch.ones(2),
        }
    )
    sparse_delta = TinyModel(
        {
            "layer.weight": torch.ones(2, 2),
        },
        sparse_zero_missing=True,
    )

    skip_layers = get_skip_layers(
        target_model=target_model,
        base_models=[sparse_delta],
        sub_models=[],
        unmatch_size_layer_op="skip",
    )

    assert "layer.bias" not in skip_layers
