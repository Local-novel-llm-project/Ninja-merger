import pytest

from modules.Utils.loaders import load_config
from modules.Utils.utility import define_savename


def test_load_config_accepts_single_mapping_models(tmp_path):
    config_path = tmp_path / "single.yaml"
    config_path.write_text(
        """
models:
  target: target-model
  left: left-model
  right: right-model
  operation: sub
""".strip()
        + "\n",
        encoding="utf-8",
    )

    app_config = load_config(str(config_path))

    assert app_config.use_scaling is False
    assert len(app_config.models) == 1
    assert app_config.models[0].base_model_names == ["left-model"]
    assert app_config.models[0].sub_model_names == ["right-model"]
    assert app_config.models[0].target_value == "target-model"
    assert app_config.models[0].velocity == 1.0
    assert app_config.models[0].post_velocity == 1.0


def test_load_config_rejects_non_mapping_model_entries(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("models:\n  - just-a-string\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Each entry in 'models' must be a mapping"):
        load_config(str(config_path))


def test_load_config_rejects_duplicate_names(tmp_path):
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(
        """
models:
  - name: shared-name
    left: left-a
    right: right-a
    operation: sub
  - name: shared-name
    left: left-b
    right: right-b
    operation: add
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate model name 'shared-name'"):
        load_config(str(config_path))


def test_define_savename_creates_name_subdirectory(tmp_path):
    save_name = define_savename(
        base=["left-model"],
        sub=["right-model"],
        target="/models/target-model",
        out_dir=str(tmp_path),
        index=0,
        model_dict={"name": "merge-alpha", "operation": "sub"},
    )

    assert save_name == str(tmp_path / "target-model" / "merge-alpha")
    assert (tmp_path / "target-model").is_dir()
    assert not save_name.endswith(".safetensors")


def test_define_savename_uses_flat_vector_stem_and_adds_timestamp_on_collision(tmp_path):
    save_name = define_savename(
        base=["left-model"],
        sub=["right-model"],
        target=None,
        out_dir=str(tmp_path),
        index=0,
        model_dict={"name": "vector-alpha", "operation": "sub"},
    )

    assert save_name == str(tmp_path / "vector" / "vector-alpha")
    assert (tmp_path / "vector").is_dir()

    (tmp_path / "vector" / "vector-alpha.difftensors").write_bytes(b"existing")

    collided_name = define_savename(
        base=["left-model"],
        sub=["right-model"],
        target=None,
        out_dir=str(tmp_path),
        index=1,
        model_dict={"name": "vector-alpha", "operation": "sub"},
    )

    assert collided_name.startswith(str(tmp_path / "vector" / "vector-alpha_"))


def test_load_config_treats_none_aliases_as_empty_right_and_null_target(tmp_path):
    config_path = tmp_path / "diff_apply.yaml"
    config_path.write_text(
        """
models:
  - name: apply-vector
    target: none
    left: vector.difftensors
    right: none
    operation: none
    post_operation: add
""".strip()
        + "\n",
        encoding="utf-8",
    )

    app_config = load_config(str(config_path))
    request = app_config.models[0]

    assert request.target_value is None
    assert request.target_value_scalar is None
    assert request.sub_model_names == []
    assert request.operation == "none"


def test_prepare_models_allows_none_operation_without_right():
    from modules.Utils.models import load_and_prepare_models
    from tests.helpers import make_merge_request

    class MinimalModel:
        def state_dict(self):
            return {"weight": 1}

    request = make_merge_request(
        left=["left-model"],
        right=[],
        operation="none",
        post_operation="add",
    )

    original_resolve = load_and_prepare_models.__globals__["_resolve_model_sequence"]
    original_prepare_velocities = None
    original_prepare_post_velocities = None
    try:
        load_and_prepare_models.__globals__["_resolve_model_sequence"] = (
            lambda model_names, request, merge_models_device, torch_dtype, recurrent_model=None: [MinimalModel()]
            if model_names
            else []
        )
        layers_module = __import__(
            "modules.Utils.layers",
            fromlist=["prepare_velocities", "prepare_post_velocities"],
        )
        original_prepare_velocities = layers_module.prepare_velocities
        original_prepare_post_velocities = layers_module.prepare_post_velocities
        layers_module.prepare_velocities = lambda velocities, state_dict: velocities
        layers_module.prepare_post_velocities = lambda velocities, state_dict: velocities

        prepared = load_and_prepare_models(request, "cpu", None)
    finally:
        load_and_prepare_models.__globals__["_resolve_model_sequence"] = original_resolve
        if original_prepare_velocities is not None:
            layers_module.prepare_velocities = original_prepare_velocities
        if original_prepare_post_velocities is not None:
            layers_module.prepare_post_velocities = original_prepare_post_velocities

    assert len(prepared.base_models) == 1
    assert prepared.sub_models == []
