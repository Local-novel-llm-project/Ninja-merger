import copy

from modules.Utils.merge_types import MergeContext, MergeRequest


def make_merge_request(**overrides):
    raw_config = {
        "left": ["model_a.safetensors"],
        "right": ["model_b.safetensors"],
        "target": None,
        "operation": "add",
        "post_operation": "none",
        "preprocess": "none",
        "post_preprocess": "none",
        "normalization": "none",
        "velocity": 1.0,
        "post_velocity": 1.0,
        "unmatch_size_layer_op": "skip",
        "include_layers": None,
        "exclude_layers": None,
        "drop_layers": None,
        "use_scaling": None,
        "force_merge_single": False,
        "v2s_empty_default": "v1",
        "v2s_single_default": "auto",
        "model_config": {},
    }
    raw_config.update(overrides)

    target_value = raw_config.get("target")
    if isinstance(target_value, list) and len(target_value) == 1:
        target_value_scalar = target_value[0]
    else:
        target_value_scalar = target_value

    is_llava_next = (
        isinstance(target_value_scalar, str)
        and target_value_scalar.lower() in ["llava", "vlm", "llava-next"]
    )

    return MergeRequest(
        raw_config=copy.deepcopy(raw_config),
        name=raw_config.get("name"),
        base_model_names=list(raw_config["left"]),
        sub_model_names=list(raw_config.get("right", [])),
        target_value=target_value,
        target_value_scalar=target_value_scalar,
        operation=raw_config.get("operation", "sub"),
        post_operation=raw_config.get("post_operation", "add"),
        preprocess=raw_config.get("preprocess", "none"),
        post_preprocess=raw_config.get("post_preprocess", "none"),
        normalization=raw_config.get("normalization", "none"),
        velocity=raw_config.get("velocity"),
        velocities=raw_config.get("velocities"),
        post_velocity=raw_config.get("post_velocity", 1.0),
        post_velocities=raw_config.get("post_velocities"),
        unmatch_size_layer_op=raw_config.get("unmatch_size_layer_op", "skip"),
        include_layers=raw_config.get("include_layers"),
        exclude_layers=raw_config.get("exclude_layers"),
        drop_layers=raw_config.get("drop_layers"),
        use_scaling=raw_config.get("use_scaling"),
        force_merge_single=raw_config.get("force_merge_single", False),
        v2s_empty_default=raw_config.get("v2s_empty_default", "v1"),
        v2s_single_default=raw_config.get("v2s_single_default", "auto"),
        is_llava_next=is_llava_next,
        model_config=copy.deepcopy(raw_config.get("model_config", {})),
    )


def make_merge_context(
    *,
    base_models,
    sub_models,
    target_model=None,
    request=None,
    velocity=1.0,
    post_velocity=1.0,
    skip_layernorm=False,
    skip_layers=None,
    include_layers=None,
    exclude_layers=None,
):
    request = request or make_merge_request(
        velocity=velocity,
        post_velocity=post_velocity,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
    )
    return MergeContext(
        request=request,
        skip_layernorm=skip_layernorm,
        target_model=target_model,
        base_models=base_models,
        sub_models=sub_models,
        velocity=velocity,
        post_velocity=post_velocity,
        skip_layers=skip_layers or [],
        include_layers=include_layers,
        exclude_layers=exclude_layers,
    )
