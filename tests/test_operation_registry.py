import pytest
import torch.nn as nn

from modules.Merger.basic_merger import BasicMerger
from modules.Merger.complex_merger import ComplexMerger
from modules.Merger.custom_merger import CustomMerger
from modules.Merger.factory import MergerFactory
from modules.Merger.qeic_merger import QeicMerger
from modules.Utils.operation_dicts import (
    NORMALIZATION_DICT,
    NORMALIZATION_REGISTRY,
    OPERATION_DICT,
    OPERATION_REGISTRY,
    POST_OPERATION_DICT,
    POST_OPERATION_REGISTRY,
    PREPROCESS_DICT,
    PREPROCESS_REGISTRY,
    validate_operation_request,
)
from modules.Utils.operation_registry import (
    get_operation_short_name,
    get_operation_spec,
)
from tests.helpers import make_merge_context, make_merge_request


def test_operation_registry_exposes_expected_metadata():
    assert get_operation_spec("add").merger_kind == "basic"
    assert get_operation_spec("widen").merger_kind == "custom"
    assert get_operation_spec("angle_merge").merger_kind == "complex"
    assert get_operation_spec("qeic_add").merger_kind == "qeic"
    assert get_operation_short_name("passthrough") == "pass"
    assert get_operation_spec("none").merger_kind == "basic"
    assert get_operation_short_name("none") == "none"
    assert get_operation_short_name("missing-op") == "unk"


def test_operation_registry_rejects_unsupported_operation():
    with pytest.raises(ValueError, match="Unsupported merge operation: missing-op"):
        get_operation_spec("missing-op")


def test_callable_dicts_are_derived_from_registries():
    assert OPERATION_DICT["add"] is OPERATION_REGISTRY["add"].func
    assert PREPROCESS_DICT["pcd"] is PREPROCESS_REGISTRY["pcd"].func
    assert NORMALIZATION_DICT["none"] is NORMALIZATION_REGISTRY["none"].func
    assert POST_OPERATION_DICT["add"] is POST_OPERATION_REGISTRY["add"].func


def test_operation_registry_validates_unsupported_modifiers():
    with pytest.raises(
        ValueError,
        match="does not support post_operation='add'",
    ):
        validate_operation_request(
            make_merge_request(
                operation="widen",
                post_operation="add",
            )
        )


def test_merger_factory_uses_registry_to_select_merger_class():
    factory = MergerFactory()
    model = nn.Linear(1, 1)

    assert isinstance(
        factory.create_merger(
            make_merge_context(
                request=make_merge_request(operation="add", post_operation="none"),
                target_model=model,
                base_models=[model],
                sub_models=[model],
            )
        ),
        BasicMerger,
    )
    assert isinstance(
        factory.create_merger(
            make_merge_context(
                request=make_merge_request(operation="widen", post_operation="none"),
                target_model=model,
                base_models=[model],
                sub_models=[model],
            )
        ),
        CustomMerger,
    )
    assert isinstance(
        factory.create_merger(
            make_merge_context(
                request=make_merge_request(operation="angle_merge", post_operation="none"),
                target_model=model,
                base_models=[model],
                sub_models=[model],
            )
        ),
        ComplexMerger,
    )
    assert isinstance(
        factory.create_merger(
            make_merge_context(
                request=make_merge_request(operation="qeic_add", post_operation="none"),
                target_model=model,
                base_models=[model],
                sub_models=[model],
            )
        ),
        QeicMerger,
    )
