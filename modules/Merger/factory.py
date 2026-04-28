from ..Merger.base import Merger
from ..Merger.basic_merger import BasicMerger
from ..Merger.complex_merger import ComplexMerger
from ..Merger.custom_merger import CustomMerger
from ..Merger.qeic_merger import QeicMerger
from ..Utils.merge_types import MergeContext
from ..Utils.operation_registry import get_operation_spec


class MergerFactory:
    """Merger インスタンスを生成するファクトリークラス。"""

    def __init__(self):
        self.merger_classes: Dict[str, type[Merger]] = {
            "basic": BasicMerger,
            "custom": CustomMerger,
            "complex": ComplexMerger,
            "qeic": QeicMerger,
        }

    def create_merger(self, context: MergeContext) -> Merger:
        """指定された operation に基づいて適切な Merger インスタンスを生成。"""
        operation_spec = get_operation_spec(context.request.operation)
        merger_class = self.merger_classes[operation_spec.merger_kind]
        return merger_class(context)
