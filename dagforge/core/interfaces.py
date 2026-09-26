"""接口契约：StructureLearner Protocol + 后端探针接口。"""
from typing import Protocol, runtime_checkable

from .types import DAGResult, Dataset


@runtime_checkable
class StructureLearner(Protocol):
    """任何因果发现方法都应满足：给定 Dataset，产出 DAGResult。"""
    name: str

    def fit(self, data: Dataset) -> DAGResult:
        ...


class BackendProbe:
    """探测可选 SOTA 后端是否可用（try/except ImportError 隔离）。"""

    @staticmethod
    def available_causallearn() -> bool:
        try:
            import causallearn  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def available_scipy() -> bool:
        try:
            import scipy  # noqa: F401
            return True
        except Exception:
            return False
