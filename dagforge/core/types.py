"""核心数据类型：Dataset / DAGResult / Metrics。"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    """观测数据集。X: (n_samples, n_features)；true_adj 为可选的真值 DAG 邻接。"""
    X: np.ndarray
    feature_names: list = field(default_factory=list)
    name: str = "synthetic"
    true_adj: np.ndarray | None = None

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if not self.feature_names:
            self.feature_names = [f"X{i}" for i in range(self.X.shape[1])]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]


@dataclass
class DAGResult:
    """单方法因果发现结果。adj 为阈值化后的有向邻接（0/1），W 为原始权值矩阵。"""
    adj: np.ndarray
    W: np.ndarray | None = None
    method: str = "unknown"
    elapsed: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.adj = np.asarray(self.adj, dtype=int)


@dataclass
class Metrics:
    """结构恢复指标。SHD 越小越好；precision/recall/f1 越大越好。"""
    shd: float
    precision: float
    recall: float
    f1: float
    fdr: float
    n_edges_pred: int = 0
    n_edges_true: int = 0
    shd_directed: float | None = None
