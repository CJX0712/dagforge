"""强基线（naive）：相关阈值法。

直接用 |Pearson 相关| 阈值化得到（无向）边集合。作为对比基线：
无因果语义、不保证无环、对中介效应产生假阳性，是 NOTEARS 要显著超越的对象。
"""
import time

import numpy as np

from ..core.types import DAGResult, Dataset


def corr_threshold(
    data: Dataset,
    threshold: float = 0.3,
) -> DAGResult:
    t0 = time.perf_counter()
    C = np.corrcoef(data.X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    adj = (np.abs(C) > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    elapsed = time.perf_counter() - t0
    return DAGResult(
        adj=adj,
        W=C,
        method="corr_threshold",
        elapsed=elapsed,
        meta={"threshold": threshold},
    )


def empty_graph(data: Dataset) -> DAGResult:
    """下界基线：空图（无边）。用于锚定 SHD/f1 的理论最差对照。"""
    t0 = time.perf_counter()
    n = data.X.shape[1]
    adj = np.zeros((n, n), dtype=int)
    elapsed = time.perf_counter() - t0
    return DAGResult(adj=adj, method="empty", elapsed=elapsed, meta={})
