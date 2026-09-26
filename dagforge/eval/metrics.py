"""结构恢复指标。

主指标基于「无向骨架」边集合（所有方法公平可比）：
  - SHD（结构汉明距离，越小越好）= 漏报边 + 误报边
  - precision / recall / f1（基于边集合）
  - FDR = FP / (TP+FP)
另提供「有向 SHD」用于可定向方法（NOTEARS）的额外对比。
"""

import numpy as np

from ..core.errors import EvalError
from ..core.types import Metrics


def _undirected(adj: np.ndarray) -> np.ndarray:
    a = (np.asarray(adj) != 0).astype(int)
    return ((a + a.T) > 0).astype(int)


def _edge_set(adj: np.ndarray) -> set:
    a = _undirected(adj)
    n = a.shape[0]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j]:
                edges.add((i, j))
    return edges


def structure_metrics(pred_adj: np.ndarray, true_adj: np.ndarray) -> Metrics:
    if pred_adj.shape != true_adj.shape:
        raise EvalError(f"维度不匹配: {pred_adj.shape} vs {true_adj.shape}")
    P = _edge_set(pred_adj)
    T = _edge_set(true_adj)
    tp = len(P & T)
    fp = len(P - T)
    fn = len(T - P)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    shd = fp + fn
    return Metrics(
        shd=float(shd),
        precision=precision,
        recall=recall,
        f1=f1,
        fdr=fdr,
        n_edges_pred=len(P),
        n_edges_true=len(T),
    )


def shd_directed(pred_adj: np.ndarray, true_adj: np.ndarray) -> float:
    """有向 SHD：缺失边 + 误报边 + 方向反转。"""
    p = (np.asarray(pred_adj) != 0).astype(int)
    t = (np.asarray(true_adj) != 0).astype(int)
    n = p.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pe = p[i, j]
            te = t[i, j]
            if pe and te:
                # 方向是否一致（考虑对称边视为一致）
                if not (p[i, j] == t[i, j] or p[j, i] == t[j, i]):
                    s += 1
            elif pe and not te or (not pe) and te:
                s += 1
    return float(s)


def aggregate(metrics_list: list) -> dict[str, float]:
    """对多数据集的指标列表求均值与标准差。"""
    if not metrics_list:
        raise EvalError("空指标列表")
    keys = ["shd", "precision", "recall", "f1", "fdr"]
    out = {}
    for k in keys:
        vals = [getattr(m, k) for m in metrics_list]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out
