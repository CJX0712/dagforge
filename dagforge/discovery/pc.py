"""约束式 PC 算法（Gaussian 版）：基于偏相关的条件独立性检验恢复骨架 + 方向化。

作为对比方法之一；与 NOTEARS 评分类路线互补。纯 numpy/scipy 实现（离线可用）。
"""
import itertools
import time

import numpy as np
from scipy.stats import norm

from ..core.types import DAGResult, Dataset


def _cov_submatrix(Xc: np.ndarray, idxs: list, i: int, j: int):
    """返回 (i,j)∪S 的协方差子矩阵与偏相关系数 r_ij|S。"""
    sub = Xc[:, idxs].T
    cov = np.cov(sub)
    cov = np.atleast_2d(cov)
    if cov.shape[0] != len(idxs):
        cov = np.cov(sub, rowvar=True)
    prec = np.linalg.pinv(cov)
    ii = idxs.index(i)
    jj = idxs.index(j)
    denom = np.sqrt(max(prec[ii, ii], 1e-12) * max(prec[jj, jj], 1e-12))
    return -prec[ii, jj] / denom


def _fisher_z(r: float, n: int, n_cond: int) -> float:
    """Fisher z 统计量；|z| 越小越可能独立。"""
    if abs(r) >= 1.0:
        r = 0.999999 if r > 0 else -0.999999
    z = 0.5 * np.log((1 + r) / (1 - r))
    return z * np.sqrt(max(n - n_cond - 3, 1))


def pc(
    data: Dataset,
    alpha: float = 0.05,
    max_cond: int = 3,
) -> DAGResult:
    t0 = time.perf_counter()
    X = np.asarray(data.X, dtype=float)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    # 初始完全无向图
    skel = np.ones((d, d), dtype=int) - np.eye(d, dtype=int)
    sep = {}

    z_crit = norm.ppf(1 - alpha / 2)
    max_cond = min(max_cond, d - 2)

    for cond_size in range(max_cond + 1):
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d) if skel[i, j]]
        for (i, j) in pairs:
            # 共同邻居（当前骨架下 i,j 的公共邻接点）
            neigh = [k for k in range(d) if skel[i, k] and skel[j, k] and k != i and k != j]
            if len(neigh) < cond_size:
                continue
            removed = False
            for S in itertools.combinations(neigh, cond_size):
                idxs = [i, j] + list(S)
                r = _cov_submatrix(Xc, idxs, i, j)
                z = _fisher_z(r, n, len(S))
                if abs(z) <= z_crit:
                    skel[i, j] = skel[j, i] = 0
                    sep[(i, j)] = set(S)
                    removed = True
                    break
            if removed:
                continue

    # 方向化：v-结构检测 + 基础 Meek 规则
    adj = skel.copy()
    # 记录邻接
    def neighbors(u):
        return [k for k in range(d) if adj[u, k]]

    # v-structures: i-j-k 未屏蔽 (j not in sep(i,k)) => i->j<-k
    for j in range(d):
        for i in range(d):
            for k in range(d):
                if i == j or j == k or i == k:
                    continue
                if adj[i, j] and adj[j, k] and not adj[i, k]:
                    key = (i, k) if (i, k) in sep else (k, i)
                    if key in sep and j not in sep[key]:
                        # 定向 i->j 与 k->j
                        _orient(adj, i, j)
                        _orient(adj, k, j)
    # Meek R1: i->j & j-k => i->j (j->k)
    changed = True
    while changed:
        changed = False
        for i in range(d):
            for j in range(d):
                if adj[i, j] == 1 and adj[j, i] == 0:  # i->j
                    for k in range(d):
                        if adj[j, k] and adj[k, j]:  # j-k 无向
                            adj[j, k] = 1
                            adj[k, j] = 0
                            changed = True

    elapsed = time.perf_counter() - t0
    return DAGResult(
        adj=adj,
        method="pc",
        elapsed=elapsed,
        meta={"alpha": alpha, "max_cond": max_cond},
    )


def _orient(adj, a, b):
    """定向 a->b（adj[a,b]=1, adj[b,a]=0）。"""
    adj[a, b] = 1
    adj[b, a] = 0
