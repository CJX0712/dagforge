"""Tier-0 可选 SOTA 后端包装：causal-learn（PC / GES 官方复刻）+ gcastle（NOTEARS）。

全部用 try/except 隔离 —— 后端缺失时返回 None，pipeline 探测后自动跳过并标注，
绝不伪造数字。成功安装时，作为与自研实现的独立交叉验证参照。
"""
import numpy as np

from ..core.types import DAGResult, Dataset


def available_causallearn() -> bool:
    try:
        import causallearn  # noqa: F401
        return True
    except Exception:
        return False


def available_gcastle() -> bool:
    try:
        from castle.algorithms import Notears  # noqa: F401  (无 torch 时 import 失败)
        return True
    except Exception:
        return False


def _ensure_acyclic(W: np.ndarray) -> np.ndarray:
    adj = (np.abs(W) > 1e-12).astype(int)

    def find_cycle():
        n = adj.shape[0]
        color = [0] * n

        def dfs(u, stack):
            color[u] = 1
            stack.append(u)
            for v in range(n):
                if adj[u, v]:
                    if color[v] == 1:
                        return stack[stack.index(v):]
                    if color[v] == 0:
                        r = dfs(v, stack)
                        if r:
                            return r
            color[u] = 2
            stack.pop()
            return None

        for i in range(n):
            if color[i] == 0:
                r = dfs(i, [])
                if r:
                    return r
        return None

    while True:
        cyc = find_cycle()
        if not cyc:
            break
        best_w, best = None, None
        for k in range(len(cyc)):
            u, v = cyc[k], cyc[(k + 1) % len(cyc)]
            w = abs(W[u, v])
            if best_w is None or w < best_w:
                best_w, best = w, (u, v)
        u, v = best
        W[u, v] = 0.0
        adj[u, v] = 0
    return W


def causallearn_notears(data: Dataset, lambda1: float = 0.1, w_threshold: float = 0.3) -> DAGResult:
    """官方 NOTEARS 复刻（若存在）。本环境 causal-learn 未含 NOTEARS → 返回 None。"""
    try:
        from causallearn.search.ScoreBased import NOTEARS  # 部分版本无此模块
    except Exception:
        return None
    X = np.asarray(data.X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    model = NOTEARS(X, lambda1=lambda1, loss_type="l2")
    W = np.asarray(model.W, dtype=float)
    Wt = _ensure_acyclic(W.copy())
    Wt[np.abs(Wt) < w_threshold] = 0.0
    adj = (np.abs(Wt) > 0).astype(int)
    return DAGResult(adj=adj, W=Wt, method="cl_notears", meta={"backend": "causal-learn"})


def causallearn_pc(data: Dataset, alpha: float = 0.05) -> DAGResult:
    """官方 PC 复刻，作为约束式方法的独立交叉验证参照。缺失时返回 None。"""
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except Exception:
        return None
    X = np.asarray(data.X, dtype=float)
    cg = pc(X, alpha=alpha, indep_test="fisherz", stable=True)
    graph = np.asarray(cg.G.graph, dtype=int)
    adj = (graph != 0).astype(int)
    np.fill_diagonal(adj, 0)
    return DAGResult(adj=adj, method="cl_pc", meta={"backend": "causal-learn"})


def gcastle_notears(data: Dataset, lambda1: float = 0.1, w_threshold: float = 0.3) -> DAGResult:
    """gcastle 的 NOTEARS 实现（需 torch）。缺失时返回 None。"""
    try:
        from castle.algorithms import Notears
    except Exception:
        return None
    X = np.asarray(data.X, dtype=float)
    model = Notears()
    model.learn(X)
    W = np.asarray(model.causal_matrix, dtype=float)
    Wt = _ensure_acyclic(W.copy())
    Wt[np.abs(Wt) < w_threshold] = 0.0
    adj = (np.abs(Wt) > 0).astype(int)
    return DAGResult(adj=adj, method="gcastle_notears", meta={"backend": "gcastle"})
