import numpy as np

from dagforge.core.seed import set_all
from dagforge.core.types import Dataset
from dagforge.discovery import notears
from dagforge.eval import metrics as M

RNG = np.random.default_rng(0)


def _build_dataset(W: np.ndarray, n_samples: int = 1500, noise: float = 0.4, seed: int = 0):
    d = W.shape[0]
    gen = np.random.default_rng(seed)
    X = np.zeros((n_samples, d))
    # 拓扑序采样（保证与 W 一致）
    for j in range(d):
        parents = np.where(W[j] != 0)[0]
        if parents.size:
            X[:, j] = X[:, parents] @ W[j, parents]
        X[:, j] += noise * gen.standard_normal(n_samples)
    return Dataset(X=X, true_adj=(W != 0).astype(int), name="known")


def _is_dag(adj):
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
        if color[i] == 0 and dfs(i, []):
            return False
    return True


def test_notears_recovers_chain():
    set_all(0)
    d = 5
    W = np.zeros((d, d))
    for i in range(d - 1):
        W[i + 1, i] = 1.0  # 0->1->2->3->4
    ds = _build_dataset(W, n_samples=2000, noise=0.3, seed=1)
    res = notears.notears_linear(
        ds, lambda1=0.05, max_iter=50, inner_iter=200, lr=0.03, w_threshold=0.3)
    assert _is_dag(res.adj), "NOTEARS 输出必须为 DAG"
    met = M.structure_metrics(res.adj, ds.true_adj)
    # 链式结构强信号，应较高召回
    assert met.recall >= 0.7, f"recall 过低: {met.recall}"


def test_notears_determinism():
    set_all(7)
    d = 4
    W = np.zeros((d, d))
    W[1, 0] = 0.8
    W[2, 1] = 0.7
    W[3, 0] = -0.6
    ds = _build_dataset(W, n_samples=800, noise=0.5, seed=3)
    cfg = {"lambda1": 0.1, "max_iter": 40, "inner_iter": 200, "lr": 0.02, "w_threshold": 0.3}
    r1 = notears.notears_linear(ds, **cfg)
    set_all(7)
    r2 = notears.notears_linear(ds, **cfg)
    assert np.allclose(r1.W, r2.W), "NOTEARS 两次运行 W 不一致（确定性问题）"
    assert np.array_equal(r1.adj, r2.adj)


def test_notears_acyclicity_invariant():
    set_all(2)
    d = 6
    W = np.zeros((d, d))
    # 故意造一个可能带环的稠密图
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)]
    for a, b in edges:
        W[b, a] = 0.9
    ds = _build_dataset(W, n_samples=600, noise=0.6, seed=9)
    res = notears.notears_linear(ds, max_iter=30, inner_iter=150, lr=0.02)
    assert _is_dag(res.adj)
