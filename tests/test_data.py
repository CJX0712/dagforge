import numpy as np

from dagforge.core.seed import set_all
from dagforge.core.types import Dataset
from dagforge.data import synthetic


def _is_dag(adj: np.ndarray) -> bool:
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


def test_generate_dataset_is_dag():
    set_all(1)
    ds = synthetic.generate_dataset(n_nodes=10, edge_prob=0.4, n_samples=500, seed=1)
    assert ds.X.shape == (500, 10)
    assert ds.true_adj.shape == (10, 10)
    assert _is_dag(ds.true_adj)


def test_suite_deterministic_and_distinct():
    set_all(42)
    a = synthetic.generate_benchmark_suite(n_graphs=4, base_seed=42, n_nodes=8)
    b = synthetic.generate_benchmark_suite(n_graphs=4, base_seed=42, n_nodes=8)
    for da, db in zip(a, b):
        assert np.allclose(da.X, db.X)
        assert np.allclose(da.true_adj, db.true_adj)
    # 不同 seed 衍生出不同数据
    c = synthetic.generate_benchmark_suite(n_graphs=4, base_seed=43, n_nodes=8)
    assert not np.allclose(a[0].X, c[0].X)


def test_dataset_post_init_feature_names():
    ds = Dataset(X=np.zeros((5, 3)))
    assert ds.feature_names == ["X0", "X1", "X2"]
