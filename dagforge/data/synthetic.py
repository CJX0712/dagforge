"""合成 DAG 生成 + 线性 Gaussian 结构方程（SEM）采样。

确定性：所有随机性由 seed 派生的 numpy Generator 控制，无全局状态污染。
生成保证拓扑序（边仅由低索引指向高索引），因此 adj 必然无环。
"""

import numpy as np

from ..core.seed import get_seed
from ..core.types import Dataset


def generate_dag(n_nodes: int, edge_prob: float, gen: np.random.Generator) -> np.ndarray:
    """随机 Erdos-Renyi DAG：边 i->j 仅当 i<j，保证无环。返回 (n,n) 0/1 邻接。"""
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if gen.random() < edge_prob:
                adj[j, i] = 1.0
    return adj


def generate_dataset(
    n_nodes: int = 12,
    edge_prob: float = 0.3,
    n_samples: int = 800,
    sem_noise: float = 1.0,
    seed: int | None = None,
    name: str = "synthetic",
    weight_scale: float = 1.0,
) -> Dataset:
    """生成线性 Gaussian SEM 数据集，真值 DAG 已标注。

    X_j = sum_{i<j} W[j,i] X_i + N(0, sem_noise^2)
    其中 W 仅在边位置非零，权重符号与大小随机。
    """
    if seed is None:
        seed = get_seed()
    gen = np.random.default_rng(seed)

    adj = generate_dag(n_nodes, edge_prob, gen)
    # 仅在边位置赋予非零权重（符号与幅度随机）
    sign = gen.choice([-1.0, 1.0], size=(n_nodes, n_nodes))
    mag = gen.uniform(0.4, 1.4, size=(n_nodes, n_nodes))
    W = np.where(adj > 0, sign * mag * weight_scale, 0.0)

    X = np.zeros((n_samples, n_nodes))
    for j in range(n_nodes):
        parents = np.where(adj[j] > 0)[0]
        if parents.size:
            X[:, j] = X[:, parents] @ W[j, parents]
        X[:, j] += sem_noise * gen.standard_normal(n_samples)

    feature_names = [f"X{i}" for i in range(n_nodes)]
    return Dataset(X=X, feature_names=feature_names, name=name, true_adj=adj)


def generate_benchmark_suite(
    n_graphs: int = 8,
    base_seed: int = 42,
    n_nodes: int = 12,
    edge_prob: float = 0.3,
    n_samples: int = 800,
    sem_noise: float = 1.0,
) -> list:
    """生成一组（确定性）基准数据集，seed 由 base_seed 派生，互不重叠。"""
    suite = []
    for k in range(n_graphs):
        ds = generate_dataset(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            n_samples=n_samples,
            sem_noise=sem_noise,
            seed=base_seed * 1000 + k + 1,
            name=f"g{k:02d}",
        )
        suite.append(ds)
    return suite
