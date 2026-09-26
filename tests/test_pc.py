import numpy as np

from dagforge.core.seed import set_all
from dagforge.core.types import Dataset
from dagforge.discovery import pc
from dagforge.eval import metrics as M


def _build_dataset(W, n_samples=1500, noise=0.4, seed=0):
    d = W.shape[0]
    gen = np.random.default_rng(seed)
    X = np.zeros((n_samples, d))
    for j in range(d):
        parents = np.where(W[j] != 0)[0]
        if parents.size:
            X[:, j] = X[:, parents] @ W[j, parents]
        X[:, j] += noise * gen.standard_normal(n_samples)
    return Dataset(X=X, true_adj=(W != 0).astype(int), name="known")


def test_pc_recovers_skeleton():
    set_all(5)
    d = 6
    W = np.zeros((d, d))
    for i in range(d - 1):
        W[i + 1, i] = 1.0
    ds = _build_dataset(W, n_samples=2000, noise=0.2, seed=2)
    res = pc.pc(ds, alpha=0.05, max_cond=3)
    assert res.adj.shape == (d, d)
    met = M.structure_metrics(res.adj, ds.true_adj)
    assert met.recall >= 0.5, f"PC recall 过低: {met.recall}"


def test_pc_no_crash_on_small():
    set_all(1)
    d = 4
    W = np.zeros((d, d))
    W[1, 0] = 0.8
    W[2, 1] = -0.7
    ds = _build_dataset(W, n_samples=500, noise=0.5, seed=4)
    res = pc.pc(ds, alpha=0.1, max_cond=2)
    met = M.structure_metrics(res.adj, ds.true_adj)
    assert 0.0 <= met.f1 <= 1.0
