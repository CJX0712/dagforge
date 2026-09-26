import numpy as np

from dagforge.eval import metrics as M


def test_structure_metrics_exact():
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0->1->2
    pred = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    met = M.structure_metrics(pred, true)
    assert met.shd == 0
    assert met.f1 == 1.0
    assert met.precision == 1.0 and met.recall == 1.0


def test_structure_metrics_extra_edge():
    true = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    pred = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])  # 多了一条 0->2
    met = M.structure_metrics(pred, true)
    assert met.shd == 1
    assert met.precision == 0.5
    assert met.recall == 1.0
    assert abs(met.f1 - 2 * 0.5 * 1.0 / 1.5) < 1e-9


def test_structure_metrics_undirected_fair():
    # 无向骨架视角：方向不同但边相同 -> F1=1（公平对比）
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    pred = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])  # 方向反转
    met = M.structure_metrics(pred, true)
    assert met.f1 == 1.0
    assert met.shd == 0


def test_aggregate():
    true = np.array([[0, 1], [0, 0]])
    met_list = [
        M.structure_metrics(np.array([[0, 1], [0, 0]]), true),
        M.structure_metrics(np.array([[0, 0], [0, 0]]), true),
    ]
    agg = M.aggregate(met_list)
    assert abs(agg["f1_mean"] - 0.5) < 1e-9
    assert agg["shd_mean"] == 0.5


def test_shd_directed_counts_orientation():
    true = np.array([[0, 1], [0, 0]])   # 0->1
    pred_wrong = np.array([[0, 0], [1, 0]])  # 1->0 方向反了
    s = M.shd_directed(pred_wrong, true)
    assert s >= 1  # 方向反转应被计为错误
