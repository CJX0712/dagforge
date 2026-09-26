"""旗舰方法 NOTEARS（Zheng et al., NeurIPS 2018）—— 纯 numpy 评分式因果发现。

将 DAG 学习转化为带「无环约束」的连续优化：
    min_W  (1/2n) ||X - XW||_F^2 + λ||W||_1
    s.t.   h(W) = tr(e^{W∘W}) - d = 0
用增广 Lagrangian + L-BFGS-B（平滑 L1 近似 + 变量界 ±5 防 expm 溢出）求解，
约束通过矩阵指数 h(W) 表达（DAG ⇔ h(W)=0）。本实现不依赖任何重型库
（仅 numpy + scipy），即离线兜底本身即 SOTA。
"""
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from ..core.errors import DiscoveryError
from ..core.types import DAGResult, Dataset


def _h(W: np.ndarray) -> float:
    """无环约束 h(W) = tr(exp(W∘W)) - d，h(W)=0 当且仅当 W 对应 DAG。"""
    return float(np.trace(expm(W * W)) - W.shape[0])


def _grad_h(W: np.ndarray) -> np.ndarray:
    """h 对 W 的梯度：exp(W∘W).T ∘ (2W)。"""
    return expm(W * W).T * (2.0 * W)


def _ensure_acyclic(W: np.ndarray) -> np.ndarray:
    """若阈值后含环，反复移除环上 |W| 最小的边，直到输出为 DAG。"""
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
                        idx = stack.index(v)
                        return stack[idx:]
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


def notears_linear(
    data: Dataset,
    lambda1: float = 0.1,
    rho_init: float = 1.0,
    max_iter: int = 60,
    inner_iter: int = 300,
    lr: float = 0.02,
    h_tol: float = 1e-6,
    w_threshold: float = 0.3,
    rho_max: float = 1e16,
) -> DAGResult:
    """NOTEARS 主入口。返回阈值化后的有向邻接（保证无环）。

    增广拉格朗日：L(W)=f(W)+λ||W||₁+α·h(W)+ρ/2·h(W)²，h(W)=tr(e^{W∘W})−d。
    内层用 FISTA（加速近端梯度）对固定 (α,ρ) 最小化 L，收敛远快于 Adam；
    外层对每步重算 h(W) 更新对偶变量 (α,ρ)。
    """
    t0 = time.perf_counter()
    X = np.asarray(data.X, dtype=float)
    if X.ndim != 2:
        raise DiscoveryError("X 必须为二维矩阵")
    n, d = X.shape
    X = X - X.mean(axis=0, keepdims=True)  # 中心化

    W = np.zeros((d, d))  # 固定 0 初始化 → 完全确定性
    a, r = 0.0, float(rho_init)
    eps = 1e-3  # 平滑 L1 的 epsilon，避免次梯度在 0 处奇异
    bounds = [(-5.0, 5.0)] * (d * d)  # 限幅防止 expm(W∘W) 溢出

    def _obj_grad(w_flat):
        Wm = w_flat.reshape(d, d)
        E = X - X @ Wm
        f = 0.5 / n * np.sum(E ** 2)
        h = _h(Wm)
        smooth_l1 = lambda1 * np.sum(np.sqrt(Wm ** 2 + eps))
        obj = f + a * h + 0.5 * r * h * h + smooth_l1
        grad_f = -(X.T @ E) / n
        gh = _grad_h(Wm)
        g = grad_f + (a + r * h) * gh + lambda1 * Wm / np.sqrt(Wm ** 2 + eps)
        return obj, g.ravel()

    for _ in range(max_iter):
        # 内层：L-BFGS-B 对固定 (a, r) 最小化平滑增广拉格朗日（几十次迭代即收敛）
        res = minimize(
            _obj_grad, W.ravel(), method="L-BFGS-B", jac=True,
            bounds=bounds, options={"maxiter": inner_iter, "maxfun": inner_iter * 4, "ftol": 1e-9, "gtol": 1e-8},
        )
        W = res.x.reshape(d, d)
        h_new = _h(W)
        if h_new <= h_tol or r >= rho_max:
            break
        r = min(r * 10.0, rho_max)             # 外圈增大惩罚（与原始 NOTEARS 一致）
        a = a + r * h_new

    # 阈值化 + 保证无环（移除环上最小权值边）
    Wt = _ensure_acyclic(W.copy())
    Wt[np.abs(Wt) < w_threshold] = 0.0
    adj = (np.abs(Wt) > 0).astype(int)

    elapsed = time.perf_counter() - t0
    return DAGResult(
        adj=adj,
        W=Wt,
        method="notears_lin",
        elapsed=elapsed,
        meta={
            "lambda1": lambda1,
            "h_final": _h(W),
            "w_threshold": w_threshold,
        },
    )
