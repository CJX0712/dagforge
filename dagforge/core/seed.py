"""全局确定性：所有随机源从唯一 seed 入口设齐，保证 demo 两次运行逐位一致。"""
import os
import random

import numpy as np

_GLOBAL_SEED: int | None = None


def set_all(seed: int = 42) -> int:
    """一次性设齐 numpy / random / hash 随机源，返回生效的 seed。"""
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    os.environ.setdefault("PYTHONHASHSEED", str(_GLOBAL_SEED))
    return _GLOBAL_SEED


def get_seed() -> int:
    return _GLOBAL_SEED if _GLOBAL_SEED is not None else 42


def rng() -> np.random.Generator:
    """返回一个由当前全局 seed 派生的 numpy Generator（不污染全局状态）。"""
    return np.random.default_rng(get_seed())
