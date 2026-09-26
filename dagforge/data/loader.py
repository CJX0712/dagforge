"""从 CSV 载入观测数据 -> Dataset（供 CLI/真实数据使用）。"""
import csv
import os

import numpy as np

from ..core.errors import DataError
from ..core.types import Dataset


def load_csv(path: str, has_header: bool = True, true_adj_path: str | None = None) -> Dataset:
    if not os.path.exists(path):
        raise DataError(f"文件不存在: {path}")
    rows = []
    header = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        if has_header:
            header = next(reader, [])
        for r in reader:
            if not r:
                continue
            rows.append([float(x) for x in r])
    X = np.asarray(rows, dtype=float)
    if X.ndim != 2 or X.size == 0:
        raise DataError("CSV 必须解析为二维数值矩阵")
    feature_names = header if header else [f"X{i}" for i in range(X.shape[1])]
    true_adj = None
    if true_adj_path:
        true_adj = np.loadtxt(true_adj_path, dtype=float)
    return Dataset(X=X, feature_names=feature_names, name=os.path.basename(path), true_adj=true_adj)
