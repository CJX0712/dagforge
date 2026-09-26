"""DagForge — 模块化因果发现（DAG 结构学习）系统.

因果发现 = 从观测数据推断变量间的因果有向无环图（DAG）。
旗舰方法 NOTEARS（Zheng et al., NeurIPS 2018）以评分式 + 矩阵指数无环约束
将 DAG 学习转化为连续优化问题，是近年的里程碑式 SOTA。

作者：晨星  |  仓库：CJX0712/dagforge
"""
import os

# 强制单线程 BLAS：保证 NOTEARS 等矩阵运算跨运行逐位可复现
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

__version__ = "0.1.0"
__author__ = "晨星"
