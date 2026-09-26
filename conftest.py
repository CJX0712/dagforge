import os
import sys

# 强制单线程 BLAS：保证 NOTEARS 等矩阵运算跨运行逐位可复现（消除多线程舍入不确定性）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

# 保证 `import dagforge` 可解析（项目根目录即包所在父目录）
sys.path.insert(0, os.path.dirname(__file__))
