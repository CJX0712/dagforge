"""端到端演示：运行 DagForge 基准并把结果落盘为 benchmark.json。

用法：
  python -m dagforge.examples.run_demo [--output benchmark.json] [--seed 42]
结果真实可复现：固定 seed 下两次运行 benchmark.json 核心指标逐位一致。
"""
import argparse
import json
import os
import sys

# 强制单线程 BLAS：保证 NOTEARS 跨运行逐位可复现
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

sys.path.insert(0, ".")

from dagforge.core.config import DagForgeConfig
from dagforge.pipeline.discovery_pipeline import DagForgePipeline


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="benchmark.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sota", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
    args = ap.parse_args()

    cfg = DagForgeConfig.from_env().validate()
    cfg.seed = args.seed
    cfg.use_sota = args.sota
    pipe = DagForgePipeline(cfg)
    bench = pipe.run(verbose=True)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(bench, fh, ensure_ascii=False, indent=2)
    print(f"[ok] 演示完成，benchmark 已写入 {args.output}")


if __name__ == "__main__":
    main()
