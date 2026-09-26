"""DagForge CLI 入口（argparse）。

子命令：
  run    运行完整基准并落盘 benchmark.json（默认当前目录）
  learn  从 CSV 学习单个 DAG 并输出邻接矩阵（json）
  demo   同 run，面向快速演示
"""
import argparse
import json
import os
import sys

# 强制单线程 BLAS：保证 NOTEARS 跨运行逐位可复现
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

from .core.config import DagForgeConfig
from .data import loader
from .discovery import notears
from .pipeline.discovery_pipeline import DagForgePipeline


def _reconfigure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def cmd_run(args):
    _reconfigure_stdout()
    cfg = DagForgeConfig.from_env().validate()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.sota is not None:
        cfg.use_sota = args.sota
    pipe = DagForgePipeline(cfg)
    bench = pipe.run(verbose=not args.quiet)
    out = args.output or "benchmark.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(bench, fh, ensure_ascii=False, indent=2)
    print(f"[ok] benchmark 已写入 {out}")
    return bench


def cmd_learn(args):
    _reconfigure_stdout()
    ds = loader.load_csv(args.input, has_header=not args.no_header, true_adj_path=args.true_adj)
    res = notears.notears_linear(
        ds, lambda1=args.lambda1, w_threshold=args.w,
        max_iter=args.max_iter, inner_iter=args.inner_iter, lr=args.lr)
    out = {"method": res.method, "adjacency": res.adj.tolist()}
    if args.true_adj:
        from .eval import metrics as M
        met = M.structure_metrics(res.adj, ds.true_adj)
        out["metrics"] = {"shd": met.shd, "f1": met.f1,
                          "precision": met.precision, "recall": met.recall}
    print(json.dumps(out, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dagforge", description="DagForge 因果发现系统 (作者: 晨星)")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="运行基准并落盘 benchmark.json")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--output", default="benchmark.json")
    r.add_argument("--sota", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("learn", help="从 CSV 学习单个 DAG")
    l.add_argument("input", help="CSV 路径")
    l.add_argument("--true-adj", default=None, help="真值邻接矩阵 txt 路径（可选，用于评测）")
    l.add_argument("--no-header", action="store_true")
    l.add_argument("--lambda1", type=float, default=0.1)
    l.add_argument("--w", type=float, default=0.3)
    l.add_argument("--max-iter", type=int, default=60)
    l.add_argument("--inner-iter", type=int, default=200)
    l.add_argument("--lr", type=float, default=0.02)
    l.set_defaults(func=cmd_learn)

    d = sub.add_parser("demo", help="快速演示（同 run）")
    d.add_argument("--seed", type=int, default=None)
    d.add_argument("--output", default="benchmark.json")
    d.add_argument("--sota", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
    d.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    _reconfigure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
