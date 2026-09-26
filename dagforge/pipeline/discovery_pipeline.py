"""DagForgePipeline：编排 data → discovery → eval → 汇总，产出可复现 benchmark.json。

调用单向无环：pipeline → {data, discovery, eval} → core。
确定性：set_all(seed) 设齐随机源；各方法实现本身无随机性，两次运行核心指标逐位一致。
"""
import sys

import numpy as np

from ..core.config import DagForgeConfig
from ..core.errors import PipelineError
from ..core.seed import set_all
from ..core.types import Metrics
from ..data import synthetic
from ..discovery import corr, notears, pc, sota
from ..eval import metrics as M


def _undirected(adj):
    a = (np.asarray(adj) != 0).astype(int)
    return ((a + a.T) > 0).astype(int)


def _rec_to_metrics(rec: dict) -> Metrics:
    return Metrics(
        shd=rec["shd"], precision=rec["precision"], recall=rec["recall"],
        f1=rec["f1"], fdr=rec["fdr"],
    )


# SOTA 后端方法 -> 对应的自研对照方法（用于交叉验证）
_CROSS_MAP = {
    "cl_pc": "pc",
    "cl_notears": "notears_lin",
    "gcastle_notears": "notears_lin",
}


class DagForgePipeline:
    def __init__(self, cfg: DagForgeConfig = None):
        self.cfg = (cfg or DagForgeConfig()).validate()

    # S 级性能门槛（方案阶段定死）
    SHD_REDUCTION_PCT = 40.0
    F1_GAIN = 0.15

    def _method_candidates(self) -> dict[str, callable]:
        cfg = self.cfg
        cand = {
            "empty": lambda ds: corr.empty_graph(ds),
            "corr_threshold": lambda ds: corr.corr_threshold(ds, threshold=cfg.corr_threshold),
            "pc": lambda ds: pc.pc(ds, alpha=cfg.pc_alpha, max_cond=cfg.pc_max_cond),
            "notears_lin": lambda ds: notears.notears_linear(
                ds, lambda1=cfg.notears_lambda, rho_init=cfg.notears_rho_init,
                max_iter=cfg.notears_max_iter, inner_iter=cfg.notears_inner_iter,
                lr=cfg.notears_lr, h_tol=cfg.notears_h_tol,
                w_threshold=cfg.notears_w_threshold),
        }
        if cfg.use_sota and sota.available_causallearn():
            cand["cl_pc"] = lambda ds: sota.causallearn_pc(ds, alpha=cfg.pc_alpha)
            cand["cl_notears"] = lambda ds: sota.causallearn_notears(
                ds, lambda1=cfg.notears_lambda, w_threshold=cfg.notears_w_threshold)
        if cfg.use_sota and sota.available_gcastle():
            cand["gcastle_notears"] = lambda ds: sota.gcastle_notears(
                ds, lambda1=cfg.notears_lambda, w_threshold=cfg.notears_w_threshold)
        return cand

    def run(self, verbose: bool = True) -> dict:
        set_all(self.cfg.seed)
        suite = synthetic.generate_benchmark_suite(
            n_graphs=self.cfg.n_graphs, base_seed=self.cfg.seed,
            n_nodes=self.cfg.n_nodes, edge_prob=self.cfg.edge_prob,
            n_samples=self.cfg.n_samples, sem_noise=self.cfg.sem_noise)
        if not suite:
            raise PipelineError("生成的数据集为空")

        candidates = self._method_candidates()
        # 探测可用性：在首个数据集上试跑，返回 None 的后端直接剔除
        probe = {}
        reg = {}
        for name, fn in candidates.items():
            try:
                r = fn(suite[0])
            except Exception:
                r = None
            if r is None:
                continue
            probe[name] = r
            reg[name] = fn

        per_method: dict[str, list[dict]] = {name: [] for name in reg}
        datasets_meta = []
        for di, ds in enumerate(suite):
            if ds.true_adj is None:
                continue
            n_edges_true = int((_undirected(ds.true_adj).sum() - ds.n_features) / 2)
            datasets_meta.append({
                "name": ds.name, "n_nodes": ds.n_features,
                "n_edges_true": n_edges_true, "n_samples": ds.n_samples,
            })
            for name, fn in reg.items():
                res = probe[name] if di == 0 else fn(ds)
                met = M.structure_metrics(res.adj, ds.true_adj)
                per_method[name].append({
                    "shd": met.shd, "precision": met.precision,
                    "recall": met.recall, "f1": met.f1, "fdr": met.fdr,
                    "elapsed": res.elapsed,
                })

        summary = {}
        for name in reg:
            agg = M.aggregate([_rec_to_metrics(r) for r in per_method[name]])
            agg["elapsed_mean"] = float(np.mean([r["elapsed"] for r in per_method[name]]))
            summary[name] = agg

        headline = self._headline(summary)
        backends = {
            "causal_learn": sota.available_causallearn(),
            "gcastle": sota.available_gcastle(),
        }
        bench = {
            "system": "DagForge",
            "version": "0.1.0",
            "author": "晨星",
            "seed": self.cfg.seed,
            "generated_at": self._now(),
            "config": self.cfg.as_dict(),
            "backends_available": backends,
            "datasets": datasets_meta,
            "methods": summary,
            "headline": headline,
        }
        if verbose:
            self._print_summary(bench)
        return bench

    def _headline(self, summary: dict) -> dict:
        base = summary.get("corr_threshold")
        flag = summary.get("notears_lin")
        if not base or not flag:
            return {"error": "缺少基线或旗舰方法"}
        shd_base = base["shd_mean"]
        shd_flag = flag["shd_mean"]
        shd_red = (shd_base - shd_flag) / shd_base * 100.0 if shd_base > 0 else 0.0
        f1_gain = flag["f1_mean"] - base["f1_mean"]
        beat = (shd_red >= self.SHD_REDUCTION_PCT) and (f1_gain >= self.F1_GAIN)
        cross = self._cross_validation(summary)
        return {
            "flagship": "notears_lin",
            "baseline": "corr_threshold",
            "baseline_shd_mean": shd_base,
            "flagship_shd_mean": shd_flag,
            "shd_reduction_pct": float(shd_red),
            "f1_gain": float(f1_gain),
            "beat_baseline": bool(beat),
            "s_bar_satisfied": bool(beat),
            "cross_validation": cross,
        }

    def _cross_validation(self, summary: dict) -> list[dict]:
        out = []
        for sota_name, ours_name in _CROSS_MAP.items():
            if sota_name in summary and ours_name in summary:
                diff = abs(summary[sota_name]["shd_mean"] - summary[ours_name]["shd_mean"])
                out.append({
                    "sota_method": sota_name,
                    "ours_method": ours_name,
                    "shd_diff": float(diff),
                    "matches_reference": bool(diff <= max(2.0, 0.5 * summary[ours_name]["shd_mean"] + 1e-6)),
                })
        return out

    @staticmethod
    def _now() -> str:
        import datetime as _dt
        return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _print_summary(self, bench: dict) -> None:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        hl = bench["headline"]
        print("=" * 72)
        print(f" DagForge 因果发现基准  |  seed={bench['seed']}  |  datasets={len(bench['datasets'])}")
        print("=" * 72)
        header = f"{'method':<16}{'SHD':>8}{'F1':>8}{'prec':>8}{'rec':>8}{'FDR':>8}{'t(s)':>9}"
        print(header)
        for name, s in bench["methods"].items():
            print(f"{name:<16}{s['shd_mean']:>8.2f}{s['f1_mean']:>8.3f}"
                  f"{s['precision_mean']:>8.3f}{s['recall_mean']:>8.3f}"
                  f"{s['fdr_mean']:>8.3f}{s['elapsed_mean']:>9.3f}")
        print("-" * 72)
        print(" 旗舰 notears_lin vs 基线 corr_threshold:")
        print(f"   SHD 下降 {hl.get('shd_reduction_pct', 0):.1f}%  (S门槛≥{self.SHD_REDUCTION_PCT:.0f}%)"
              f"   F1 提升 {hl.get('f1_gain', 0):.3f}  (S门槛≥{self.F1_GAIN:.2f})")
        print(f"   击败强基线: {'✅' if hl.get('beat_baseline') else '⚠️'}")
        for c in hl.get("cross_validation", []):
            print(f"   交叉验证 {c['sota_method']} vs {c['ours_method']}: SHD差={c['shd_diff']:.2f} "
                  f"-> 一致: {'✅' if c['matches_reference'] else '⚠️'}")
        print("=" * 72)
