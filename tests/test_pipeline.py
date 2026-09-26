import json

from dagforge.core.config import DagForgeConfig
from dagforge.pipeline.discovery_pipeline import DagForgePipeline


def _small_cfg(seed=42, use_sota=True):
    c = DagForgeConfig()
    c.seed = seed
    c.use_sota = use_sota
    c.n_graphs = 3
    c.n_nodes = 6
    c.edge_prob = 0.3
    c.n_samples = 400
    c.notears_max_iter = 25
    c.notears_inner_iter = 120
    c.notears_lr = 0.03
    c.pc_max_cond = 3
    return c.validate()


def test_pipeline_runs_and_reports():
    bench = DagForgePipeline(_small_cfg()).run(verbose=False)
    assert bench["system"] == "DagForge"
    assert "notears_lin" in bench["methods"]
    assert "corr_threshold" in bench["methods"]
    # 旗舰应优于空图基线（SHD 不更差）
    assert bench["methods"]["notears_lin"]["shd_mean"] <= bench["methods"]["empty"]["shd_mean"] + 1e-6
    assert "beat_baseline" in bench["headline"]


def _strip_timing(methods):
    out = {}
    for k, v in methods.items():
        out[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("elapsed")}
    return out


def test_pipeline_deterministic():
    b1 = DagForgePipeline(_small_cfg(seed=11)).run(verbose=False)
    b2 = DagForgePipeline(_small_cfg(seed=11)).run(verbose=False)
    # 核心指标（SHD/F1/precision/recall/fdr 及其 std）逐位一致；elapsed 为墙钟耗时，允许不同
    assert json.dumps(_strip_timing(b1["methods"]), sort_keys=True) == \
        json.dumps(_strip_timing(b2["methods"]), sort_keys=True)
    assert json.dumps(b1["headline"], sort_keys=True) == json.dumps(b2["headline"], sort_keys=True)


def test_offline_fallback_no_sota():
    bench = DagForgePipeline(_small_cfg(use_sota=False)).run(verbose=False)
    assert "cl_pc" not in bench["methods"]
    assert "notears_lin" in bench["methods"]  # 纯 numpy 兜底仍可用


def test_cli_run_smoke(tmp_path):
    from dagforge.cli import main
    out = tmp_path / "bench.json"
    main(["run", "--output", str(out), "--seed", "5", "--quiet"])
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["system"] == "DagForge"
    assert "methods" in data
