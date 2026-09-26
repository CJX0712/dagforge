"""配置：支持 ENV_DAGFORGE_* 覆盖 + 基础校验（schema 校验，无第三方依赖）。"""
import os
from dataclasses import dataclass

from .errors import ConfigError

_PREFIX = "DAGFORGE_"


def _coerce(value: str, target_type):
    if target_type is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    return target_type(value)


@dataclass
class DagForgeConfig:
    # 全局
    seed: int = 42
    use_sota: bool = True
    # 合成数据
    n_graphs: int = 8
    n_nodes: int = 12
    edge_prob: float = 0.3
    n_samples: int = 800
    sem_noise: float = 1.0
    # NOTEARS（旗舰）
    notears_lambda: float = 0.1
    notears_rho_init: float = 1.0
    notears_max_iter: int = 60
    notears_inner_iter: int = 100
    notears_lr: float = 0.02
    notears_h_tol: float = 1e-6
    notears_w_threshold: float = 0.2
    # PC（约束式基线）
    pc_alpha: float = 0.05
    pc_max_cond: int = 3
    # 相关阈值（强基线 naive）
    corr_threshold: float = 0.3
    # 评测
    n_bootstrap: int = 0

    @classmethod
    def from_env(cls) -> "DagForgeConfig":
        cfg = cls()
        mapping = {
            "SEED": "seed", "USE_SOTA": "use_sota",
            "N_GRAPHS": "n_graphs", "N_NODES": "n_nodes", "EDGE_PROB": "edge_prob",
            "N_SAMPLES": "n_samples", "SEM_NOISE": "sem_noise",
            "NOTEARS_LAMBDA": "notears_lambda", "NOTEARS_RHO_INIT": "notears_rho_init",
            "NOTEARS_MAX_ITER": "notears_max_iter", "NOTEARS_INNER_ITER": "notears_inner_iter",
            "NOTEARS_LR": "notears_lr", "NOTEARS_H_TOL": "notears_h_tol",
            "NOTEARS_W_THRESHOLD": "notears_w_threshold",
            "PC_ALPHA": "pc_alpha", "PC_MAX_COND": "pc_max_cond",
            "CORR_THRESHOLD": "corr_threshold", "N_BOOTSTRAP": "n_bootstrap",
        }
        for env_key, attr in mapping.items():
            raw = os.environ.get(_PREFIX + env_key)
            if raw is None:
                continue
            cur = getattr(cfg, attr)
            try:
                setattr(cfg, attr, _coerce(raw, type(cur)))
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"无法解析 {_PREFIX}{env_key}={raw!r} 为 {type(cur).__name__}") from exc
        return cfg

    def validate(self) -> "DagForgeConfig":
        if self.seed < 0:
            raise ConfigError("seed 必须 >= 0")
        if not (0 < self.edge_prob <= 1):
            raise ConfigError("edge_prob 必须在 (0, 1]")
        if self.n_nodes < 2 or self.n_samples < 10:
            raise ConfigError("n_nodes>=2 且 n_samples>=10")
        if not (0 < self.pc_alpha < 1):
            raise ConfigError("pc_alpha 必须在 (0,1)")
        if self.notears_lambda < 0 or self.notears_w_threshold <= 0:
            raise ConfigError("notears 超参非法")
        if self.corr_threshold <= 0:
            raise ConfigError("corr_threshold 必须 > 0")
        return self

    def as_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
