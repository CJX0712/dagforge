import numpy as np
import pytest

from dagforge.core.config import ConfigError, DagForgeConfig
from dagforge.core.errors import ConfigError as CE2
from dagforge.core.errors import DagForgeError
from dagforge.core.seed import get_seed, rng, set_all


def test_seed_determinism():
    set_all(123)
    a = rng().standard_normal(5)
    set_all(123)
    b = rng().standard_normal(5)
    assert np.allclose(a, b)


def test_set_all_returns_seed():
    assert set_all(7) == 7
    assert get_seed() == 7


def test_config_default_valid():
    cfg = DagForgeConfig()
    cfg.validate()  # 不应抛异常


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("DAGFORGE_SEED", "99")
    monkeypatch.setenv("DAGFORGE_N_NODES", "20")
    monkeypatch.setenv("DAGFORGE_USE_SOTA", "false")
    cfg = DagForgeConfig.from_env().validate()
    assert cfg.seed == 99
    assert cfg.n_nodes == 20
    assert cfg.use_sota is False


def test_config_validation_bounds(monkeypatch):
    monkeypatch.setenv("DAGFORGE_EDGE_PROB", "1.5")
    with pytest.raises(ConfigError):
        DagForgeConfig.from_env().validate()


def test_error_hierarchy():
    assert issubclass(CE2, DagForgeError)
    e = CE2("boom")
    assert e.code == "E100"
    assert "E100" in str(e)
