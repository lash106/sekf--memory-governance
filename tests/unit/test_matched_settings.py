import pytest

from sekf.baselines import assert_matched_settings, compare_run_configs
from sekf.contracts import RunConfig, Thresholds


def test_compare_run_configs_detects_model_mismatch() -> None:
    left = RunConfig(model="gpt-4o-mini", temperature=0.0, seed=1)
    right = RunConfig(model="gpt-4o", temperature=0.0, seed=1)
    violations = compare_run_configs(left, right)
    assert len(violations) == 1
    assert violations[0].field == "model"


def test_assert_matched_settings_ok() -> None:
    cfg = RunConfig(model="gpt-4o-mini", temperature=0.0, seed=7)
    thr = Thresholds()
    assert_matched_settings(cfg, cfg, council_thresholds=thr, baseline_thresholds=thr)


def test_assert_matched_settings_raises() -> None:
    left = RunConfig(model="a", temperature=0.0, seed=1)
    right = RunConfig(model="a", temperature=0.2, seed=1)
    with pytest.raises(ValueError, match="temperature"):
        assert_matched_settings(left, right)
