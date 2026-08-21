import numpy as np

from ask_or_act.config import ExperimentConfig
from ask_or_act.simulation import run_replicate
from ask_or_act.statistics import mean_t_interval, paired_cost_difference


def test_t_interval_for_constant_replicate_values_has_zero_width() -> None:
    interval = mean_t_interval([2.0, 2.0, 2.0, 2.0])
    assert interval == {"mean": 2.0, "lower": 2.0, "upper": 2.0}


def test_paired_difference_is_computed_before_interval() -> None:
    threshold = np.array([1.0, 2.0, 4.0, 8.0])
    baseline = np.array([2.0, 4.0, 8.0, 16.0])
    result = paired_cost_difference(threshold, baseline)
    expected = mean_t_interval([-1.0, -2.0, -4.0, -8.0])
    assert result == expected


def test_small_replicate_is_reproducible_and_complete() -> None:
    config = ExperimentConfig(task_count=50, replicate_count=2, master_seed=31)
    first = run_replicate(config, replicate_index=0)
    second = run_replicate(config, replicate_index=0)

    assert first == second
    assert len(first) == 12
    assert {record["regime"] for record in first} == {
        "calibrated",
        "overconfident",
        "underconfident",
        "noisy",
    }
    assert {record["policy"] for record in first} == {
        "always_act",
        "always_ask",
        "threshold",
    }
