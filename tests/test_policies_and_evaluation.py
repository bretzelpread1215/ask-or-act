import numpy as np
import pytest

from ask_or_act.confidence import temperature_scale
from ask_or_act.config import ExperimentConfig
from ask_or_act.evaluation import evaluate_tasks, summarize_outcomes
from ask_or_act.models import DecisionBatch, ObservationBatch, TaskBatch
from ask_or_act.policies import (
    always_act,
    always_ask,
    cost_derived_threshold,
    threshold_policy,
)


def test_cost_derived_thresholds_match_agreed_settings() -> None:
    assert cost_derived_threshold(0.0, 1.0, 2.0) == 0.5
    assert cost_derived_threshold(0.0, 1.0, 5.0) == 0.8
    assert cost_derived_threshold(0.0, 1.0, 10.0) == 0.9


def test_cost_derived_threshold_supports_nonzero_correct_action_cost() -> None:
    assert cost_derived_threshold(1.0, 2.0, 5.0) == 0.75
    config = ExperimentConfig(
        correct_action_cost=1.0,
        clarification_cost=2.0,
        incorrect_action_cost=5.0,
    )
    assert config.threshold == 0.75


@pytest.mark.parametrize(
    "costs",
    [
        (2.0, 1.0, 5.0),
        (0.0, 6.0, 5.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_cost_derived_threshold_rejects_invalid_cost_order(costs: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        cost_derived_threshold(*costs)
    with pytest.raises(ValueError):
        ExperimentConfig(
            correct_action_cost=costs[0],
            clarification_cost=costs[1],
            incorrect_action_cost=costs[2],
        )


def test_configuration_and_threshold_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ExperimentConfig(noise_sigma=float("nan"))
    with pytest.raises(ValueError, match="nonnegative"):
        cost_derived_threshold(0.0, 1.0, float("inf"))


def test_threshold_policy_acts_at_exact_equality() -> None:
    reported = np.array([[0.8, 0.1, 0.1], [0.79, 0.11, 0.1]])
    decisions = threshold_policy(ObservationBatch(reported), 0.8)

    np.testing.assert_array_equal(decisions.asks, [False, True])
    np.testing.assert_array_equal(decisions.actions, [0, 0])


def test_temperature_directions_are_relative_to_calibrated_threshold_policy() -> None:
    probabilities = np.array([[0.75, 0.15, 0.1], [0.85, 0.1, 0.05]])
    calibrated = threshold_policy(ObservationBatch(probabilities), 0.8)
    overconfident = threshold_policy(
        ObservationBatch(temperature_scale(probabilities, 0.5)), 0.8
    )
    underconfident = threshold_policy(
        ObservationBatch(temperature_scale(probabilities, 2.0)), 0.8
    )

    assert overconfident.asks.mean() < calibrated.asks.mean()
    assert underconfident.asks.mean() > calibrated.asks.mean()


def test_clarification_is_successful_and_cannot_also_be_incorrect() -> None:
    tasks = TaskBatch(
        np.full((3, 3), 1 / 3),
        np.array([0, 1, 2]),
    )
    decisions = DecisionBatch(
        np.array([False, False, True]),
        np.array([0, 0, 0]),
    )

    outcomes = evaluate_tasks(tasks, decisions)

    np.testing.assert_array_equal(outcomes["task_success"], [True, False, True])
    np.testing.assert_array_equal(outcomes["incorrect_action"], [False, True, False])
    np.testing.assert_allclose(outcomes["total_cost"], [0.0, 5.0, 1.0])
    summary = summarize_outcomes(outcomes)
    assert summary["task_success_rate"] == 2 / 3
    assert summary["incorrect_action_rate"] == 1 / 3
    assert summary["clarification_frequency"] == 1 / 3


def test_baseline_policies_have_expected_ask_behavior() -> None:
    reported = np.array([[0.6, 0.3, 0.1], [0.4, 0.35, 0.25]])
    observation = ObservationBatch(reported)
    act_decisions = always_act(observation)
    ask_decisions = always_ask(observation)

    assert not act_decisions.asks.any()
    assert ask_decisions.asks.all()
    np.testing.assert_array_equal(act_decisions.actions, [0, 0])
    np.testing.assert_array_equal(ask_decisions.actions, [0, 0])
