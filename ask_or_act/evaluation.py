import numpy as np
from numpy.typing import NDArray

from ask_or_act.models import DecisionBatch, TaskBatch


OutcomeArrays = dict[str, NDArray[np.float64] | NDArray[np.bool_]]

METRIC_DEFINITIONS = {
    "expected_total_cost": "mean total cost per generated task",
    "task_success_rate": "successful outcomes divided by all generated tasks",
    "incorrect_action_rate": "incorrect unclarified actions divided by all generated tasks",
    "clarification_frequency": "clarification requests divided by all generated tasks",
}


def evaluate_tasks(
    tasks: TaskBatch,
    decisions: DecisionBatch,
    correct_action_cost: float = 0.0,
    clarification_cost: float = 1.0,
    incorrect_action_cost: float = 5.0,
) -> OutcomeArrays:
    """evaluate final outcomes, assuming clarification always resolves the target."""
    intended = tasks.intended_targets
    asked = decisions.asks
    chosen = decisions.actions
    if not (len(intended) == len(asked) == len(chosen)):
        raise ValueError("tasks and decisions must have equal length")
    if np.any(chosen >= tasks.true_probabilities.shape[1]):
        raise ValueError("actions must index a target column")
    costs = np.asarray(
        [correct_action_cost, clarification_cost, incorrect_action_cost], dtype=float
    )
    if not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("costs must be nonnegative")
    if incorrect_action_cost <= correct_action_cost:
        raise ValueError("incorrect_action_cost must exceed correct_action_cost")
    if not correct_action_cost <= clarification_cost <= incorrect_action_cost:
        raise ValueError("costs must satisfy correct <= clarification <= incorrect")

    incorrect = ~asked & (chosen != intended)
    success = ~incorrect
    total_cost = np.where(
        asked,
        clarification_cost,
        np.where(incorrect, incorrect_action_cost, correct_action_cost),
    ).astype(float)
    return {
        "total_cost": total_cost,
        "task_success": success,
        "incorrect_action": incorrect,
        "clarification": asked,
    }


def summarize_outcomes(outcomes: OutcomeArrays) -> dict[str, float]:
    required = {"total_cost", "task_success", "incorrect_action", "clarification"}
    if set(outcomes) != required:
        raise ValueError(f"outcomes must contain exactly {sorted(required)}")
    if len(outcomes["total_cost"]) == 0:
        raise ValueError("outcomes cannot be empty")
    return {
        "expected_total_cost": float(np.mean(outcomes["total_cost"])),
        "task_success_rate": float(np.mean(outcomes["task_success"])),
        "incorrect_action_rate": float(np.mean(outcomes["incorrect_action"])),
        "clarification_frequency": float(np.mean(outcomes["clarification"])),
    }
