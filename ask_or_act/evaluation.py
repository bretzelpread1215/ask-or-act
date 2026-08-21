import numpy as np
from numpy.typing import NDArray


OutcomeArrays = dict[str, NDArray[np.float64] | NDArray[np.bool_]]


def evaluate_tasks(
    intended_targets: NDArray[np.int64],
    asks: NDArray[np.bool_],
    actions: NDArray[np.int64],
    correct_action_cost: float = 0.0,
    clarification_cost: float = 1.0,
    incorrect_action_cost: float = 5.0,
) -> OutcomeArrays:
    """evaluate final outcomes, assuming clarification always resolves the target."""
    intended = np.asarray(intended_targets, dtype=np.int64)
    asked = np.asarray(asks, dtype=bool)
    chosen = np.asarray(actions, dtype=np.int64)
    if intended.ndim != 1 or asked.ndim != 1 or chosen.ndim != 1:
        raise ValueError("intended_targets, asks, and actions must be one-dimensional")
    if not (len(intended) == len(asked) == len(chosen)):
        raise ValueError("intended_targets, asks, and actions must have equal length")
    if min(correct_action_cost, clarification_cost, incorrect_action_cost) < 0:
        raise ValueError("costs must be nonnegative")

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
