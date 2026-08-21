import numpy as np
from numpy.typing import NDArray


Decision = tuple[NDArray[np.bool_], NDArray[np.int64]]


def _reported_matrix(reported: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(reported, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("reported confidence must be a two-dimensional target matrix")
    if np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("reported confidence rows must be nonnegative and sum to 1")
    return values


def always_act(reported: NDArray[np.float64]) -> Decision:
    values = _reported_matrix(reported)
    asks = np.zeros(values.shape[0], dtype=bool)
    actions = np.argmax(values, axis=1).astype(np.int64)
    return asks, actions


def always_ask(reported: NDArray[np.float64]) -> Decision:
    values = _reported_matrix(reported)
    asks = np.ones(values.shape[0], dtype=bool)
    actions = np.argmax(values, axis=1).astype(np.int64)
    return asks, actions


def threshold_policy(reported: NDArray[np.float64], threshold: float) -> Decision:
    values = _reported_matrix(reported)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    asks = values.max(axis=1) < threshold
    actions = np.argmax(values, axis=1).astype(np.int64)
    return asks, actions


def cost_derived_threshold(
    correct_action_cost: float,
    clarification_cost: float,
    incorrect_action_cost: float,
) -> float:
    if min(correct_action_cost, clarification_cost, incorrect_action_cost) < 0:
        raise ValueError("costs must be nonnegative")
    if incorrect_action_cost <= correct_action_cost:
        raise ValueError("incorrect_action_cost must exceed correct_action_cost")
    if not correct_action_cost <= clarification_cost <= incorrect_action_cost:
        raise ValueError("costs must satisfy correct <= clarification <= incorrect")
    return (incorrect_action_cost - clarification_cost) / (
        incorrect_action_cost - correct_action_cost
    )
