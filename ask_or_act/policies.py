import numpy as np

from ask_or_act.models import DecisionBatch, ObservationBatch


def always_act(observation: ObservationBatch) -> DecisionBatch:
    values = observation.reported_probabilities
    asks = np.zeros(values.shape[0], dtype=bool)
    actions = np.argmax(values, axis=1).astype(np.int64)
    return DecisionBatch(asks, actions)


def always_ask(observation: ObservationBatch) -> DecisionBatch:
    values = observation.reported_probabilities
    asks = np.ones(values.shape[0], dtype=bool)
    actions = np.argmax(values, axis=1).astype(np.int64)
    return DecisionBatch(asks, actions)


def threshold_policy(
    observation: ObservationBatch, threshold: float
) -> DecisionBatch:
    values = observation.reported_probabilities
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    asks = values.max(axis=1) < threshold
    actions = np.argmax(values, axis=1).astype(np.int64)
    return DecisionBatch(asks, actions)


def cost_derived_threshold(
    correct_action_cost: float,
    clarification_cost: float,
    incorrect_action_cost: float,
) -> float:
    costs = np.asarray(
        [correct_action_cost, clarification_cost, incorrect_action_cost], dtype=float
    )
    if not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("costs must be nonnegative")
    if incorrect_action_cost <= correct_action_cost:
        raise ValueError("incorrect_action_cost must exceed correct_action_cost")
    if not correct_action_cost <= clarification_cost <= incorrect_action_cost:
        raise ValueError("costs must satisfy correct <= clarification <= incorrect")
    return (incorrect_action_cost - clarification_cost) / (
        incorrect_action_cost - correct_action_cost
    )
