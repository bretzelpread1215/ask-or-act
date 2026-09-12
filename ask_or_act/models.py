from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def validate_probability_matrix(
    values: NDArray[np.float64], *, positive: bool, label: str
) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{label} must be a nonempty two-dimensional target matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be finite")
    invalid = matrix <= 0 if positive else matrix < 0
    if np.any(invalid) or not np.allclose(matrix.sum(axis=1), 1.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} rows must be {qualifier} and sum to 1")
    return matrix


def _integer_vector(values: NDArray[np.int64], label: str) -> NDArray[np.int64]:
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be a nonempty one-dimensional array")
    if not np.issubdtype(vector.dtype, np.integer):
        raise ValueError(f"{label} must contain integers")
    return vector.astype(np.int64, copy=False)


@dataclass(frozen=True)
class TaskBatch:
    true_probabilities: NDArray[np.float64]
    intended_targets: NDArray[np.int64]

    def __post_init__(self) -> None:
        probabilities = validate_probability_matrix(
            self.true_probabilities, positive=True, label="true probabilities"
        )
        targets = _integer_vector(self.intended_targets, "intended targets")
        if len(targets) != len(probabilities):
            raise ValueError("true probabilities and intended targets must have equal length")
        if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
            raise ValueError("intended targets must index a target column")
        object.__setattr__(self, "true_probabilities", probabilities)
        object.__setattr__(self, "intended_targets", targets)


@dataclass(frozen=True)
class ObservationBatch:
    reported_probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        probabilities = validate_probability_matrix(
            self.reported_probabilities, positive=False, label="reported probabilities"
        )
        object.__setattr__(self, "reported_probabilities", probabilities)


@dataclass(frozen=True)
class DecisionBatch:
    asks: NDArray[np.bool_]
    actions: NDArray[np.int64]

    def __post_init__(self) -> None:
        asks = np.asarray(self.asks)
        if asks.ndim != 1 or asks.size == 0 or asks.dtype != np.bool_:
            raise ValueError("asks must be a nonempty one-dimensional boolean array")
        actions = _integer_vector(self.actions, "actions")
        if len(asks) != len(actions):
            raise ValueError("asks and actions must have equal length")
        if np.any(actions < 0):
            raise ValueError("actions must be nonnegative")
        object.__setattr__(self, "asks", asks)
        object.__setattr__(self, "actions", actions)
