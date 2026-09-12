import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ask_or_act.models import TaskBatch, validate_probability_matrix


def generate_tasks(
    n_tasks: int,
    concentration: float,
    rng: Generator,
    n_targets: int = 3,
) -> TaskBatch:
    """draw true target probabilities and one intended target per task."""
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    if n_targets < 2:
        raise ValueError("n_targets must be at least 2")
    if concentration <= 0:
        raise ValueError("concentration must be positive")

    alpha = np.full(n_targets, concentration, dtype=float)
    probabilities = rng.dirichlet(alpha, size=n_tasks)
    probabilities = np.maximum(probabilities, np.finfo(float).tiny)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random(n_tasks)
    intended_targets = (draws[:, None] > cumulative).sum(axis=1).astype(np.int64)
    return TaskBatch(probabilities, intended_targets)


def true_ambiguity(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """return one minus the largest true target probability per task."""
    values = validate_probability_matrix(
        probabilities, positive=True, label="probabilities"
    )
    return 1.0 - values.max(axis=1)
