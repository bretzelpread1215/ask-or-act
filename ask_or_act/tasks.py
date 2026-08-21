import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def generate_tasks(
    n_tasks: int,
    concentration: float,
    rng: Generator,
    n_targets: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
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
    return probabilities, intended_targets


def true_ambiguity(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """return one minus the largest true target probability per task."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional target matrix")
    if np.any(probabilities <= 0) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("probability rows must be positive and sum to 1")
    return 1.0 - probabilities.max(axis=1)
