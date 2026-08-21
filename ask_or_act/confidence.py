import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def _probability_matrix(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional target matrix")
    if np.any(values <= 0) or not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("probability rows must be positive and sum to 1")
    return values


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def calibrated_confidence(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """copy the true distribution without distortion."""
    return _probability_matrix(probabilities).copy()


def temperature_scale(
    probabilities: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    """apply q_i proportional to p_i raised to 1 / temperature."""
    values = _probability_matrix(probabilities)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _softmax(np.log(values) / temperature)


def gaussian_logit_noise(
    probabilities: NDArray[np.float64], sigma: float, rng: Generator
) -> NDArray[np.float64]:
    """add independent gaussian noise to each log probability."""
    values = _probability_matrix(probabilities)
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0:
        return values.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=values.shape)
    return _softmax(np.log(values) + noise)


def confidence_regimes(
    probabilities: NDArray[np.float64],
    noise_rng: Generator,
    overconfident_temperature: float = 0.5,
    underconfident_temperature: float = 2.0,
    noise_sigma: float = 0.5,
) -> dict[str, NDArray[np.float64]]:
    """build the four preregistered reported-confidence regimes."""
    return {
        "calibrated": calibrated_confidence(probabilities),
        "overconfident": temperature_scale(probabilities, overconfident_temperature),
        "underconfident": temperature_scale(probabilities, underconfident_temperature),
        "noisy": gaussian_logit_noise(probabilities, noise_sigma, noise_rng),
    }
