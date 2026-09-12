import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ask_or_act.models import ObservationBatch, validate_probability_matrix


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def calibrated_confidence(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """copy the true distribution without distortion."""
    return validate_probability_matrix(
        probabilities, positive=True, label="probabilities"
    ).copy()


def temperature_scale(
    probabilities: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    """apply q_i proportional to p_i raised to 1 / temperature."""
    values = validate_probability_matrix(
        probabilities, positive=True, label="probabilities"
    )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _softmax(np.log(values) / temperature)


def gaussian_logit_noise(
    probabilities: NDArray[np.float64], sigma: float, rng: Generator
) -> NDArray[np.float64]:
    """add independent gaussian noise to each log probability."""
    values = validate_probability_matrix(
        probabilities, positive=True, label="probabilities"
    )
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
) -> dict[str, ObservationBatch]:
    """build the four preregistered reported-confidence regimes."""
    return {
        "calibrated": ObservationBatch(calibrated_confidence(probabilities)),
        "overconfident": ObservationBatch(
            temperature_scale(probabilities, overconfident_temperature)
        ),
        "underconfident": ObservationBatch(
            temperature_scale(probabilities, underconfident_temperature)
        ),
        "noisy": ObservationBatch(
            gaussian_logit_noise(probabilities, noise_sigma, noise_rng)
        ),
    }
