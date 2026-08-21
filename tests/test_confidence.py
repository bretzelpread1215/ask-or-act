import numpy as np

from ask_or_act.confidence import (
    calibrated_confidence,
    gaussian_logit_noise,
    temperature_scale,
)


def test_calibrated_confidence_equals_true_distribution() -> None:
    probabilities = np.array([[0.6, 0.3, 0.1], [0.34, 0.33, 0.33]])
    np.testing.assert_array_equal(calibrated_confidence(probabilities), probabilities)


def test_temperature_scaling_preserves_ranking_and_changes_sharpness() -> None:
    probabilities = np.array([[0.6, 0.3, 0.1], [0.45, 0.35, 0.2]])
    overconfident = temperature_scale(probabilities, 0.5)
    underconfident = temperature_scale(probabilities, 2.0)

    np.testing.assert_array_equal(np.argsort(overconfident), np.argsort(probabilities))
    np.testing.assert_array_equal(np.argsort(underconfident), np.argsort(probabilities))
    assert np.all(overconfident.max(axis=1) > probabilities.max(axis=1))
    assert np.all(underconfident.max(axis=1) < probabilities.max(axis=1))
    np.testing.assert_allclose(overconfident.sum(axis=1), 1.0)
    np.testing.assert_allclose(underconfident.sum(axis=1), 1.0)


def test_zero_logit_noise_returns_true_distribution() -> None:
    probabilities = np.array([[0.6, 0.3, 0.1]])
    result = gaussian_logit_noise(probabilities, 0.0, np.random.default_rng(2))
    np.testing.assert_array_equal(result, probabilities)


def test_logit_noise_is_reproducible_and_can_change_top_target() -> None:
    probabilities = np.array([[0.34, 0.33, 0.33]])
    first = gaussian_logit_noise(probabilities, 0.5, np.random.default_rng(4))
    second = gaussian_logit_noise(probabilities, 0.5, np.random.default_rng(4))

    np.testing.assert_array_equal(first, second)
    assert np.argmax(first, axis=1)[0] != np.argmax(probabilities, axis=1)[0]
