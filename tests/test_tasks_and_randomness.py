import numpy as np

from ask_or_act.randomness import replicate_streams
from ask_or_act.tasks import generate_tasks, true_ambiguity


def test_generated_tasks_are_valid_and_reproducible() -> None:
    task_rng_a, _ = replicate_streams(17, 3)
    task_rng_b, _ = replicate_streams(17, 3)

    probabilities_a, targets_a = generate_tasks(200, 1.0, task_rng_a)
    probabilities_b, targets_b = generate_tasks(200, 1.0, task_rng_b)

    np.testing.assert_array_equal(probabilities_a, probabilities_b)
    np.testing.assert_array_equal(targets_a, targets_b)
    assert probabilities_a.shape == (200, 3)
    assert np.all(probabilities_a > 0)
    np.testing.assert_allclose(probabilities_a.sum(axis=1), 1.0)
    assert np.all((0 <= targets_a) & (targets_a < 3))


def test_task_and_noise_streams_are_distinct_and_reproducible() -> None:
    task_a, noise_a = replicate_streams(23, 4)
    task_b, noise_b = replicate_streams(23, 4)

    np.testing.assert_array_equal(task_a.random(10), task_b.random(10))
    np.testing.assert_array_equal(noise_a.random(10), noise_b.random(10))

    task_again, noise_again = replicate_streams(23, 4)
    assert not np.array_equal(task_again.random(10), noise_again.random(10))


def test_true_ambiguity_is_one_minus_largest_probability() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    np.testing.assert_allclose(true_ambiguity(probabilities), [0.3, 0.6])
