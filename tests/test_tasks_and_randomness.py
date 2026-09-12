import numpy as np
import pytest

from ask_or_act.randomness import replicate_streams
from ask_or_act.tasks import generate_tasks, true_ambiguity


def test_generated_tasks_are_valid_and_reproducible() -> None:
    task_rng_a, _ = replicate_streams(17, 3)
    task_rng_b, _ = replicate_streams(17, 3)

    tasks_a = generate_tasks(200, 1.0, task_rng_a)
    tasks_b = generate_tasks(200, 1.0, task_rng_b)

    np.testing.assert_array_equal(tasks_a.true_probabilities, tasks_b.true_probabilities)
    np.testing.assert_array_equal(tasks_a.intended_targets, tasks_b.intended_targets)
    assert tasks_a.true_probabilities.shape == (200, 3)
    assert np.all(tasks_a.true_probabilities > 0)
    np.testing.assert_allclose(tasks_a.true_probabilities.sum(axis=1), 1.0)
    assert np.all((0 <= tasks_a.intended_targets) & (tasks_a.intended_targets < 3))


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


def test_generate_tasks_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_tasks(0, 1.0, np.random.default_rng(1))
