import numpy as np
import pytest

from ask_or_act.models import DecisionBatch, ObservationBatch, TaskBatch


def test_task_batch_rejects_target_outside_probability_columns() -> None:
    with pytest.raises(ValueError, match="index a target column"):
        TaskBatch(np.array([[0.6, 0.4]]), np.array([2]))


@pytest.mark.parametrize(
    "probabilities",
    [
        np.empty((0, 3)),
        np.array([[0.5, np.nan, 0.5]]),
        np.array([[0.6, -0.1, 0.5]]),
        np.array([[0.6, 0.3, 0.2]]),
    ],
)
def test_observation_batch_rejects_invalid_probabilities(
    probabilities: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        ObservationBatch(probabilities)


def test_decision_batch_rejects_silent_dtype_coercion() -> None:
    with pytest.raises(ValueError, match="boolean"):
        DecisionBatch(np.array([0, 1]), np.array([0, 1]))
    with pytest.raises(ValueError, match="integers"):
        DecisionBatch(np.array([False, True]), np.array([0.0, 1.0]))
