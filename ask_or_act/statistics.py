import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t


def mean_t_interval(values: ArrayLike, confidence: float = 0.95) -> dict[str, float]:
    """calculate a two-sided t interval across replicate-level values."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError("values must contain at least two replicate-level results")
    if not np.all(np.isfinite(sample)):
        raise ValueError("values must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    mean = float(sample.mean())
    standard_error = float(sample.std(ddof=1) / np.sqrt(sample.size))
    critical_value = float(t.ppf((1.0 + confidence) / 2.0, df=sample.size - 1))
    margin = critical_value * standard_error
    return {"mean": mean, "lower": mean - margin, "upper": mean + margin}


def paired_cost_difference(
    threshold_costs: ArrayLike,
    baseline_costs: ArrayLike,
    confidence: float = 0.95,
) -> dict[str, float]:
    """summarize threshold-minus-baseline replicate-level cost differences."""
    threshold = np.asarray(threshold_costs, dtype=float)
    baseline = np.asarray(baseline_costs, dtype=float)
    if threshold.shape != baseline.shape:
        raise ValueError("paired cost arrays must have the same shape")
    return mean_t_interval(threshold - baseline, confidence)
