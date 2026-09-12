from dataclasses import dataclass
import math

from ask_or_act.policies import cost_derived_threshold


@dataclass(frozen=True)
class ExperimentConfig:
    n_targets: int = 3
    dirichlet_concentration: float = 1.0
    overconfident_temperature: float = 0.5
    underconfident_temperature: float = 2.0
    noise_sigma: float = 0.5
    correct_action_cost: float = 0.0
    clarification_cost: float = 1.0
    incorrect_action_cost: float = 5.0
    task_count: int = 10_000
    replicate_count: int = 100
    master_seed: int = 20260820

    def __post_init__(self) -> None:
        integer_fields = {
            "n_targets": self.n_targets,
            "task_count": self.task_count,
            "replicate_count": self.replicate_count,
            "master_seed": self.master_seed,
        }
        if any(type(value) is not int for value in integer_fields.values()):
            raise ValueError("counts and seeds must be integers")
        numeric_fields = (
            self.dirichlet_concentration,
            self.overconfident_temperature,
            self.underconfident_temperature,
            self.noise_sigma,
            self.correct_action_cost,
            self.clarification_cost,
            self.incorrect_action_cost,
        )
        if not all(math.isfinite(value) for value in numeric_fields):
            raise ValueError("configuration values must be finite")
        if self.n_targets < 2:
            raise ValueError("n_targets must be at least 2")
        if self.dirichlet_concentration <= 0:
            raise ValueError("dirichlet_concentration must be positive")
        if min(
            self.overconfident_temperature,
            self.underconfident_temperature,
        ) <= 0:
            raise ValueError("temperatures must be positive")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be nonnegative")
        if self.task_count <= 0 or self.replicate_count <= 0:
            raise ValueError("task_count and replicate_count must be positive")
        if self.master_seed < 0:
            raise ValueError("master_seed must be nonnegative")
        if min(
            self.correct_action_cost,
            self.clarification_cost,
            self.incorrect_action_cost,
        ) < 0:
            raise ValueError("costs must be nonnegative")
        if self.incorrect_action_cost <= self.correct_action_cost:
            raise ValueError("incorrect_action_cost must exceed correct_action_cost")
        if not (
            self.correct_action_cost
            <= self.clarification_cost
            <= self.incorrect_action_cost
        ):
            raise ValueError("costs must satisfy correct <= clarification <= incorrect")

    @property
    def threshold(self) -> float:
        return cost_derived_threshold(
            self.correct_action_cost,
            self.clarification_cost,
            self.incorrect_action_cost,
        )
