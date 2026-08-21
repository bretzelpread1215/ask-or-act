from typing import Any

from ask_or_act.confidence import confidence_regimes
from ask_or_act.config import ExperimentConfig
from ask_or_act.evaluation import evaluate_tasks, summarize_outcomes
from ask_or_act.policies import always_act, always_ask, threshold_policy
from ask_or_act.randomness import replicate_streams
from ask_or_act.tasks import generate_tasks

REGIME_ORDER = ("calibrated", "overconfident", "underconfident", "noisy")
POLICY_ORDER = ("always_act", "always_ask", "threshold")
METRIC_NAMES = (
    "expected_total_cost",
    "task_success_rate",
    "incorrect_action_rate",
    "clarification_frequency",
)


def run_replicate(config: ExperimentConfig, replicate_index: int) -> list[dict[str, Any]]:
    """run every regime and policy on one shared task batch."""
    task_rng, noise_rng = replicate_streams(config.master_seed, replicate_index)
    probabilities, intended_targets = generate_tasks(
        n_tasks=config.task_count,
        concentration=config.dirichlet_concentration,
        rng=task_rng,
        n_targets=config.n_targets,
    )
    regimes = confidence_regimes(
        probabilities,
        noise_rng=noise_rng,
        overconfident_temperature=config.overconfident_temperature,
        underconfident_temperature=config.underconfident_temperature,
        noise_sigma=config.noise_sigma,
    )

    records: list[dict[str, Any]] = []
    for regime_name in REGIME_ORDER:
        reported = regimes[regime_name]
        decisions = {
            "always_act": always_act(reported),
            "always_ask": always_ask(reported),
            "threshold": threshold_policy(reported, config.threshold),
        }
        for policy_name in POLICY_ORDER:
            asks, actions = decisions[policy_name]
            outcomes = evaluate_tasks(
                intended_targets,
                asks,
                actions,
                correct_action_cost=config.correct_action_cost,
                clarification_cost=config.clarification_cost,
                incorrect_action_cost=config.incorrect_action_cost,
            )
            records.append(
                {
                    "replicate": replicate_index,
                    "regime": regime_name,
                    "policy": policy_name,
                    **summarize_outcomes(outcomes),
                }
            )
    return records
