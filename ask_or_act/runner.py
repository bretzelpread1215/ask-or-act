import csv
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ask_or_act.config import ExperimentConfig
from ask_or_act.simulation import METRIC_NAMES, POLICY_ORDER, REGIME_ORDER, run_replicate
from ask_or_act.statistics import mean_t_interval, paired_cost_difference

REPLICATE_FIELDS = ("replicate", "regime", "policy", *METRIC_NAMES)
BASELINE_ORDER = ("always_act", "always_ask")


def collect_replicates(config: ExperimentConfig) -> list[dict[str, Any]]:
    """run every configured replicate in deterministic order."""
    records: list[dict[str, Any]] = []
    for replicate_index in range(config.replicate_count):
        records.extend(run_replicate(config, replicate_index))
    return records


def summarize_replicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """summarize each regime-policy group across replicate-level values."""
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        for policy in POLICY_ORDER:
            group = [
                record
                for record in records
                if record["regime"] == regime and record["policy"] == policy
            ]
            if len(group) < 2:
                raise ValueError("each regime-policy group needs at least two replicates")
            row: dict[str, Any] = {
                "regime": regime,
                "policy": policy,
                "n_replicates": len(group),
            }
            for metric in METRIC_NAMES:
                interval = mean_t_interval([record[metric] for record in group])
                row[f"{metric}_mean"] = interval["mean"]
                row[f"{metric}_ci_lower"] = interval["lower"]
                row[f"{metric}_ci_upper"] = interval["upper"]
            rows.append(row)
    return rows


def calculate_paired_cost_differences(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """summarize threshold-minus-baseline costs within each regime."""
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        threshold_by_replicate = {
            record["replicate"]: record["expected_total_cost"]
            for record in records
            if record["regime"] == regime and record["policy"] == "threshold"
        }
        for baseline in BASELINE_ORDER:
            baseline_by_replicate = {
                record["replicate"]: record["expected_total_cost"]
                for record in records
                if record["regime"] == regime and record["policy"] == baseline
            }
            if threshold_by_replicate.keys() != baseline_by_replicate.keys():
                raise ValueError("paired policies must contain the same replicates")
            replicate_ids = sorted(threshold_by_replicate)
            if len(replicate_ids) < 2:
                raise ValueError("paired comparisons need at least two replicates")
            interval = paired_cost_difference(
                [threshold_by_replicate[index] for index in replicate_ids],
                [baseline_by_replicate[index] for index in replicate_ids],
            )
            rows.append(
                {
                    "regime": regime,
                    "baseline_policy": baseline,
                    "n_replicates": len(replicate_ids),
                    "mean_difference": interval["mean"],
                    "ci_lower": interval["lower"],
                    "ci_upper": interval["upper"],
                }
            )
    return rows


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metadata(config: ExperimentConfig) -> dict[str, Any]:
    configuration = asdict(config)
    configuration["threshold"] = config.threshold
    return {
        "configuration": configuration,
        "seed": config.master_seed,
        "interval_method": "two-sided 95% t interval across replicate-level values",
        "paired_difference": "threshold minus baseline",
        "versions": {
            "numpy": version("numpy"),
            "python": platform.python_version(),
            "scipy": version("scipy"),
        },
    }


def run_primary_experiment(
    config: ExperimentConfig,
    output_directory: str | Path = "results",
) -> dict[str, Path]:
    """run, summarize, and save the primary experiment."""
    if config.replicate_count < 2:
        raise ValueError("replicate_count must be at least 2 for t intervals")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    replicates = collect_replicates(config)
    summary = summarize_replicates(replicates)
    paired = calculate_paired_cost_differences(replicates)

    paths = {
        "replicates": output_path / "primary_replicates.csv",
        "summary": output_path / "primary_summary.csv",
        "paired": output_path / "paired_cost_differences.csv",
        "metadata": output_path / "run_metadata.json",
    }
    _write_csv(paths["replicates"], REPLICATE_FIELDS, replicates)

    summary_fields = ["regime", "policy", "n_replicates"]
    for metric in METRIC_NAMES:
        summary_fields.extend(
            [f"{metric}_mean", f"{metric}_ci_lower", f"{metric}_ci_upper"]
        )
    _write_csv(paths["summary"], tuple(summary_fields), summary)
    _write_csv(
        paths["paired"],
        (
            "regime",
            "baseline_policy",
            "n_replicates",
            "mean_difference",
            "ci_lower",
            "ci_upper",
        ),
        paired,
    )
    paths["metadata"].write_text(
        json.dumps(_metadata(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
