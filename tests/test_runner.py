import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ask_or_act.config import ExperimentConfig
from ask_or_act.runner import (
    calculate_paired_cost_differences,
    collect_replicates,
    run_primary_experiment,
    summarize_replicates,
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def test_small_run_has_expected_rows_groups_and_pairing() -> None:
    config = ExperimentConfig(task_count=80, replicate_count=4, master_seed=91)
    records = collect_replicates(config)
    summary = summarize_replicates(records)
    paired = calculate_paired_cost_differences(records)

    assert len(records) == 48
    assert len(summary) == 12
    assert len(paired) == 8
    assert Counter(record["replicate"] for record in records) == {
        0: 12,
        1: 12,
        2: 12,
        3: 12,
    }
    assert all(row["n_replicates"] == 4 for row in summary)

    for row in paired:
        threshold = [
            record["expected_total_cost"]
            for record in records
            if record["regime"] == row["regime"]
            and record["policy"] == "threshold"
        ]
        baseline = [
            record["expected_total_cost"]
            for record in records
            if record["regime"] == row["regime"]
            and record["policy"] == row["baseline_policy"]
        ]
        expected = float(np.mean(np.asarray(threshold) - np.asarray(baseline)))
        assert row["mean_difference"] == expected


def test_primary_outputs_are_complete_and_deterministic(tmp_path: Path) -> None:
    config = ExperimentConfig(task_count=50, replicate_count=3, master_seed=72)
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    first_paths = run_primary_experiment(config, first_directory)
    second_paths = run_primary_experiment(config, second_directory)

    expected_names = {
        "primary_replicates.csv",
        "primary_summary.csv",
        "paired_cost_differences.csv",
        "run_metadata.json",
    }
    assert {path.name for path in first_paths.values()} == expected_names
    assert {path.name for path in first_directory.iterdir()} == expected_names
    for name in expected_names:
        assert (first_directory / name).read_bytes() == (second_directory / name).read_bytes()

    assert len(_csv_rows(first_paths["replicates"])) == 36
    assert len(_csv_rows(first_paths["summary"])) == 12
    assert len(_csv_rows(first_paths["paired"])) == 8

    metadata = json.loads(first_paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["configuration"]["task_count"] == 50
    assert metadata["configuration"]["replicate_count"] == 3
    assert metadata["configuration"]["threshold"] == 0.8
    assert metadata["seed"] == 72
    assert set(metadata["versions"]) == {"numpy", "python", "scipy"}
