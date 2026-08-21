import argparse
import time
from collections.abc import Sequence

from ask_or_act.config import ExperimentConfig
from ask_or_act.runner import run_primary_experiment


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="run the primary ask-or-act experiment")
    parser.add_argument("--task-count", type=int, default=defaults.task_count)
    parser.add_argument("--replicate-count", type=int, default=defaults.replicate_count)
    parser.add_argument("--seed", type=int, default=defaults.master_seed)
    parser.add_argument("--output-directory", default="results")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    config = ExperimentConfig(
        task_count=args.task_count,
        replicate_count=args.replicate_count,
        master_seed=args.seed,
    )
    started = time.perf_counter()
    run_primary_experiment(config, args.output_directory)
    elapsed = time.perf_counter() - started
    print(
        f"wrote {config.replicate_count * 12} replicate rows "
        f"to {args.output_directory} in {elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
