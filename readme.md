# ask or act

a small simulation study of when an indoor delivery robot should ask for clarification before choosing among three possible targets.

the project provides task generation, confidence transformations, three policies, task-level evaluation, reproducible random streams, and replicate-level summaries.

the simulation keeps latent tasks, reported observations, and policy decisions separate. policies receive only reported target probabilities. they never receive the true distribution or intended target.

## setup

python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

## run

```powershell
python -m ask_or_act
```

the primary run writes replicate results, group summaries, paired cost differences, and run metadata to `results/`. use `--task-count`, `--replicate-count`, `--seed`, or `--output-directory` to override those settings.

each row in `primary_replicates.csv` is one replicate mean over the configured number of tasks. `primary_summary.csv` and `paired_cost_differences.csv` aggregate those replicate-level values. intervals use replicate means as the independent monte carlo units. policy comparisons are paired within replicate because each policy sees the same generated tasks.

## primary setting

- three candidate targets
- true probabilities drawn from `dirichlet(1, 1, 1)`
- correct action cost `0`
- clarification cost `1`
- incorrect action cost `5`
- threshold policy asks below confidence `0.80`
- 10,000 tasks per replicate
- 100 replicates

see `experiment_spec.md` for the preregistered comparisons and assumptions. decisions and postponed work are tracked in `notes/research_log.md`.
