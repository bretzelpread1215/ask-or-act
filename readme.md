# ask or act

a small simulation study of when an indoor delivery robot should ask for clarification before choosing among three possible targets.

the project provides task generation, confidence transformations, three policies, task-level evaluation, reproducible random streams, and replicate-level summaries.

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
