# ask or act

a small simulation study of when an indoor delivery robot should ask for clarification before choosing among three possible targets.

the first milestone provides the simulation core: task generation, confidence transformations, three policies, task-level evaluation, reproducible random streams, and replicate-level confidence intervals. it does not run the full experiment or create figures.

## setup

python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

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
