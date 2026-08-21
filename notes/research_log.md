# research log

## substantive decisions

- use three targets and draw true probabilities from `dirichlet(1, 1, 1)`
- sample the intended target from the true distribution; policies see only reported confidence
- use abstract costs `0` for a correct action, `1` for clarification, and `5` for an incorrect action
- test incorrect-action costs `2`, `5`, and `10` with cost-derived thresholds
- compare calibrated, overconfident, underconfident, and noisy confidence
- use temperature scaling at `t = 0.5` and `t = 2`, and gaussian logit noise at `sigma = 0.5`
- run 100 replicates of 10,000 shared tasks and use paired comparisons
- use two-sided 95% t intervals across replicate-level results without p-values

## reasons

- expected total cost makes the interaction-error tradeoff explicit
- shared tasks reduce monte carlo noise in policy comparisons
- temperature scaling isolates clarification changes because it preserves ranking
- logit noise tests errors in both confidence magnitude and target ranking
- a fixed cost-derived threshold makes the robustness claim falsifiable

## framing issue

with perfectly calibrated confidence and known costs, `tau = 0.80` is analytically optimal when clarification costs `1` and an incorrect action costs `5`. the experiment therefore studies threshold robustness under noisy or miscalibrated confidence rather than only checking expected-cost arithmetic.

## assumptions and limitations

- clarification reveals the intended target perfectly and cannot be followed by an incorrect action
- the dirichlet task generator is a modeling choice, not a claim about human instructions
- monte carlo intervals describe simulation uncertainty, not real-world behavioral uncertainty

## postponed ideas

- vary symmetric dirichlet concentration below and above `1`
- test additional temperature and noise severities
- transfer the fixed `0.80` threshold to other costs as a labeled cost-mismatch analysis

## open questions

- how conclusions change when clarification is imperfect
- which task-distribution and confidence parameters could be estimated from later empirical data

## observations

the full experiment has not been run.
