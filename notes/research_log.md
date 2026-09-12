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

the code represents latent task batches, reported observations, and policy decisions as separate validated objects. this makes the information boundary testable: a policy can inspect reported probabilities but cannot inspect the intended target or true probabilities through its input.

replicate output records include their task count and are validated as a complete panel before summaries or paired comparisons are calculated. duplicate or missing regime-policy rows are rejected.

the primary run completed with 100 replicates of 10,000 tasks and seed `20260820`. paired differences are threshold cost minus baseline cost, with two-sided 95% monte carlo intervals:

| regime | baseline | mean difference | 95% monte carlo interval |
|---|---|---:|---:|
| calibrated | always act | -0.982699 | [-0.986991, -0.978407] |
| calibrated | always ask | -0.040539 | [-0.041764, -0.039314] |
| overconfident | always act | -0.818036 | [-0.821890, -0.814182] |
| overconfident | always ask | +0.124124 | [+0.121277, +0.126971] |
| underconfident | always act | -0.945381 | [-0.949981, -0.940781] |
| underconfident | always ask | -0.003221 | [-0.003372, -0.003070] |
| noisy | always act | -1.109893 | [-1.114706, -1.105080] |
| noisy | always ask | +0.004962 | [+0.003408, +0.006516] |

the csv files retain full precision.

the primary calibrated-confidence claim was supported against both baselines. the threshold policy beat always act in all four regimes. full robustness was not supported because the threshold policy was more costly than always ask under overconfident and noisy confidence. the underconfident advantage over always ask and the noisy disadvantage were both small in magnitude despite narrow monte carlo intervals. within this simulation, the results suggest that a cost-derived threshold depends on the quality of the confidence estimates it receives.
