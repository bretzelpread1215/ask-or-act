# experiment specification

## question

when does a confidence-threshold policy reduce expected cost relative to always acting or always asking, and how robust is it to distorted confidence?

## task model

each task has three candidate targets. draw the true distribution and intended target as

\[
p \sim \operatorname{Dirichlet}(1,1,1), \qquad y \sim \operatorname{Categorical}(p).
\]

the true ambiguity of a task is `1 - max(p)`. probabilities are kept strictly positive so `log(p)` is defined. the symmetric dirichlet concentration is configurable.

## reported confidence

policies observe `q`, not `p` or `y`.

- calibrated: `q = p`
- overconfident: `q_i = p_i^(1/t) / sum_j p_j^(1/t)`, with `t = 0.5`
- underconfident: the same transformation, with `t = 2`
- noisy: `q = softmax(log(p) + epsilon)`, where each `epsilon_i` is independently drawn from `normal(0, 0.5^2)`

temperature scaling preserves target ranking. gaussian logit noise can change both reported confidence and the top-ranked target. temperatures and noise severity are configurable.

## policies

- always act on `argmax(q)`
- always ask
- ask when `max(q) < tau`; act at exact equality

the cost-derived threshold is

\[
\tau = \frac{c_{wrong} - c_{ask}}{c_{wrong} - c_{correct}}.
\]

require `c_correct <= c_ask <= c_wrong` and `c_wrong > c_correct`. the primary costs are `c_correct = 0`, `c_ask = 1`, and `c_wrong = 5`, giving `tau = 0.80`.

## outcomes

task-level total cost is

\[
c_{ask} I(asked) + c_{wrong} I(incorrect).
\]

report expected total cost, task-success rate, incorrect-action rate, and clarification frequency. all rates use all tasks as the denominator.

clarification is assumed to reveal the intended target perfectly. an asked task incurs only clarification cost and ends successfully. this is an idealized assumption, not a model of real clarification reliability. task success is therefore the complement of incorrect-action rate.

## simulation

run 100 independent replicates of 10,000 tasks from a fixed master seed. within a replicate, generate `p` and `y` once and reuse them for every regime and policy. generate noisy confidence from a separate reproducible random stream.

calculate two-sided 95% t intervals across the 100 replicate-level values. calculate paired policy differences within each replicate before applying the same interval method. these are monte carlo intervals, not estimates of uncertainty about real robots or people.

## claims

primary claim: at `c_wrong = 5`, the fixed `tau = 0.80` policy has lower expected total cost than both baselines under calibrated confidence.

robustness claim: the same threshold policy has lower expected total cost than both baselines in each miscalibrated regime.

define paired differences as `threshold cost - baseline cost`. a comparison supports its claim only when its entire 95% monte carlo interval is below zero. call the policy fully robust only when this holds against both baselines in all four regimes. otherwise, report where the claim holds and where it breaks. do not use p-values.

directional predictions compare each distorted regime's threshold policy with the calibrated threshold policy:

- overconfidence lowers clarification frequency and raises incorrect-action rate
- underconfidence raises clarification frequency and lowers incorrect-action rate

## required analysis

- primary cost figure by policy and confidence regime, with 95% monte carlo intervals
- one aligned three-panel figure for success, incorrect-action, and clarification rates
- descriptive threshold sweep from `0.00` through `1.00` in steps of `0.01`, with `0.80` marked
- compact sensitivity table for incorrect-action costs `2`, `5`, and `10`, using thresholds `0.50`, `0.80`, and `0.90`
- numerical means, intervals, and paired cost differences

the threshold sweep does not tune or replace the preregistered policy. transferring `tau = 0.80` to other cost settings is optional and must be labeled as cost mismatch.
