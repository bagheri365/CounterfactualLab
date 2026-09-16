# CounterfactualLab

**From predicting who will engage to estimating who will engage because of the recommendation.**

CounterfactualLab is a hands-on causal inference lab for learning uplift modeling, heterogeneous treatment effects, confounding, propensity methods, and causal targeting from first principles.

## The question

A conventional response model asks:

`P(Y=1 | X, T=1)` — who is likely to respond if treated?

Causal targeting asks:

`tau(x) = E[Y(1) - Y(0) | X=x]` — whose outcome changes because of treatment?

Those rankings need not agree. A user can have high baseline response probability and little incremental treatment effect. Under a treatment budget, the final object is a policy `pi(x)` that decides whom to treat.

## Learning path

| Milestone | Question | Main lesson |
| --- | --- | --- |
| M0 | What are we building? | Separate data, models, estimates, policies, evaluation, and diagnostics. |
| M1 | What is a counterfactual? | Only one potential outcome is factual for each user. |
| M2 | Is response prediction uplift? | No: high response can reflect high baseline risk rather than treatment effect. |
| M3 | How can CATE be estimated? | A T-Learner models treated and control outcomes separately. |
| M4 | How can uplift be evaluated? | Randomized factual outcomes support group/ranking evaluation without individual `tau_true`. |
| M5 | What does confounding do? | Treatment-control differences can be badly biased when assignment depends on outcome predictors. |
| M6 | What do propensity scores/IPW do? | Reweighting can balance observed confounders under identification and overlap assumptions. |
| M7 | What is doubly robust estimation? | AIPW combines propensity and outcome nuisance models; cross-fitting keeps nuisance predictions out of fold. |
| M8 | What if overlap is weak? | Extreme propensities create unstable weights and low effective sample size; better ML cannot create support. |
| M9 | What if a confounder is hidden? | Excellent observed balance does not establish exchangeability; IPW/DR do not remove unobserved confounding. |
| M10 | How does CATE become a decision? | Rank estimated effects, impose a budget, and evaluate the resulting treatment policy. |
| M11 | What changes on real randomized data? | Hillstrom permits experimental uplift evaluation, but true individual effects, PEHE, and oracle policy value are unavailable. |

## Selected results

The synthetic experiments deliberately change treatment assignment and other controlled conditions so causal assumptions and failure modes can be inspected directly.

- **M2:** response and oracle-uplift top-20% targeting overlapped only 41.2%. Oracle uplift targeting produced 201.09 expected incremental conversions per 1,000 treatments versus 124.15 for response targeting.
- **M3:** the T-Learner reached CATE correlation 0.906 and 81.8% top-20% overlap with the oracle ranking in this favorable randomized synthetic setting.
- **M5:** observed confounding moved the naive ATE estimate to 0.2532 while the true ATE was 0.0607.
- **M6:** estimated-propensity IPW produced an ATE estimate of 0.0736 and reduced maximum observed-covariate SMD from 0.769 to 0.020.
- **M8:** weak overlap reduced IPW effective sample size to about 6.2% of 20,000 observations and substantially increased across-seed IPW variability.
- **M9:** weighting reduced maximum observed-X SMD from 0.647 to 0.004, yet hidden-confounder imbalance remained and IPW/DR estimates stayed far from the true ATE.
- **M10:** at a 20% treatment budget, T-Learner targeting achieved synthetic oracle incremental gain of 0.0391 per population member versus 0.0248 for response targeting.
- **M11:** Men's Email versus No Email contained 42,613 Hillstrom customers. The observed randomized conversion effect was 0.0068; pedagogical AUUC was 171.45 for T-Learner ranking, 163.84 for response ranking, and 140.74 for one random ranking.

These are experiment outputs, not universal performance claims.

## Assumptions and boundaries

**Prediction is not causation.** Predicting response under treatment does not identify the difference between treatment and no treatment.

**Identification is not estimation.** Propensity methods, outcome models, and doubly robust estimators operate after causal identification assumptions are stated. More flexible ML cannot repair an unidentified causal effect.

**Observed adjustment is not protection from hidden confounding.** IPW and AIPW/DR in observational scenarios rely on observed-confounding assumptions. Double robustness concerns nuisance-model specification under those assumptions; it does not mean robustness to unobserved confounding.

**Overlap matters.** If comparable treated and untreated users are absent in parts of feature space, weighting and regression cannot manufacture missing counterfactual support.

**Individual treatment-effect labels are latent.** Individual `Y(1)-Y(0)` and categories such as “persuadable” are not ordinarily observed. Synthetic data expose both potential outcomes only for learning and validation.

**Real data remove the oracle.** Hillstrom is randomized, so factual treatment/control outcomes support causal evaluation at group and policy levels. They do not reveal each customer's missing counterfactual, true individual CATE, or PEHE.

**AUUC here is pedagogical.** The M4/M11 implementation integrates cumulative estimated incremental outcomes. It should not be interpreted as a universally standardized Qini coefficient.

## Repository structure

```text
configs/                           synthetic scenario configurations
src/counterfactuallab/data/        synthetic and Hillstrom data
src/counterfactuallab/models/      response, T-Learner, propensity, DR
src/counterfactuallab/evaluation/  CATE, uplift, IPW, and policy evaluation
src/counterfactuallab/policies/    response, uplift, and oracle policies
src/counterfactuallab/diagnostics/ balance and overlap diagnostics
scripts/                           runnable milestone experiments
tests/                             unit and learning-invariant tests
data/raw/                          local third-party raw data (not committed)
```

The organizing flow is:

`DATA -> MODELS -> ESTIMATES -> POLICIES -> EVALUATION`

Diagnostics inspect data and estimates rather than changing the causal target.

## Setup

Create and activate a Python 3.12 environment, then install the project:

```bash
python -m pip install -e ".[dev]"
pytest
```

Run individual experiments from the repository root, for example:

```bash
python scripts/run_t_learner.py
python scripts/run_ipw.py
python scripts/run_overlap_failure.py
python scripts/run_hidden_confounding.py
python scripts/run_cate_to_policy.py
```

## Hillstrom data

M11 uses Kevin Hillstrom's randomized email-marketing experiment. Download the original CSV and save it as:

```text
data/raw/hillstrom.csv
```

Challenge page: `https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html`

Original CSV: `https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`

Then run:

```bash
python scripts/run_hillstrom.py
```

M11 starts with the binary comparison **Men's Email vs No Email** and uses **conversion** as the outcome. The Women's Email arm remains available for later extensions.

## What this lab does not claim

CounterfactualLab is a teaching repository, not a benchmark claiming state-of-the-art causal performance. Synthetic oracle metrics exist to make assumptions and failure modes visible. Results from one DGP, one random seed, or one randomized marketing experiment should not be generalized without further validation, uncertainty analysis, and domain-specific causal reasoning.
