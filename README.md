# CounterfactualLab

**From predicting who will engage to estimating who will engage because of the recommendation.**

CounterfactualLab is a hands-on causal inference lab for learning uplift modeling, heterogeneous treatment effects, confounding, propensity methods, and causal targeting from first principles.

> **Core principle:** Change the causal assumption, measure what breaks, and make the missing counterfactual explicit.

## At a Glance

|                          |                                                                                                                               |
|:-------------------------|:------------------------------------------------------------------------------------------------------------------------------|
| **Research question**    | Why isn’t predicting who will engage the same as predicting who will engage **because of** a recommendation?                  |
| **Causal target**        | `tau(x) = E[Y(1) - Y(0) | X=x]`                                                                                               |
| **Synthetic setting**    | Controlled potential outcomes expose true CATE, treatment propensities, counterfactuals, and oracle policy value.             |
| **Real experiment**      | 42,613 Hillstrom customers in the Men’s Email vs No Email comparison, using conversion as the outcome.                        |
| **Main pattern**         | Response probability and treatment effect can rank users very differently.                                                    |
| **Key failure modes**    | Observed confounding, weak overlap, hidden confounding, and evaluation leakage.                                               |
| **Evaluation principle** | Learned uplift rankings are scored out of fold so a row’s factual outcome is not used to train the model that ranks that row. |

## Why This Project Exists

A conventional response model asks:

`P(Y=1 | X, T=1)` — **who is likely to respond if treated?**

Causal targeting asks:

`tau(x) = E[Y(1) - Y(0) | X=x]` — **whose outcome changes because of treatment?**

These are different targets.

A user can have a high probability of responding to a recommendation even if the recommendation changes nothing for that user. Conversely, a user with modest baseline response probability can have a large incremental treatment effect.

Under a treatment budget, the final object is not merely a prediction. It is a policy:

`pi(x)` — **whom should we treat?**

|                             | Response prediction          | Causal targeting                      |
|-----------------------------|:-----------------------------|:--------------------------------------|
| **Target**                  | `P(Y=1 | X, T=1)`            | `E[Y(1)-Y(0) | X=x]`                  |
| **Question**                | Who will respond if treated? | Who responds because of treatment?    |
| **Needs a counterfactual?** | No                           | Yes                                   |
| **Typical use**             | Response scoring             | Incremental targeting / uplift policy |

## Key Findings

### 1. Prediction and uplift can rank different users

In the randomized synthetic experiment, response targeting and oracle-uplift targeting shared only **41.2%** of their top-20% users.

At the same treatment budget:

| Targeting rule        | Expected incremental conversions / 1,000 treatments |
|:----------------------|----------------------------------------------------:|
| Response ranking      |                                              124.15 |
| Oracle uplift ranking |                                              201.09 |

The T-Learner recovered much of this structure in the favorable randomized synthetic setting, reaching **0.906 CATE correlation** and **81.8% top-20% overlap** with the oracle ranking.

The lesson is structural: high baseline response probability is not the same thing as high treatment effect.

### 2. Confounding can make naive treatment comparisons look causal

When treatment assignment depended on observed outcome predictors, the synthetic experiment had:

| Quantity                         |  Value |
|:---------------------------------|-------:|
| True ATE                         | 0.0607 |
| Naive treated-control difference | 0.2532 |
| Estimated-propensity IPW         | 0.0736 |

IPW also reduced the maximum observed-covariate standardized mean difference from **0.769** to **0.020**.

This demonstrates what propensity methods are trying to repair: imbalance in observed variables related to both treatment assignment and outcomes.

### 3. Better estimation cannot manufacture overlap

Under weak overlap, extreme propensity scores produced unstable inverse-probability weights.

| Diagnostic               | Confounded setting | Weak-overlap setting |
|:-------------------------|-------------------:|---------------------:|
| Extreme propensity share |               1.2% |                41.9% |
| Maximum weight           |               31.2 |                496.8 |
| Effective sample size    |              67.9% |                 6.2% |

Across seeds, IPW variability increased substantially under weak overlap.

The problem is not simply model quality. If comparable treated and untreated observations are missing from part of feature space, stronger machine learning cannot create the missing support.

### 4. Excellent observed balance does not rule out hidden confounding

The hidden-confounder experiment deliberately creates a variable that affects treatment and outcome but is unavailable to the estimator.

| Quantity                                |  Value |
|:----------------------------------------|-------:|
| True ATE                                | 0.0605 |
| Naive estimate                          | 0.3319 |
| Observed-X IPW                          | 0.2054 |
| Observed-X DR                           | 0.2046 |
| Maximum observed-X SMD before weighting |  0.647 |
| Maximum observed-X SMD after weighting  |  0.004 |

Observed covariates look extremely well balanced after weighting, yet the causal estimates remain far from the truth because the hidden confounder remains unaddressed.

Double robustness is robustness to nuisance-model specification under the relevant identification assumptions. It is **not** robustness to unobserved confounding.

### 5. CATE becomes useful through a policy

At a 20% treatment budget in the synthetic randomized setting:

| Policy              | Population-average incremental gain |
|:--------------------|------------------------------------:|
| Response targeting  |                              0.0248 |
| T-Learner targeting |                              0.0391 |
| Oracle targeting    |                              0.0402 |

The practical objective is not merely to estimate heterogeneous effects. It is to turn those estimates into a treatment decision under constraints such as budget or capacity.

### 6. Honest evaluation can change the apparent conclusion

The original M4/M11 implementations fitted learned rankings and evaluated them using factual outcomes from the same observations. M4 and M11 now use **five-fold out-of-fold ranking scores**.

On the synthetic experiment, the main lesson remained stable:

| Ranking   | Out-of-fold pedagogical AUUC | Oracle incremental conversions / 1,000 |
|:----------|-----------------------------:|---------------------------------------:|
| Response  |                       820.89 |                                 124.47 |
| T-Learner |                      1101.24 |                                 195.46 |
| Random    |                       558.17 |                                  60.52 |

On Hillstrom, however, honest evaluation materially changed the apparent ranking:

| Ranking   | Previous in-sample AUUC | Out-of-fold AUUC | Out-of-fold top-20% estimated effect |
|:----------|-------------------------|-----------------:|-------------------------------------:|
| Response  | 163.84                  |           146.19 |                               0.0067 |
| T-Learner | 171.45                  |           136.55 |                               0.0078 |
| Random    | 140.74                  |           140.74 |                               0.0078 |

The observed randomized conversion effect for Men’s Email vs No Email was **0.0068**.

These are point estimates from one experiment and one random ranking. They do **not** establish that one ranking method is superior; uncertainty was not estimated.

The methodological lesson is more important than the ordering: **randomization addresses treatment assignment, while cross-fitting separately prevents each row’s factual outcome from helping train the model that produces that row’s evaluation score.**

## Learning Path

CounterfactualLab is organized as a sequence of causal questions rather than a single benchmark.

| Milestone | Question                                             | Main lesson                                                                                                                         |
|:----------|:-----------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| M0        | What are we building?                                | Separate data, models, estimates, policies, evaluation, and diagnostics.                                                            |
| M1        | What is a counterfactual?                            | Only one potential outcome is factual for each user.                                                                                |
| M2        | Is response prediction uplift?                       | No: high response can reflect high baseline risk rather than treatment effect.                                                      |
| M3        | How can CATE be estimated?                           | A T-Learner models treated and control outcomes separately.                                                                         |
| M4        | How can uplift be evaluated?                         | Randomized factual outcomes support group/ranking evaluation; learned rankings should be scored out of fold.                        |
| M5        | What does confounding do?                            | Treatment-control differences can be badly biased when assignment depends on outcome predictors.                                    |
| M6        | What do propensity scores/IPW do?                    | Reweighting can balance observed confounders under identification and overlap assumptions.                                          |
| M7        | What is doubly robust estimation?                    | AIPW combines propensity and outcome nuisance models; cross-fitting keeps nuisance predictions out of fold.                         |
| M8        | What if overlap is weak?                             | Extreme propensities create unstable weights and low effective sample size; better ML cannot create support.                        |
| M9        | What if a confounder is hidden?                      | Excellent observed balance does not establish exchangeability; IPW/DR do not remove unobserved confounding.                         |
| M10       | How does CATE become a decision?                     | Rank estimated effects, impose a budget, and evaluate the resulting treatment policy.                                               |
| M11       | What changes on real randomized data?                | Hillstrom permits experimental uplift evaluation, but honest learned-ranking evaluation still requires train/evaluation separation. |
| M12       | How should the lessons fit together?                 | Synthesize identification, estimation, diagnostics, policy, and evaluation into one causal workflow.                                |
| M13       | How do we evaluate learned uplift rankings honestly? | Use out-of-fold scores to separate ranking learning from factual-outcome evaluation.                                                |

## Assumptions and Boundaries

**Prediction is not causation.** Predicting response under treatment does not identify the difference between treatment and no treatment.

**Identification is not estimation.** Propensity methods, outcome models, and doubly robust estimators operate after causal identification assumptions are stated. More flexible ML cannot repair an unidentified causal effect.

**Observed adjustment is not protection from hidden confounding.** IPW and AIPW/DR in observational scenarios rely on observed-confounding assumptions. Double robustness concerns nuisance-model specification under those assumptions; it does not mean robustness to unobserved confounding.

**Overlap matters.** If comparable treated and untreated users are absent in parts of feature space, weighting and regression cannot manufacture missing counterfactual support.

**Individual treatment-effect labels are latent.** Individual `Y(1)-Y(0)` and categories such as “persuadable” are not ordinarily observed. Synthetic data expose both potential outcomes only for learning and validation.

**Real data remove the oracle.** Hillstrom is randomized, so factual treatment/control outcomes support causal evaluation at group and policy levels. They do not reveal each customer’s missing counterfactual, true individual CATE, or PEHE.

**Randomization does not remove evaluation leakage.** M4 and M11 use five-fold cross-fitted response and T-Learner scores so each row is ranked by models fit without that row. Randomization addresses treatment assignment; cross-fitting separately keeps the row’s factual outcome out of the model that produces its evaluation score.

**AUUC here is pedagogical.** The M4/M11 implementation integrates cumulative estimated incremental outcomes. It should not be interpreted as a universally standardized Qini coefficient.

## Repository Structure

``` text
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

## Reproduce the Experiments

Create and activate a Python 3.12 environment, then install the project:

``` bash
python -m pip install -e ".[dev]"
pytest
```

Run individual experiments from the repository root:

``` bash
python scripts/run_response_vs_uplift.py
python scripts/run_t_learner.py
python scripts/run_uplift_evaluation.py
python scripts/run_confounding.py
python scripts/run_ipw.py
python scripts/run_dr.py
python scripts/run_overlap_failure.py
python scripts/run_hidden_confounding.py
python scripts/run_cate_to_policy.py
python scripts/run_hillstrom.py
```

## Hillstrom Data

M11 uses Kevin Hillstrom’s randomized email-marketing experiment. Download the original CSV and save it as:

``` text
data/raw/hillstrom.csv
```

Challenge page: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

Original CSV: https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Then run:

``` bash
python scripts/run_hillstrom.py
```

M11 starts with the binary comparison **Men’s Email vs No Email** and uses **conversion** as the outcome. The Women’s Email arm remains available for later extensions.

## What This Lab Does Not Claim

CounterfactualLab is a teaching repository, not a benchmark claiming state-of-the-art causal performance.

Synthetic oracle metrics exist to make assumptions and failure modes visible. Results from one data-generating process, one random seed, or one randomized marketing experiment should not be generalized without further validation, uncertainty analysis, and domain-specific causal reasoning.
