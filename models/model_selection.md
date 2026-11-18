# Model selection and primary metric (Person B)

## Decision rule

Use **five-fold trip-grouped CV Top-1 accuracy across all observations** as the
primary classifier-selection metric. A prediction is correct only when the
highest-scored candidate arc equals the reference arc. An observation whose
correct arc is absent from the candidate set remains an error. This matches the
project's operational question: did the model choose the right road for each
GPS observation?

Use the CV slice excluding trips `00000033` and `00000054` as a diagnostic, not
as a substitute primary score. Report student-test Top-1 once for the final
frozen configuration; it is a ten-trip holdout, and it should not be used to
tune or choose hyperparameters. Candidate log loss is a key secondary metric
because C consumes candidate probabilities as HMM emissions. Also report
inference time and fold variation.

## Baseline comparison

| Model | CV Top-1, all trips | CV Top-1, excluding 33/54 | CV candidate log loss | Student-test Top-1 | Student-test candidate log loss |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 74.10% | 82.91% | 0.2316 | 84.61% | 0.2082 |
| KNN, distance weighted | 73.51% | 82.21% | 0.2977 | 84.43% | 0.2189 |
| Random Forest | 74.62% | 83.52% | 0.1121 | 85.18% | 0.0815 |
| MLP | 74.54% | 83.41% | 0.0951 | 84.97% | 0.0759 |
| HistGradientBoosting | **74.78%** | **83.70%** | **0.0922** | **85.52%** | **0.0721** |

The committed, machine-generated combined table is
`results/models/model_comparison_all.csv`; its companion chart is
`results/models/model_comparison_top1.png`. After Bayesian tuning, that table
also includes a separately labelled tuned HistGradientBoosting row.

## Current selection

The baseline evidence favors **HistGradientBoosting**: it has the best baseline
all-trip CV Top-1, best student-test Top-1, and lowest candidate log loss of
the five. Its Top-1 margin over Random Forest is only 0.16 percentage points,
so this is a metric-based choice, not proof of meaningful superiority.

The ten-trial Bayesian search used a three-fold grouped-CV objective for
runtime, then evaluated its best candidate with the agreed five-fold CV. The
best search candidate was `max_iter=100`, `learning_rate=0.05`,
`max_leaf_nodes=15`, `min_samples_leaf=10`, and `l2_regularization=0`. It scored
74.75% five-fold all-trip Top-1 and 0.09231 candidate log loss. The baseline
scored 74.78% and 0.09218 on the same five-fold setup, so the baseline wins the
primary metric and its probabilities are marginally better on the secondary
metric. The **selected export is therefore baseline HistGradientBoosting**:
150 iterations, learning rate 0.1, 31 leaf nodes, minimum leaf size 20, and
L2 regularization 1.0. Student-test was not used to make this decision. Full
details are in `results/bayes/hgb_bayesopt_20261118/selection_decision.json`;
the fitted artifact and handoff schema are in `models/artifacts/`.

## The 95% target and current candidate limit

With the current 50 m table, 11.16% of training observations have no reference
arc among their candidates. Therefore the **maximum possible all-trip Top-1 is
88.84%**, even for a perfect ranker. Hyperparameter tuning alone cannot produce
95% under this evaluation definition. A's existing radius scan shows that 150
m reduces the miss rate to 4.46% and 200 m to 1.19%; broader candidate pools
must be rebuilt and compared with grouped CV because they also add distractor
roads, training rows, and inference work. Do not present a 95% figure from an
outlier-excluded or candidate-conditioned subset as the all-trip score.
