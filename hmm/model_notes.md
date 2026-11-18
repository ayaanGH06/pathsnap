# Model notes: Logistic Regression and HistGradientBoosting (Person C)

Everything here comes from code that was run. Official numbers are in
`results/models/run_20261118_115731_logistic_regression_hist_gradient_boosting/`.
Extra evidence (coefficients, permutation importance, per-trip Top-1, fit-set scores) comes from
`hmm/model_notes_evidence.py`, which retrained both models on the same tables.

Setup shared with the rest of the team: seed 42, radius 50 m candidate tables, the same 10 features
(`dist_to_arc, proj_frac, arc_len, heading_diff, speed, dt_prev, dist_rank, dist_minus_min, n_cand, conn_prev`),
5-fold trip-grouped CV over 80 train trips, 10 student_test trips. B's `model_run_config.json`
(random_forest_mlp_20261118) lists the same seed, folds and feature columns.

## A. Why each model suits PathSnap

Each GPS point has about 9 candidate arcs in train and 11 in student_test; about 9-10% of candidate rows are the true arc.
The classifier gives each candidate a probability, and the HMM/Viterbi step needs those probabilities as emission scores.
That is why log loss is reported next to Top-1.

**Logistic Regression.** Gives one probability per candidate from a weighted sum, so it is fast
(3.7 s to fit on 1.87M rows, 0.016 s to score all 209k student_test candidates) and its weights can be read.
Fitted standardised weights on the full train set: `dist_minus_min` -2.87, `dist_rank` -2.86, `heading_diff` -1.36,
`conn_prev` -0.33, `arc_len` +0.22. So it mostly says: pick the nearest arc whose heading agrees with the vehicle.
Good as a transparent baseline and a sanity check on the features.

**HistGradientBoosting.** Trees can combine features (for example distance matters differently
depending on heading and on how many candidates compete), and it handles the skewed feature scales without scaling.
Permutation importance on student_test (log-loss increase when a feature is shuffled): `heading_diff` 0.183,
`dist_minus_min` 0.132, `proj_frac` 0.052, `dist_to_arc` 0.010, `speed` 0.007. For LR the same test shows
`dist_minus_min` 0.134 and `heading_diff` 0.091. HGB uses `proj_frac` (position along the arc), which LR cannot use well.

## B. Hyperparameters, initial values and loss

| | Logistic Regression | HistGradientBoosting |
|---|---|---|
| Set by us | `max_iter=500`, `class_weight="balanced"` | `max_iter=150`, `learning_rate=0.1`, `l2_regularization=1.0` |
| Library defaults left as is (sklearn 1.7.2) | `C=1.0`, L2 penalty, `lbfgs`, `tol=1e-4` | `max_leaf_nodes=31`, `min_samples_leaf=20`, `max_depth=None`, `max_bins=255`, `early_stopping="auto"`, `n_iter_no_change=10`, `validation_fraction=0.1` |
| Loss | Binary log loss (cross-entropy) with L2 penalty | Binary log loss (`loss="log_loss"`) |
| Preprocessing | Median imputer, StandardScaler | Median imputer, no scaling |
| Iterations used in the re-run | 24 of 500 (converged early) | 150 of 150 (hit the cap) |

All values are starting points, not tuned. HGB used all 150 rounds, so 150 is a cap and not a
tuned stopping point; tuning belongs to Review 2. `balanced` weights in LR shift probabilities
upward for the rare positive class, which may explain part of its higher log loss (0.23 vs 0.09).
That is a hypothesis; we did not test unweighted LR.

## C. Results and fit assessment

Student-test (10 trips, 19,160 observations, 27 candidate misses = 0.14%), official run:

| | Top-1 accuracy | Macro F1 | Candidate log loss |
|---|---|---|---|
| Logistic Regression | 0.8461 | 0.7178 | 0.2082 |
| HistGradientBoosting | 0.8552 | 0.7467 | 0.0721 |

CV over 80 train trips, all trips (11.16% candidate misses) / excluding trips 33 and 54 (1.39%):

| | Top-1 | Macro F1 | Log loss |
|---|---|---|---|
| LR | 0.7410 / 0.8291 | 0.7346 / 0.7381 | 0.2316 / 0.2219 |
| HGB | 0.7478 / 0.8370 | 0.7633 / 0.7672 | 0.0922 / 0.0870 |

Reading these without overclaiming:

- **Student-test is easier than CV for a known reason.** Top-1 counts a candidate miss as wrong. Train has 11.16% misses,
  student_test 0.14%. Among observations whose true arc is a candidate, Top-1 is LR 0.834 (CV) vs 0.847 (student_test)
  and HGB 0.842 vs 0.856. So the real gap is about 1.3-1.5 points, not the 10 points the headline numbers suggest. The cause of the miss-rate difference is still unexplained (A is checking).
- **Overfitting (evidence-limited).** Fit-set scores are in-sample: the model was scored on its own training rows with the same
  `observation_metrics` code and the same denominators as CV (all 80 train trips, 208,417 observations, candidate log loss over candidate rows),
  so the numbers are comparable in units. For LR, fit-set Top-1 0.741 and log loss 0.232 match CV (0.741, 0.232), so we see no clear overfit signal.
  Matching metrics do not prove there is none. HGB has a fit-vs-CV gap: log loss 0.074 vs 0.092 (student_test 0.072) and Top-1 0.769 vs 0.748.
  The gap is small, and we did not test it further (for example with a different number of rounds or leaf size).
- **Probability quality.** LR's candidate log loss is about 2.5 to 3 times HGB's in both CV and student_test, so its probabilities are weaker.
  We have not diagnosed why. Possible reasons are its linear form, the `balanced` class weights, or both; we did not test them separately.
  We do not call this underfitting. HGB's Top-1 is only 0.9 points higher on student_test, and Top-1 is limited by candidate misses.
- **Fold stability.** Excluding trips 33 and 54, per-fold Top-1 is LR 0.759-0.891 and HGB 0.765-0.897 (fold 4 is lowest in both).
  Including all trips, fold 1 drops to 0.597 (LR) and 0.603 (HGB) because 31.8% of its observations are candidate misses.
  On student_test, per-trip Top-1 ranges 0.680-0.954 (LR) and 0.703-0.947 (HGB); trip 00000077 is lowest for both.
  The model ranking is stable in direction (HGB ahead of LR in every CV fold on Top-1 and log loss) but the spread across trips is larger than the LR-HGB gap.
- **Comparison with B's models** (same tables, student_test Top-1): RF 0.8518, MLP 0.8497. These four are within about 1 point of each other.
  Treat the ranking as descriptive, not a decisive generalisation result.

### Caveat on Macro F1

`evaluate.py` computes Macro F1 over arc IDs as class labels. Arc IDs are row numbers inside each trip's own `.arcs` file,
so arc 12 in one trip is a different road from arc 12 in another, yet both count as one class. In student_test there are
2,697 distinct reference arc values but 2,890 distinct (trip, arc) pairs. Missing predictions use a `__NO_CANDIDATE__`
label that never appears as a true class. Macro averaging gives each arc ID the same weight, so rarely seen arcs dominate.
Use Macro F1 only as a rough secondary number. It should not be read as a per-road-type quality score, and Top-1 plus log loss are better for choosing between models.

### Reproducibility note

Retraining here (sklearn 1.7.2, other machine) gave student_test Top-1 0.8459 (LR) and 0.8565 (HGB), log loss 0.2084 and 0.0715,
a few thousandths different from the official run. The official CSVs above are the source of record.
Route continuity is not measured yet; that needs the graph-aware HMM/Viterbi decoder.
