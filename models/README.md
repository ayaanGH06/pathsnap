# Model comparison (Person B)

`evaluate.py` compares candidate-arc classifiers. Candidate tables have one row per `(trip_id, obs_idx, arc_id)`, numeric feature columns, and a binary `label`. Observation sidecars have one row per observation and must include `trip_id`, `obs_idx`, `ref_arc`, and `in_cand`.

For each GPS observation, the model scores the candidate arcs and the highest-scoring arc is the classifier-only prediction. The observation sidecar is the scoring denominator, so a missing true candidate remains an incorrect Top-1 prediction instead of disappearing from evaluation. Arc IDs are normalized to nullable integers before comparison, preventing values such as `123` and `123.0` from comparing unequal as strings.

## Baseline roster

| Model key | Family | Starting configuration |
|---|---|---|
| `logistic_regression` | Linear | 500 iterations, balanced class weights |
| `knn_distance_weighted` | Instance based | 15 neighbours, distance weights |
| `random_forest` | Tree ensemble | 300 trees, leaf size 2 |
| `hist_gradient_boosting` | Boosting | 150 iterations, learning rate 0.1 |
| `mlp` | Neural network | 64/32 hidden units, early stopping |

Current team split: C runs Logistic Regression and HistGradientBoosting; D runs KNN; B runs Random Forest and MLP.

## Synthetic smoke check

These generated tables are artificial and their scores must never be used as project results. They include candidate misses and trips 33/54 to check denominator and reporting behavior.

```bash
python models/make_demo_data.py
python models/evaluate.py \
  --train /tmp/pathsnap-demo-candidates/candidates_train_demo.csv \
  --student-test /tmp/pathsnap-demo-candidates/candidates_student_demo.csv \
  --train-obs /tmp/pathsnap-demo-candidates/obs_train_demo.csv \
  --student-test-obs /tmp/pathsnap-demo-candidates/obs_student_demo.csv \
  --folds 5 --models random_forest mlp \
  --results-dir /tmp/pathsnap-demo-results
```

## Real candidate-table runs

Run from the repository root after A completes the build:

```bash
python models/evaluate.py \
  --train data/processed/candidates_train.parquet \
  --student-test data/processed/candidates_student_test.parquet \
  --train-obs data/processed/obs_train.parquet \
  --student-test-obs data/processed/obs_student_test.parquet \
  --folds 5 --models random_forest mlp \
  --results-dir results/models
```

All models must use the same candidate-table version and fold count. Grouped CV keeps trips together. Each run writes to a unique subfolder with a CV summary, per-fold timing/results, student-test results, and run configuration.

## Metric interpretation

- Top-1 accuracy divides correct predictions by all rows in the observation sidecar. A candidate miss counts as incorrect.
- Candidate miss rate is computed from `in_cand` in the sidecar.
- Macro F1, precision, and recall include every observation; missing predictions use a no-candidate sentinel and are not dropped.
- Candidate Log Loss is binary loss over candidate rows, not route-level loss.
- CV and student-test summaries include both all trips and a separate slice excluding trips `00000033` and `00000054`.
- Route continuity is N/A here; C measures that with the graph-aware HMM/Viterbi decoder.

The five models are starting candidates, not selected winners. Choose the final model using actual CV and student-test results, fold stability, probability quality, inference time, and decoder compatibility. Do not treat synthetic smoke-check scores as experiment results.

## Person B: tuning, comparison, and export

The Bayesian search tunes HistGradientBoosting on trip-grouped training CV only;
it does not load student-test data. Its primary objective is all-trip,
observation-level Top-1, with trips 33 and 54 retained in the main objective.
The outlier-excluded slice is diagnostic. Search history, sampled
configurations, and the selected configuration are saved in `results/bayes/`.

```bash
python -m bayes_opt.optimize_hgb
python -m bayes_opt.finalize_hgb
python models/export_selected_model.py
python models/build_model_comparison.py
```

The selected `Pipeline` (including median imputation) and its handoff contract
are saved under `models/artifacts/`. See `models/model_notes.md` for B's
Random Forest/MLP/KNN evidence and `models/model_selection.md` for the primary
metric, comparison, and selection rationale. The held-out student-test result
is evaluated after the hyperparameters are frozen; it is not part of the search.
