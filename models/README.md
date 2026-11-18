# Model comparison (Person B)

`evaluate.py` compares candidate-arc classifiers. Each row is one `(trip_id, obs_idx, arc_id)` candidate. Numeric feature columns are used to fit a binary classifier (`label=1` for the reference arc, `label=0` otherwise). For each observation, the candidate with the highest positive-class probability is the classifier-only prediction.

## Baseline roster

| Model key | Family | Starting configuration |
|---|---|---|
| `logistic_regression` | Linear | 500 iterations, balanced class weights |
| `knn_distance_weighted` | Instance based | 15 neighbours, distance weights |
| `random_forest` | Tree ensemble | 300 trees, leaf size 2 |
| `hist_gradient_boosting` | Boosting | 150 iterations, learning rate 0.1 |
| `mlp` | Neural network | 64/32 hidden units, early stopping |

These are initial candidates, not selected winners. Confirm the exact boosting implementation with the teammate assigned to boosting. Current team split: C runs Logistic Regression and HistGradientBoosting; D runs KNN; B runs Random Forest and the MLP. Confirm that C’s boosting model is HistGradientBoosting before running it.

## Dummy-table smoke check

The generator writes artificial tables under `/tmp` by default. They are only for checking that the evaluator runs; never include their scores in the report or project results.

```bash
python models/make_demo_data.py
python models/evaluate.py \
  --train /tmp/pathsnap-demo-candidates/candidates_train_demo.csv \
  --student-test /tmp/pathsnap-demo-candidates/candidates_student_demo.csv \
  --folds 5 \
  --models random_forest mlp \
  --results-dir /tmp/pathsnap-demo-results
```

## Real candidate-table runs

Run from the repository root after A delivers matching train and student-test tables:

```bash
python models/evaluate.py \
  --train data/processed/candidates_train.parquet \
  --student-test data/processed/candidates_student_test.parquet \
  --folds 5 \
  --models random_forest \
  --results-dir results/models
```

Each invocation creates a uniquely named run folder, so different members can run their assigned models without overwriting one another. Use exactly the same candidate-table version and fold count for all model runs. The shared `config.RANDOM_SEED` (42) is used by model constructors. `GroupKFold` keeps all rows from a trip in one fold.

## Metrics and interpretation

- Top-1 accuracy chooses the highest-scored candidate per observation. A candidate miss counts as incorrect.
- Candidate miss rate counts observations with no positive candidate label.
- Arc-level Macro F1, precision, and recall are computed on observations with a known positive candidate; misses are excluded because their true arc is absent from the candidate rows. Report the miss rate alongside these metrics.
- Log Loss is binary candidate-level log loss, not route-level log loss.
- Route continuity is reported as N/A here; agree its definition and calculate it with C's graph-aware decoder evaluation.

Do not select a final model from training accuracy. Compare CV and student-test results, stability, probability quality, inference time, and compatibility with the decoder. Log actual runs only; do not invent report results.
