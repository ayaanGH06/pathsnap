# Model notes: Random Forest, MLP, and KNN (Person B)

## Shared experiment setup

All three models used A's 50 m candidate tables, the same ten frozen v1 features,
seed 42, and five-fold GroupKFold with whole trips held out. Top-1 accuracy is
computed over every observation-sidecar row; if the correct road is absent from
the candidate list, that observation is wrong. The second CV slice excludes
trips `00000033` and `00000054` only as a diagnostic, while the all-trip number
remains the full-data result. Student test contains 10 trips and is not used for
hyperparameter selection.

Official run files are under `results/models/` in the corresponding dated run
folders. The CSV values below are fractions converted to percentages here.

## Results

| Model | CV Top-1, all trips | CV Top-1, excluding 33/54 | CV macro F1, all trips | Student-test Top-1 | Student-test macro F1 | Candidate log loss (CV / test) |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 74.62% | 83.52% | 76.05% | 85.18% | 74.19% | 0.1121 / 0.0815 |
| MLP | 74.54% | 83.41% | 75.61% | 84.97% | 73.07% | 0.0951 / 0.0759 |
| KNN, distance weighted | 73.51% | 82.21% | 72.47% | 84.43% | 70.77% | 0.2977 / 0.2189 |

The candidate-miss rate is 11.16% across the training partition and 0.14% on
student test. These are misses where the reference arc is not in the candidate
set; they do not necessarily mean the observation has no candidate roads at
all. With the current 50 m candidate pool, the theoretical ceiling for the
all-training-observation Top-1 score is 88.84%, even if every observation with
its correct arc available were ranked perfectly. A's radius analysis reports
miss rates of 4.46% at 150 m and 1.19% at 200 m, but these radii need a new
end-to-end evaluation before anyone can claim a higher accuracy.

## Random Forest

The pipeline median-imputes features and fits 300 trees with
`class_weight="balanced_subsample"` and `min_samples_leaf=2`. Each tree sees
resampled feature/row choices; averaging the trees reduces variance and learns
nonlinear interactions. Scaling is not needed for tree splits. Its CV Top-1 is
nearly tied with the baseline HistGradientBoosting result. Full-train fit took
about 281 seconds and student-test inference about 1.22 seconds, making it the
slowest fit among B's three but much faster at inference than KNN.

## MLP

The pipeline median-imputes and standardizes features, then fits a feed-forward
network with hidden layers of 64 and 32 units, up to 300 iterations, and early
stopping. Scaling matters because gradient-based neural-network training is
sensitive to feature magnitudes. MLP reached nearly the same Top-1 as Random
Forest but had slightly lower macro F1 and higher candidate log loss. Full-train
fit took about 46.5 seconds; student-test inference took about 0.038 seconds.

## KNN

The pipeline median-imputes and standardizes features, then uses the 15 nearest
training candidate rows with distance weighting. Nearby examples have greater
influence on the candidate's positive-class probability. KNN fit took about
1.9 seconds, but prediction on student test took about 50.5 seconds because it
must compare against stored training examples. In these runs it had the lowest
Top-1 and macro F1 of the five baseline models and the weakest candidate
log-loss, so its speed advantage during fitting did not translate to inference.

## Interpretation and limits

- The all-trip CV numbers for these models are around 73.5–74.6%; excluding the
  two known hard trips raises them to 82.2–83.5%. Always show both slices.
- Student-test Top-1 is around 84.4–85.2%, but the test has only 10 trips and a
  much lower candidate-miss rate. Treat it as supporting evidence, not a stable
  estimate of future performance.
- Macro F1 is secondary: `arc_id` values are local to each trip's road network,
  so the same numeric ID across trips does not represent the same road. The
  metric is not a clean measure of road-type balance.
- The classifier scores candidates independently for each GPS point. Route
  continuity is not measured here; C's graph-aware HMM/Viterbi decoder is
  responsible for sequence-level path selection.
- Bayesian tuning uses training-trip CV only. It must not use student-test
  metrics to choose hyperparameters.
