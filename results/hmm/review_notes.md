# HMM review notes (C), development run - for review before anything is published

## Status
Local, unpushed. All numbers are single development readouts, not independent held-out evidence for decoder
design: decoder settings were selected on 20 training trips (out-of-fold scores of a same-configuration refit),
then run once on student-test with the config frozen. Student-test errors were inspected afterwards for diagnostics
only; no setting changed after that.

## Metric definitions (hmm/experiment.py)
- **Pointwise accuracy**: observations whose predicted arc_id equals the reference arc_id, divided by ALL GPS observations
  of the partition. An observation with no prediction counts wrong. A filled prediction counts correct only if it matches.
- **Route precision / recall** (columns `route_prec_sum` / `route_rec_sum` in the json, pooled over trips): over the sets of unique
  directed arc ids, precision = |predicted ∩ reference| / |predicted|, recall = |predicted ∩ reference| / |reference|.
  This is SET overlap: it ignores order, repeats and where along the trip an arc was used. It is not sequence accuracy.
- **Segment ratio**: number of runs of consecutive identical arc ids in the predicted per-observation sequence divided by the
  same count for the reference sequence (1.0 = same fragmentation; higher = more switching). Missing predictions (-1) count as an arc value.
- **Direction flips**: consecutive observations where the prediction changes to the reverse twin of the previous arc (same
  nodes, opposite direction). **Non-adjacent changes per 1,000 changes**: changes where the new arc does not start at the previous
  arc's end node; also counts legitimate changes across skipped points, so only compare between methods.
- **Reverse-twin error**: predicted arc is the reverse twin of the reference arc for that observation.

## Denominator diagnostics
| Partition | Observations | No candidate | True arc not a candidate (A's miss) | True arc lost only to top-20 pruning | Filled | Candidate rows pruned |
|---|---|---|---|---|---|---|
| student_test (10 trips) | 19,160 | 0 | 27 (0.14%) | 0 | 0 | 25,708 rows |
| 20 tuning trips (train, dev) | 64,761 | 16,788 | 17,593 (27.2%) | 2 | 776 (69 correct) | 42,669 rows |
Full-track mode was used for every run (exact observation count and times). The four-argument mode cannot recover leading or
trailing candidate-less observations (see README "Decoder contract").

## Leakage audit
- `hmm/decoder.py` never reads `label`, `ref_arc` or the route file. Inputs: candidate rows (features, score), nodes, arcs, optional track.
- `hmm/experiment.py` and `viz/make_prediction_map.py` read the reference only AFTER decoding, to compute metrics or draw errors.
  `n_truth_lost_to_pruning` uses `label` for a diagnostic count only.
- `hmm/score_candidates.py` uses `label` only to fit the same-configuration refit on training folds; train scores are out-of-fold.
- Student-test was not used for decoder selection.

## Versions and provenance
- Scores: (a) out-of-fold train scores and interim student scores from `hmm/score_candidates.py` (models/evaluate.py roster,
  untuned HGB, seed 42, GroupKFold(5) over sorted train trip ids) under Python 3.10.12, scikit-learn 1.7.2, numpy 2.2.6,
  pandas 2.3.3; (b) student-test scores of B's artifact `models/artifacts/pathsnap_selected_model.joblib` computed in a separate
  Python 3.12.16 venv with scikit-learn 1.9.1 (the version in B's manifest; numpy 2.5.5, pandas 3.1.1), loaded with warnings
  as errors. Decoding and metrics ran under 3.10 (the decoder does not use scikit-learn).
- Artifact student-test Top-1 85.80% (this check) vs 85.52% in B's results/bayes/selected_hgb_student_test.csv, which comes from a
  separate refit in `export_selected_model.py`. Refits in two environments gave 85.65% and 85.64%. Model fits therefore vary by
  about 0.15-0.3 points between fits and environments; the cause (early stopping validation split, thread order, version) is not isolated.
- Candidate tables: rebuilt locally from the dataset archive with A's `features.build_tables` (v1, 50 m): 1,867,152 train rows, matching B's manifest.

## What the trip 7 map shows
`pred_map_00000007_zoom_worst_run` (local file, raw coordinates, not for commit) shows the longest run of wrong predictions is a dense cluster
of GPS points at one location away from the road centreline; the decoder chose the reverse-direction twin arc there. The vehicle may have been
stopped, but the speed feature is below 1 m/s for only 3 of the trip's observations, so this is not established. Not a claim about cause.

## Tests
`python -m pytest hmm -q`: 22 tests (9 Viterbi core, 13 decoder). Synthetic hand-built networks only; real-trip behaviour is covered by the experiments, not unit tests.
