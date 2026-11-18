# C9: classifier-only vs classifier + HMM (DEVELOPMENT RUN, interim scores)

**Status: not final.** Emissions come from an interim local reproduction of the untuned
HistGradientBoosting model in `models/evaluate.py` (same settings, features, seed 42 and
GroupKFold(5) folds as B's runs), because B's exported model (handoff H3) is not on main yet
(`models/artifacts/` only has `.gitkeep`). Re-run `hmm/experiment.py` with B's export before
quoting these numbers as final. Config: `hmm/hmm_config.json`, tuned on training trips only.

## What these numbers are
- **Tuning/development numbers, not independent held-out performance.** The 20 training trips were used to *select* decoder settings (out-of-fold scores keep the classifier honest, but the decoder choices were picked on these trips).
- **Student-test:** the frozen config was run once. Its errors were then inspected for diagnostics (reverse-twin share, trip 7). No decoder change was made after seeing them and none should be, so this stays a single development readout until the final protocol with B's export is agreed.
- Interim scores (see above): replace with B's export before anything is quoted as final.

## Artifact-based frozen run (supersedes the refit-based table for reporting)
Same frozen config, emissions from B's saved artifact (`results/hmm/final_student_test_artifact.json`): classifier-only 85.80%,
classifier + HMM 93.52% pointwise over all 19,160 observations (mean per-trip 87.4% to 93.9%); direction flips 1,308 to 9; segment ratio
1.85 to 1.04. Trip 7 is again worse with the HMM (96.0% to 89.1%). Definitions, provenance and leakage audit: `review_notes.md`.
The tables below come from the earlier same-configuration refit scores and are kept for the log.

## Setup
- Tuning: 20 of the 80 training trips (every 4th sorted id), out-of-fold scores, objective =
  pointwise accuracy excluding trips 33 and 54. Log: `tuning_log_hist_gradient_boosting.csv`
  (one-at-a-time search then a combined run then an extension; 18 runs). Student-test was not used.
- One student-test run with the frozen config: `final_student_test_REFIT_DEVELOPMENT_superseded.json` (superseded by `final_student_test_artifact.json`).
- Denominator: every GPS observation. No prediction counts as wrong. Student-test has 19,160
  observations, 27 whose true arc is not a candidate (0.14%); all have at least one candidate, so
  no observation was filled. Trips 33 and 54 are in train only.

## Student-test results (10 trips, 19,160 observations)
| Metric | Classifier only | Classifier + HMM |
|---|---|---|
| Pointwise accuracy (all obs) | 85.65% | 92.81% |
| Mean per-trip accuracy | 87.15% | 93.64% |
| Route precision / recall (unique arcs) | 0.810 / 0.968 | 0.959 / 0.989 |
| Arc changes between consecutive points | 5742 | 3226 |
| Changes to a non-adjacent arc, per 1,000 changes | 262 | 112 |
| Direction flips (change to reverse twin) | 1296 | 9 |
| Predicted arc segments / reference segments | 1.85 | 1.04 |

"Non-adjacent" means the new arc does not start where the old one ends. It also counts legitimate
changes after skipped points, so read it as a relative measure.
Classifier-only 85.65% agrees with B's HGB student-test Top-1 (85.52%, `results/models/`),
the small difference comes from refitting here.

Decoder events on student-test: 1 long-gap restart, 6 dead-end restarts, 0 penalised jumps
(forbid-and-restart policy), 25708 candidate rows pruned by the top-20 limit.

### Per trip
| Trip | Classifier | HMM | Change (pts) |
|---|---|---|---|
| 00000000 | 93.9% | 97.8% | +3.8 |
| 00000007 | 94.0% | 89.0% | -5.0 |
| 00000009 | 86.2% | 91.2% | +5.1 |
| 00000040 | 91.0% | 95.4% | +4.4 |
| 00000052 | 94.7% | 100.0% | +5.3 |
| 00000056 | 88.5% | 96.7% | +8.2 |
| 00000065 | 81.5% | 91.7% | +10.2 |
| 00000076 | 86.9% | 97.2% | +10.3 |
| 00000077 | 70.3% | 83.2% | +12.9 |
| 00000091 | 84.6% | 94.3% | +9.7 |

## Where the HMM helped
Removes scattered off-route picks (orange off-route spurs in the local plot for trip 77), cuts direction flips
from 1296 to 9 and fragmentation from 1.85x to 1.04x the
reference segment count. Biggest gains: trip 77 (+12.9 pts), 76 (+10.3), 65 (+10.2), 91 (+9.7).

## Where it did not help
- **Trip 00000007: HMM 89.0% vs classifier 94.0%.** All 149 HMM errors are the reverse twin of the
  correct arc: the decoder commits to the wrong direction for a stretch and stays there
  (local plot, not committed: raw coordinates; the overlap hides direction).
- Across student-test, 643 of 1,378 HMM errors (47%) are the reverse twin of the reference arc and
  244 more are other arcs of the reference route, i.e. the right road at the wrong place or direction.
  Few direction *flips* (9) with many twin *errors* means long consistent wrong-direction runs.
  Direction is the main remaining weakness; candidate ideas (not tested): stronger heading evidence
  in the emission, direction prior from the start of a segment.
- Trips with no candidates cannot be fixed by the decoder. Training trip 54: 16,519 of 19,313
  observations have no candidate within 50 m; trip 33 has 3,449 reference-arc misses of 5,844
  (`results/data/miss_rate_per_trip.csv`). Carry-forward fill (copy neighbouring prediction within
  30 s) filled 776 observations on the 20 tuning trips and only 69 were correct (about 9%), so it
  adds almost nothing; observations that stay unfilled count as wrong.

## Training-trip tuning (20 trips, out-of-fold) - development numbers
| Accuracy | Classifier only | HMM default settings | HMM tuned |
|---|---|---|---|
| All 20 trips (includes trip 54 with 16,519 of 19,313 points lacking candidates) | 61.3% | 67.6% | 68.5% |
| Excluding trips 33 and 54 (tuning objective) | 84.1% | 92.7% | 93.8% |

Why the objective excludes 33 and 54: of the 18,230 training observations with no candidate, 16,564 (91%) belong to these two trips (`results/data/per_trip_coverage_train.csv`). On the all-trip number one trip decides the result and decoder settings cannot change it, so selecting on it would mostly measure candidate coverage. This is a disclosed choice: all-trip numbers are reported above and in the log (`acc` column), nothing is removed from any denominator, and the fill/abstain choice is logged separately.
Biggest effects (excluding 33/54, tuning trips): forbidding unreachable moves with restart (+0.6 pts vs penalty 20), emission weight 0.5
(+0.7), U-turn penalty 15 (+0.2, direction flips 53 to 20). Settings at the edge of the grid were
extended once (stage 3); no further gain, so the search stopped. Smoothing choices are validated only
on these 20 trips.
