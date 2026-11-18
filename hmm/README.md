# HMM design (C1)

## Hidden states
One directed candidate arc per GPS observation: the road segment (with travel direction) the vehicle is actually on. Candidates come from A's candidate-generation step; the decoder outputs one arc per observation, matching the playbook `decode()` contract.

## Why directed states
The task brief reports that about 92% of arcs have a reverse twin with identical geometry, so distance features alone cannot separate directions. Direction comes from heading features in the emission score and from direction-aware transitions that penalise implausible flips.

## Emission score
Baseline: Gaussian on perpendicular distance from GPS point to candidate arc, initial sigma 7.5 m (midpoint of the 5-10 m range the brief suggests from its reported noise measurements: median 4-5 m, p99 about 30 m). This is a starting value to tune in C7-C8. Scores are computed in log space with a small probability floor as numerical safeguards against underflow, not as part of the noise model.

Planned upgrade: use the classifier posterior (`predict_proba` from the shared harness in models/evaluate.py) as a practical candidate-quality score, without assuming it equals a generative likelihood. The C9 experiment decides whether it beats the baseline.

## Transition score
Heuristic starting preferences, to be validated, not fixed rules:
- Same-arc stays are usually plausible (the brief reports about 8 GPS points per route arc), without overriding travel-distance evidence.
- Reachable moves are scored by network (shortest-path) distance versus what is travelable in the actual elapsed dt, which handles irregular sampling such as trip 52's 2-second steps.
- Truly unreachable candidate pairs: see "Implemented transitions and recovery (C3/C4)" below.

## Decoding
Viterbi over the observation sequence picks the highest-scoring sequence instead of the best candidate per point, aiming to reduce "jumping between streets". Parameters live in `hmm/hmm_config.json`.

## Decisions log
| Date | Decision | By | Reason |
|---|---|---|---|
| 2026-11-18 | Hidden state = one directed candidate arc per observation | C (Ayaan) | Matches the playbook decode() contract |
| 2026-11-18 | Direction from heading features + direction-aware transitions | C (Ayaan) | Brief reports ~92% of arcs have identical-geometry reverse twins |
| 2026-11-18 | Emission baseline Gaussian, initial sigma 7.5 m, tuned later | C (Ayaan) | Brief's suggested 5-10 m range; log-space + floor are numerical safeguards |
| 2026-11-18 | Classifier posterior as practical emission score | C (Ayaan) | Uses features beyond distance; C9 decides vs baseline |
| 2026-11-18 | Transitions prefer same-arc / network-reachable moves as a heuristic, scored by network distance vs actual dt | C (Ayaan) | Brief reports ~8 points per arc; heuristic to be validated |
| 2026-11-18 | Unreachable pairs deferred to C3/C4 | C (Ayaan) | No jump/fallback rule before the transition model is designed |
| 2026-11-18 | C2 skeleton interface: per-observation candidate lists (variable counts, limit 20), -inf marks impossible moves, an unreachable step raises NoFeasiblePathError; gap recovery stays with C4 | C (Ayaan) | Matches the decode() contract without a network-sized matrix; fails loudly instead of guessing a policy |

## Implemented transitions and recovery (C3/C4) - `hmm/decoder.py`
- Transition log score = -|route distance - straight GPS distance| / beta_m, with route distance from a bounded Dijkstra on the directed network using along-arc positions (`proj_frac` x arc length) of both candidates. Same arc moving forward = distance along the arc; tiny backward movement (within back_tol_m) = 0; same arc via a loop is allowed; repeated arcs are never merged or removed. Reverse-twin moves get `uturn_penalty`.
- Zero-candidate observation: not a state. The chain bridges over it using the real elapsed time and the distance between the neighbouring observations. `decode_detailed` returns one slot per GPS point (-1 = no candidate and not filled). Optional `fill_missing: carry` copies the nearest decoded neighbour within `fill_max_gap_s`; a filled arc only counts correct if it equals the reference. Filling is weak evidence: 9% of filled observations were correct on the tuning trips.
- Long gap (> `gap_restart_s`): the chain restarts as an independent segment.
- No path within the search bound or implausible speed: `unreachable_penalty` (a number) allows the move with that penalty, or `null` forbids it and the chain restarts at the dead end. Tuned value: `null`.
- More than 20 candidates at a point (up to 148 occur): the decoder keeps the top 20 by classifier score and counts the pruned rows (`n_pruned_rows`); retained-truth loss is counted separately from candidate misses (2 observations on the 20 tuning trips).
- Distance-only emissions exist as `emission_mode: distance` (baseline for tests); the tuned config uses classifier probabilities.
- Interface: `decode(trip_candidates_with_scores, nodes, arcs, hmm_config, track=None)` returns arc_ids; pass `track` for exact times (the API has it).

## Decisions log (C3-C9, 2026-11-18)
| Decision | Why / evidence |
|---|---|
| Newson-Krumm style exponential on |route - GPS distance| | Standard map-matching form; beta tuned (5 m best of 2.5/5/20 on 20 train trips) |
| Bounded Dijkstra (cutoff min(45 m/s x dt + 60 m, 3000 m)) | Cost; trip 1 has 165,790 arcs. Cutoffs are our choices, not from the brief |
| Forbid unreachable moves and restart (penalty null) | Accuracy excl. 33/54: 93.3% vs 92.0% for penalty 8 on tuning trips |
| emission_weight 0.5 | 93.5% vs 92.7% at 1.0; 0.25 and 0.1 gave no gain |
| uturn_penalty 15 | Flips 53 -> 20; 30 gave no further gain |
| gap_restart_s 120 | 30/120 differed by 0.02 pts, within noise |
| Tuning on 20 train trips (every 4th), out-of-fold scores, no student-test | Playbook C8 allows student-test, but the earlier team rule keeps it for evaluation; frozen config run once on student-test |
| Tuning objective excludes trips 33 and 54 | They are 92% of tuning-set observations without candidates and would swamp the metric; all-trip numbers are still reported |
| Emissions from an interim HGB reproduction | B's export (H3) not on main; re-run `hmm/experiment.py` with B's export before quoting results as final |

## Decoder contract (H4) as implemented
`decode(trip_candidates_with_scores, nodes, arcs, hmm_config, track=None)` - the four contract arguments work as specified; `track` is an optional fifth keyword, so a caller that passes only the four still works.
- `trip_candidates_with_scores`: one trip's candidate rows from A's table (columns `obs_idx`, `arc_id`, `proj_frac`, `dt_prev`, `speed`, `dist_to_arc`) plus a `score` column with B's probability for that same row (keep row-to-arc correspondence).
- `hmm_config`: dict (see `hmm/hmm_config.json`; missing keys fall back to defaults; `load_config()` reads the file).
- **One slot per GPS observation needs the observation count.** The candidate table lists only observations that have candidates, so leading/trailing/all-zero-candidate observations cannot be recovered from it. Pass `track` (lon, lat, t for every observation: exact times, exact count, enables fill) or at least `n_obs`. With neither, the length is inferred as the last observation with candidates + 1 (`stats["n_obs_inferred"]` is True), trailing observations are absent and no fill happens, so the four-argument mode does NOT meet a one-per-GPS-point contract. Elapsed times without `track` are approximated from `dt_prev` and `speed`. On four student-test trips the modes gave identical predictions, but the API should pass `track`.
- Returns arc_ids in observation order for observations with a decoded or filled arc. `decode_detailed` returns one slot per GPS point (-1 = none) plus stats. Accuracy must be computed over every observation.

## Reproducing the artifact-based student-test run
1. A's tables: `data/processed/candidates_student_test.parquet` (built by A's `features.build_tables`, not tracked).
2. `python hmm/score_with_artifact.py artifact` (needs scikit-learn matching `models/artifacts/model_manifest.json`; no fitting, no label input).
3. `HMM_CACHE=/tmp/hmm_cache_art python -m hmm.experiment cache student_test artifact` then
   `HMM_CACHE=/tmp/hmm_cache_art python -m hmm.experiment final artifact hmm/hmm_config.json` writes `results/hmm/final_student_test_artifact.json`.
4. Maps: `python viz/make_prediction_map.py <trip> artifact <out_dir> --png` (outputs contain raw coordinates; do not commit).
