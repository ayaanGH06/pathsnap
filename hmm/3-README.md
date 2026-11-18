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
- Truly unreachable candidate pairs: no jump rule committed here; recovery/fallback behaviour is deferred to C3/C4.

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
