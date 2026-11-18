# HMM section (C11) - DRAFT, numbers are from interim scores and must be re-run with B's export

**Hidden states.** One directed candidate arc per GPS observation, from A's candidate table (50 m radius). Directed because about 92% of arcs have a reverse twin with the same geometry. At most 20 candidates per point are kept (top 20 by classifier score; up to 148 exist).

**Emissions.** Log of the classifier probability for each candidate, scaled by 0.5 (tuned), floor 1e-6. A Gaussian-on-distance baseline exists for tests. Current numbers use an interim reproduction of the untuned HistGradientBoosting model; B's export replaces it.

**Transitions.** -|network route distance - straight GPS distance| / 5 m. Route distance uses the along-arc position of each candidate and a bounded shortest-path search on the directed network, using the real time gap. Same-arc stays cost the along-arc distance, repeated arcs are allowed, reverse-twin moves get a penalty of 15.

**Missing candidates and large gaps.** An observation with no candidate is bridged (time and distance span the gap) and shown as missing in the output, never dropped; it counts as wrong in accuracy unless the optional fill produces the reference arc (9% of filled points were right on the tuning trips). A gap over 120 s restarts the chain. Trip 54 (16,519 of 19,313 points without candidates) shows the limit: the decoder cannot recover roads it was never offered.

**Disconnected segments.** If no path exists within the search bound, or it would need more than 45 m/s, the move is forbidden and the chain restarts at that step (tuned over a penalty of 8, 20 or 40, which all scored lower). On student-test this happened 6 times in 19,160 observations, plus 1 long-gap restart.

**Did the HMM improve the route? (development run)** Student-test pointwise accuracy 85.65% to 92.81%, direction flips 1,296 to 9, segment count 1.85x to 1.04x the reference. Improved on 9 of 10 trips. It did not help on trip 7 (94.0% to 89.0%): every error was the reverse twin of the correct arc. Across student-test 47% of the remaining errors are reverse twins, so direction is the main weakness. Details and per-trip table: `results/hmm/c9_comparison.md`. Maps are generated locally by `viz/make_prediction_map.py` (they contain raw coordinates, so they are not committed); direction errors need an arrow or colour-by-direction view, the current plots do not show them.

**Caveats.** Tuned on 20 training trips only. Interim scores. Trips 33 and 54 (train) have candidate-coverage problems from candidate generation, not from the decoder.
