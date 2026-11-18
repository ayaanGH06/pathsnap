# Data processing notes (Member A)

All numbers here come from the CSVs in `results/data/`, produced by
`data_processing/inventory.py` (A1-A3) and `data_processing/reference.py` (A4).
Only the `train` (80 trips) and `student_test` (10 trips) partitions were used.
Evaluator data was never opened.

## 1. Dataset inventory (A1) -> results/data/inventory.csv
- Trips analysed: 80 train + 10 student_test = 90.
- Files per trip: 4 (`<id>.track`, `<id>.nodes`, `<id>.arcs`, `<id>.route`).
- Trip ids are zero-padded strings (e.g. `00000001`).
- GPS observations per trip: min 262, median 1,496.5, max 19,313 (all 90 trips).
- Examples: trip 00000001 has 1,315 observations, 88,809 nodes, 165,790 arcs
  and a 525-arc reference route. Trip 00000004 has 1,064 observations,
  7,350 nodes, 15,499 arcs and a 320-arc route.
- Each trip has its own road network, so `node_id` and `arc_id` are only
  valid within one trip.

## 2. Data quality (A2) -> results/data/quality.csv
- Missing/NaN values: 0 across all 90 trips.
- Timestamp ordering: 0 unsorted trips, 0 duplicate timestamps.
- Irregular sampling: 66 of 90 trips have at least one gap longer than 1 s.
- Share of exactly 1 s steps per trip: median 99.14 %, mean 94.32 %,
  lowest 53.07 % (train trip 00000022).
- Largest gap: 153,064 s (about 42 hours) in train trip 00000090. The mean of the
  per-trip maximum gap is 2,408 s, so a few trips have extreme gaps.
- Step length between consecutive points: median of per-trip medians 13.18 m,
  highest per-trip median 41.27 m (trip 00000001, which looks like a
  motorway trip).
- Coordinate order: files are (longitude, latitude). Confirmed from the
  values (e.g. 8.595, 49.844 for trip 00000001).
- Trips lie in different regions (trips 1-4 are in four different areas),
  so a single projection cannot suit all trips.

### Decisions on quality
- No observations are removed. Every observation needs a prediction.
- Speed and distance features use the real time difference `dt`, and `dt`
  is kept as a feature so models can see large gaps.
- The HMM (Member C) must handle large gaps.

## 3. Road network (A3) -> results/data/network.csv
- Arcs are directed. The share of arcs that have a reverse arc is
  median 95.11 %, minimum 35.10 %, maximum 100 %.
- Isolated nodes: 0 trips with any isolated node (all 90 trips).
- Connected components per trip (undirected view): min 1, median 19,
  mean 37.2, max 421.
- Consequence: some arcs are unreachable from others, so connectivity
  features must handle disconnected candidates.

## 4. Projection (A5)
- Decision: use the UTM zone computed from each trip's own track centroid
  (EPSG 326xx for the northern hemisphere, 327xx for the southern).
- Reason: trips lie in different countries, so one fixed CRS would distort
  distances. UTM gives metres, and all distances are computed in metres,
  never in degrees.

## 5. Reference route (A4) -> results/data/reference_check.csv
- Command: `python -m data_processing.reference all` (the committed reference check covers all 80 train trips)
- Route arcs form a connected chain: 100 % of consecutive arcs (median across 80 train trips).
- Distance from GPS points to the nearest route arc: median of trip medians 4.5 m; median of trip-level
  95th percentiles 14.25 m (80 train trips, not pooled percentiles).
- Median per-trip monotone-snap agreement with nearest-arc is 97.5 % (80 train trips, not pooled).
- Labelling rule used for the frozen v1 tables: each observation is assigned
  the route arc found by an in-order (monotone) snap to the route.
- Limitations: the reference route is not perfect truth. GPS has no direct
  row link to the route. Gaps mean some route arcs receive no observation.
  Labels near junctions can be ambiguous.

## 6. Candidate generation (A7)
- Method (`features/candidates.py`): for every GPS observation, all arcs whose point-to-segment distance is at most the radius are candidates. A spatial index (shapely STRtree) pre-filters by bounding box, then the exact distance is computed in metres in the per-trip UTM projection.
- Candidate miss rate = share of observations whose reference arc is not in the candidate set. Measured on the `train` partition (80 trips) with `data_processing/miss_rate.py`:

| Radius | 25 m | 50 m | 100 m | 150 m | 200 m |
|---|---|---|---|---|---|
| Miss rate (train) | 13.2 % | 11.2 % | 9.9 % | 4.5 % | 1.2 % |

- Chosen radius: **50 m (v1, provisional)**. A larger radius lowers the miss rate but gives every observation more candidates to rank. With 50 m the table has 1,867,152 rows for 190,187 observations that have at least one candidate (about 9.8 candidates per observation, computed from `results/data/tables_run.json` and `label_check_train.json`).
- Larger radii (150 m, 200 m) were **not** tested with the models. Changing the radius after the feature freeze is a v2 change: version tag in `features/FEATURES.md`, regenerate the tables, re-run every model.
- Observations with no arc inside the radius have **no rows** in the candidate table. They are listed in `data/processed/obs_<partition>.parquet` (`n_cand = 0`) and counted as misses everywhere.

## 7. Decisions log
| Decision | By | Reason |
|---|---|---|
| Keep all observations, including those after large gaps | Member A | Every observation needs a prediction |
| Per-trip UTM projection | Member A | Trips are in different countries |
| Speed uses real dt, dt kept as a feature | Member A | 66 of 90 trips have irregular gaps |
| Label = in-order (monotone) snap of each observation to the reference route | Member A | Keeps labels consistent with the route order. Checked on trips 54, 33 and 2: the distance to the nearest arc anywhere on the route matches the distance to the labelled arc (0.0 % of observations differ by more than 50 m for trips 33 and 2, 1.6 % for trip 54), so the snap is not the cause of the far labels |
| Radius 50 m for v1 | Member A | See section 6. The 11.16 % train miss rate is mostly trips 54 and 33. Larger radii untested |
| Features v1 frozen | Member A | See `features/FEATURES.md`. No learned transforms at this stage |
| Restored a damaged local copy of `student_test/00000000/00000000.arcs` | Member A | Line 1 read `q0 2433` instead of `0 2433` (stray character). Checked against the original `Student_folder.zip`; no values were changed. In the original zip every `.arcs` line has two integers and every `.route` line has one integer |

## 8. Candidate tables and checks (A8-A9)
- Build: `python -m features.build_tables 50` (seed 42, radius 50 m, 10 features) writes `data/processed/candidates_<partition>.parquet` and `obs_<partition>.parquet`, and logs `results/data/tables_run.json`. Parquet files are not committed.

| Partition | Trips | Observations | Candidate rows | Miss rate |
|---|---|---|---|---|
| train | 80 | 208,417 | 1,867,152 | 11.16 % |
| student_test | 10 | 19,160 | 209,121 | 0.14 % |

- Label check (`features/check_labels.py`, results in `results/data/label_check_*.json`): PASS on both partitions. No duplicate (trip, observation, arc) rows, labels are only 0 or 1, and at most one `label = 1` per observation.
- Per-trip coverage (`features/per_trip_coverage.py`, `results/data/per_trip_coverage_*.csv`): train has 18,230 observations with no candidate in 50 m plus 5,038 with candidates but no `label = 1`, which gives 23,268 misses and matches `tables_run.json`. Student-test has 0 and 27. 57 of 80 train trips have zero misses. All 27 student-test misses are in trip 56.

## 9. Trips with far reference routes (train only)
Diagnostic: `features/diagnose_trips.py`, results in `results/data/trip_diagnostic_train.json`, plots in `results/data/figures/` (each titled with its partition).

| Trip | Observations | Share >50 m from labelled arc | Median distance to nearest arc in network | Median distance to labelled arc |
|---|---|---|---|---|
| 54 | 19,313 | 89.4 % | 84.2 m | 140.1 m |
| 33 | 5,844 | 59.0 % | 7.0 m | 122.4 m |
| 2 | 7,243 | 12.0 % | 5.4 m | 5.8 m |

- Trip 54: many GPS observations are far from every road in its network (median nearest-road distance 84.2 m, maximum 121.9 m). The root cause is not established.
- Trip 33: the GPS is on a road, but the reference route is far from it for much of the trip. In 50 windows of 60 consecutive far observations, the median GPS path length is 451.6 m and the median net displacement is 8.1 m, which indicates GPS jitter at a stop. Why the reference route does not cover that location is not established.
- Trip 2: 41 time gaps over 10 s (largest 160 s); 76 % of the last tenth of the trip is far from the reference.
- Because the reference route itself is far from the GPS in these observations, a model based on geometry features is unlikely to predict the reference arc there. Pooled accuracy is strongly affected by these trips (trip 54 alone is 19,313 of the 208,417 train observations), so the team should decide whether to also report results without them.

## 10. Inference and EDA report handoff (A12-A13)

- Label-free frozen-v1 pipeline: `features/pipeline.py`; API and checks are documented in `features/PIPELINE.md`.
- Report section source: `data_processing/eda_report.md`; aggregate figure: `results/data/figures/eda_summary.png`.
- `features/FEATURES.md` is the ten-column feature dictionary. No feature or radius change was made.
