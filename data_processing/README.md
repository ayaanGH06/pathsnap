# Data processing notes (Member A)

All numbers here come from the CSVs in `results/data/`, produced by
`data_processing/inventory.py` (A1-A3) and `data_processing/reference.py` (A4).
Only the `train` (80 trips) and `student_test` (10 trips) partitions were used.
Evaluator data was never opened.

## 1. Dataset inventory (A1) -> results/data/inventory.csv
- Trips analysed: 80 train + 10 student_test = 90.
- Files per trip: 4 (`<id>.track`, `<id>.nodes`, `<id>.arcs`, `<id>.route`).
- Trip ids are zero-padded strings (e.g. `00000001`).
- GPS observations per trip: min ___, median ___, max ___.
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
  lowest 53.07 % (trip ___).
- Largest gap: 153,064 s (about 42 hours) in trip ___. The mean of the
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
- Isolated nodes: ___ (count of trips with any isolated node).
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
- Command: `python -m data_processing.reference 5`
- Route arcs form a connected chain: ___ % of consecutive arcs (median).
- Distance from GPS points to the nearest route arc: median ___ m,
  95th percentile ___ m.
- Monotone snap agrees with nearest-arc for ___ % of observations.
- Labelling rule (to confirm after the check): each observation is assigned
  the route arc found by an in-order (monotone) snap to the route.
- Limitations: the reference route is not perfect truth. GPS has no direct
  row link to the route. Gaps mean some route arcs receive no observation.
  Labels near junctions can be ambiguous.

## 6. Candidate generation (A7)
- (fill in after running: radius tested, candidate miss rate per radius)

## 7. Decisions log
| Decision | By | Reason |
|---|---|---|
| Keep all observations, including those after large gaps | Member A | Every observation needs a prediction |
| Per-trip UTM projection | Member A | Trips are in different countries |
| Speed uses real dt, dt kept as a feature | Member A | 66 of 90 trips have irregular gaps |