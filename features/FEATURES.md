# Frozen features v1

Status: **v1 frozen**, 50 m candidate radius. This dictionary describes the
existing training code in `features/build_tables.py`, not a new feature version.
No feature, radius, label or model was changed by the A12 inference handoff.

Each row is one `(trip_id, obs_idx, arc_id)` candidate. Trip ids identify their
own road network; arc ids are zero-based row indices within that trip only.
Candidates are sorted by observation and then ascending distance. Equal-distance
ties keep the builder's existing order; inference does not reorder them.

## Model columns, in required order

| Column | Stored dtype | Definition | Reason |
|---|---|---|---|
| `dist_to_arc` | float32 | Exact point-to-segment distance in metres, in the trip's UTM projection | Nearby roads are more plausible |
| `proj_frac` | float32 | Position of the perpendicular projection along the directed segment, clipped to [0,1] | Distinguishes endpoints from segment interiors |
| `arc_len` | float32 | Segment length in metres; squared length is floored at 1e-12 before square root | Describes local road geometry |
| `heading_diff` | float32 | Absolute wrapped difference [0,180] degrees between arc bearing and GPS movement bearing | Distinguishes opposite road directions |
| `speed` | float32 | Distance from previous GPS point divided by its real timestamp gap, with denominator floor 1e-6 seconds; first observation copies first step; singleton is 0 | Motion context |
| `dt_prev` | float32 | Actual previous timestamp gap in seconds; first copies first step; singleton is 1 | Makes irregular sampling visible |
| `dist_rank` | int16 | One-based distance rank among this observation's candidates | Relative proximity |
| `dist_minus_min` | float32 | Distance minus this observation's minimum candidate distance, in metres | Relative separation from best geometric candidate |
| `n_cand` | int16 | Number of arcs within 50 m of this observation | Local ambiguity |
| `conn_prev` | int8 | 1 if this arc is also a previous-observation candidate, or starts at the end node of any previous-observation candidate; otherwise 0 | Directed one-hop continuity without reference labels |

Vehicle heading uses the displacement from the preceding to following GPS
observation (one-sided at the ends). If its length is under 1 m, heading difference
is set to the neutral value 90 degrees. Features therefore use the whole trip,
including the next GPS point, and are not a causal live-streaming API.

Distances use one UTM zone selected from the trip track centroid, with longitude
and latitude in their original order. The same projection and geometry helpers
are called during training and inference. No learned transforms are fitted here;
B's saved model contains its own preprocessing.

## Metadata and labels

- `trip_id`: string; `obs_idx` and `arc_id`: int32 in the candidate table.
- `label`: int8 in training/student-test tables only. It identifies the arc
  assigned by in-order snapping to the reference route. It is not a feature.
- Inference candidates have no `label` or `ref_arc`, and load no `.route` file.
- Zero-candidate observations have no candidate rows. Keep the full track and
  observation sidecar; never use the candidate rows alone as the denominator.
- Candidate counts/ranks preserve the original builder's dtypes. They are not
  recast or expanded by this handoff.

## Reproduction and version policy

Training tables: `python -m features.build_tables 50` with
`PATHSNAP_DATA_ROOT` pointing at the original dataset root.
Inference: see [PIPELINE.md](PIPELINE.md).

The train reference miss rate at 50 m is 11.16%; student-test is 0.14%.
A missing reference candidate cannot be recovered by a candidate-only classifier.
The 150 m and 200 m settings were not model-tested. Any radius/feature change
requires an explicit v2 decision, rebuilt tables and rerun models. Do not silently
remove train trips 33 or 54 or reinterpret their labels.

This file replaces an accidental copy of older diagnostic Python code. The
maintained diagnostic remains `features/diagnose_trips.py`; it was not changed.
