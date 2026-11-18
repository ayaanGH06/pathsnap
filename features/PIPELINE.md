# A12 / H5: server-side inference handoff

`features.pipeline.prepare_trip(trip_id, partition='student_test')` returns a
`TripFeatures` object. `PATHSNAP_DATA_ROOT` selects the folder containing `train/`
and `student_test/`; no local path is hardcoded. Only these two partitions are
supported by the existing loader. Evaluator support requires the team's explicit
organiser-approved loader change, not renaming or inspecting evaluator files now.

If the API already loaded the trip with `with_route=False`, use
`prepare_trip_frames(track, nodes, arcs, trip_id=trip_id, partition=partition)`.
This takes the same label-free feature path without loading the trip twice.

## Returned object

- `candidates`: `trip_id`, `obs_idx`, `arc_id`, then the ten v1 features, in the
  unchanged training-builder row order and dtypes. No label or reference column.
- `model_features`: only the ten ordered model columns.
- `track`: every original GPS observation, with `lon, lat, t`; no filtering.
- `observations`: `trip_id, obs_idx, n_cand`, one row for every GPS observation.
- `nodes`, `arcs`: the original directed network frames.
- `n_obs`, `feature_columns`, `feature_version='v1'`, `radius_m=50.0`.

These are server objects, not a UI payload. Reference routes are never loaded.
The per-trip projection and feature code are shared with training; this wrapper
fits nothing and changes neither features nor candidate radius.

## D's scoring / decoding adapter

The C decoder must be present before this snippet can run. Load the artifact and
config once at server startup under B's compatible Python/scikit-learn environment.

```python
import json
from pathlib import Path
import joblib
from features.pipeline import prepare_trip
# Import decode from the delivered C package before running this example.

manifest = json.loads(Path('models/artifacts/model_manifest.json').read_text())
model = joblib.load('models/artifacts/pathsnap_selected_model.joblib')
cfg = json.loads(Path('hmm/hmm_config.json').read_text())

trip = prepare_trip('00000007', 'student_test')
assert list(trip.feature_columns) == manifest['feature_columns_in_required_order']
candidates = trip.candidates.copy()
# No predict_proba call on an empty candidate table.
positive = list(model.classes_).index(1)
candidates['score'] = (model.predict_proba(trip.model_features)[:, positive]
                       if len(candidates) else [])
predictions = decode(candidates, trip.nodes, trip.arcs, cfg, track=trip.track)
assert len(predictions) == trip.n_obs
```

A12 does not import C; the integration layer supplies its decoder. Retain the unmodified per-observation output, including
`-1` unmatched. If a display route is needed, derive it separately; never use a
shortened route list to compute observation accuracy. All observations, including
zero-candidate ones, belong in the denominator.

Only **after predictions are complete** may the server separately load its route
and compute reference labels/metrics. Never pass labels to the model, decoder or
UI. Do not return the reference route as the prediction. The two-laptop client
needs only the API address, not the dataset or reference files.

## Checks

```sh
python -m unittest features.test_pipeline -v
python -m features.verify_pipeline
```

The second command needs the dataset root. It guards against `.route` opens,
compares exact values/dtypes/order with the existing builder, and additionally
compares existing `data/processed/candidates_*.parquet` when present. To rebuild
those tables for the parity check, run `python -m features.build_tables 50`
separately; that offline training-table command reads labels by design.

Verified on the original source archive: 80 train trips, 208,417 observations,
1,867,152 candidate rows; 10 student-test trips, 19,160 observations, 209,121 rows.
The full-track output preserves all 18,230 train observations with zero candidates.
Student-test has zero zero-candidate observations (27 have candidates but no
reference match). Full-data summary: `results/data/pipeline_check.json`.
