# Section 7: System Integration and Evaluation

## Implemented system functionality

The Person D contribution provides a FastAPI foundation with health checking,
partition-aware trip listing and prediction request validation. The API loads
student-test or training trips through the existing shared loader with
`with_route=False`, so reference route labels are not loaded or exposed.

The frontend provides trip selection, prediction controls, loading, success
and error states, predicted-arc rendering, GPS observations and an optional
server metrics display. It uses no synthetic routes or invented performance
numbers.

The system supports two-laptop deployment through an environment-configurable
backend URL and CORS origin list.

## Prediction path (integrated)

`POST /predict` loads the trip without its reference route, builds the frozen v1 features (`features/pipeline.py`, 50 m),
scores every candidate with the saved HistGradientBoosting artifact (`models/artifacts/`), and decodes with the HMM
(`hmm/decoder.py`, `decode_detailed`) over the full GPS track, so every observation gets an arc id or -1. Only after the
prediction does the server read the reference route to compute accuracy; the route is never sent to the model or the UI.
The artifact and decoder configuration are loaded once at first request.

## Results (all ten student-test trips through the API)

Source: `results/api/api_check_student_test.json`, produced by `scripts/check_all_trips.py`. Pointwise accuracy against
the project reference-label rule, pooled over 19,160 observations: 93.52%. Per trip: 00000000 98.9%, 00000007 89.1%,
00000009 91.2%, 00000040 95.4%, 00000052 100%, 00000056 96.7%, 00000065 90.5%, 00000076 97.0%, 00000077 86.2%,
00000091 94.4%. Each prediction takes 0.1 to 2.3 s on a laptop-class CPU. No observation was dropped and none ended
unmatched. This is pointwise accuracy, not full-route sequence accuracy.

## Ground-truth separation check

`scripts/check_all_trips.py` records when the `.route` file is opened: once per trip, after the prediction, and no
response contains a reference route. `tests/test_integration.py` checks trip-id validation (digits only) and the CORS rule.

## Deployment

API on Laptop 1 (`uvicorn api.main:app --host 0.0.0.0 --port 8000`), UI on Laptop 2 with a static file server; the UI has an API
address box. Browser origins are limited to localhost and private networks. See README.md and DEMO.md. The two-laptop run over a real
network is to be confirmed by the team; the single-machine run used a non-loopback address and a different port.

## KNN

KNN results by D are in `results/models/knn_distance_weighted_20261118/`; the model discussion belongs to D's text in section 3.
