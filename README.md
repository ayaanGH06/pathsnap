# PathSnap

Map matching from noisy GPS: for one trip, find which directed road arcs the vehicle travelled.
Pipeline: candidate arcs within 50 m of each GPS point -> 10 features per candidate (A) ->
saved HistGradientBoosting classifier scores each candidate (B) -> HMM/Viterbi picks a connected
sequence of arcs (C) -> FastAPI serves it, a browser UI draws it (D).

Version 1 is frozen: 50 m radius, 10 features, seed 42. The model is not retrained at run time.

## 1. Install (Python 3.11 or newer; tested on 3.12)

```bash
git clone https://github.com/ayaanGH06/pathsnap.git
cd pathsnap
python3.12 -m venv .venv          # any Python >= 3.11
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins scikit-learn 1.9.1, the version recorded in
`models/artifacts/model_manifest.json` for the saved model. A different scikit-learn may load the
file with a warning; use the pin.

## 2. Data

The raw dataset is not in git. Unzip it so that this layout exists, and point `PATHSNAP_DATA_ROOT` at it:

```
<DATA_ROOT>/student_test/00000000/00000000.{track,nodes,arcs,route}
<DATA_ROOT>/train/...
```

```bash
export PATHSNAP_DATA_ROOT="/path/to/Student folder"      # Windows PowerShell: $env:PATHSNAP_DATA_ROOT="C:\path\Student folder"
```

At the evaluation, put the ten evaluator trips in the folder named by `PATHSNAP_PARTITION`
(default `student_test`) under that root. The `.route` file is optional: without it the API still
predicts and just returns no accuracy.

## 3. Run

Laptop 1 (API):
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Check `http://localhost:8000/health` and `http://localhost:8000/docs`. Every request is logged in this console.

Laptop 2 (UI), or the same machine for a quick look:
```bash
python -m http.server 5173 --directory ui
```
Open `http://localhost:5173`, type the API address in the first box (for two laptops:
`http://<laptop1-ip>:8000`; remembered by the browser), choose a trip, press Predict route.
You can also open `http://<laptop2-ip>:5173/?api=http://<laptop1-ip>:8000&trip=00000052`.

Full two-laptop checklist, trips to show and fallbacks: [DEMO.md](DEMO.md).

## 4. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_DATA_ROOT` | `data` | folder holding `train/` and `student_test/` |
| `PATHSNAP_PARTITION` | `student_test` | partition the API lists and predicts |
| `PATHSNAP_CORS_ORIGINS` | empty | extra comma-separated browser origins. localhost and private-network (10.x, 172.16-31.x, 192.168.x) origins are always allowed |
| `PATHSNAP_API_HOST` / `PATHSNAP_API_PORT` | `0.0.0.0` / `8000` | documented defaults |

## 5. API

- `GET /health`: status, partition, data root.
- `GET /trips`: trip ids only.
- `POST /predict` with `{"trip_id": "00000007"}` (1-16 digits): returns `predicted_route` (arc ids, consecutive repeats merged), `per_observation` (one arc id per GPS point, `-1` = unmatched), GPS points, nearby road geometry, `evaluation_metrics` and `timing`.

Ground truth: the trip is loaded with `with_route=False` and predicted first. Only afterwards the server reads the reference
route to compute accuracy. The reference route is never given to the model and never returned.
`evaluation_metrics` are pointwise accuracy against the project's reference-label rule, plus directed-arc-set precision/recall.
They are not full-route sequence accuracy.

## 6. Verify

```bash
python -m pytest -q tests hmm features        # unit tests, no dataset needed
python scripts/check_all_trips.py             # needs the dataset; runs all trips through the API,
                                              # checks nothing is dropped and the reference is read only after prediction
```

## 7. Repository map

`data_processing/` loader, EDA (A) | `features/` candidates, features, `pipeline.py` (A) | `models/` training, selection, saved model (B) |
`bayes_opt/` HGB tuning | `hmm/` decoder, config, comparison (C) | `api/` backend, `ui/` browser UI (D) | `viz/` maps |
`results/` aggregate results only | `report/` report sections | `scripts/` end-to-end check.
Generated maps and screenshots show raw coordinates and are not committed.

## 8. Pretrained model (nothing is trained when you run the demo)

The API loads the already-trained model `models/artifacts/pathsnap_selected_model.joblib` (HistGradientBoosting with its
median imputer, scikit-learn 1.9.1; description in `models/artifacts/model_manifest.json`) and the decoder settings in
`hmm/hmm_config.json`. No `fit` runs at start-up or per request (`tests/test_integration.py` enforces this).
Integrity: `cd models/artifacts && sha256sum -c CHECKSUMS.sha256`.
Inference path for one trip: `features/pipeline.py` (50 m, 10 features) -> saved model `predict_proba` (class 1) ->
`hmm/decoder.py decode_detailed` -> route. Training and tuning scripts (`models/`, `bayes_opt/`, `hmm/experiment.py`) are
included for reproducibility only; they are not needed to run predictions.
Reported result: 93.52% pointwise accuracy over all 19,160 student-test observations (classifier alone 85.80%).
