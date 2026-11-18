# pathsnap
PathSnap: distributed map-matching and route reconstruction from noisy GPS data (UE24CS352A Machine Learning hackathon)

## API setup

1. Create and activate a Python environment.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Set `PATHSNAP_DATA_ROOT` to the folder containing `train/` and `student_test/`.
4. Start the backend: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`.
5. Open `http://localhost:8000/docs` for interactive API documentation.

Environment variables:

- `PATHSNAP_DATA_ROOT`: dataset root, default `data`.
- `PATHSNAP_PARTITION`: `train` or `student_test`, default `student_test`.
- `PATHSNAP_CORS_ORIGINS`: comma-separated frontend origins.
- `PATHSNAP_API_HOST` and `PATHSNAP_API_PORT`: documented startup defaults.

## Frontend setup

The supplied UI is dependency-free. On Laptop 2, serve [ui/](/ui/) with:
`python -m http.server 5173 --directory ui`

Set the backend address for two-laptop use before loading `app.js` by setting
`window.PATHSNAP_API_URL` in a local script, or change `API_URL` in `ui/app.js`.
The value must be the Laptop 1 LAN URL, for example `http://192.168.1.20:8000`.

## Two-laptop deployment

Laptop 1:

1. Place the dataset at a local path.
2. Set `PATHSNAP_DATA_ROOT` and `PATHSNAP_PARTITION`.
3. Set `PATHSNAP_CORS_ORIGINS=http://<laptop2-ip>:5173`.
4. Run `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`.
5. Confirm `http://<laptop1-ip>:8000/health`.

Laptop 2:

1. Run `python -m http.server 5173 --directory ui`.
2. Set `window.PATHSNAP_API_URL` to `http://<laptop1-ip>:8000`.
3. Open `http://<laptop2-ip>:5173`.

Do not use `localhost` on Laptop 2 for the Laptop 1 backend. Also permit ports
8000 and 5173 in any host firewall.

## API endpoints

- `GET /health`: API availability, configured partition and data root.
- `GET /trips`: trip IDs only for the configured partition.
- `POST /predict`: loads a trip with `with_route=False`, runs the delivered
  prediction pipeline, and returns geometry, GPS observations and optional
  server metrics/timing. Ground-truth route labels are never sent to the
  predictor or serialized by this endpoint.

When the real pipeline is absent, `/predict` returns HTTP 503 with an explicit
integration-not-ready message.
