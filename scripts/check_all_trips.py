"""D7/D8: run every trip in the configured partition through the real API app,
record aggregate results, and prove the reference route is read only AFTER the
prediction (never by the predictor).

    PATHSNAP_DATA_ROOT=/path/to/'Student folder' python scripts/check_all_trips.py
Writes results/api/api_check_<partition>.json (aggregate numbers only, no coordinates).
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import pandas as pd

from api.main import create_app

events = []  # ordered log of ("predict_done"|"route_open", trip)
_real_read_csv = pd.read_csv


def spy_read_csv(path, *a, **k):
    if str(path).endswith(".route"):
        events.append(("route_open", Path(str(path)).stem))
    return _real_read_csv(path, *a, **k)


pd.read_csv = spy_read_csv
import api.main as api_main
_real_predict = api_main.run_prediction_pipeline


def spy_predict(*a, **k):
    out = _real_predict(*a, **k)
    events.append(("predict_done", None))
    return out


api_main.run_prediction_pipeline = spy_predict

client = TestClient(create_app())
part = client.get("/health").json()["partition"]
trips = client.get("/trips").json()["trip_ids"]
rows, ok = [], True
for t in trips:
    events.clear()
    t0 = time.time()
    r = client.post("/predict", json={"trip_id": t})
    if r.status_code != 200:
        print(t, "FAILED", r.status_code, r.text[:200]); ok = False; continue
    b = r.json()
    # the route may only be opened once (by server-side evaluation, after prediction)
    opened = [e for e in events if e[0] == "route_open"]
    names = [e[0] for e in events]
    after = "predict_done" in names and all(
        names.index("predict_done") < i for i, n in enumerate(names) if n == "route_open")
    leak = "reference_route" in b or "route" in b
    n_obs = len(b["gps_observations"])
    full = b["per_observation"] is not None and len(b["per_observation"]) == n_obs
    m = b["evaluation_metrics"] or {}
    rows.append(dict(trip_id=t, observations=n_obs, per_obs_complete=full,
                     route_arcs=len(b["predicted_route"]), no_candidate_obs=b["n_no_candidates"],
                     unmatched=m.get("unmatched_observations"), route_files_opened=len(opened), route_read_only_after_prediction=after,
                     reference_in_response=leak, seconds=round(time.time() - t0, 2),
                     **{k: m.get(k) for k in ("observation_accuracy", "route_precision", "route_recall")}))
    ok &= full and not leak and len(opened) <= 1 and after
    print(rows[-1])
tot = sum(r["observations"] for r in rows)
acc = sum(r["observation_accuracy"] * r["observations"] for r in rows) / tot if tot else None
out = dict(partition=part, trips=len(rows), of=len(trips), all_ok=bool(ok),
           pooled_observation_accuracy=round(acc, 4) if acc else None,
           note="pointwise accuracy on A's label rule, not full-route accuracy", per_trip=rows)
p = Path("results/api") / f"api_check_{part}.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2))
print("wrote", p, "all_ok =", ok, "pooled acc", out["pooled_observation_accuracy"])
sys.exit(0 if ok else 1)
