import sklearn.ensemble, sklearn.pipeline
"""End-to-end checks on synthetic data (no dataset needed) plus the ground-truth
separation guarantee. Real-data runs: python scripts/check_all_trips.py"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import pipeline as P
from api.main import create_app


def test_collapse_keeps_revisits_and_drops_unmatched():
    assert P.collapse_consecutive([1, 1, 2, 2, 1]) == [1, 2, 1]
    assert P.collapse_consecutive([-1, 3, -1, -1, 3]) == [3]


def test_predict_never_reads_route(monkeypatch, tmp_path):
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path))
    track = pd.DataFrame({"lon": [77.1, 77.2], "lat": [28.1, 28.2], "t": [0.0, 1.0]})
    nodes = pd.DataFrame({"lon": [77.1, 77.2], "lat": [28.1, 28.2]})
    arcs = pd.DataFrame({"from_node": [0], "to_node": [1]})
    seen = []

    def fake_load(trip_id, partition, with_route=True):
        seen.append(with_route)
        return track, nodes, arcs, (np.array([0]) if with_route else None)

    class Fake:
        def predict(self, track, nodes, arcs, trip_id=None):
            return {"predicted_route": [0], "per_observation": [0, 0], "n_no_candidates": 0}

    monkeypatch.setattr("api.main.load_trip", fake_load)
    monkeypatch.setattr("api.evaluation.load_trip", fake_load)
    monkeypatch.setattr(P, "_instance", Fake())
    client = TestClient(create_app())
    body = client.post("/predict", json={"trip_id": "00000001"}).json()
    # predictor load first without route; evaluation reads it afterwards
    assert seen[0] is False
    assert "reference_route" not in body and "route" not in body


@pytest.mark.parametrize("bad", ["../etc/passwd", "..", "0000/../0001", "a b", "1" * 17, "00000007.route"])
def test_trip_id_must_be_digits(monkeypatch, tmp_path, bad):
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path))
    r = TestClient(create_app()).post("/predict", json={"trip_id": bad})
    assert r.status_code == 422


def test_cors_lan_only(monkeypatch, tmp_path):
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path))
    c = TestClient(create_app())
    for origin, allowed in [("http://192.168.1.20:5173", True), ("http://localhost:5173", True),
                            ("https://evil.example", False), ("http://8.8.8.8:5173", False)]:
        h = c.get("/health", headers={"Origin": origin}).headers
        assert (h.get("access-control-allow-origin") == origin) is allowed


def test_inference_never_trains(monkeypatch):
    """Loading and predicting must use the saved model only: any fit call fails the test."""
    import sklearn.base

    def no_fit(*a, **k):
        raise AssertionError("fit() called during inference")

    for cls in (sklearn.base.BaseEstimator,):
        monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit", no_fit)
        monkeypatch.setattr(sklearn.ensemble.HistGradientBoostingClassifier, "fit", no_fit)
    pipe = P.PathSnapPipeline()
    assert list(pipe.model.classes_) == [0, 1]
    out = pipe.predict(
        pd.DataFrame({"lon": [77.1, 77.2], "lat": [28.1, 28.2], "t": [0.0, 1.0]}),
        pd.DataFrame({"lon": [77.1, 77.2], "lat": [28.1, 28.2]}),
        pd.DataFrame({"from_node": [0], "to_node": [1]}),
        trip_id="00000001",
    )
    assert len(out["per_observation"]) == 2
