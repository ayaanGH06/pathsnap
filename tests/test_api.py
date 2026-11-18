import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.dependencies import PipelineResult


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_PARTITION", "student_test")
    monkeypatch.delenv("PATHSNAP_CORS_ORIGINS", raising=False)
    return TestClient(create_app())


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_trips_uses_loader(monkeypatch, client):
    monkeypatch.setattr("api.main.list_trips", lambda partition: ["00000001", "00000002"])
    response = client.get("/trips")
    assert response.status_code == 200
    assert response.json() == {
        "partition": "student_test",
        "trip_ids": ["00000001", "00000002"],
    }


def test_valid_prediction_response_and_no_route_leak(monkeypatch, client):
    track = pd.DataFrame({"lon": [77.1, 77.2], "lat": [28.1, 28.2], "t": [0.0, 1.0]})
    nodes = pd.DataFrame({"lon": [77.1, 77.2], "lat": [28.1, 28.2]})
    arcs = pd.DataFrame({"from_node": [0], "to_node": [1]})

    def fake_load(trip_id, partition, with_route=True):
        assert with_route is False
        return track, nodes, arcs, None

    monkeypatch.setattr("api.main.load_trip", fake_load)
    monkeypatch.setattr(
        "api.main.run_prediction_pipeline",
        lambda track, nodes, arcs, trip_id=None: PipelineResult(
            predicted_route=[0],
            evaluation_metrics={"accuracy": 0.75},
            timing={"total_ms": 12.5},
        ),
    )

    response = client.post("/predict", json={"trip_id": "00000001"})
    assert response.status_code == 200
    body = response.json()
    assert body["trip_id"] == "00000001"
    assert body["predicted_route"] == [0]
    assert len(body["gps_observations"]) == 2
    assert body["map_geometry"]["nodes"][0]["node_id"] == 0
    assert body["evaluation_metrics"] == {"accuracy": 0.75}
    assert "reference_route" not in body


def test_missing_trip(client, monkeypatch):
    monkeypatch.setattr(
        "api.main.load_trip",
        lambda trip_id, partition, with_route=True: (_ for _ in ()).throw(
            FileNotFoundError(trip_id)
        ),
    )
    response = client.post("/predict", json={"trip_id": "00000099"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_pipeline_not_ready(monkeypatch, client):
    track = pd.DataFrame({"lon": [77.1], "lat": [28.1], "t": [0.0]})
    nodes = pd.DataFrame({"lon": [77.1], "lat": [28.1]})
    arcs = pd.DataFrame({"from_node": [0], "to_node": [0]})
    monkeypatch.setattr(
        "api.main.load_trip",
        lambda trip_id, partition, with_route=True: (track, nodes, arcs, None),
    )
    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr("api.pipeline.get_pipeline", boom)
    response = client.post("/predict", json={"trip_id": "00000001"})
    assert response.status_code == 503
    assert "could not be loaded" in response.json()["detail"].lower()


@pytest.mark.parametrize("payload", [{}, {"trip_id": ""}, {"trip_id": " " * 2}])
def test_invalid_request_schema(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code in (422, 422)
