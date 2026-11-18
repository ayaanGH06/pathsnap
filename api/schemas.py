"""Pydantic request and response contracts for the public API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128)


class GPSObservation(BaseModel):
    lon: float
    lat: float
    t: float | None = None


class MapNode(BaseModel):
    node_id: int
    lon: float
    lat: float


class MapArc(BaseModel):
    arc_id: int
    from_node: int
    to_node: int


class MapGeometry(BaseModel):
    nodes: list[MapNode]
    arcs: list[MapArc]


class Timing(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_ms: float | None = None


class PredictionResponse(BaseModel):
    trip_id: str
    predicted_route: list[int]
    gps_observations: list[GPSObservation]
    map_geometry: MapGeometry
    per_observation: list[int] | None = None
    n_no_candidates: int = 0
    evaluation_metrics: dict[str, Any] | None = None
    timing: Timing | None = None


class TripListResponse(BaseModel):
    partition: str
    trip_ids: list[str]


class HealthResponse(BaseModel):
    status: str
    partition: str
    data_root: str


class ErrorResponse(BaseModel):
    detail: str
