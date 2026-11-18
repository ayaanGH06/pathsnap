"""FastAPI application for PathSnap route prediction."""

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from data_processing.loader import list_trips, load_trip

from .config import get_settings
from .dependencies import PredictionNotReadyError, run_prediction_pipeline
from .schemas import (
    ErrorResponse,
    GPSObservation,
    HealthResponse,
    MapArc,
    MapGeometry,
    MapNode,
    PredictionRequest,
    PredictionResponse,
    Timing,
    TripListResponse,
)

logger = logging.getLogger("pathsnap.api")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="PathSnap API",
        version="0.1.0",
        description="Ground-truth-safe API for PathSnap route prediction.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        logger.info(
            "request_started request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_finished request_id=%s status=%s elapsed_ms=%.2f",
            request_id,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(PredictionNotReadyError)
    async def pipeline_not_ready(_request: Request, exc: PredictionNotReadyError):
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def invalid_configuration(_request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            partition=settings.partition,
            data_root=str(settings.data_root),
        )

    @app.get("/trips", response_model=TripListResponse)
    def trips() -> TripListResponse:
        try:
            trip_ids = list_trips(settings.partition)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Trip data is unavailable: {exc}",
            ) from exc
        return TripListResponse(partition=settings.partition, trip_ids=trip_ids)

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            404: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def predict(request: PredictionRequest) -> PredictionResponse:
        trip_id = request.trip_id.strip()
        if not trip_id:
            raise HTTPException(status_code=422, detail="trip_id must not be blank")

        try:
            track, nodes, arcs, _reference_route = load_trip(
                trip_id,
                settings.partition,
                with_route=False,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Trip {trip_id!r} was not found in partition {settings.partition!r}",
            ) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Trip data is unavailable: {exc}",
            ) from exc

        result = run_prediction_pipeline(track=track, nodes=nodes, arcs=arcs)

        gps_observations = [
            GPSObservation(
                lon=float(row.lon),
                lat=float(row.lat),
                t=float(row.t) if "t" in track.columns and pd_notna(row.t) else None,
            )
            for row in track.itertuples(index=False)
        ]
        map_geometry = MapGeometry(
            nodes=[
                MapNode(node_id=index, lon=float(row.lon), lat=float(row.lat))
                for index, row in enumerate(nodes.itertuples(index=False))
            ],
            arcs=[
                MapArc(
                    arc_id=index,
                    from_node=int(row.from_node),
                    to_node=int(row.to_node),
                )
                for index, row in enumerate(arcs.itertuples(index=False))
            ],
        )
        return PredictionResponse(
            trip_id=trip_id,
            predicted_route=result.predicted_route,
            gps_observations=gps_observations,
            map_geometry=map_geometry,
            evaluation_metrics=result.evaluation_metrics,
            timing=Timing(**result.timing) if result.timing else None,
        )

    return app


def pd_notna(value) -> bool:
    import pandas as pd

    return bool(pd.notna(value))


app = create_app()
