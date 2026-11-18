"""Prediction pipeline integration without changing teammate-owned interfaces."""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class PredictionNotReadyError(RuntimeError):
    pass


class PredictionPipeline(Protocol):
    def predict(self, track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame) -> dict:
        """Run prediction using no reference route."""


@dataclass(frozen=True)
class PipelineResult:
    predicted_route: list[int]
    evaluation_metrics: dict | None = None
    timing: dict | None = None
    per_observation: list[int] | None = None
    n_no_candidates: int = 0


def run_prediction_pipeline(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    pipeline: PredictionPipeline | None = None,
    trip_id: str | None = None,
) -> PipelineResult:
    """Call the delivered pipeline only.

    It deliberately accepts no route/reference labels and never constructs
    synthetic predictions. A pipeline is intentionally injected for tests
    and later person-B/person-C integration.
    """
    if pipeline is None:
        try:
            from .pipeline import get_pipeline

            pipeline = get_pipeline()
        except Exception as exc:  # model/artifact could not be loaded
            raise PredictionNotReadyError(
                f"Prediction pipeline could not be loaded: {exc}"
            ) from exc

    try:
        output = pipeline.predict(track=track, nodes=nodes, arcs=arcs, trip_id=trip_id)
    except TypeError:  # injected test pipelines without trip_id
        output = pipeline.predict(track=track, nodes=nodes, arcs=arcs)
    predicted_route = output.get("predicted_route")
    if not isinstance(predicted_route, list):
        raise RuntimeError("The prediction pipeline returned an invalid predicted_route.")
    if any(isinstance(arc_id, bool) or not isinstance(arc_id, int) for arc_id in predicted_route):
        raise RuntimeError("The prediction pipeline returned non-integer arc IDs.")
    return PipelineResult(
        predicted_route=predicted_route,
        evaluation_metrics=output.get("evaluation_metrics"),
        timing=output.get("timing"),
        per_observation=output.get("per_observation"),
        n_no_candidates=int(output.get("n_no_candidates", 0)),
    )
