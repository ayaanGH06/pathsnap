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


def run_prediction_pipeline(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    pipeline: PredictionPipeline | None = None,
) -> PipelineResult:
    """Call the delivered pipeline only.

    It deliberately accepts no route/reference labels and never constructs
    synthetic predictions. A pipeline is intentionally injected for tests
    and later person-B/person-C integration.
    """
    if pipeline is None:
        raise PredictionNotReadyError(
            "Prediction integration is not ready: the real feature pipeline, "
            "model, and/or decoder have not been delivered."
        )

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
    )
