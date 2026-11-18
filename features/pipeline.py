"""A12/H5: label-free, frozen-v1 features for one complete trip.

This module does not score candidates or evaluate predictions. Reference routes
are deliberately not loaded. See features/PIPELINE.md for the API handoff.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from features.build_tables import FEATURES, RADIUS_M, build_trip_features

FEATURE_VERSION = "v1"
FEATURE_COLUMNS = tuple(FEATURES)


@dataclass(frozen=True)
class TripFeatures:
    """Server-side inference inputs, not a JSON response for the UI.

    Candidate row order is the training builder's order. Observations and track
    include every GPS point, even when it has no candidate rows.
    """
    trip_id: str
    partition: str
    candidates: pd.DataFrame
    observations: pd.DataFrame
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    feature_version: str = FEATURE_VERSION
    radius_m: float = RADIUS_M

    @property
    def n_obs(self) -> int:
        return len(self.track)

    @property
    def feature_columns(self) -> tuple:
        return FEATURE_COLUMNS

    @property
    def model_features(self) -> pd.DataFrame:
        """Exactly the ten model inputs, without ids, labels or reference data."""
        return self.candidates.loc[:, list(FEATURE_COLUMNS)]


def prepare_trip(trip_id: str, partition: str = "student_test") -> TripFeatures:
    """Load one train/student_test trip and build unchanged 50 m v1 features.

    Accept a numeric id or an eight-digit folder id. PATHSNAP_DATA_ROOT selects
    the data root through the shared loader. Evaluator data is not supported by
    that loader and is never accessed here. No .route file is opened or needed.
    """
    raw_id = str(trip_id)
    if not raw_id.isascii() or not raw_id.isdecimal() or len(raw_id) > 8:
        raise ValueError("trip_id must contain 1 to 8 ASCII digits")
    tid = raw_id.zfill(8)
    track, nodes, arcs, _ = load_trip(tid, partition, with_route=False)
    return prepare_trip_frames(track, nodes, arcs, trip_id=tid, partition=partition)


def prepare_trip_frames(track: pd.DataFrame, nodes: pd.DataFrame,
                        arcs: pd.DataFrame, *, trip_id: str,
                        partition: str = "student_test") -> TripFeatures:
    """Same feature path for server callers that already loaded a trip.

    Pass only track/nodes/arcs loaded with with_route=False. trip_id is metadata,
    not a model input. This function never reads any files.
    """
    tid = str(trip_id)
    if track.empty:
        raise ValueError("Cannot build features for an empty GPS track")
    candidates, n_cand = build_trip_features(track, nodes, arcs, radius=RADIUS_M)
    # Do not sort, round, cast features, reset rows or add labels here.
    candidates.insert(0, "trip_id", tid)
    observations = pd.DataFrame({
        "trip_id": tid,
        "obs_idx": np.arange(len(track)),
        "n_cand": n_cand,
    })
    return TripFeatures(tid, partition, candidates, observations, track, nodes, arcs)
