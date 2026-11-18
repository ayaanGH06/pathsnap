"""Server-side accuracy. Runs only AFTER prediction; the reference route is
never passed to the predictor, the model, or the response."""
import numpy as np

from data_processing.loader import load_trip
from features.build_tables import reference_arcs


def evaluate(trip_id, partition, track, nodes, arcs, per_observation):
    _t, _n, _a, route = load_trip(trip_id, partition, with_route=True)
    if route is None or len(route) == 0:
        return None
    ref = reference_arcs(track, nodes, arcs, route)
    pred = np.asarray(per_observation, dtype=np.int64)
    n = len(ref)
    correct = int((pred[:n] == ref).sum())
    pset, rset = {int(a) for a in pred if a >= 0}, {int(a) for a in route}
    inter = len(pset & rset)
    return {
        "observation_accuracy": round(correct / n, 4),
        "observations_correct": correct,
        "observations_total": n,
        "unmatched_observations": int((pred < 0).sum()),
        "route_precision": round(inter / len(pset), 4) if pset else 0.0,
        "route_recall": round(inter / len(rset), 4) if rset else 0.0,
    }
