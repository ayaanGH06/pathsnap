"""Real PathSnap inference: A's v1 features -> B's saved model -> C's decoder.

No reference route is read or accepted here. The artifact and decoder config
are loaded once and reused for every request.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features.pipeline import prepare_trip_frames
from hmm.decoder import decode_detailed

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "artifacts" / "pathsnap_selected_model.joblib"
MANIFEST = ROOT / "models" / "artifacts" / "model_manifest.json"
HMM_CONFIG = ROOT / "hmm" / "hmm_config.json"


def collapse_consecutive(per_obs):
    """Route display form: drop unmatched (-1) and consecutive repeats only.
    A later revisit of the same arc is kept."""
    route = []
    for arc in per_obs:
        arc = int(arc)
        if arc < 0:
            continue
        if not route or route[-1] != arc:
            route.append(arc)
    return route


class PathSnapPipeline:
    def __init__(self, artifact=ARTIFACT, manifest=MANIFEST, hmm_config=HMM_CONFIG):
        self.model = joblib.load(artifact)
        self.manifest = json.loads(Path(manifest).read_text())
        self.cols = list(self.manifest["feature_columns_in_required_order"])
        self.cfg = json.loads(Path(hmm_config).read_text())
        self.pos = list(self.model.classes_).index(1)

    def predict(self, track, nodes, arcs, trip_id=None):
        t0 = time.perf_counter()
        tf = prepare_trip_frames(track, nodes, arcs, trip_id=str(trip_id or ""))
        t1 = time.perf_counter()
        cands = tf.candidates.copy()
        if len(cands):
            cands["score"] = self.model.predict_proba(tf.model_features[self.cols])[:, self.pos]
        else:
            cands["score"] = np.array([], dtype=float)
        t2 = time.perf_counter()
        per_obs, stats = decode_detailed(cands, nodes, arcs, self.cfg, track=track)
        t3 = time.perf_counter()
        per_obs = [int(a) for a in per_obs]
        return {
            "predicted_route": collapse_consecutive(per_obs),
            "per_observation": per_obs,
            "n_observations": int(len(track)),
            "n_no_candidates": int(stats.get("n_no_candidates", 0)),
            "timing": {
                "total_ms": (t3 - t0) * 1000,
                "features_ms": (t1 - t0) * 1000,
                "scoring_ms": (t2 - t1) * 1000,
                "decoding_ms": (t3 - t2) * 1000,
            },
        }


_instance = None


def get_pipeline():
    global _instance
    if _instance is None:
        _instance = PathSnapPipeline()
    return _instance
