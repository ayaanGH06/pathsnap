"""Reproduce real-data A12 parity checks, without reading reference routes.

Run: set PATHSNAP_DATA_ROOT to the dataset, then python -m features.verify_pipeline
Optionally checks existing processed candidate parquets as well. Outputs summary
counts only to results/data/pipeline_check.json. No coordinates are exported.
"""
import builtins
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data_processing.loader import list_trips
from features.build_tables import build_trip_features
from features.pipeline import FEATURE_COLUMNS, prepare_trip


def verify():
    manifest = json.loads(Path("models/artifacts/model_manifest.json").read_text())
    assert list(FEATURE_COLUMNS) == manifest["feature_columns_in_required_order"]
    assert manifest["candidate_radius_m"] == 50
    original_open = builtins.open
    def no_routes(file, *args, **kwargs):
        if str(file).endswith(".route"):
            raise AssertionError("Reference route opened during inference validation")
        return original_open(file, *args, **kwargs)
    summary = {"feature_version": "v1", "radius_m": 50,
               "reference_route_reads": 0, "partitions": {}}
    with patch("builtins.open", side_effect=no_routes):
        for part in ("train", "student_test"):
            parquet = Path(f"data/processed/candidates_{part}.parquet")
            cached = pd.read_parquet(parquet) if parquet.exists() else None
            counts = dict(trips=0, n_obs=0, candidate_rows=0, zero_candidate_obs=0,
                          builder_exact_parity=True, processed_parquet_checked=cached is not None)
            for tid in list_trips(part):
                result = prepare_trip(tid, part)
                expected, n_cand = build_trip_features(result.track, result.nodes, result.arcs)
                expected.insert(0, "trip_id", tid)
                assert_frame_equal(result.candidates, expected, check_exact=True)
                np.testing.assert_array_equal(result.observations.n_cand, n_cand)
                assert result.observations.obs_idx.tolist() == list(range(result.n_obs))
                assert "label" not in result.candidates and "ref_arc" not in result.observations
                if cached is not None:
                    rows = cached.loc[cached.trip_id == tid].drop(columns=["label"]).reset_index(drop=True)
                    assert_frame_equal(result.candidates, rows, check_exact=True)
                counts["trips"] += 1
                counts["n_obs"] += result.n_obs
                counts["candidate_rows"] += len(result.candidates)
                counts["zero_candidate_obs"] += int((n_cand == 0).sum())
                print(part, tid, result.n_obs, len(result.candidates), flush=True)
            summary["partitions"][part] = counts
    return summary


if __name__ == "__main__":
    output = verify()
    target = Path("results/data/pipeline_check.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
