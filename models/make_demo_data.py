"""Write tiny synthetic candidate and observation tables for a harness smoke check.

The data are artificial and must never be reported as PathSnap experiment results.
Files go to /tmp unless another directory is supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_partition(trip_ids: list[str], observations_per_trip: int, seed: int):
    rng = np.random.default_rng(seed)
    candidate_rows = []
    observation_rows = []
    for trip_number, trip_id in enumerate(trip_ids):
        for obs_idx in range(observations_per_trip):
            true_candidate = int(rng.integers(0, 4))
            speed = float(rng.uniform(2.0, 20.0))
            # Include missed references in trips 33 and 54 so the denominator
            # and with/without-outliers reporting can be smoke-checked.
            in_candidate_set = not (trip_id in {"00000033", "00000054"} and obs_idx < 5)
            base_arc = trip_number * 10000 + obs_idx * 4
            ref_arc = base_arc + true_candidate if in_candidate_set else base_arc + 999
            observation_rows.append({
                "trip_id": trip_id,
                "obs_idx": obs_idx,
                "ref_arc": ref_arc,
                "n_cand": 4,
                "in_cand": in_candidate_set,
            })
            for candidate_number in range(4):
                is_true = in_candidate_set and candidate_number == true_candidate
                candidate_rows.append({
                    "trip_id": trip_id,
                    "obs_idx": obs_idx,
                    "arc_id": base_arc + candidate_number,
                    "dist_to_arc": float(rng.uniform(0.5, 8.0) if is_true else rng.uniform(10.0, 100.0)),
                    "proj_frac": float(rng.uniform(0.0, 1.0)),
                    "arc_len": float(rng.uniform(20.0, 300.0)),
                    "heading_diff": float(rng.uniform(0.0, 25.0) if is_true else rng.uniform(30.0, 180.0)),
                    "speed": speed,
                    "dt_prev": 1.0,
                    "dist_rank": float(1 if is_true else candidate_number + 1),
                    "dist_minus_min": float(rng.uniform(0.0, 80.0)),
                    "n_cand": 4,
                    "conn_prev": int(bool(rng.integers(0, 2))),
                    "label": int(is_true),
                })
    return pd.DataFrame(candidate_rows), pd.DataFrame(observation_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/pathsnap-demo-candidates"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = ["00000033", "00000054", "00000001", "00000002", "00000003",
                 "00000004", "00000005", "00000006", "00000008", "00000010"]
    test_ids = ["00000007", "00000076", "00000077"]
    train_candidates, train_obs = make_partition(train_ids, 20, seed=42)
    test_candidates, test_obs = make_partition(test_ids, 15, seed=43)
    train_candidates.to_csv(args.out_dir / "candidates_train_demo.csv", index=False)
    train_obs.to_csv(args.out_dir / "obs_train_demo.csv", index=False)
    test_candidates.to_csv(args.out_dir / "candidates_student_demo.csv", index=False)
    test_obs.to_csv(args.out_dir / "obs_student_demo.csv", index=False)
    print(f"Wrote SYNTHETIC demo tables to {args.out_dir}; do not use them as project results.")


if __name__ == "__main__":
    main()
