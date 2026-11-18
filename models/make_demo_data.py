"""Create tiny synthetic candidate tables to check the evaluator mechanics.

These data are artificial. They must never be reported as PathSnap experiments.
By default files are written under /tmp, outside the repository.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_partition(trip_ids: list[str], observations_per_trip: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for trip_number, trip_id in enumerate(trip_ids):
        for obs_idx in range(observations_per_trip):
            true_candidate = int(rng.integers(0, 4))
            speed = float(rng.uniform(2.0, 20.0))
            for candidate_number in range(4):
                is_true = candidate_number == true_candidate
                rows.append(
                    {
                        "trip_id": trip_id,
                        "obs_idx": obs_idx,
                        "arc_id": trip_number * 10000 + obs_idx * 4 + candidate_number,
                        "distance_to_arc_m": float(rng.uniform(0.5, 8.0) if is_true else rng.uniform(10.0, 100.0)),
                        "heading_difference_deg": float(rng.uniform(0.0, 25.0) if is_true else rng.uniform(30.0, 180.0)),
                        "speed_mps": speed,
                        "candidate_rank": float(1 if is_true else candidate_number + 1),
                        "label": int(is_true),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/pathsnap-demo-candidates"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = make_partition([f"demo_train_{i:02d}" for i in range(10)], 20, seed=42)
    student = make_partition([f"demo_student_{i:02d}" for i in range(3)], 15, seed=43)
    train.to_csv(args.out_dir / "candidates_train_demo.csv", index=False)
    student.to_csv(args.out_dir / "candidates_student_demo.csv", index=False)
    print(f"Wrote SYNTHETIC demo tables to {args.out_dir}; do not use them as project results.")


if __name__ == "__main__":
    main()
