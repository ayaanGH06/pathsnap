"""features/check_labels.py

Checks on a candidate table (contract 4.3):
  1. No duplicate (trip_id, obs_idx, arc_id) rows.
  2. At most one label=1 per (trip_id, obs_idx).
  3. Label values are only 0/1.
Also reports, from the table itself, how many observations have no label=1
(these are the candidate misses, or observations with no reference arc).

Usage (from repo root, with PATHSNAP_DATA_ROOT exported):
    python -m features.check_labels train
    python -m features.check_labels student_test

Never run this on evaluator data: only 'train' and 'student_test' are accepted.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ALLOWED = {"train", "student_test"}
KEYS = ["trip_id", "obs_idx"]


def load_table(partition):
    base = Path("data/processed") / f"candidates_{partition}"
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = base.with_suffix(suffix)
        if p.exists():
            print(f"Reading {p}")
            return reader(p, dtype={"trip_id": str}) if suffix == ".csv" else reader(p)
    raise FileNotFoundError(f"No candidates_{partition}.parquet/.csv in data/processed/. "
                            f"Run: python -m features.build_tables 50")


def main(partition):
    if partition not in ALLOWED:
        sys.exit(f"Partition must be one of {sorted(ALLOWED)}")
    df = load_table(partition)
    df["trip_id"] = df["trip_id"].astype(str)

    n_rows = len(df)
    n_trips = df["trip_id"].nunique()
    n_obs = len(df.drop_duplicates(KEYS))

    dup_rows = int(df.duplicated(KEYS + ["arc_id"]).sum())
    bad_values = int((~df["label"].isin([0, 1])).sum())

    pos = df.groupby(KEYS)["label"].sum()
    multi = pos[pos > 1]
    zero = int((pos == 0).sum())

    result = {
        "partition": partition,
        "rows": n_rows,
        "trips": n_trips,
        "observations_with_candidates": n_obs,
        "duplicate_trip_obs_arc_rows": dup_rows,
        "label_values_not_0_or_1": bad_values,
        "observations_with_more_than_one_positive": int(len(multi)),
        "max_positives_per_observation": int(pos.max()),
        "observations_with_zero_positive": zero,
    }
    print(json.dumps(result, indent=2))

    if len(multi):
        print("\nFirst offending observations:")
        print(multi.head(10))
        bad = multi.index[0]
        print(f"\nRows for {bad}:")
        mask = (df["trip_id"] == bad[0]) & (df["obs_idx"] == bad[1])
        print(df.loc[mask, KEYS + ["arc_id", "label"]])

    out = Path("results/data")
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"label_check_{partition}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out_file}  (commit with: git add -f {out_file.as_posix()})")

    ok = dup_rows == 0 and bad_values == 0 and len(multi) == 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")