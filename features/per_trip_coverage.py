"""features/per_trip_coverage.py

Per-trip candidate coverage at the radius the table was built with.

Why this exists: the candidate table has no rows for observations that have ZERO
candidates inside the radius, so a miss rate computed from the table alone undercounts.
Here the true number of observations per trip comes from the raw track (loader), and

    n_miss = n_no_candidate (absent from table) + n_zero_positive (in table, no label=1)

Columns written per trip:
  n_obs_total, n_obs_in_table, n_no_candidate, n_zero_positive, n_miss, miss_rate,
  d_label_median / p95 / max  (distance to the labelled arc where one was found;
                               capped by the radius, so it understates real offsets)

It also compares its totals with results/data/tables_run.json as a consistency check.

Usage (repo root, PATHSNAP_DATA_ROOT exported):
    python -m features.per_trip_coverage train
    python -m features.per_trip_coverage student_test

Only 'train' and 'student_test' are accepted. Never run on evaluator data.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from data_processing.loader import list_trips, load_trip

ALLOWED = {"train", "student_test"}
KEYS = ["trip_id", "obs_idx"]


def load_table(partition):
    base = Path("data/processed") / f"candidates_{partition}"
    for suffix in (".parquet", ".csv"):
        p = base.with_suffix(suffix)
        if p.exists():
            print(f"Reading {p}")
            df = pd.read_parquet(p) if suffix == ".parquet" else pd.read_csv(p, dtype={"trip_id": str})
            df["trip_id"] = df["trip_id"].astype(str)
            return df
    raise FileNotFoundError(f"No candidates_{partition} table in data/processed/.")


def main(partition):
    if partition not in ALLOWED:
        sys.exit(f"Partition must be one of {sorted(ALLOWED)}")
    df = load_table(partition)

    # observations per trip from the raw tracks (all observations, kept as decided)
    totals = {}
    for tid in list_trips(partition):
        track = load_trip(tid, partition, with_route=False)[0]
        totals[str(tid)] = len(track)
    total = pd.Series(totals, name="n_obs_total")

    obs = df.groupby(KEYS)["label"].max().rename("has_label").reset_index()
    in_table = obs.groupby("trip_id").size().rename("n_obs_in_table")
    zero_pos = obs[obs["has_label"] == 0].groupby("trip_id").size().rename("n_zero_positive")

    out = pd.concat([total, in_table, zero_pos], axis=1).fillna(0).astype(int)
    out.index.name = "trip_id"
    out["n_no_candidate"] = out["n_obs_total"] - out["n_obs_in_table"]
    out["n_miss"] = out["n_no_candidate"] + out["n_zero_positive"]
    out["miss_rate"] = out["n_miss"] / out["n_obs_total"]

    pos = df[df["label"] == 1]
    d = pos.groupby("trip_id")["dist_to_arc"].agg(
        d_label_median="median", d_label_p95=lambda s: s.quantile(0.95), d_label_max="max")
    out = out.join(d).sort_values("miss_rate", ascending=False)

    N = int(out["n_obs_total"].sum())
    M = int(out["n_miss"].sum())
    print(f"\nPartition: {partition}")
    print(f"Trips: {len(out)}  observations (raw tracks): {N}  in table: {int(out['n_obs_in_table'].sum())}")
    print(f"No candidate in radius: {int(out['n_no_candidate'].sum())}   "
          f"in table but no label=1: {int(out['n_zero_positive'].sum())}")
    print(f"Total misses: {M}   true miss rate: {M / N:.4f}   "
          f"(ceiling on all-observation accuracy for a one-candidate-per-observation classifier: {1 - M / N:.4f})")

    log = Path("results/data/tables_run.json")
    if log.exists():
        ref = json.loads(log.read_text()).get(partition, {})
        print(f"tables_run.json says: n_obs={ref.get('n_obs')}  miss_rate={ref.get('miss_rate')}  "
              f"-> {'MATCH' if ref.get('n_obs') == N and abs(ref.get('miss_rate', -1) - M / N) < 1e-9 else 'MISMATCH, investigate'}")

    print("\nWorst 10 trips by miss rate:")
    cols = ["n_obs_total", "n_no_candidate", "n_zero_positive", "n_miss", "miss_rate",
            "d_label_median", "d_label_p95"]
    print(out[cols].head(10).round(4).to_string())

    by_n = out.sort_values("n_miss", ascending=False)
    print(f"\nShare of all misses from the 5 trips with most misses: "
          f"{by_n['n_miss'].head(5).sum() / max(M, 1):.1%}")
    print(f"Trips with zero misses: {int((out['n_miss'] == 0).sum())} of {len(out)}")
    print(f"Median per-trip miss rate: {out['miss_rate'].median():.4f}   max: {out['miss_rate'].max():.4f}")

    dest = Path("results/data")
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"per_trip_coverage_{partition}.csv"
    out.to_csv(f)
    print(f"\nSaved {f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")