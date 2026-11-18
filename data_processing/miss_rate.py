"""A7: candidate miss rate by radius, on TRAIN trips only.
Label rule (A4): monotone snap of each observation to the reference route.
Usage: python -m data_processing.miss_rate
"""
from pathlib import Path
import numpy as np
import pandas as pd

from data_processing.loader import list_trips, load_trip
from data_processing.geometry import TripProjector, arc_endpoints, point_seg_dist
from data_processing.reference import monotone_snap
from features.candidates import arc_candidates

OUT = Path("results/data"); OUT.mkdir(parents=True, exist_ok=True)
RADII = [25, 50, 75, 100, 150, 200]

rows = []
for tid in list_trips("train"):
    track, nodes, arcs, route = load_trip(tid, "train")
    proj = TripProjector.from_track(track)
    P = proj.xy(track["lon"].values, track["lat"].values)
    nxy = proj.xy(nodes["lon"].values, nodes["lat"].values)
    A, B = arc_endpoints(nxy, arcs)

    ra = route.astype(int)
    snap = monotone_snap(point_seg_dist(P, A[ra], B[ra]))
    label_arc = ra[snap]                              # reference arc per observation

    oi, ai, d = arc_candidates(P, A, B, max(RADII))
    is_label = ai == label_arc[oi]
    label_d = np.full(len(P), np.inf)
    label_d[oi[is_label]] = d[is_label]
    for r in RADII:
        n_cand = np.bincount(oi[d <= r], minlength=len(P))
        rows.append(dict(trip_id=tid, radius_m=r, n_obs=len(P),
                         n_missed=int((label_d > r).sum()),
                         mean_candidates=float(n_cand.mean()),
                         max_candidates=int(n_cand.max())))
    print(tid, "done", flush=True)

per_trip = pd.DataFrame(rows)
per_trip.to_csv(OUT / "miss_rate_per_trip.csv", index=False)
agg = per_trip.groupby("radius_m").agg(
    n_obs=("n_obs", "sum"), n_missed=("n_missed", "sum"),
    mean_candidates=("mean_candidates", "mean"),
    max_candidates=("max_candidates", "max")).reset_index()
agg["miss_rate"] = agg.n_missed / agg.n_obs
agg.to_csv(OUT / "miss_rate_by_radius.csv", index=False)
print(agg.to_string(index=False))