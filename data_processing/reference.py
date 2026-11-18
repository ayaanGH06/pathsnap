"""A4: how .route relates to the GPS track.
Usage: python -m data_processing.reference [n_trips | all]
Writes results/data/reference_check.csv and results/data/ref_trip_<id>.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

from data_processing.loader import list_trips, load_trip

OUT = Path("results/data")
OUT.mkdir(parents=True, exist_ok=True)


def utm_transformer(lon, lat):
    """One UTM zone per trip, chosen from the trip's own centroid."""
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True), epsg


def point_seg_dist(P, A, B):
    """Distance from each point (N,2) to each segment (M,2)-(M,2) -> (N,M)."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = ((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None]
    t = np.clip(t, 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(2))


def monotone_snap(d):
    """Assign each obs a route-arc index, never going backwards, minimising total distance."""
    N, M = d.shape
    D = np.empty_like(d)
    D[0] = d[0]
    for i in range(1, N):
        D[i] = d[i] + np.minimum.accumulate(D[i - 1])
    idx = np.empty(N, dtype=int)
    idx[-1] = int(np.argmin(D[-1]))
    for i in range(N - 1, 0, -1):
        idx[i - 1] = int(np.argmin(D[i - 1, : idx[i] + 1]))
    return idx


def check(tid, part):
    track, nodes, arcs, route = load_trip(tid, part)
    tr, epsg = utm_transformer(track["lon"].mean(), track["lat"].mean())
    px, py = tr.transform(track["lon"].values, track["lat"].values)
    nx_, ny_ = tr.transform(nodes["lon"].values, nodes["lat"].values)
    P = np.c_[px, py]

    ra = arcs.loc[route]
    A = np.c_[nx_[ra["from_node"].values], ny_[ra["from_node"].values]]
    B = np.c_[nx_[ra["to_node"].values], ny_[ra["to_node"].values]]

    consecutive = (ra["to_node"].values[:-1] == ra["from_node"].values[1:]).mean() * 100
    d = point_seg_dist(P, A, B)
    nearest = d.argmin(1)
    snap = monotone_snap(d)
    dmin = d.min(1)
    dsnap = d[np.arange(len(P)), snap]

    row = dict(
        trip_id=tid, partition=part, epsg=epsg, n_obs=len(P), route_len=len(route),
        pct_route_consecutive=round(consecutive, 1),
        nearest_route_dist_median_m=round(float(np.median(dmin)), 1),
        nearest_route_dist_p95_m=round(float(np.percentile(dmin, 95)), 1),
        nearest_route_dist_max_m=round(float(dmin.max()), 1),
        pct_nearest_is_monotone=round(float((np.diff(nearest) >= 0).mean() * 100), 1),
        pct_snap_equals_nearest=round(float((snap == nearest).mean() * 100), 1),
        snap_dist_median_m=round(float(np.median(dsnap)), 1),
        snap_dist_p95_m=round(float(np.percentile(dsnap, 95)), 1),
        pct_route_arcs_visited=round(100 * len(np.unique(snap)) / len(route), 1),
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    for a, b in zip(A, B):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="green", lw=1.5)
    ax.scatter(px, py, s=4, color="blue")
    ax.set_title(f"Trip {tid} ({part}): GPS (blue) vs reference route arcs (green)")
    ax.set_aspect("equal")
    fig.savefig(OUT / f"ref_trip_{tid}.png", dpi=120)
    plt.close(fig)
    return row


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "5"
    ids = list_trips("train")
    ids = ids if arg == "all" else ids[: int(arg)]
    rows = []
    for tid in ids:
        rows.append(check(tid, "train"))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "reference_check.csv", index=False)
    print("saved", OUT / "reference_check.csv")