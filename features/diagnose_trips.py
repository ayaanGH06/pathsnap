"""features/diagnose_trips.py

Why do some trips (e.g. 54, 33, 2) miss so many reference arcs at 50 m?

For each chosen trip, per GPS observation (no radius limit):
  d_ref   distance (m) to the observation's labelled reference arc
  d_near  distance (m) to the nearest arc anywhere in that trip's network
and a few trip-level checks (time gaps, track vs network extent, where along the trip
the far observations sit).

How to read the result:
  * d_near large for the far observations  -> GPS is far from EVERY road in that trip's
    network (network/extract mismatch or GPS error); a bigger radius mostly finds wrong roads.
  * d_near small but d_ref large           -> GPS sits on some road, but the reference route
    follows a different one (reference/label issue, or a parallel road).
  * d_ref - d_near > 50 for many obs        -> same idea, counted directly.
  * d_route_min small but d_ref large      -> the reference ROUTE passes near the GPS, so the
    in-order (monotone) snap in reference.py picked the wrong arc: a labelling problem.
  * d_route_min also large                 -> the reference route itself is far from the GPS.

Usage (repo root, PATHSNAP_DATA_ROOT exported):
    python -m features.diagnose_trips train 54 33 2
    python -m features.diagnose_trips train            (defaults to 54 33 2)

Only 'train' and 'student_test' are accepted. Never run on evaluator data.
Writes: results/data/trip_diagnostic_<partition>.json (summary, commit this),
        results/data/figures/trip_<id>_diag_<partition>.png (plot, partition in title),
        data/processed/diag_<partition>_<id>.parquet (per-observation, not committed).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from data_processing.geometry import point_seg_dist
from data_processing.loader import list_trips, load_trip
from features.build_tables import _geometry, reference_arcs
from features.candidates import pair_dist

ALLOWED = {"train", "student_test"}


def pct(a, q):
    return float(np.nanpercentile(a, q))


def diagnose(tid, part):
    track, nodes, arcs, route = load_trip(tid, part)
    P, A, B = _geometry(track, nodes, arcs)
    ref = reference_arcs(track, nodes, arcs, route)
    n = len(P)

    d_ref = pair_dist(P, A[ref], B[ref])

    # distance to the nearest arc OF THE REFERENCE ROUTE, ignoring the monotone-order rule.
    # If this is small while d_ref is large, the monotone snap is mislabelling.
    rt = route.astype(int)
    d_route_min = point_seg_dist(P, A[rt], B[rt]).min(1)

    tree = STRtree(shapely.linestrings(np.stack([A, B], axis=1)))
    (inp, tr), dn = tree.query_nearest(shapely.points(P), return_distance=True, all_matches=False)
    near_arc = np.full(n, -1, dtype=np.int64)
    d_near = np.full(n, np.nan)
    near_arc[inp] = tr
    d_near[inp] = dn

    t = track["t"].values.astype(float)
    dt = np.diff(t) if n > 1 else np.zeros(1)

    net = np.vstack([A, B])
    lo, hi = net.min(0), net.max(0)
    outside = ((P < lo) | (P > hi)).any(1)

    chunks = np.array_split(np.arange(n), 10)
    chunk_far = [float((d_ref[c] > 50).mean()) for c in chunks]

    s = {
        "partition": part, "trip_id": tid,
        "n_obs": int(n), "n_arcs": int(len(arcs)), "n_route_arcs": int(len(route)),
        "max_gap_s": float(dt.max()), "n_gaps_gt_10s": int((dt > 10).sum()),
        "frac_track_outside_network_bbox": float(outside.mean()),
        "d_ref_m": {"median": pct(d_ref, 50), "p90": pct(d_ref, 90), "p99": pct(d_ref, 99), "max": float(d_ref.max())},
        "d_near_m": {"median": pct(d_near, 50), "p90": pct(d_near, 90), "p99": pct(d_near, 99), "max": float(np.nanmax(d_near))},
        "d_route_min_m": {"median": pct(d_route_min, 50), "p90": pct(d_route_min, 90),
                          "p99": pct(d_route_min, 99), "max": float(d_route_min.max())},
        "frac_d_route_min_gt_50": float((d_route_min > 50).mean()),
        "frac_ref_farther_than_route_min_by_gt_50": float(((d_ref - d_route_min) > 50).mean()),
        "frac_d_ref_gt_50": float((d_ref > 50).mean()),
        "frac_d_ref_gt_150": float((d_ref > 150).mean()),
        "frac_d_near_gt_50": float((d_near > 50).mean()),
        "frac_d_near_gt_150": float((d_near > 150).mean()),
        "frac_nearest_is_reference_arc": float((near_arc == ref).mean()),
        "frac_ref_farther_than_nearest_by_gt_50": float(((d_ref - d_near) > 50).mean()),
        "frac_far_by_tenth_of_trip": [round(x, 3) for x in chunk_far],
    }

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"obs_idx": np.arange(n), "t": t, "ref_arc": ref, "d_ref": d_ref,
                  "near_arc": near_arc, "d_near": d_near, "d_route_min": d_route_min}
                 ).to_parquet(f"data/processed/diag_{part}_{tid}.parquet", index=False)
    return s, (P, A, B, route, d_ref)


def plot(tid, part, P, A, B, route, d_ref):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    r = np.unique(route.astype(int))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(np.stack([A, B], axis=1), colors="lightgray",
                                     linewidths=0.4, label="road network"))
    ax.add_collection(LineCollection(np.stack([A[r], B[r]], axis=1), colors="tab:blue",
                                     linewidths=1.3, label="reference route"))
    far = d_ref > 50
    ax.scatter(P[~far, 0], P[~far, 1], s=2, c="tab:green", label="GPS, within 50 m of its reference arc")
    ax.scatter(P[far, 0], P[far, 1], s=2, c="tab:red", label="GPS, farther than 50 m")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (m, per-trip UTM)")
    ax.set_ylabel("y (m, per-trip UTM)")
    ax.set_title(f"Trip {tid} ({part} partition): GPS vs reference route")
    ax.legend(loc="best", markerscale=4, fontsize=8)
    out = Path("results/data/figures")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"trip_{tid}_diag_{part}.png"
    fig.savefig(f, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f}")


def main(part, trips):
    if part not in ALLOWED:
        sys.exit(f"Partition must be one of {sorted(ALLOWED)}")
    known = set(map(str, list_trips(part)))
    results = []
    for tid in trips:
        tid = str(tid).zfill(8)
        if tid not in known:
            print(f"Trip {tid} is not in the {part} partition, skipping")
            continue
        s, geo = diagnose(tid, part)
        results.append(s)
        print(json.dumps(s, indent=2))
        plot(tid, part, *geo)
    out = Path("results/data")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"trip_diagnostic_{part}.json"
    f.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {f}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "train"
    trips = sys.argv[2:] or ["54", "33", "2"]
    main(part, trips)