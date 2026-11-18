"""A8-A9: candidate tables with features (contract 4.3).
Usage: python -m features.build_tables [radius_m] [limit_trips]
Writes data/processed/candidates_<partition>.parquet and obs_<partition>.parquet
(with a limit, writes *_smoke files instead).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from config import RANDOM_SEED
from data_processing.loader import list_trips, load_trip
from data_processing.geometry import (TripProjector, arc_endpoints, bearing_deg,
                                      step_features, point_seg_dist)
from data_processing.reference import monotone_snap
from features.candidates import arc_candidates

RADIUS_M = 50.0
FEATURES = ["dist_to_arc", "proj_frac", "arc_len", "heading_diff", "speed",
            "dt_prev", "dist_rank", "dist_minus_min", "n_cand", "conn_prev"]
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)


def _geometry(track, nodes, arcs):
    proj = TripProjector.from_track(track)
    P = proj.xy(track["lon"].values, track["lat"].values)
    nxy = proj.xy(nodes["lon"].values, nodes["lat"].values)
    A, B = arc_endpoints(nxy, arcs)
    return P, A, B


def build_trip_features(track, nodes, arcs, radius=RADIUS_M):
    """Candidates + features for one trip. Uses NO labels. Returns (df, n_cand)."""
    P, A, B = _geometry(track, nodes, arcs)
    n = len(P)
    oi, ai, d = arc_candidates(P, A, B, radius)
    o = np.lexsort((d, oi)); oi, ai, d = oi[o], ai[o], d[o]

    AB = B[ai] - A[ai]
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    proj_frac = np.clip(((P[oi] - A[ai]) * AB).sum(1) / L2, 0, 1)
    arc_b = bearing_deg(AB[:, 0], AB[:, 1])

    # vehicle heading from neighbouring observations
    idx = np.arange(n)
    mv = P[np.minimum(idx + 1, n - 1)] - P[np.maximum(idx - 1, 0)]
    veh_b = bearing_deg(mv[:, 0], mv[:, 1])
    still = np.hypot(mv[:, 0], mv[:, 1]) < 1.0
    hd = np.abs((arc_b - veh_b[oi] + 180.0) % 360.0 - 180.0)
    hd = np.where(still[oi], 90.0, hd)           # heading unreliable -> neutral

    # speed and time since previous obs, using the REAL dt
    if n > 1:
        _, dt, v = step_features(P, track["t"].values)
        speed, dt_prev = np.r_[v[0], v], np.r_[dt[0], dt]
    else:
        speed, dt_prev = np.zeros(1), np.ones(1)

    n_cand = np.bincount(oi, minlength=n)
    start = np.r_[0, np.cumsum(n_cand)[:-1]]
    dmin = np.full(n, np.nan); has = n_cand > 0; dmin[has] = d[start[has]]
    rank = np.arange(len(oi)) - start[oi] + 1

    # connectivity to previous observation's candidate set (1 hop, directed)
    frm, to = arcs["from_node"].values, arcs["to_node"].values
    conn = np.zeros(len(oi), dtype=np.int8)
    ends = start + n_cand
    for i in range(1, n):
        if n_cand[i] == 0 or n_cand[i - 1] == 0:
            continue
        prev = ai[start[i - 1]:ends[i - 1]]
        cur = ai[start[i]:ends[i]]
        conn[start[i]:ends[i]] = np.isin(cur, prev) | np.isin(frm[cur], to[prev])

    df = pd.DataFrame({
        "obs_idx": oi.astype(np.int32), "arc_id": ai.astype(np.int32),
        "dist_to_arc": d.astype(np.float32), "proj_frac": proj_frac.astype(np.float32),
        "arc_len": np.sqrt(L2).astype(np.float32), "heading_diff": hd.astype(np.float32),
        "speed": speed[oi].astype(np.float32), "dt_prev": dt_prev[oi].astype(np.float32),
        "dist_rank": rank.astype(np.int16),
        "dist_minus_min": (d - dmin[oi]).astype(np.float32),
        "n_cand": n_cand[oi].astype(np.int16), "conn_prev": conn,
    })
    return df, n_cand


def reference_arcs(track, nodes, arcs, route):
    """Label rule (A4): in-order snap of each observation to the reference route."""
    P, A, B = _geometry(track, nodes, arcs)
    ra = route.astype(int)
    return ra[monotone_snap(point_seg_dist(P, A[ra], B[ra]))]


if __name__ == "__main__":
    radius = float(sys.argv[1]) if len(sys.argv) > 1 else RADIUS_M
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    suffix = "_smoke" if limit else ""
    log = dict(seed=RANDOM_SEED, radius_m=radius, features=FEATURES)
    for part in ["train", "student_test"]:
        t0 = time.time(); cand, obs = [], []
        ids = list_trips(part)[:limit]
        for tid in ids:
            track, nodes, arcs, route = load_trip(tid, part)
            df, n_cand = build_trip_features(track, nodes, arcs, radius)
            ref = reference_arcs(track, nodes, arcs, route)
            df["label"] = (df["arc_id"].values == ref[df["obs_idx"].values]).astype(np.int8)
            hit = np.bincount(df["obs_idx"].values[df["label"].values == 1], minlength=len(track)) > 0
            df.insert(0, "trip_id", tid)
            cand.append(df)
            obs.append(pd.DataFrame({"trip_id": tid, "obs_idx": np.arange(len(track)),
                                     "ref_arc": ref, "n_cand": n_cand, "in_cand": hit}))
            print(part, tid, "rows", len(df), flush=True)
        cand = pd.concat(cand, ignore_index=True); obs = pd.concat(obs, ignore_index=True)
        cand.to_parquet(OUT / f"candidates_{part}{suffix}.parquet", index=False)
        obs.to_parquet(OUT / f"obs_{part}{suffix}.parquet", index=False)
        log[part] = dict(trips=len(ids), n_obs=len(obs), n_rows=len(cand),
                         miss_rate=float(1 - obs["in_cand"].mean()),
                         seconds=round(time.time() - t0, 1))
        print(part, log[part], flush=True)
    if not limit:
        Path("results/data/tables_run.json").write_text(json.dumps(log, indent=2))