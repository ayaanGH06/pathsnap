"""A1-A3: inventory, data quality and road-network checks.
Writes results/data/inventory.csv, quality.csv, network.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from data_processing.loader import list_trips, load_trip, get_data_root

OUT = Path("results/data")
OUT.mkdir(parents=True, exist_ok=True)


def haversine_m(lon1, lat1, lon2, lat2):
    """Approximate metres between lon/lat points (for QUALITY CHECKS only)."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


inv_rows, q_rows, n_rows = [], [], []
root = get_data_root()

for part in ["train", "student_test"]:
    for tid in list_trips(part):
        tdir = root / part / tid
        track, nodes, arcs, route = load_trip(tid, part)

        # ---- A1 inventory
        files = sorted(p.name for p in tdir.iterdir())
        inv_rows.append(dict(
            trip_id=tid, partition=part, n_files=len(files),
            n_obs=len(track), n_nodes=len(nodes), n_arcs=len(arcs),
            route_len=len(route) if route is not None else np.nan,
            track_bytes=(tdir / f"{tid}.track").stat().st_size,
            nodes_bytes=(tdir / f"{tid}.nodes").stat().st_size,
            arcs_bytes=(tdir / f"{tid}.arcs").stat().st_size,
            route_bytes=(tdir / f"{tid}.route").stat().st_size,
        ))

        # ---- A2 quality
        dt = np.diff(track["t"].to_numpy())
        step_m = haversine_m(track["lon"].values[:-1], track["lat"].values[:-1],
                             track["lon"].values[1:], track["lat"].values[1:])
        q_rows.append(dict(
            trip_id=tid, partition=part,
            n_nan=int(track.isna().sum().sum()),
            is_sorted=bool((dt > 0).all()),
            n_dup_t=int((dt == 0).sum()),
            pct_1s_steps=float((dt == 1).mean() * 100),
            max_gap_s=float(dt.max()) if len(dt) else np.nan,
            n_gaps_gt1s=int((dt > 1).sum()),
            lon_min=track["lon"].min(), lon_max=track["lon"].max(),
            lat_min=track["lat"].min(), lat_max=track["lat"].max(),
            max_step_m=float(step_m.max()) if len(step_m) else np.nan,
            median_step_m=float(np.median(step_m)) if len(step_m) else np.nan,
        ))

        # ---- A3 network
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(arcs[["from_node", "to_node"]].itertuples(index=False, name=None))
        used = set(arcs["from_node"]) | set(arcs["to_node"])
        comps = list(nx.connected_components(G))
        arc_pairs = set(zip(arcs["from_node"], arcs["to_node"]))
        n_rev = sum((b, a) in arc_pairs for a, b in arc_pairs)
        n_rows.append(dict(
            trip_id=tid, partition=part, n_nodes=len(nodes), n_arcs=len(arcs),
            n_isolated_nodes=len(nodes) - len(used),
            n_components=len(comps),
            largest_component=max(len(c) for c in comps),
            pct_arcs_with_reverse=100 * n_rev / len(arc_pairs),
        ))
        print(part, tid, "done", flush=True)

pd.DataFrame(inv_rows).to_csv(OUT / "inventory.csv", index=False)
pd.DataFrame(q_rows).to_csv(OUT / "quality.csv", index=False)
pd.DataFrame(n_rows).to_csv(OUT / "network.csv", index=False)
print("saved to", OUT.resolve())