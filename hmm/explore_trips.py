"""Task 1 sanity check: load two student_test trips and inspect them.

This is member C's first task: get familiar with the raw data files before
building the HMM. It loads a trip's four files, checks them against the
base document, projects lon/lat to local metres, and saves plots and a
short findings report to results/hmm/.

It deliberately does NOT use data_processing/loader.py (owned by member A).
Reading is simple: every file is whitespace-separated with no header.

Usage:
    python hmm/explore_trips.py --data-root data --trips 00000052 00000007

Expected layout:
    data/student_test/<trip_id>/<trip_id>.track   lon, lat, t (seconds, ~1 Hz)
    data/student_test/<trip_id>/<trip_id>.nodes   lon, lat   (row index = node id, 0-based)
    data/student_test/<trip_id>/<trip_id>.arcs    from_node_id, to_node_id (row index = arc id)
    data/student_test/<trip_id>/<trip_id>.route   reference arc ids in travel order

Raw data stays local (data/ is git-ignored). Only this script and the
findings in results/hmm/ are committed.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only save PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# A trip is one folder with these four files (base document, Section 1).
FILE_SPECS = {
    "track": 3,  # lon, lat, t
    "nodes": 2,  # lon, lat
    "arcs": 2,   # from_node_id, to_node_id
    "route": 1,  # arc id per line
}

# Expected row counts from the team brief, so we can confirm the files are
# the ones we think they are.
EXPECTED_COUNTS = {
    "00000052": {"track": 262, "nodes": 1770, "arcs": 3577, "route": 95},
    "00000007": {"track": 1354, "nodes": 2061, "arcs": 4369, "route": 123},
}

EARTH_RADIUS_M = 6_371_000.0


def load_trip(trip_dir: Path, trip_id: str) -> dict:
    """Read a trip's four files. Whitespace-separated, no header."""
    data = {}
    for kind, ncols in FILE_SPECS.items():
        path = trip_dir / f"{trip_id}.{kind}"
        df = pd.read_csv(path, sep=r"\s+", header=None)
        if df.shape[1] != ncols:
            raise ValueError(f"{path} has {df.shape[1]} columns, expected {ncols}")
        data[kind] = df
    data["track"].columns = ["lon", "lat", "t"]
    data["nodes"].columns = ["lon", "lat"]
    data["arcs"].columns = ["from_node", "to_node"]
    data["route"].columns = ["arc_id"]
    return data


def to_metres(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple:
    """Project lon/lat to local metres around a reference point.

    Simple equirectangular approximation: good for a single trip because
    every trip covers only a small area. Each trip needs its own reference
    point - the two trips here are in different parts of the world, so one
    shared UTM zone (for example EPSG:32643) cannot cover both.
    """
    lat0_rad = np.radians(lat0)
    x = EARTH_RADIUS_M * np.radians(lon - lon0) * np.cos(lat0_rad)
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return x, y


def check_trip(trip_id: str, data: dict) -> dict:
    """Run the sanity checks from the task brief and return the findings."""
    track, nodes, arcs, route = data["track"], data["nodes"], data["arcs"], data["route"]
    findings = {"trip_id": trip_id}

    # 1. Row counts, compared with the brief where we have expected values.
    counts = {kind: len(df) for kind, df in data.items()}
    findings["counts"] = counts
    expected = EXPECTED_COUNTS.get(trip_id, {})
    findings["counts_match_brief"] = all(
        counts[k] == v for k, v in expected.items()
    )

    # 2. Arc endpoints must be valid node ids (0-based row index into nodes).
    max_node_id = int(max(arcs["from_node"].max(), arcs["to_node"].max()))
    min_node_id = int(min(arcs["from_node"].min(), arcs["to_node"].min()))
    findings["node_id_range_in_arcs"] = [min_node_id, max_node_id]
    findings["arc_endpoints_valid"] = min_node_id >= 0 and max_node_id < len(nodes)

    # 3. Route arc ids must be valid (0-based row index into arcs).
    findings["route_arc_id_range"] = [int(route["arc_id"].min()), int(route["arc_id"].max())]
    findings["route_arc_ids_valid"] = bool(
        route["arc_id"].min() >= 0 and route["arc_id"].max() < len(arcs)
    )

    # 4. Timestamps should be non-decreasing, nominally 1 second apart.
    t = track["t"].to_numpy()
    steps = np.diff(t)
    findings["timestamps_non_decreasing"] = bool((steps >= 0).all())
    findings["timestamp_step_counts"] = {
        str(int(step)): int((steps == step).sum()) for step in np.unique(steps)
    }

    # 5. Consecutive reference arcs should connect: the to-node of one arc
    #    should be the from-node of the next.
    route_arcs = arcs.iloc[route["arc_id"].to_numpy()]
    connected = (route_arcs["to_node"].to_numpy()[:-1] == route_arcs["from_node"].to_numpy()[1:])
    findings["route_consecutive_arcs_connected"] = bool(connected.all())
    findings["route_disconnections"] = int((~connected).sum())

    # 6. Geographic extent, so the projection choice can be checked.
    findings["track_lon_range"] = [float(track["lon"].min()), float(track["lon"].max())]
    findings["track_lat_range"] = [float(track["lat"].min()), float(track["lat"].max())]
    findings["nodes_lon_range"] = [float(nodes["lon"].min()), float(nodes["lon"].max())]
    findings["nodes_lat_range"] = [float(nodes["lat"].min()), float(nodes["lat"].max())]

    # 7. Route arcs should be inside the network's area (sanity check that
    #    the route references the same network).
    route_nodes = nodes.to_numpy()[
        np.unique(route_arcs[["from_node", "to_node"]].to_numpy().ravel())
    ]
    findings["route_within_network_extent"] = bool(
        route_nodes[:, 0].min() >= nodes["lon"].min()
        and route_nodes[:, 0].max() <= nodes["lon"].max()
        and route_nodes[:, 1].min() >= nodes["lat"].min()
        and route_nodes[:, 1].max() <= nodes["lat"].max()
    )
    return findings


def plot_trip(trip_id: str, data: dict, out_path: Path) -> None:
    """Plot the road network, GPS points and reference route in metres."""
    track, nodes, arcs, route = data["track"], data["nodes"], data["arcs"], data["route"]

    # Per-trip reference point: the middle of the GPS track.
    lon0 = float(track["lon"].mean())
    lat0 = float(track["lat"].mean())
    nx, ny = to_metres(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), lon0, lat0)
    tx, ty = to_metres(track["lon"].to_numpy(), track["lat"].to_numpy(), lon0, lat0)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Road network: one straight segment per arc (arcs are straight lines
    # between nodes, per the task brief).
    from_xy = np.column_stack([nx[arcs["from_node"]], ny[arcs["from_node"]]])
    to_xy = np.column_stack([nx[arcs["to_node"]], ny[arcs["to_node"]]])
    segments = np.stack([from_xy, to_xy], axis=1)
    ax.add_collection(
        matplotlib.collections.LineCollection(segments, colors="lightgrey", linewidths=0.4)
    )

    # GPS observations.
    ax.plot(tx, ty, "r.", markersize=4, label="GPS track")
    ax.plot(tx[0], ty[0], "go", markersize=8, label="start")

    # Reference route, drawn through its arcs in travel order, on top of
    # the GPS points so it stays visible.
    route_arcs = arcs.iloc[route["arc_id"].to_numpy()]
    rx = np.concatenate([[nx[f], nx[t]] for f, t in zip(route_arcs["from_node"], route_arcs["to_node"])])
    ry = np.concatenate([[ny[f], ny[t]] for f, t in zip(route_arcs["from_node"], route_arcs["to_node"])])
    ax.plot(rx, ry, "b-", linewidth=1.2, label="reference route")

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x (m, local projection)")
    ax.set_ylabel("y (m, local projection)")
    ax.set_title(f"Trip {trip_id}: network, GPS track and reference route")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_findings(all_findings: list, out_path: Path) -> None:
    """Write a short human-readable summary next to the JSON log."""
    lines = ["# Task 1 findings: two-trip loading sanity check", ""]
    for f in all_findings:
        lines.append(f"## Trip {f['trip_id']}")
        c = f["counts"]
        lines.append(
            f"- Counts: {c['track']} GPS points, {c['nodes']} nodes, "
            f"{c['arcs']} arcs, {c['route']} reference arcs "
            f"(matches brief: {f['counts_match_brief']})"
        )
        lines.append(f"- Arc endpoints reference valid node ids: {f['arc_endpoints_valid']}")
        lines.append(f"- Route arc ids valid: {f['route_arc_ids_valid']}")
        lines.append(
            f"- Timestamps non-decreasing: {f['timestamps_non_decreasing']}; "
            f"step sizes in seconds: {f['timestamp_step_counts']}"
        )
        lines.append(
            f"- Consecutive reference arcs connect: {f['route_consecutive_arcs_connected']} "
            f"({f['route_disconnections']} disconnections)"
        )
        lines.append(
            f"- Track lon/lat range: {f['track_lon_range']} x {f['track_lat_range']}"
        )
        lines.append("")
    lines += [
        "## Notes for the team",
        "- The two trips are in different parts of the world (see lon/lat ranges above),",
        "  so one shared projected CRS such as EPSG:32643 cannot cover both. This script",
        "  uses a per-trip local metre projection. Flag to A for the shared projection (A5).",
        "- Timestamp step sizes above show any sampling gaps for A's data-quality notes.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="folder containing student_test/")
    parser.add_argument("--trips", nargs="+", default=["00000052", "00000007"])
    parser.add_argument("--out", default="results/hmm", help="output folder for logs and plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_findings = []
    for trip_id in args.trips:
        trip_dir = Path(args.data_root) / "student_test" / trip_id
        data = load_trip(trip_dir, trip_id)
        findings = check_trip(trip_id, data)
        all_findings.append(findings)
        plot_path = out_dir / f"task1_trip_{trip_id}.png"
        plot_trip(trip_id, data, plot_path)
        print(json.dumps(findings, indent=2))
        print(f"saved plot -> {plot_path}")

    (out_dir / "task1_findings.json").write_text(json.dumps(all_findings, indent=2) + "\n")
    write_findings(all_findings, out_dir / "task1_findings.md")
    print(f"saved findings -> {out_dir}/task1_findings.json and task1_findings.md")


if __name__ == "__main__":
    main()
