"""Data loader for PathSnap (contract 4.2).

Only the 'train' and 'student_test' partitions can be loaded here.
The data root comes from the PATHSNAP_DATA_ROOT environment variable,
or from config.DATA_ROOT if the team adds it.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

VALID_PARTITIONS = ("train", "student_test")


def get_data_root() -> Path:
    root = os.environ.get("PATHSNAP_DATA_ROOT")
    if not root:
        try:
            import config
            root = getattr(config, "DATA_ROOT", None)
        except ImportError:
            root = None
    if not root:
        raise RuntimeError(
            "Data root not set. Set the PATHSNAP_DATA_ROOT environment "
            "variable to the folder that contains train/ and student_test/."
        )
    return Path(root)


def _partition_dir(partition: str) -> Path:
    if partition not in VALID_PARTITIONS:
        raise ValueError(f"partition must be one of {VALID_PARTITIONS}, got {partition!r}")
    d = get_data_root() / partition
    if not d.is_dir():
        raise FileNotFoundError(f"Partition folder not found: {d}")
    return d


def list_trips(partition: str) -> list:
    """Trip ids (zero-padded strings) in a partition, sorted."""
    d = _partition_dir(partition)
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def _read_table(path: Path, names: list) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def load_trip(trip_id: str, partition: str, with_route: bool = True):
    """Return (track, nodes, arcs, route).

    track : DataFrame [lon, lat, t]   one row per GPS observation
    nodes : DataFrame [lon, lat]      index = node_id
    arcs  : DataFrame [from_node, to_node]   index = arc_id
    route : np.ndarray of reference arc_ids, or None if with_route=False
            or the file is missing
    """
    trip_id = str(trip_id)
    tdir = _partition_dir(partition) / trip_id
    if not tdir.is_dir():
        raise FileNotFoundError(f"Trip folder not found: {tdir}")

    track = _read_table(tdir / f"{trip_id}.track", ["lon", "lat", "t"])

    nodes = _read_table(tdir / f"{trip_id}.nodes", ["lon", "lat"])
    nodes.index.name = "node_id"

    arcs = _read_table(tdir / f"{trip_id}.arcs", ["from_node", "to_node"]).astype("int64")
    arcs.index.name = "arc_id"

    route = None
    route_path = tdir / f"{trip_id}.route"
    if with_route and route_path.exists():
        route = pd.read_csv(route_path, header=None).iloc[:, 0].to_numpy(dtype=np.int64)

    return track, nodes, arcs, route
