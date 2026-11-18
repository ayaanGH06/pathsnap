"""A5: projection and distance helpers (shared by A, B, C).

Rule: all distances are in metres in a per-trip UTM projection chosen from the
trip's own track centroid. Never use degrees as distances.
Only the track is used to pick the zone, so this also works at inference time.
"""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


class TripProjector:
    """Per-trip lon/lat -> metres."""

    def __init__(self, lon_center: float, lat_center: float):
        self.epsg = utm_epsg(lon_center, lat_center)
        self._tr = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)

    @classmethod
    def from_track(cls, track):
        return cls(float(track["lon"].mean()), float(track["lat"].mean()))

    def xy(self, lon, lat):
        x, y = self._tr.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.c_[x, y]


def arc_endpoints(nodes_xy, arcs):
    """Return A, B (M,2) start/end points for every arc, in arc_id order."""
    return nodes_xy[arcs["from_node"].values], nodes_xy[arcs["to_node"].values]


def point_seg_dist(P, A, B):
    """Distance (m) from points P (N,2) to segments A-B (M,2) -> (N,M).
    Use only for small M. For a whole network use a spatial index (A7)."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(2))


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) from metre offsets."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def step_features(xy, t):
    """Per-step distance, dt and speed using the REAL dt (not assumed 1 s).
    Returns arrays of length N-1."""
    d = np.hypot(*np.diff(xy, axis=0).T)
    dt = np.diff(np.asarray(t, float))
    speed = d / np.maximum(dt, 1e-6)
    return d, dt, speed