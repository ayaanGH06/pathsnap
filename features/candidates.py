"""A7: candidate arcs near each GPS point, using a spatial index.
All distances in metres in the per-trip UTM projection."""
import numpy as np
import shapely
from shapely import STRtree


def pair_dist(P, A, B):
    """Distance from point P[i] to segment A[i]-B[i] (all (K,2))."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((P - A) * AB).sum(1) / L2, 0, 1)
    proj = A + t[:, None] * AB
    return np.hypot(*(P - proj).T)


def arc_candidates(P, A, B, radius):
    """Return (obs_idx, arc_idx, dist_m) for every arc within `radius` m of each point."""
    lines = shapely.linestrings(np.stack([A, B], axis=1))
    tree = STRtree(lines)
    boxes = shapely.box(P[:, 0] - radius, P[:, 1] - radius,
                        P[:, 0] + radius, P[:, 1] + radius)
    obs_idx, arc_idx = tree.query(boxes)          # bounding-box prefilter
    d = pair_dist(P[obs_idx], A[arc_idx], B[arc_idx])
    keep = d <= radius                            # exact distance filter
    return obs_idx[keep], arc_idx[keep], d[keep]