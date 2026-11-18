"""First Folium map: one student_test trip with GPS, reference route, limited network.

Usage (from repo root):
    PATHSNAP_DATA_ROOT="/path/to/Student folder" python viz/make_map.py [trip_id] [buffer_m] [out.html]

Defaults: trip 00000007, 100 m buffer, results/viz/map_trip_00000007.html.

Notes
- Raw files store (lon, lat). Folium wants (lat, lon), so every coordinate is
  flipped exactly once, in `to_latlon`.
- The reference route is supplied ground truth, not a model prediction.
- The network is limited: only arcs within `buffer_m` of a GPS point are drawn,
  except reference arcs, which are always drawn on their own layer.
"""
import sys
from pathlib import Path

import folium
import numpy as np
import shapely
from shapely import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_processing.loader import load_trip  # noqa: E402

PARTITION = "student_test"


def to_latlon(lon, lat):
    """Folium order: (lat, lon)."""
    return [float(lat), float(lon)]


def limited_arc_ids(track, nodes, arcs, buffer_m):
    """Arc ids within buffer_m of any GPS point (local equirectangular metres)."""
    lat0 = np.deg2rad(track["lat"].mean())
    kx, ky = 111320.0 * np.cos(lat0), 110540.0

    def xy(lon, lat):
        return np.column_stack([np.asarray(lon) * kx, np.asarray(lat) * ky])

    a = xy(nodes["lon"].to_numpy()[arcs["from_node"]], nodes["lat"].to_numpy()[arcs["from_node"]])
    b = xy(nodes["lon"].to_numpy()[arcs["to_node"]], nodes["lat"].to_numpy()[arcs["to_node"]])
    lines = shapely.linestrings(np.stack([a, b], axis=1))
    pts = shapely.points(xy(track["lon"], track["lat"]))
    tree = STRtree(lines)
    _, arc_idx = tree.query(pts, predicate="dwithin", distance=buffer_m)
    return np.unique(arc_idx)


def arc_coords(arc_id, nodes, arcs):
    f, t = arcs.iloc[arc_id]
    return [to_latlon(nodes.iloc[f]["lon"], nodes.iloc[f]["lat"]),
            to_latlon(nodes.iloc[t]["lon"], nodes.iloc[t]["lat"])]


def build_map(trip_id="00000007", buffer_m=100.0):
    track, nodes, arcs, route = load_trip(trip_id, PARTITION)
    near = limited_arc_ids(track, nodes, arcs, buffer_m)
    ref = np.unique(route)

    m = folium.Map(location=to_latlon(track["lon"].mean(), track["lat"].mean()),
                   zoom_start=13, tiles="OpenStreetMap", control_scale=True)

    net = folium.FeatureGroup(name=f"Road network (arcs within {buffer_m:.0f} m of GPS, {len(near)} of {len(arcs)})")
    for a in near:
        folium.PolyLine(arc_coords(a, nodes, arcs), color="#888888", weight=2, opacity=0.7).add_to(net)
    net.add_to(m)

    rl = folium.FeatureGroup(name=f"Reference route ({len(route)} arcs, supplied truth)")
    for i, a in enumerate(route):
        folium.PolyLine(arc_coords(a, nodes, arcs), color="#d62728", weight=5, opacity=0.85,
                        tooltip=f"ref #{i} arc_id {a}").add_to(rl)
    rl.add_to(m)

    gl = folium.FeatureGroup(name=f"GPS points ({len(track)})")
    folium.PolyLine([to_latlon(r.lon, r.lat) for r in track.itertuples()],
                    color="#1f77b4", weight=1, opacity=0.5).add_to(gl)
    for i, r in enumerate(track.itertuples()):
        if i % 5 == 0 or i == len(track) - 1:  # every 5th point keeps the HTML small
            folium.CircleMarker(to_latlon(r.lon, r.lat), radius=2, color="#1f77b4", fill=True,
                                fill_opacity=0.9, weight=1,
                                tooltip=f"obs {i}, t={r.t:.0f}s").add_to(gl)
    gl.add_to(m)

    first, last = track.iloc[0], track.iloc[-1]
    folium.Marker(to_latlon(first.lon, first.lat), tooltip="GPS start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(to_latlon(last.lon, last.lat), tooltip="GPS end",
                  icon=folium.Icon(color="black", icon="stop")).add_to(m)

    south, north = track["lat"].min(), track["lat"].max()
    west, east = track["lon"].min(), track["lon"].max()
    m.fit_bounds([[south, west], [north, east]])

    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
         padding: 10px 12px; border: 1px solid #999; border-radius: 4px; font: 13px sans-serif;">
      <b>student_test trip {trip_id}</b><br>
      <span style="color:#1f77b4">&#9679;</span> GPS points (every 5th)<br>
      <span style="color:#d62728">&#9473;</span> Reference route (truth)<br>
      <span style="color:#888888">&#9473;</span> Road network, {buffer_m:.0f} m around GPS only
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


if __name__ == "__main__":
    trip = sys.argv[1] if len(sys.argv) > 1 else "00000007"
    buf = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "results" / "viz" / f"map_trip_{trip}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_map(trip, buf).save(str(out))
    print("saved", out)
