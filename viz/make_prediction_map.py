"""C10: observed GPS vs HMM-predicted route vs reference, with direction and errors shown.

Usage (repo root):
  PATHSNAP_DATA_ROOT="/path/to/Student folder" python viz/make_prediction_map.py <trip_id> <score_name> [out_dir]
  (add  --png  to also render two screenshots with headless Chrome: overview + zoom on
   the biggest run of wrong predictions; needs google-chrome and internet for map tiles)

Needs data/processed/candidates_student_test.parquet, data/processed/scores_<score_name>_student_test.parquet
(hmm/score_candidates.py or B's artifact scores) and hmm/hmm_config.json.
The HTML and PNGs contain raw GPS coordinates: they are local files, not for commit or upload.

Reading the map (every arc has a direction, small triangles point along travel):
  red line + red triangle   reference route arc, correct direction
  green                     predicted arc equal to the reference arc for that observation
  magenta triangle          predicted arc is the REVERSE TWIN of the reference arc (right road, wrong direction)
  orange                    predicted arc not on the reference route / other wrong arc
  black ring                GPS observation whose predicted arc differs from the reference arc
  blue dots                 GPS (every 5th)
Reverse-twin arcs share a geometry with the reference arc, so they overlap the red line: only the
triangle direction and the magenta colour separate them.
"""
import math
import subprocess
import sys
from pathlib import Path
import folium
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_processing.loader import load_trip  # noqa: E402
from features.build_tables import reference_arcs  # noqa: E402
from hmm import decoder as dec  # noqa: E402

PART = "student_test"


def predict(trip, score_name):
    cand = pd.read_parquet(ROOT / "data/processed/candidates_student_test.parquet")
    sc = pd.read_parquet(ROOT / f"data/processed/scores_{score_name}_student_test.parquet")
    assert (sc["arc_id"].values == cand["arc_id"].values).all()
    cand["score"] = sc["score"].values
    c = cand[cand["trip_id"] == trip]
    track, nodes, arcs, route = load_trip(trip, PART)
    cfg = dec.load_config(str(ROOT / "hmm/hmm_config.json"))
    full, stats = dec.decode_detailed(c, nodes, arcs, cfg, track)
    ref = reference_arcs(track, nodes, arcs, route)   # evaluation only, never used by the decoder
    top = c.sort_values(["obs_idx", "score"], ascending=[True, False]).drop_duplicates("obs_idx")
    clf = np.full(len(track), -1); clf[top["obs_idx"].values] = top["arc_id"].values
    return track, nodes, arcs, route, full, clf, ref, stats


def arrow(m, nodes, arcs, a, color, size=14):
    f, t = arcs.iloc[a]
    (x0, y0), (x1, y1) = (nodes.iloc[f].lon, nodes.iloc[f].lat), (nodes.iloc[t].lon, nodes.iloc[t].lat)
    bearing = math.degrees(math.atan2((x1 - x0) * math.cos(math.radians(y0)), y1 - y0)) % 360
    html = (f'<div style="font-size:{size}px;color:{color};transform:rotate({bearing:.0f}deg);'
            f'text-shadow:0 0 2px #fff;line-height:{size}px;">&#9650;</div>')
    folium.Marker([(y0 + y1) / 2, (x0 + x1) / 2],
                  icon=folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size // 2, size // 2)),
                  tooltip=f"arc {a} ({f}->{t})").add_to(m)


def line(g, nodes, arcs, a, color, w, tip):
    f, t = arcs.iloc[a]
    folium.PolyLine([[nodes.iloc[f].lat, nodes.iloc[f].lon], [nodes.iloc[t].lat, nodes.iloc[t].lon]],
                    color=color, weight=w, opacity=0.8, tooltip=tip).add_to(g)


def worst_run(wrong):
    """(start, end) of the longest run of consecutive wrong observations."""
    best, s = (0, 0), None
    for i, w in enumerate(list(wrong) + [False]):
        if w and s is None:
            s = i
        if not w and s is not None:
            best = max(best, (i - s, s, i - 1)); s = None
    return best[1], best[2]


def build(trip, score_name, out, view="overview"):
    track, nodes, arcs, route, full, clf, ref, stats = predict(trip, score_name)
    frm, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    wrong = full != ref
    twin = wrong & (full >= 0) & (frm[full] == to[ref]) & (to[full] == frm[ref])
    n, nt, nw = len(ref), int(twin.sum()), int(wrong.sum())
    ref_arcs = sorted(set(map(int, route)))
    m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=15, control_scale=True)
    g = folium.FeatureGroup(name=f"Reference route ({len(ref_arcs)} arcs)")
    for a in ref_arcs:
        line(g, nodes, arcs, a, "#d62728", 7, f"ref arc {a}")
    g.add_to(m)
    right = sorted(set(map(int, full[(~wrong) & (full >= 0)])))
    g = folium.FeatureGroup(name="Predicted = reference arc")
    for a in right:
        line(g, nodes, arcs, a, "#2ca02c", 3, f"correct arc {a}")
    g.add_to(m)
    tw = sorted(set(map(int, full[twin])))
    oth = sorted(set(map(int, full[wrong & ~twin & (full >= 0)])) - set(tw))
    g = folium.FeatureGroup(name=f"Predicted reverse twin of reference ({len(tw)} arcs)")
    for a in tw:
        line(g, nodes, arcs, a, "#c51b8a", 3, f"reverse-twin arc {a}")
    g.add_to(m)
    g = folium.FeatureGroup(name=f"Predicted other wrong arc ({len(oth)} arcs)")
    for a in oth:
        line(g, nodes, arcs, a, "#ff7f0e", 4, f"wrong arc {a}")
    g.add_to(m)
    g = folium.FeatureGroup(name="Direction triangles")
    for a in ref_arcs:
        arrow(g, nodes, arcs, a, "#d62728")
    for a in tw:
        arrow(g, nodes, arcs, a, "#c51b8a", 12)
    g.add_to(m)
    g = folium.FeatureGroup(name=f"Wrong observations ({nw})")
    for i in np.flatnonzero(wrong):
        if i % 3 == 0 or nw < 300:
            kind = "reverse twin" if twin[i] else ("no prediction" if full[i] < 0 else "other arc")
            folium.CircleMarker([track.lat.iloc[i], track.lon.iloc[i]], radius=5, color="black",
                                weight=1, fill=False, tooltip=f"obs {i}: {kind}").add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup(name="GPS (every 5th)")
    for i in range(0, n, 5):
        folium.CircleMarker([track.lat.iloc[i], track.lon.iloc[i]], radius=2, color="#1f77b4",
                            weight=1, fill=True).add_to(g)
    g.add_to(m)
    folium.LayerControl().add_to(m)
    legend = (f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:#fff;padding:8px 10px;'
              f'border:1px solid #999;font:12px sans-serif;max-width:360px">'
              f'<b>student_test trip {trip}: HMM vs reference</b> (dev run, scores: {score_name})<br>'
              f'wrong {nw} of {n} observations ({100 * nw / n:.1f}%); reverse-twin {nt}<br>'
              f'<span style="color:#d62728">&#9473;</span> reference + direction triangle &nbsp;'
              f'<span style="color:#2ca02c">&#9473;</span> correct &nbsp;'
              f'<span style="color:#c51b8a">&#9473;</span> reverse twin (wrong direction) &nbsp;'
              f'<span style="color:#ff7f0e">&#9473;</span> other wrong<br>'
              f'Reverse twins overlap the red line: look at triangle direction and colour. '
              f'&#9711; wrong GPS point, <span style="color:#1f77b4">&#9679;</span> GPS</div>')
    m.get_root().html.add_child(folium.Element(legend))
    south, north, west, east = track.lat.min(), track.lat.max(), track.lon.min(), track.lon.max()
    out.mkdir(parents=True, exist_ok=True)
    if view == "zoom_worst_run" and nw:
        s0, e0 = worst_run(wrong)
        sl = slice(max(s0 - 15, 0), min(e0 + 15, n - 1) + 1)
        bounds = [[track.lat.iloc[sl].min(), track.lon.iloc[sl].min()],
                  [track.lat.iloc[sl].max(), track.lon.iloc[sl].max()]]
    else:
        bounds = [[south, west], [north, east]]
    m.fit_bounds(bounds, max_zoom=18)
    p = out / f"pred_map_{trip}_{view}.html"
    m.save(str(p))
    return p, dict(n=n, wrong=nw, twin=nt)


def screenshot(html, png):
    subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
                    "--window-size=1400,900", "--virtual-time-budget=9000",
                    f"--screenshot={png}", f"file://{html}"], check=True, capture_output=True, timeout=60)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--png"]
    out = Path(args[2]) if len(args) > 2 else Path("/tmp/pathsnap_maps")
    for view in ("overview", "zoom_worst_run"):
        p, info = build(args[0], args[1], out, view)
        print(info, "saved", p)
        if "--png" in sys.argv:
            screenshot(p.resolve(), p.with_suffix(".png").resolve())
            print("saved", p.with_suffix(".png"))
