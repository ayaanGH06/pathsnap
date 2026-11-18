"""Tests for hmm/decoder.py (C3-C7): transitions, recovery, repeat arcs, coverage.

Run from the repo root:  python -m pytest hmm/test_decoder.py -v
All networks are tiny synthetic ones laid out in metres and converted to lon/lat,
so expected answers are known by hand. These are NOT project results.
"""
import numpy as np
import pandas as pd
from pyproj import Transformer

from data_processing.geometry import TripProjector
from hmm import decoder as dec

CFG = {**dec.DEFAULT_CONFIG, "unreachable_penalty": None, "fill_missing": "none"}


def world(node_xy, arcs, obs_xy, t):
    """Build (nodes, arcs, track) from metre coordinates around a fixed origin."""
    lon0, lat0 = 8.6, 49.8
    proj = TripProjector(lon0, lat0)
    inv = Transformer.from_crs(f"EPSG:{proj.epsg}", "EPSG:4326", always_xy=True)
    ox, oy = proj.xy(lon0, lat0)[0]
    def ll(pts):
        pts = np.asarray(pts, float)
        lon, lat = inv.transform(pts[:, 0] + ox, pts[:, 1] + oy)
        return np.asarray(lon), np.asarray(lat)
    nl, na = ll(node_xy)
    ol, oa = ll(obs_xy)
    nodes = pd.DataFrame({"lon": nl, "lat": na})
    arcs = pd.DataFrame(arcs, columns=["from_node", "to_node"])
    track = pd.DataFrame({"lon": ol, "lat": oa, "t": np.asarray(t, float)})
    # tracks are projected with the track mean, so offsets are tiny but tests
    # only rely on relative distances
    return nodes, arcs, track


def table(rows):
    """rows: (obs_idx, arc_id, score, proj_frac)"""
    df = pd.DataFrame(rows, columns=["obs_idx", "arc_id", "score", "proj_frac"])
    df["dist_to_arc"] = 5.0
    return df


def line_world():
    # 0 --a0--> 1 --a1--> 2 --a2--> 3 along x; a3: 1->4 dead-end spur; reverse twins a4: 1->0
    node_xy = [(0, 0), (100, 0), (200, 0), (300, 0), (100, 60)]
    arcs = [(0, 1), (1, 2), (2, 3), (1, 4), (1, 0)]
    return node_xy, arcs


def test_network_distance_beats_pointwise_best():
    node_xy, arcs = line_world()
    nodes, arcs, track = world(node_xy, arcs, [(50, 0), (110, 2), (160, 0)], [0, 1, 2])
    c = table([(0, 0, .9, .5), (1, 1, .4, .1), (1, 3, .6, .05), (2, 1, .9, .6)])
    # pointwise best at obs 1 is the spur (arc 3); the spur is a dead end so
    # the route 0 -> 3 -> 1 does not exist and the decoder must pick arc 1.
    assert dec.decode(c, nodes, arcs, CFG, track) == [0, 1, 1]


def test_repeated_arc_is_kept_not_collapsed():
    # square loop 0->1->2->3->0, one arc per side (100 m). A vehicle goes around and
    # comes back onto arc 0: the arc must appear twice in the predicted sequence.
    node_xy = [(0, 0), (100, 0), (100, 100), (0, 100)]
    arcs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    obs = [(50, 0), (100, 50), (50, 100), (0, 50), (50, 0)]
    nodes, arcs, track = world(node_xy, arcs, obs, [0, 8, 16, 24, 32])
    c = table([(i, a, .9, .5) for i, a in enumerate([0, 1, 2, 3, 0])])
    cfg = {**CFG, "beta_m": 20.0}
    assert dec.decode(c, nodes, arcs, cfg, track) == [0, 1, 2, 3, 0]


def test_same_arc_loop_route_distance_is_finite():
    node_xy = [(0, 0), (100, 0), (100, 100), (0, 100)]
    arcs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    nodes, arcs, track = world(node_xy, arcs, [(80, 0), (20, 0)], [0, 40])
    c = table([(0, 0, .9, .8), (1, 0, .9, .2)])
    P = dec.prepare_trip(c, nodes, arcs, track, CFG)
    D, _, _, _ = dec.route_distances(P, 1, CFG)
    # 20 m to the end of arc 0, 300 m round the loop... minus: head(1)->tail(0) = 300, plus 20 m
    assert abs(D[0, 0] - (20 + 300 + 20)) < 1.0


def test_zero_candidate_obs_bridged_and_reported_not_dropped():
    node_xy, arcs = line_world()
    nodes, arcs, track = world(node_xy, arcs, [(50, 0), (100, 0), (150, 0)], [0, 1, 2])
    c = table([(0, 0, .9, .5), (2, 1, .9, .5)])           # obs 1 has no candidate
    full, st = dec.decode_detailed(c, nodes, arcs, CFG, track)
    assert list(full) == [0, -1, 1]                       # absent stays visible
    assert st["n_no_candidates"] == 1 and not st["filled"].any()
    cfg = {**CFG, "fill_missing": "carry"}
    full, st = dec.decode_detailed(c, nodes, arcs, cfg, track)
    assert list(full) == [0, 0, 1] and st["filled"].tolist() == [False, True, False]


def test_fill_only_counts_when_it_matches_truth():
    # carry-forward fills obs 1 with arc 0 but truth is arc 1: must count wrong
    full = np.array([0, -1, 1])
    out, filled = dec.fill_missing(full, np.array([0., 1., 2.]), "carry", 30.0)
    assert out.tolist() == [0, 0, 1] and filled.tolist() == [False, True, False]
    ref = np.array([0, 1, 1])
    assert int((out == ref).sum()) == 2 and int((filled & (out == ref)).sum()) == 0
    # beyond the time limit it stays absent
    out, filled = dec.fill_missing(full, np.array([0., 100., 200.]), "carry", 30.0)
    assert out.tolist() == [0, -1, 1] and not filled.any()


def test_long_gap_restarts_chain():
    node_xy, arcs = line_world()
    # second observation is 1000 s later and 'teleports' to an unrelated place
    nodes, arcs, track = world(node_xy, arcs, [(50, 0), (250, 0)], [0, 1000])
    c = table([(0, 0, .9, .5), (1, 2, .9, .5)])
    full, st = dec.decode_detailed(c, nodes, arcs, CFG, track)
    assert list(full) == [0, 2] and st["n_gap_restarts"] == 1


def test_disconnected_pair_restarts_or_penalises():
    # two separate roads, no path between them, close in time
    node_xy = [(0, 0), (100, 0), (500, 0), (600, 0)]
    arcs = [(0, 1), (2, 3)]
    nodes, arcs, track = world(node_xy, arcs, [(50, 0), (550, 0)], [0, 5])
    c = table([(0, 0, .9, .5), (1, 1, .9, .5)])
    full, st = dec.decode_detailed(c, nodes, arcs, CFG, track)       # forbid + restart
    assert list(full) == [0, 1] and st["n_dead_restarts"] == 1
    full, st = dec.decode_detailed(c, nodes, arcs, {**CFG, "unreachable_penalty": 20.0}, track)
    assert list(full) == [0, 1] and st["n_penalised_jumps"] == 1


def test_uturn_penalty_prefers_staying_on_arc():
    node_xy, arcs = line_world()
    nodes, arcs, track = world(node_xy, arcs, [(60, 0), (62, 0)], [0, 1])
    # arc 4 is arc 0's reverse twin; equal emissions, so the U-turn penalty decides
    c = table([(0, 0, .9, .6), (1, 4, .9, .38), (1, 0, .9, .62)])
    assert dec.decode(c, nodes, arcs, CFG, track) == [0, 0]


def test_prunes_to_max_candidates_by_score_and_counts_it():
    node_xy, arcs = line_world()
    nodes, arcs, track = world(node_xy, arcs, [(50, 0)], [0])
    rows = [(0, 0, .5 + i * 1e-4, .5) for i in range(25)]
    c = table(rows)
    c["arc_id"] = [0, 1, 2, 3, 4] * 5
    P = dec.prepare_trip(c, nodes, arcs, track, CFG)
    assert len(P["sets"][0]) == 20 and P["n_pruned"] == 5


def test_every_observation_has_an_answer_slot():
    node_xy, arcs = line_world()
    nodes, arcs, track = world(node_xy, arcs, [(10, 0), (50, 0), (90, 0), (120, 0)], [0, 1, 2, 3])
    c = table([(0, 0, .9, .1), (3, 1, .9, .2)])
    full, _ = dec.decode_detailed(c, nodes, arcs, CFG, track)
    assert len(full) == 4          # full observation denominator preserved


def test_decode_without_track_matches_with_track_when_features_given():
    node_xy, arcs = line_world()
    nodes, arcs, track = world(node_xy, arcs, [(10, 0), (50, 0), (90, 0), (150, 0)], [0, 1, 2, 3])
    c = table([(0, 0, .9, .1), (1, 0, .9, .5), (2, 0, .9, .9), (3, 1, .9, .5)])
    c["dt_prev"] = 1.0
    c["speed"] = [0.0, 40.0, 40.0, 60.0]
    assert dec.decode(c, nodes, arcs, CFG) == dec.decode(c, nodes, arcs, CFG, track) == [0, 0, 0, 1]


def test_leading_trailing_and_all_zero_candidate_observations():
    node_xy, arcs = line_world()
    nodes, arcs, track = world(node_xy, arcs, [(10, 0), (40, 0), (60, 0), (150, 0), (190, 0)],
                               [0, 1, 2, 3, 4])
    c = table([(1, 0, .9, .4), (2, 0, .9, .6)])        # obs 0, 3, 4 have none
    c["dt_prev"], c["speed"] = 1.0, 20.0
    full, st = dec.decode_detailed(c, nodes, arcs, CFG, track)
    assert list(full) == [-1, 0, 0, -1, -1] and st["n_no_candidates"] == 3
    assert not st["n_obs_inferred"]
    # n_obs alone also keeps the trailing slots; with neither, trailing ones are not seen
    full, st = dec.decode_detailed(c, nodes, arcs, CFG, None, n_obs=5)
    assert len(full) == 5 and full[0] == -1 and full[4] == -1
    full, st = dec.decode_detailed(c, nodes, arcs, CFG)
    assert len(full) == 3 and st["n_obs_inferred"]       # documented limitation
    # no candidates at all: every slot is -1, nothing raises
    empty = c.iloc[0:0]
    full, st = dec.decode_detailed(empty, nodes, arcs, CFG, track)
    assert list(full) == [-1] * 5 and st["n_no_candidates"] == 5
    assert dec.decode(empty, nodes, arcs, CFG, track) == []


def test_no_track_without_timing_columns_fails_clearly():
    import pytest
    node_xy, arcs = line_world()
    nodes, arcs, track = world(node_xy, arcs, [(10, 0), (40, 0)], [0, 1])
    c = table([(0, 0, .9, .1), (1, 0, .9, .4)])
    with pytest.raises(ValueError, match="dt_prev and speed"):
        dec.decode(c, nodes, arcs, CFG)
