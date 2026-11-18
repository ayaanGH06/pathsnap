"""C3-C7: real transition and recovery layer, and the decode() wrapper (contract 4.5).

decode(trip_candidates_with_scores, nodes, arcs, hmm_config, track=None)
    -> list of predicted arc_ids, one per GPS observation that has candidates.

Model
-----
State: one directed candidate arc per observation (up to max_candidates kept,
ranked by score; the rest are pruned and counted).
Emission: log of the classifier probability (emission_mode "classifier",
weighted by emission_weight), or a clearly separate distance baseline
(emission_mode "distance", Gaussian on dist_to_arc).
Transition (Newson-Krumm style): the route distance through the directed road
network, using the along-arc positions (proj_frac * arc length) of both
candidates, is compared with the straight-line distance between the two GPS
points:
    log P = -|route_distance - gps_distance| / beta_m
Route distance for arc a at fraction fa to arc b at fraction fb:
    same arc, moving forward:   (fb - fa) * len
    same arc, tiny backward:    0 (within back_tol_m, GPS noise)
    otherwise:                  (1 - fa) * len_a + shortest_path(head_a, tail_b) + fb * len_b
Repeated arcs are allowed: going round a loop back onto the same arc is a valid
route (shortest path head_a -> tail_a), and no repeated arc is removed or merged.
Reverse-twin moves (U-turns) get an extra uturn_penalty.

Recovery policy (all counted in the returned stats):
- observation with zero candidates: not a state; the chain bridges over it using
  the real elapsed time and the straight-line distance between the neighbouring
  observations that do have candidates. It is never dropped silently:
  decode_detailed reports -1 for it.
- time gap longer than gap_restart_s: restart the chain (independent segment,
  uniform start prior); a route over a long gap says nothing about the vehicle.
- candidate pair with no network path within the search bound, or faster than
  max_speed_mps * dt + slack_m: if unreachable_penalty is a number the move is
  allowed with that log penalty (counted as a "penalised jump"); if it is null
  the move is forbidden, and when no path survives the chain restarts there.
Shortest-path search is Dijkstra bounded by min(search_speed_mps * dt + slack_m,
search_cap_m), so cost stays small on big networks (trip 1 has 165,790 arcs).
Cut-offs, penalties and speeds are tunable choices logged in results/hmm/, not
established constants.
"""
import heapq
import json
import numpy as np

from data_processing.geometry import TripProjector
from hmm.viterbi import viterbi_decode, NoFeasiblePathError

DEFAULT_CONFIG = {
    "emission_mode": "classifier",   # "classifier" or "distance" (baseline)
    "emission_weight": 1.0,
    "prob_floor": 1e-6,
    "sigma_m": 7.5,                  # distance baseline only
    "max_candidates": 20,
    "beta_m": 10.0,
    "uturn_penalty": 5.0,
    "unreachable_penalty": 20.0,     # null = forbid and restart
    "gap_restart_s": 60.0,
    "max_speed_mps": 45.0,
    "search_speed_mps": 45.0,
    "slack_m": 60.0,
    "search_cap_m": 3000.0,
    "back_tol_m": 15.0,
    "fill_missing": "carry",         # "none" or "carry": see fill_missing()
    "fill_max_gap_s": 30.0,
}


def load_config(path="hmm/hmm_config.json"):
    cfg = dict(DEFAULT_CONFIG)
    with open(path) as f:
        cfg.update(json.load(f))
    return cfg


class Network:
    """Directed road graph with arc lengths in metres and bounded Dijkstra."""

    def __init__(self, nodes, arcs, projector):
        xy = projector.xy(nodes["lon"].values, nodes["lat"].values)
        self.frm = arcs["from_node"].to_numpy(dtype=np.int64)
        self.to = arcs["to_node"].to_numpy(dtype=np.int64)
        self.length = np.hypot(*(xy[self.to] - xy[self.frm]).T)
        order = np.argsort(self.frm, kind="stable")
        self._order = order
        self._ptr = np.searchsorted(self.frm[order], np.arange(len(nodes) + 1))
        self._cache = {}

    def _dijkstra(self, src, cutoff):
        dist = {src: 0.0}
        heap = [(0.0, src)]
        done = set()
        ptr, order, to, ln = self._ptr, self._order, self.to, self.length
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                a = order[k]
                nd = d + ln[a]
                if nd <= cutoff:
                    v = int(to[a])
                    if nd < dist.get(v, 1e18):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
        return dist

    def reach(self, src, cutoff):
        """dict node -> distance, complete up to `cutoff` metres (cached)."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        if len(self._cache) > 5000:
            self._cache.clear()
        d = self._dijkstra(src, cutoff)
        self._cache[src] = (cutoff, d)
        return d


def prepare_trip(cands, nodes, arcs, track=None, cfg=None, n_obs=None):
    """Turn a scored candidate table for ONE trip into decode-ready structures.

    cands needs obs_idx, arc_id, score, proj_frac and, for the distance baseline,
    dist_to_arc; without `track` it also needs dt_prev and speed (used to
    approximate elapsed time and straight-line distance).
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    if len(cands) == 0:
        raise ValueError("trip has no candidate rows")
    if track is not None:
        proj = TripProjector.from_track(track)
        t = track["t"].to_numpy(float)
        pxy = proj.xy(track["lon"].values, track["lat"].values)
        n_obs = len(track)
    else:
        proj = TripProjector(float(nodes["lon"].mean()), float(nodes["lat"].mean()))
        t = pxy = None
        n_obs = int(n_obs) if n_obs is not None else int(cands["obs_idx"].max()) + 1
    net = Network(nodes, arcs, proj)

    c = cands.sort_values(["obs_idx", "score"], ascending=[True, False], kind="stable")
    c = c.assign(_rank=c.groupby("obs_idx").cumcount())
    n_pruned = int((c["_rank"] >= cfg["max_candidates"]).sum())
    c = c[c["_rank"] < cfg["max_candidates"]]
    obs = c["obs_idx"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, obs[1:] != obs[:-1], True])
    chain = obs[bounds[:-1]]
    sets = [c.iloc[bounds[i]:bounds[i + 1]] for i in range(len(chain))]
    first = c.drop_duplicates("obs_idx").set_index("obs_idx")
    return dict(net=net, chain=chain, sets=sets, t=t, pxy=pxy, n_obs=n_obs,
                n_pruned=n_pruned, n_no_cand=n_obs - len(chain),
                dt_prev=first["dt_prev"] if "dt_prev" in first else None,
                speed=first["speed"] if "speed" in first else None)


def _gap_info(P, p, q):
    """(elapsed seconds, straight-line metres) from observation p to q."""
    if P["t"] is not None:
        return float(P["t"][q] - P["t"][p]), float(np.hypot(*(P["pxy"][q] - P["pxy"][p])))
    span = q - p  # no track: approximate from the features of q
    if P["dt_prev"] is None or P["speed"] is None:
        raise ValueError("without `track`, the candidate table needs dt_prev and speed columns")
    dt = float(P["dt_prev"].loc[q]) * span
    return dt, float(P["speed"].loc[q]) * dt


def route_distances(P, i, cfg):
    """Raw route-distance matrix (K_prev, K_cur) between chain items i-1 and i.

    inf where no path exists within the search bound. It does not depend on
    beta, penalties, or the gap/speed thresholds, so it can be cached while those
    are tuned. Returns (D, uturn_mask, dt, gps_dist).
    """
    cfg = {**DEFAULT_CONFIG, **cfg}
    net = P["net"]
    a, b = P["sets"][i - 1], P["sets"][i]
    dt, dg = _gap_info(P, int(P["chain"][i - 1]), int(P["chain"][i]))
    cutoff = min(cfg["search_speed_mps"] * max(dt, 0.0) + cfg["slack_m"], cfg["search_cap_m"])
    aid, bid = a["arc_id"].to_numpy(), b["arc_id"].to_numpy()
    fa, fb = a["proj_frac"].to_numpy(float), b["proj_frac"].to_numpy(float)
    La, Lb = net.length[aid], net.length[bid]
    head, tail = net.to[aid], net.frm[bid]
    D = np.full((len(aid), len(bid)), np.inf)
    long_gap = dt > cfg["gap_restart_s"]
    if not long_gap:   # a long gap restarts the chain, no distances needed
        for r in range(len(aid)):
            reach = net.reach(int(head[r]), cutoff)
            rest = (1.0 - fa[r]) * La[r]
            for s in range(len(bid)):
                if aid[r] == bid[s]:
                    delta = (fb[s] - fa[r]) * Lb[s]
                    if delta >= -cfg["back_tol_m"]:
                        D[r, s] = max(delta, 0.0)
                        continue
                sp = reach.get(int(tail[s]))
                if sp is not None:
                    D[r, s] = rest + sp + fb[s] * Lb[s]
    uturn = ((net.frm[aid][:, None] == net.to[bid][None, :]) &
             (net.to[aid][:, None] == net.frm[bid][None, :]) &
             (aid[:, None] != bid[None, :]))
    return D, uturn, dt, dg


def log_transition(D, uturn, dt, dg, cfg):
    bad = ~np.isfinite(D) | (D > cfg["max_speed_mps"] * dt + cfg["slack_m"])
    with np.errstate(invalid="ignore"):
        lt = -np.abs(D - dg) / cfg["beta_m"] - cfg["uturn_penalty"] * uturn
    pen = cfg["unreachable_penalty"]
    return np.where(bad, -np.inf if pen is None else -float(pen), lt)


def log_emission(S, cfg):
    if cfg["emission_mode"] == "distance":
        return -0.5 * (S["dist_to_arc"].to_numpy(float) / cfg["sigma_m"]) ** 2
    p = np.clip(S["score"].to_numpy(float), cfg["prob_floor"], 1.0)
    return cfg["emission_weight"] * np.log(p)


def decode_prepared(P, cfg, route_cache=None):
    """Decode a prepared trip. route_cache: optional list of route_distances()
    results (element i-1 is link i) to reuse across parameter settings.
    Returns (arc_id per chain item, stats)."""
    cfg = {**DEFAULT_CONFIG, **cfg}
    sets, chain = P["sets"], P["chain"]
    n = len(chain)
    stats = dict(n_obs=P["n_obs"], n_no_candidates=P["n_no_cand"], n_pruned_rows=P["n_pruned"],
                 n_gap_restarts=0, n_dead_restarts=0, n_penalised_jumps=0)
    links, seg = {}, [0]
    for i in range(1, n):
        links[i] = route_cache[i - 1] if route_cache is not None else route_distances(P, i, cfg)
        if links[i][2] > cfg["gap_restart_s"]:
            seg.append(i)
            stats["n_gap_restarts"] += 1
    seg.append(n)
    out = np.empty(n, dtype=np.int64)
    ems = [log_emission(S, cfg) for S in sets]
    pen = cfg["unreachable_penalty"]

    def run(lo, hi):
        while lo < hi:
            trans = [log_transition(*links[i], cfg) for i in range(lo + 1, hi)]
            try:
                path, _ = viterbi_decode(ems[lo:hi], iter(trans),
                                         max_candidates=cfg["max_candidates"])
            except NoFeasiblePathError as e:
                if not e.step:
                    raise
                stats["n_dead_restarts"] += 1
                run(lo, lo + e.step)      # decode up to the dead end, then restart there
                lo += e.step
                continue
            for k, j in enumerate(path):
                out[lo + k] = sets[lo + k]["arc_id"].iloc[j]
            if pen is not None:
                for k in range(1, len(path)):
                    if trans[k - 1][path[k - 1], path[k]] == -float(pen):
                        stats["n_penalised_jumps"] += 1
            return

    for s0, s1 in zip(seg[:-1], seg[1:]):
        run(s0, s1)
    return out, stats


def fill_missing(full, t, mode, max_gap_s):
    """Fill observations that have no candidate with a neighbouring prediction.

    mode "none": leave -1 (the observation then counts wrong in accuracy).
    mode "carry": copy the arc predicted at the nearest earlier observation, or
    the nearest later one if there is no earlier one, but only if that
    observation is within max_gap_s seconds. Beyond that stays -1.
    A filled arc only counts as correct if it equals the reference arc.
    Returns (new array, boolean mask of filled observations).
    """
    out = full.copy()
    filled = np.zeros(len(full), dtype=bool)
    if mode == "none" or t is None:
        return out, filled
    known = np.flatnonzero(full >= 0)
    if len(known) == 0:
        return out, filled
    for i in np.flatnonzero(full < 0):
        k = np.searchsorted(known, i)
        for j in ((known[k - 1],) if k > 0 else ()) + ((known[k],) if k < len(known) else ()):
            if abs(t[i] - t[j]) <= max_gap_s:
                out[i], filled[i] = full[j], True
                break
    return out, filled


def decode_detailed(trip_candidates_with_scores, nodes, arcs, hmm_config=None, track=None, n_obs=None):
    """Returns (arc_id per observation, stats). An observation with no candidate
    is -1 unless fill_missing filled it; stats["filled"] marks those.

    One slot per GPS observation is only guaranteed when the total is known:
    pass `track` (lon, lat, t for every observation; exact times too) or at least
    `n_obs`. With neither, the length is inferred as last-observation-with-candidates + 1
    (stats["n_obs_inferred"] is True), so trailing observations without candidates
    are not represented, and with no `track` no fill happens.
    """
    cfg = {**DEFAULT_CONFIG, **(hmm_config or {})}
    known = track is not None or n_obs is not None
    total = len(track) if track is not None else n_obs
    if len(trip_candidates_with_scores) == 0:       # every observation lacks a candidate
        n = int(total) if known else 0
        return (np.full(n, -1, dtype=np.int64),
                dict(n_obs=n, n_no_candidates=n, n_pruned_rows=0, n_gap_restarts=0,
                     n_dead_restarts=0, n_penalised_jumps=0, n_obs_inferred=not known,
                     filled=np.zeros(n, dtype=bool)))
    P = prepare_trip(trip_candidates_with_scores, nodes, arcs, track, cfg, n_obs=total)
    arcs_out, stats = decode_prepared(P, cfg)
    full = np.full(P["n_obs"], -1, dtype=np.int64)
    full[P["chain"]] = arcs_out
    full, filled = fill_missing(full, P["t"], cfg["fill_missing"], cfg["fill_max_gap_s"])
    stats["filled"] = filled
    stats["n_obs_inferred"] = not known
    return full, stats


def decode(trip_candidates_with_scores, nodes, arcs, hmm_config=None, track=None, n_obs=None):
    """Contract 4.5: predicted arc_ids in observation order, one per GPS observation
    that has a decoded or filled arc. Observations left at -1 (no candidate and
    not filled) are OMITTED here, so the list can be shorter than the track.
    Use decode_detailed for one slot per GPS observation (-1 = no prediction);
    accuracy over all observations must be computed from that, not from this list."""
    full, _ = decode_detailed(trip_candidates_with_scores, nodes, arcs, hmm_config, track, n_obs)
    return [int(a) for a in full if a >= 0]
