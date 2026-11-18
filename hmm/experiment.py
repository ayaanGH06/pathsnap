"""C8-C9: decoder tuning and the classifier-only vs classifier+HMM comparison.

Steps (all resumable; route distances are cached because they do not depend on
the parameters being tuned):
  python -m hmm.experiment cache  <partition> <score_name> [seconds]
  python -m hmm.experiment tune   <score_name>     # grid on TRAINING trips only
  python -m hmm.experiment final  <score_name> <config.json> # one run on student_test

`score_name` selects data/processed/scores_<score_name>_{train_oof,student_test}.parquet
(see hmm/score_candidates.py). Training scores are out-of-fold, so tuning never
uses a model that saw the trip being decoded. student_test is not used for tuning.
"""
import itertools
import json
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd

from data_processing.loader import list_trips, load_trip
from hmm import decoder as dec

D = "data/processed/"
CACHE = os.environ.get("HMM_CACHE", "/tmp/hmm_cache")
OUT = "results/hmm/"
# cache is built with the most permissive search settings in the grid
CACHE_CFG = {**dec.DEFAULT_CONFIG, "gap_restart_s": 120.0, "search_speed_mps": 45.0,
             "slack_m": 60.0, "search_cap_m": 3000.0}


def load_scores(score_name, partition):
    cand = pd.read_parquet(D + f"candidates_{partition}.parquet")
    suffix = "train_oof" if partition == "train" else "student_test"
    sc = pd.read_parquet(D + f"scores_{score_name}_{suffix}.parquet")
    keys = ["trip_id", "obs_idx", "arc_id"]
    assert len(sc) == len(cand) and (sc[keys].to_numpy() == cand[keys].to_numpy()).all(), \
        "score rows must align one-to-one with the candidate table on trip_id, obs_idx and arc_id"
    cand["score"] = sc["score"].values
    return cand


def cache_path(partition, trip):
    return f"{CACHE}/{partition}_{trip}.pkl"


def cmd_cache(partition, score_name, budget=100):
    os.makedirs(CACHE, exist_ok=True)
    cand = load_scores(score_name, partition)
    obs = pd.read_parquet(D + f"obs_{partition}.parquet")
    t0 = time.time(); done = 0
    for trip in list_trips(partition):
        if os.path.exists(cache_path(partition, trip)):
            continue
        if time.time() - t0 > budget:
            print("budget used; rerun to continue"); return
        track, nodes, arcs, _ = load_trip(trip, partition)
        c = cand[cand["trip_id"] == trip]
        if len(c) == 0:
            pickle.dump(None, open(cache_path(partition, trip), "wb")); continue
        P = dec.prepare_trip(c, nodes, arcs, track, CACHE_CFG)
        links = [dec.route_distances(P, i, CACHE_CFG) for i in range(1, len(P["chain"]))]
        net = P.pop("net")
        P["frm"], P["to"] = net.frm, net.to
        P["ref"] = obs[obs["trip_id"] == trip].sort_values("obs_idx")["ref_arc"].to_numpy()
        # classifier-only prediction over ALL candidates (before pruning)
        top = c.sort_values(["obs_idx", "score"], ascending=[True, False]).drop_duplicates("obs_idx")
        top1 = np.full(P["n_obs"], -1); top1[top["obs_idx"].values] = top["arc_id"].values
        P["top1"] = top1
        # observations whose true arc was a candidate but fell outside the top-20 kept
        r = c.sort_values(["obs_idx", "score"], ascending=[True, False])
        r = r.assign(_rank=r.groupby("obs_idx").cumcount())
        P["n_truth_lost_to_pruning"] = int(((r["_rank"] >= dec.DEFAULT_CONFIG["max_candidates"]) & (r["label"] == 1)).sum())
        pickle.dump((P, links), open(cache_path(partition, trip), "wb"))
        done += 1
        print(partition, trip, f"{time.time() - t0:.0f}s", flush=True)
    print("all cached", partition)


def trip_metrics(pred, P, filled=None, trip=None):
    """Pointwise and sequence metrics for one trip. Missing predictions (-1)
    count as wrong; the denominator is every GPS observation."""
    ref = P["ref"].astype(np.int64)
    out = {"trip": trip, "n_obs": len(ref), "correct": int((pred == ref).sum()),
           "n_no_cand": int(P["n_no_cand"]), "n_filled": 0, "n_filled_correct": 0,
           "n_truth_lost_pruning": int(P.get("n_truth_lost_to_pruning", 0))}
    if filled is not None:
        out["n_filled"] = int(filled.sum())
        out["n_filled_correct"] = int((filled & (pred == ref)).sum())
    ok = pred >= 0
    a, b = pred[:-1], pred[1:]
    both = (a >= 0) & (b >= 0)
    a, b = a[both], b[both]
    changed = a != b
    frm, to = P["frm"], P["to"]
    adj = to[a] == frm[b]
    twin = (frm[a] == to[b]) & (to[a] == frm[b]) & changed
    out.update(n_pairs=int(len(a)), n_changes=int(changed.sum()),
               n_disconnected=int((changed & ~adj).sum()), n_flips=int(twin.sum()))
    ps, rs = set(pred[ok].tolist()), set(ref.tolist())
    out.update(n_pred_arcs=len(ps), n_ref_arcs=len(rs), n_hit_arcs=len(ps & rs))
    seg = lambda x: int((x[1:] != x[:-1]).sum() + 1)
    out.update(pred_segments=seg(pred), ref_segments=seg(ref))
    return out


def summarize(rows):
    df = pd.DataFrame(rows)
    s = df.sum(numeric_only=True)
    ex = df[~df["trip"].isin(["00000033", "00000054"])]
    prec = s.n_hit_arcs / max(s.n_pred_arcs, 1); rec = s.n_hit_arcs / max(s.n_ref_arcs, 1)
    return dict(acc=s.correct / s.n_obs,
                acc_ex3354=ex.correct.sum() / ex.n_obs.sum(),
                filled=int(s.n_filled), filled_correct=int(s.n_filled_correct),
                no_cand=int(s.n_no_cand), truth_lost_pruning=int(s.n_truth_lost_pruning),
                acc_trip_mean=float((df.correct / df.n_obs).mean()),
                disc_per_1k_changes=1000 * s.n_disconnected / max(s.n_changes, 1),
                disconnected=int(s.n_disconnected), changes=int(s.n_changes), flips=int(s.n_flips),
                route_prec_sum=float(prec), route_rec_sum=float(rec),
                segs_ratio=s.pred_segments / s.ref_segments)


def load_cached(partition, trips=None):
    for trip in trips or list_trips(partition):
        p = cache_path(partition, trip)
        if os.path.exists(p):
            v = pickle.load(open(p, "rb"))
            if v is not None:
                yield trip, v


def check_cache_compatible(cfg):
    """The route cache was built with CACHE_CFG; settings that need a wider search or a
    longer gap than the cache holds would silently decode wrongly, so refuse them."""
    c = {**dec.DEFAULT_CONFIG, **cfg}
    for k in ("gap_restart_s", "slack_m", "search_cap_m", "search_speed_mps"):
        if c[k] > CACHE_CFG[k]:
            raise ValueError(f"{k}={c[k]} exceeds the cache build setting {CACHE_CFG[k]}; rebuild the cache with a larger CACHE_CFG")


def run_config(cfg, items):
    check_cache_compatible(cfg)
    rows, tot = [], dict(n_gap_restarts=0, n_dead_restarts=0, n_penalised_jumps=0, n_pruned_rows=0)
    for trip, (P, links) in items:
        arcs_out, st = dec.decode_prepared(P, cfg, route_cache=links)
        full = np.full(P["n_obs"], -1, dtype=np.int64); full[P["chain"]] = arcs_out
        full, filled = dec.fill_missing(full, P["t"], cfg["fill_missing"], cfg["fill_max_gap_s"])
        rows.append(trip_metrics(full, P, filled, trip))
        for k in tot:
            tot[k] += st[k]
    return rows, tot


def baseline_rows(items):
    return [trip_metrics(P["top1"], P, None, t) for t, (P, _) in items]


# One-at-a-time search around the defaults, then one combined run. A full grid
# would cost hours on one CPU; this is a deliberate, logged simplification.
VARY = dict(
    beta_m=[5.0, 20.0],
    emission_weight=[0.5, 2.0],
    unreachable_penalty=[8.0, 40.0, None],
    gap_restart_s=[30.0, 120.0],
    uturn_penalty=[0.0, 15.0],
    fill_missing=["none"],
)
VARY_EXT = dict(beta_m=[2.5], emission_weight=[0.25, 0.1], uturn_penalty=[30.0])
TUNE_EVERY = 4   # tune on every 4th sorted training trip (20 of 80), out-of-fold scores


def _key(g):
    return json.dumps(g, sort_keys=True)


def run_stage(score_name, configs, tag):
    """Evaluate `configs` (dicts of overrides) on the tuning trips, streaming one
    trip at a time. Returns {key: (rows, totals)} when all trips are done, else None."""
    prog_f = f"{CACHE}/tune_{score_name}_{tag}.pkl"
    prog = pickle.load(open(prog_f, "rb")) if os.path.exists(prog_f) else {"trips": [], "rows": {}, "base": [], "tot": {}}
    trips = list_trips("train")[::TUNE_EVERY]
    t0 = time.time()
    for trip, (P, links) in load_cached("train", trips):
        if trip in prog["trips"]:
            continue
        if time.time() - t0 > 80:
            print(tag, "budget used; rerun to continue:", len(prog["trips"]), "of", len(trips)); return None
        if not prog["base"] or len(prog["base"]) <= len(prog["trips"]):
            prog["base"] += baseline_rows([(trip, (P, links))])
        for g in configs:
            rows, tot = run_config({**dec.DEFAULT_CONFIG, **g}, [(trip, (P, links))])
            k = _key(g)
            prog["rows"].setdefault(k, []).extend(rows)
            T = prog["tot"].setdefault(k, dict.fromkeys(tot, 0))
            for kk in tot:
                T[kk] += tot[kk]
        prog["trips"].append(trip)
        pickle.dump(prog, open(prog_f, "wb"))
    return {_key(g): (prog["rows"][_key(g)], prog["tot"][_key(g)]) for g in configs}, prog["base"]


def cmd_tune(score_name):
    os.makedirs(OUT, exist_ok=True)
    stage1 = [{}] + [{k: v} for k, vals in VARY.items() for v in vals]
    r1 = run_stage(score_name, stage1, "s1")
    if r1 is None:
        return
    res1, base = r1
    log = []
    def rec(stage, g, rows, tot):
        log.append(dict(stage=stage, partition="train_oof_every4th", score=score_name,
                        provisional_interim_scores=True, overrides=_key(g), **summarize(rows), **tot))
    for g in stage1:
        rec(1, g, *res1[_key(g)])
    best = {}
    for k, vals in VARY.items():
        cand = [{}] + [{k: v} for v in vals]
        top = max(cand, key=lambda g: summarize(res1[_key(g)][0])["acc_ex3354"])
        best.update(top)
    # Stage 2 settings sat on the edge of the grid (lowest beta and weight, highest
    # U-turn penalty), so stage 3 extends those directions around the stage-2 best.
    ext = [{**best, **{k: v}} for k, vals in VARY_EXT.items() for v in vals]
    stage23 = [best] + ext
    r2 = run_stage(score_name, stage23, "s3")
    if r2 is None:
        return
    for i, g in enumerate(stage23):
        rec(2 if i == 0 else 3, g, *r2[0][_key(g)])
    df = pd.DataFrame(log)
    df.to_csv(OUT + f"tuning_log_{score_name}.csv", index=False)
    print("classifier-only on tuning trips:", {k: summarize(base)[k] for k in ("acc", "acc_ex3354", "truth_lost_pruning")})
    print(df[["stage", "overrides", "acc", "acc_ex3354", "filled", "filled_correct", "disc_per_1k_changes", "flips", "n_penalised_jumps",
              "n_gap_restarts"]].to_string())
    top = df.sort_values("acc_ex3354", ascending=False).iloc[0]
    print("BEST (by accuracy excluding trips 33/54):", top["overrides"], top["acc_ex3354"])


def cmd_final(score_name, cfg_path):
    cfg = dec.load_config(cfg_path)
    items = list(load_cached("student_test"))
    rows, tot = run_config(cfg, items)
    base = baseline_rows(items)
    res = dict(score=score_name, config=cfg, hmm=summarize(rows), classifier_only=summarize(base),
               per_trip=[dict(trip=t, hmm_acc=r["correct"] / r["n_obs"], clf_acc=b["correct"] / b["n_obs"])
                         for (t, _), r, b in zip(items, rows, base)], **tot)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(OUT + f"final_student_test_{score_name}.json", "w"), indent=2, default=float)
    print(json.dumps({k: res[k] for k in ("hmm", "classifier_only")}, indent=1, default=float))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "cache":
        cmd_cache(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 100)
    elif c == "tune":
        cmd_tune(sys.argv[2])
    elif c == "final":
        cmd_final(sys.argv[2], sys.argv[3])
