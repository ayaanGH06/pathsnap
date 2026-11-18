"""Evidence for hmm/model_notes.md: LR and HGB, same tables/seed as the task 4 run.

Run from repo root after `python -m features.build_tables 50`:
    python hmm/model_notes_evidence.py
Prints JSON. Trains on `train` only; student_test is scored, never tuned on.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
import evaluate as ev  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

P = ROOT / "data" / "processed"
FEATS = ["dist_to_arc", "proj_frac", "arc_len", "heading_diff", "speed", "dt_prev",
         "dist_rank", "dist_minus_min", "n_cand", "conn_prev"]
tr, st = pd.read_parquet(P / "candidates_train.parquet"), pd.read_parquet(P / "candidates_student_test.parquet")
otr, ost = pd.read_parquet(P / "obs_train.parquet"), pd.read_parquet(P / "obs_student_test.parquet")
out = {"train_rows": len(tr), "train_positive_rate": float(tr["label"].mean()),
       "student_rows": len(st), "student_positive_rate": float(st["label"].mean()),
       "train_candidates_per_obs": len(tr) / len(otr), "student_candidates_per_obs": len(st) / len(ost),
       "student_trip_ids": sorted(ost["trip_id"].unique().tolist())}
roster = ev.model_roster()
for name in ["logistic_regression", "hist_gradient_boosting"]:
    pipe = roster[name][0]
    pipe.fit(tr[FEATS], tr["label"].astype(int))
    m = pipe.named_steps["model"]
    r = {"n_iter": int(np.max(m.n_iter_))}
    # resubstitution (fit-set) vs student_test, same metric code as evaluate.py
    for tag, c, o in [("train_fit", tr, otr), ("student_test", st, ost)]:
        s = ev.positive_probability(pipe, c[FEATS])
        met = ev.observation_metrics(c, s, o)
        r[tag] = {k: met[k] for k in ["top1_accuracy", "macro_f1"]}
        r[tag]["candidate_log_loss"] = float(log_loss(c["label"].astype(int), s, labels=[0, 1]))
        if tag == "student_test":
            base = r[tag]["candidate_log_loss"]
            imp = {}
            rng = np.random.default_rng(42)
            for f in FEATS:
                X = st[FEATS].copy(); X[f] = rng.permutation(X[f].to_numpy())
                imp[f] = float(log_loss(st["label"].astype(int), ev.positive_probability(pipe, X), labels=[0, 1]) - base)
            r["permutation_logloss_increase_student_test"] = imp
            # per-trip Top-1
            ev_obs = []
            for t, ot in o.groupby("trip_id"):
                mk = c["trip_id"] == t
                ev_obs.append((t, ev.observation_metrics(c[mk.to_numpy()].reset_index(drop=True), s[mk.to_numpy()], ot)["top1_accuracy"]))
            r["student_top1_per_trip"] = {t: round(a, 4) for t, a in ev_obs}
    if name == "logistic_regression":
        coef = m.coef_[0]
        r["standardised_coef"] = dict(zip(FEATS, map(float, coef)))
    else:
        r["max_iter"] = 150
    out[name] = r
# macro-F1 label space: arc_id is a trip-local row index
out["distinct_ref_arc_values_student"] = int(ost["ref_arc"].nunique())
out["distinct_trip_arc_pairs_student"] = int(ost[["trip_id", "ref_arc"]].drop_duplicates().shape[0])
print(json.dumps(out, indent=1))
