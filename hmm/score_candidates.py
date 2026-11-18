"""C7 (interim): classifier probabilities for every candidate row.

Uses the shared roster in models/evaluate.py with the same GroupKFold folds
(5 folds over the sorted training trip ids) as B's CV, so:
- train scores are out-of-fold: each trip is scored by a model that never saw
  it, which is what decoder tuning needs;
- student_test scores come from a model fitted on all 80 training trips.

This stands in for B's exported untuned model (handoff H3) until models/artifacts/
is populated. Nothing is retrained with new settings or features. Output files
carry trip_id, obs_idx, arc_id, score and stay under data/processed/ (not committed).

Usage: python -m hmm.score_candidates hist_gradient_boosting [logistic_regression ...]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from models.evaluate import model_roster, positive_probability

FEATURES = ["dist_to_arc", "proj_frac", "arc_len", "heading_diff", "speed",
            "dt_prev", "dist_rank", "dist_minus_min", "n_cand", "conn_prev"]
D = "data/processed/"


def main(names):
    tr = pd.read_parquet(D + "candidates_train.parquet")
    st = pd.read_parquet(D + "candidates_student_test.parquet")
    trips = np.array(sorted(tr["trip_id"].unique()))
    roster = model_roster()
    for name in names:
        template = roster[name][0]
        oof = np.full(len(tr), np.nan)
        for fit_i, val_i in GroupKFold(n_splits=5).split(np.zeros(len(trips)), groups=trips):
            fit = tr["trip_id"].isin(set(trips[fit_i])).to_numpy()
            val = tr["trip_id"].isin(set(trips[val_i])).to_numpy()
            m = clone(template).fit(tr.loc[fit, FEATURES], tr.loc[fit, "label"].astype(int))
            oof[val] = positive_probability(m, tr.loc[val, FEATURES])
        m = clone(template).fit(tr[FEATURES], tr["label"].astype(int))
        key = ["trip_id", "obs_idx", "arc_id"]
        a = tr[key].assign(score=oof); b = st[key].assign(score=positive_probability(m, st[FEATURES]))
        a.to_parquet(D + f"scores_{name}_train_oof.parquet", index=False)
        b.to_parquet(D + f"scores_{name}_student_test.parquet", index=False)
        print(name, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
