"""Score the student-test candidate table with B's saved artifact (H3).

Usage (from repo root, after A's tables exist in data/processed/):
    python hmm/score_with_artifact.py [out_name=artifact]
Writes data/processed/scores_<out_name>_student_test.parquet (trip_id, obs_idx, arc_id, score).
Needs a scikit-learn compatible with models/artifacts/model_manifest.json (1.9.1 there).
Does no fitting; the label column is not passed to the model.
"""
import json, sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "artifact"
manifest = json.loads((ROOT / "models/artifacts/model_manifest.json").read_text())
model = joblib.load(ROOT / "models/artifacts/pathsnap_selected_model.joblib")
cols = ["dist_to_arc", "proj_frac", "arc_len", "heading_diff", "speed", "dt_prev",
        "dist_rank", "dist_minus_min", "n_cand", "conn_prev"]
c = pd.read_parquet(ROOT / "data/processed/candidates_student_test.parquet")
pos = list(model.classes_).index(1)
out = c[["trip_id", "obs_idx", "arc_id"]].copy()
out["score"] = model.predict_proba(c[cols])[:, pos]
p = ROOT / f"data/processed/scores_{name}_student_test.parquet"
out.to_parquet(p, index=False)
print("wrote", p, len(out))
