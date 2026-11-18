"""Fit and export the selected candidate-probability pipeline for C and D.

Uses the HGB configuration selected by matched five-fold train CV, fits only on
the train candidate table, and stores the imputer/model together in one
scikit-learn Pipeline. Student-test is evaluated once after configuration
selection; it is not used for hyperparameter tuning or selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import sklearn

from config import RANDOM_SEED
from models.evaluate import (
    OUTLIER_TRIPS,
    evaluate_student,
    model_roster,
    read_table,
    validate_candidates,
    validate_observations,
    validate_pair,
)

TRAIN_PATH = ROOT / "data/processed/candidates_train.parquet"
OBS_PATH = ROOT / "data/processed/obs_train.parquet"
STUDENT_PATH = ROOT / "data/processed/candidates_student_test.parquet"
STUDENT_OBS_PATH = ROOT / "data/processed/obs_student_test.parquet"
ARTIFACT_DIR = ROOT / "models/artifacts"
OPT_DIR = ROOT / "results/bayes/hgb_bayesopt_20261118"
TEST_OUTPUT = ROOT / "results/bayes/selected_hgb_student_test.csv"


def main() -> None:
    search_summary = json.loads((OPT_DIR / "optimization_summary.json").read_text())
    selection = json.loads((OPT_DIR / "selection_decision.json").read_text())
    config = selection["selected_config"]
    train = read_table(TRAIN_PATH)
    train_obs = read_table(OBS_PATH)
    student = read_table(STUDENT_PATH)
    student_obs = read_table(STUDENT_OBS_PATH)
    features = validate_candidates(train, "train")
    student_features = validate_candidates(student, "student_test")
    if features != student_features:
        raise ValueError("Train and student-test feature columns differ")
    validate_observations(train_obs, "train_obs")
    validate_observations(student_obs, "student_obs")
    validate_pair(train, train_obs, "train")
    validate_pair(student, student_obs, "student_test")

    pipeline, baseline = model_roster()["hist_gradient_boosting"]
    pipeline.set_params(**{f"model__{name}": value for name, value in config.items()})
    pipeline.fit(train[features], train["label"].astype(int))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "pathsnap_selected_model.joblib"
    joblib.dump(pipeline, artifact_path, compress=3)
    manifest = {
        "model_name": f"HistGradientBoostingClassifier ({selection['selected_model']})",
        "artifact": artifact_path.name,
        "purpose": "Score each candidate road; class 1 probability means candidate matches reference road.",
        "positive_class": 1,
        "feature_columns_in_required_order": features,
        "preprocessing_included": [
            "SimpleImputer(strategy='median')",
            "no feature scaling (tree model)",
        ],
        "model_config": {**baseline, **config},
        "selection_objective": selection["primary_metric"],
        "selection_reason": selection["reason"],
        "optimization_best_config": search_summary["best_config"],
        "selection_cv": {
            "search_folds": search_summary["search_cv_folds"],
            "final_folds": search_summary["final_cv_folds"],
            "grouped_by": "trip_id",
            "seed": RANDOM_SEED,
        },
        "candidate_radius_m": 50,
        "student_test_used_for_tuning": False,
        "training_candidate_rows": int(len(train)),
        "training_observations": int(len(train_obs)),
        "python_sklearn_version": sklearn.__version__,
        "load_note": "Load with a compatible scikit-learn version; see artifacts/README.md.",
    }
    (ARTIFACT_DIR / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    student_rows, _ = evaluate_student(
        selection["selected_model"], pipeline, manifest["model_config"],
        train, student, student_obs, features, OUTLIER_TRIPS,
    )
    pd.DataFrame(student_rows).to_csv(TEST_OUTPUT, index=False)
    print(f"Exported {artifact_path.relative_to(ROOT)}")
    print(f"Saved one-time held-out student-test metrics to {TEST_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
