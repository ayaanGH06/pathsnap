"""Compare tuned and untuned HGB on the same 5-fold training CV and select."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config import RANDOM_SEED
from models.evaluate import (
    OUTLIER_TRIPS,
    model_roster,
    read_table,
    run_cv,
    validate_candidates,
    validate_observations,
    validate_pair,
)

TRAIN_PATH = ROOT / "data/processed/candidates_train.parquet"
OBS_PATH = ROOT / "data/processed/obs_train.parquet"
OUT_DIR = ROOT / "results/bayes/hgb_bayesopt_20261118"


def metric(path: Path, slice_name: str, metric_name: str) -> float:
    frame = pd.read_csv(path)
    row = frame.loc[frame["evaluation_slice"] == slice_name].iloc[0]
    return float(row[metric_name])


def main() -> None:
    train = read_table(TRAIN_PATH)
    observations = read_table(OBS_PATH)
    features = validate_candidates(train, "train")
    validate_observations(observations, "train_obs")
    validate_pair(train, observations, "train")

    baseline_pipeline, baseline_metadata = model_roster()["hist_gradient_boosting"]
    baseline_config = {
        "max_iter": 150,
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    }
    baseline_summary, baseline_folds = run_cv(
        "hist_gradient_boosting_baseline", baseline_pipeline, baseline_metadata,
        train, observations, features, 5, OUTLIER_TRIPS,
    )
    pd.DataFrame(baseline_summary).to_csv(OUT_DIR / "baseline_5fold_cv.csv", index=False)
    pd.DataFrame(baseline_folds).to_csv(OUT_DIR / "baseline_5fold_folds.csv", index=False)

    tuned_path = OUT_DIR / "best_5fold_cv.csv"
    baseline_path = OUT_DIR / "baseline_5fold_cv.csv"
    tuned_top1 = metric(tuned_path, "all_trips", "top1_accuracy")
    baseline_top1 = metric(baseline_path, "all_trips", "top1_accuracy")
    tuned_loss = metric(tuned_path, "all_trips", "candidate_log_loss")
    baseline_loss = metric(baseline_path, "all_trips", "candidate_log_loss")

    search_summary = json.loads((OUT_DIR / "optimization_summary.json").read_text())
    tuned_config = search_summary["best_config"]
    if tuned_top1 > baseline_top1 + 1e-12:
        selected_name = "hist_gradient_boosting_bayes_optimized"
        selected_config = {**baseline_config, **tuned_config}
        reason = "Bayesian-tuned HGB had higher 5-fold all-trip Top-1."
    elif baseline_top1 > tuned_top1 + 1e-12:
        selected_name = "hist_gradient_boosting_baseline"
        selected_config = baseline_config
        reason = "Baseline HGB had higher 5-fold all-trip Top-1; tuning did not improve the primary metric."
    elif tuned_loss < baseline_loss:
        selected_name = "hist_gradient_boosting_bayes_optimized"
        selected_config = {**baseline_config, **tuned_config}
        reason = "Top-1 tied; tuned HGB had lower candidate log loss, the secondary emission-quality criterion."
    else:
        selected_name = "hist_gradient_boosting_baseline"
        selected_config = baseline_config
        reason = "Top-1 tied and baseline log loss was no worse; prefer the simpler, untuned configuration."

    decision = {
        "primary_metric": "5-fold trip-grouped CV observation-level Top-1 across all training observations",
        "secondary_metric": "candidate log loss on candidate rows",
        "student_test_used_for_selection": False,
        "seed": RANDOM_SEED,
        "baseline_5fold_cv_top1_all": baseline_top1,
        "tuned_5fold_cv_top1_all": tuned_top1,
        "baseline_5fold_candidate_log_loss": baseline_loss,
        "tuned_5fold_candidate_log_loss": tuned_loss,
        "selected_model": selected_name,
        "selected_config": selected_config,
        "reason": reason,
    }
    (OUT_DIR / "selection_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
