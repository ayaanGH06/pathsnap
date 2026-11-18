"""Evaluate candidate-arc classifiers for PathSnap.

Expected input: one row per (trip_id, obs_idx, arc_id), engineered numeric
features, and a binary label (1 for the reference arc, 0 otherwise).
All learned preprocessing is inside each sklearn Pipeline so it is fitted only
on the training trips in each fold.

This module is a starter harness. It does not contain or imply any experiment
results. Route continuity is left to the HMM/decoder evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from config import RANDOM_SEED

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = RANDOM_SEED
KEY_COLUMNS = {"trip_id", "obs_idx", "arc_id", "label"}


def model_roster() -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    """Return five baseline models from distinct model families."""
    roster: dict[str, tuple[Pipeline, dict[str, Any]]] = {}

    def add(name: str, estimator: Any, scale: bool, config: dict[str, Any]) -> None:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        roster[name] = (Pipeline(steps), config)

    add(
        "logistic_regression",
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED),
        True,
        {"family": "linear", "max_iter": 500, "class_weight": "balanced"},
    )
    add(
        "knn_distance_weighted",
        KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        True,
        {"family": "instance-based", "n_neighbors": 15, "weights": "distance", "n_jobs": -1},
    )
    add(
        "random_forest",
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        False,
        {
            "family": "tree ensemble",
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
        },
    )
    add(
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, l2_regularization=1.0, random_state=SEED
        ),
        False,
        {
            "family": "boosting",
            "max_iter": 150,
            "learning_rate": 0.1,
            "l2_regularization": 1.0,
        },
    )
    add(
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=True,
            random_state=SEED,
        ),
        True,
        {
            "family": "neural network",
            "hidden_layer_sizes": [64, 32],
            "max_iter": 300,
            "early_stopping": True,
        },
    )
    return roster


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", dtype={"trip_id": "string"})
    else:
        raise ValueError(f"Unsupported table type: {path}. Use Parquet, CSV, or TSV.")
    if "trip_id" in frame:
        frame["trip_id"] = frame["trip_id"].astype("string")
    return frame


def validate_table(frame: pd.DataFrame, name: str) -> list[str]:
    required = KEY_COLUMNS
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} has no candidate rows")
    if frame[["trip_id", "obs_idx", "arc_id"]].isna().any().any():
        raise ValueError(f"{name} contains missing trip/observation/arc identifiers")
    if frame["label"].isna().any() or not set(frame["label"].unique()).issubset({0, 1, False, True}):
        raise ValueError(f"{name}.label must contain only 0/1 values")
    features = [column for column in frame.columns if column not in KEY_COLUMNS]
    if not features:
        raise ValueError(f"{name} has no feature columns")
    non_numeric = [column for column in features if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ValueError(f"Features must be numeric; encode or remove: {non_numeric}")
    if frame["label"].nunique() != 2:
        raise ValueError(f"{name} must contain both positive and negative candidate labels")
    return features


def observation_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    """Score the highest-probability candidate for each observation.

    Candidate misses count as incorrect in Top-1 accuracy. Arc-level Macro F1,
    precision, and recall use observations with a known positive candidate;
    candidate miss rate is reported separately because the true arc is absent
    from those rows.
    """
    scored = frame[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    scored["score"] = scores
    group_cols = ["trip_id", "obs_idx"]
    positive_counts = scored.groupby(group_cols, sort=False)["label"].sum()
    misses = positive_counts.eq(0)
    ambiguous = positive_counts.gt(1)
    if ambiguous.any():
        raise ValueError("Each observation may have at most one positive candidate label")

    best_rows = scored.loc[scored.groupby(group_cols, sort=False)["score"].idxmax()]
    truth_rows = scored.loc[scored["label"].eq(1), group_cols + ["arc_id"]]
    truth_rows = truth_rows.rename(columns={"arc_id": "true_arc_id"})
    evaluated = best_rows.merge(truth_rows, on=group_cols, how="left", validate="one_to_one")
    covered = evaluated["true_arc_id"].notna()
    correct = (evaluated.loc[covered, "arc_id"].astype(str).to_numpy() == evaluated.loc[covered, "true_arc_id"].astype(str).to_numpy())
    total_observations = len(evaluated)
    correct_count = int(correct.sum())

    if covered.any():
        y_true = evaluated.loc[covered, "true_arc_id"].astype(str)
        y_pred = evaluated.loc[covered, "arc_id"].astype(str)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        macro_f1 = macro_precision = macro_recall = float("nan")

    return {
        "observation_count": int(total_observations),
        "covered_observations": int(covered.sum()),
        "candidate_misses": int(misses.sum()),
        "candidate_miss_rate": float(misses.mean()),
        "top1_accuracy_all_observations": float(correct_count / total_observations),
        "macro_f1_covered_observations": macro_f1,
        "macro_precision_covered_observations": macro_precision,
        "macro_recall_covered_observations": macro_recall,
        "route_continuity": None,
        "route_continuity_note": "N/A until HMM/Viterbi decoder evaluation",
    }


def positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return model.predict_proba(features)[:, classes.index(1)]


def run_cv(
    model_name: str,
    model_template: Pipeline,
    config: dict[str, Any],
    train: pd.DataFrame,
    feature_columns: list[str],
    folds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = train["trip_id"].astype(str)
    group_count = groups.nunique()
    if folds < 2 or folds > group_count:
        raise ValueError(f"folds must be between 2 and number of training trips ({group_count})")
    splitter = GroupKFold(n_splits=folds)
    fold_rows = []
    oof_scores = np.full(len(train), np.nan, dtype=float)

    for fold, (fit_idx, valid_idx) in enumerate(
        splitter.split(train[feature_columns], train["label"], groups), start=1
    ):
        model = sklearn_clone(model_template)
        started = time.perf_counter()
        model.fit(train.iloc[fit_idx][feature_columns], train.iloc[fit_idx]["label"].astype(int))
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = positive_probability(model, train.iloc[valid_idx][feature_columns])
        inference_seconds = time.perf_counter() - started
        oof_scores[valid_idx] = scores
        fold_metrics = observation_metrics(train.iloc[valid_idx], scores)
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "validation_trip_count": int(groups.iloc[valid_idx].nunique()),
                "fit_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
                **fold_metrics,
            }
        )

    if np.isnan(oof_scores).any():
        raise RuntimeError("Some training rows did not receive out-of-fold scores")
    pooled_metrics = observation_metrics(train, oof_scores)
    try:
        candidate_log_loss = float(log_loss(train["label"].astype(int), oof_scores, labels=[0, 1]))
    except ValueError:
        candidate_log_loss = float("nan")
    summary = {
        "model": model_name,
        "algorithm_family": config["family"],
        "configuration": json.dumps(config, sort_keys=True),
        "cv_folds": folds,
        "cv_trip_count": int(group_count),
        "candidate_log_loss_oof": candidate_log_loss,
        "seed": SEED,
        **pooled_metrics,
    }
    return summary, fold_rows


def sklearn_clone(estimator: Pipeline) -> Pipeline:
    # Imported here to keep the public API of this script small and explicit.
    from sklearn.base import clone

    return clone(estimator)


def evaluate_student(
    model_name: str,
    model_template: Pipeline,
    train: pd.DataFrame,
    student: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Any], Pipeline]:
    model = sklearn_clone(model_template)
    started = time.perf_counter()
    model.fit(train[feature_columns], train["label"].astype(int))
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = positive_probability(model, student[feature_columns])
    inference_seconds = time.perf_counter() - started
    metrics = observation_metrics(student, scores)
    try:
        candidate_log_loss = float(log_loss(student["label"].astype(int), scores, labels=[0, 1]))
    except ValueError:
        candidate_log_loss = float("nan")
    summary = {
        "model": model_name,
        "fit_seconds_full_train": fit_seconds,
        "inference_seconds_student_test": inference_seconds,
        "inference_ms_per_candidate": inference_seconds * 1000 / len(student),
        "candidate_log_loss": candidate_log_loss,
        **metrics,
    }
    return summary, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Training candidate table (Parquet/CSV/TSV)")
    parser.add_argument("--student-test", type=Path, required=True, help="Student-test candidate table (Parquet/CSV/TSV)")
    parser.add_argument("--results-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", choices=sorted(model_roster()), help="Run only these model keys; omit to run all five")
    args = parser.parse_args()

    train = read_table(args.train)
    student = read_table(args.student_test)
    feature_columns = validate_table(train, "train")
    student_features = validate_table(student, "student_test")
    if feature_columns != student_features:
        raise ValueError("Train and student-test feature columns/order must match")
    if set(train["trip_id"].astype(str)) & set(student["trip_id"].astype(str)):
        raise ValueError("A trip_id appears in both train and student_test")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    available_models = model_roster()
    selected_names = args.models or list(available_models)
    selected_models = {name: available_models[name] for name in selected_names}
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S_") + "_".join(selected_names)
    run_dir = args.results_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)
    cv_summaries: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    student_summaries: list[dict[str, Any]] = []
    run_metadata = {
        "seed": SEED,
        "folds": args.folds,
        "train_path": str(args.train),
        "student_test_path": str(args.student_test),
        "feature_columns": feature_columns,
        "note": "Results are generated only when this script is run on the supplied candidate tables.",
    }

    for model_name, (template, config) in selected_models.items():
        print(f"Evaluating {model_name}...")
        cv_summary, folds = run_cv(model_name, template, config, train, feature_columns, args.folds)
        student_summary, _ = evaluate_student(model_name, template, train, student, feature_columns)
        student_summary["algorithm_family"] = config["family"]
        student_summary["configuration"] = json.dumps(config, sort_keys=True)
        student_summary["seed"] = SEED
        cv_summaries.append(cv_summary)
        fold_summaries.extend(folds)
        student_summaries.append(student_summary)

    pd.DataFrame(cv_summaries).to_csv(run_dir / "model_comparison_cv.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(run_dir / "model_cv_folds.csv", index=False)
    pd.DataFrame(student_summaries).to_csv(run_dir / "model_comparison_student_test.csv", index=False)
    run_metadata["models"] = selected_names
    run_metadata["run_timestamp"] = datetime.now().astimezone().isoformat()
    (run_dir / "model_run_config.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(f"Saved run outputs to {run_dir}")


if __name__ == "__main__":
    main()
