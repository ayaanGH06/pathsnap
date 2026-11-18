"""Compare PathSnap candidate-arc classifiers with trip-grouped CV.

Candidate tables contain one row per (trip_id, obs_idx, arc_id). Observation
sidecars provide the full evaluation denominator and reference arc, including
observations whose true arc was not generated as a candidate.
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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = RANDOM_SEED
CANDIDATE_KEYS = ["trip_id", "obs_idx", "arc_id"]
OBSERVATION_KEYS = ["trip_id", "obs_idx"]
CANDIDATE_META = set(CANDIDATE_KEYS + ["label"])
OUTLIER_TRIPS = ["00000033", "00000054"]


def model_roster() -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    """Five baseline models; settings are starting points, not selected results."""
    roster: dict[str, tuple[Pipeline, dict[str, Any]]] = {}

    def add(name: str, estimator: Any, scale: bool, config: dict[str, Any]) -> None:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        roster[name] = (Pipeline(steps), config)

    add("logistic_regression", LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED), True,
        {"family": "linear", "max_iter": 500, "class_weight": "balanced"})
    add("knn_distance_weighted", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1), True,
        {"family": "instance-based", "n_neighbors": 15, "weights": "distance", "n_jobs": -1})
    add("random_forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), False,
        {"family": "tree ensemble", "n_estimators": 300, "min_samples_leaf": 2,
         "class_weight": "balanced_subsample"})
    add("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
        l2_regularization=1.0, random_state=SEED), False,
        {"family": "boosting", "max_iter": 150, "learning_rate": 0.1, "l2_regularization": 1.0})
    add("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
        random_state=SEED), True,
        {"family": "neural network", "hidden_layer_sizes": [64, 32], "max_iter": 300,
         "early_stopping": True})
    return roster


def normalize_trip_id(value: Any) -> str:
    text = str(value).strip()
    return text.zfill(8) if text.isdigit() else text


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                            dtype={"trip_id": "string"})
    else:
        raise ValueError(f"Unsupported table type: {path}. Use Parquet, CSV, or TSV.")
    if "trip_id" in frame:
        frame["trip_id"] = frame["trip_id"].map(normalize_trip_id)
    if "obs_idx" in frame:
        obs_idx = pd.to_numeric(frame["obs_idx"], errors="raise")
        if not np.equal(obs_idx, np.floor(obs_idx)).all():
            raise ValueError(f"{path}: obs_idx values must be integers")
        frame["obs_idx"] = obs_idx.astype("int64")
    return frame


def normalize_arc_column(frame: pd.DataFrame, column: str, name: str) -> None:
    values = pd.to_numeric(frame[column], errors="raise")
    if values.isna().any() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name}.{column} must contain integer arc IDs")
    # Nullable integer dtype avoids float coercion when a merge has missing predictions.
    frame[column] = values.astype("Int64")


def validate_candidates(frame: pd.DataFrame, name: str) -> list[str]:
    required = CANDIDATE_META
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} has no candidate rows")
    if frame[["trip_id", "obs_idx", "arc_id", "label"]].isna().any().any():
        raise ValueError(f"{name} contains missing candidate IDs or labels")
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError(f"{name}.label must contain both 0 and 1")
    frame["label"] = labels.astype("int8")
    normalize_arc_column(frame, "arc_id", name)
    if frame.duplicated(CANDIDATE_KEYS).any():
        raise ValueError(f"{name} has duplicate (trip_id, obs_idx, arc_id) rows")
    features = [column for column in frame.columns if column not in CANDIDATE_META]
    if not features:
        raise ValueError(f"{name} has no feature columns")
    non_numeric = [column for column in features
                   if not (pd.api.types.is_numeric_dtype(frame[column]) or
                           pd.api.types.is_bool_dtype(frame[column]))]
    if non_numeric:
        raise ValueError(f"Features must be numeric; encode or remove: {non_numeric}")
    positives = frame.groupby(OBSERVATION_KEYS, sort=False)["label"].sum()
    if positives.gt(1).any():
        raise ValueError(f"{name} has observations with more than one positive candidate")
    return features


def validate_observations(frame: pd.DataFrame, name: str) -> None:
    required = {"trip_id", "obs_idx", "ref_arc", "in_cand"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if frame.empty or frame[["trip_id", "obs_idx", "ref_arc", "in_cand"]].isna().any().any():
        raise ValueError(f"{name} has no rows or has missing required observation values")
    normalize_arc_column(frame, "ref_arc", name)
    if not pd.api.types.is_bool_dtype(frame["in_cand"]):
        values = frame["in_cand"].astype(str).str.strip().str.lower()
        mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        converted = values.map(mapping)
        if converted.isna().any():
            raise ValueError(f"{name}.in_cand must be boolean or 0/1")
        frame["in_cand"] = converted.astype(bool)
    else:
        frame["in_cand"] = frame["in_cand"].astype(bool)
    if frame.duplicated(OBSERVATION_KEYS).any():
        raise ValueError(f"{name} must have one row per (trip_id, obs_idx)")


def validate_pair(candidates: pd.DataFrame, observations: pd.DataFrame, name: str) -> None:
    candidate_keys = pd.MultiIndex.from_frame(candidates[OBSERVATION_KEYS].drop_duplicates())
    observation_keys = pd.MultiIndex.from_frame(observations[OBSERVATION_KEYS])
    missing_obs = candidate_keys.difference(observation_keys)
    if len(missing_obs):
        raise ValueError(f"{name}: candidate table has observations absent from its obs sidecar")

    has_positive = candidates.groupby(OBSERVATION_KEYS, sort=False)["label"].max().rename("has_positive")
    check = observations.set_index(OBSERVATION_KEYS).join(has_positive, how="left")
    check["has_positive"] = check["has_positive"].fillna(0).astype(bool)
    mismatch = check["in_cand"].astype(bool) != check["has_positive"]
    if mismatch.any():
        raise ValueError(f"{name}: in_cand disagrees with whether a candidate row has label=1")


def positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return model.predict_proba(features)[:, classes.index(1)]


def observation_metrics(candidates: pd.DataFrame, scores: np.ndarray,
                        observations: pd.DataFrame) -> dict[str, Any]:
    """Evaluate against every obs-sidecar row; misses stay in the denominator."""
    scored = candidates[["trip_id", "obs_idx", "arc_id"]].copy().reset_index(drop=True)
    scored["score"] = scores
    if len(scored):
        best_idx = scored.groupby(OBSERVATION_KEYS, sort=False)["score"].idxmax()
        best = scored.loc[best_idx, OBSERVATION_KEYS + ["arc_id"]].rename(columns={"arc_id": "pred_arc"})
    else:
        best = pd.DataFrame(columns=OBSERVATION_KEYS + ["pred_arc"])
    evaluated = observations[OBSERVATION_KEYS + ["ref_arc", "in_cand"]].merge(
        best, on=OBSERVATION_KEYS, how="left", validate="one_to_one"
    )
    # Cast after merge to nullable integers; this prevents 123 vs 123.0 text mismatches.
    evaluated["pred_arc"] = pd.to_numeric(evaluated["pred_arc"], errors="coerce").astype("Int64")
    evaluated["ref_arc"] = pd.to_numeric(evaluated["ref_arc"], errors="raise").astype("Int64")
    correct = evaluated["in_cand"] & evaluated["pred_arc"].notna() & evaluated["pred_arc"].eq(evaluated["ref_arc"]).fillna(False)
    y_true = evaluated["ref_arc"].astype(str)
    y_pred = evaluated["pred_arc"].astype("string").fillna("__NO_CANDIDATE__").astype(str)
    return {
        "observation_count": int(len(evaluated)),
        "observations_with_candidate": int(evaluated["pred_arc"].notna().sum()),
        "candidate_misses": int((~evaluated["in_cand"]).sum()),
        "candidate_miss_rate": float((~evaluated["in_cand"]).mean()),
        "top1_accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "route_continuity": None,
        "route_continuity_note": "N/A until graph-aware HMM/Viterbi evaluation",
    }


def scored_slices(candidates: pd.DataFrame, scores: np.ndarray, observations: pd.DataFrame,
                  excluded_trips: list[str]) -> list[dict[str, Any]]:
    slices = [("all_trips", observations)]
    kept = observations.loc[~observations["trip_id"].isin(excluded_trips)]
    slices.append(("excluding_33_54", kept))
    output = []
    for slice_name, obs_slice in slices:
        trip_ids = set(obs_slice["trip_id"])
        mask = candidates["trip_id"].isin(trip_ids).to_numpy()
        cand_slice = candidates.loc[mask].reset_index(drop=True)
        score_slice = scores[mask]
        row = {"evaluation_slice": slice_name, **observation_metrics(cand_slice, score_slice, obs_slice)}
        if len(cand_slice):
            row["candidate_log_loss"] = float(log_loss(cand_slice["label"].astype(int), score_slice, labels=[0, 1]))
        else:
            row["candidate_log_loss"] = float("nan")
        output.append(row)
    return output


def run_cv(model_name: str, template: Pipeline, config: dict[str, Any],
           candidates: pd.DataFrame, observations: pd.DataFrame,
           feature_columns: list[str], folds: int, excluded_trips: list[str]
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trips = np.array(sorted(observations["trip_id"].unique()))
    if folds < 2 or folds > len(trips):
        raise ValueError(f"folds must be between 2 and number of training trips ({len(trips)})")
    splitter = GroupKFold(n_splits=folds)
    oof_scores = np.full(len(candidates), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    # Split the trip list itself, then select all candidate rows for those trips.
    for fold, (fit_trip_idx, valid_trip_idx) in enumerate(
        splitter.split(np.zeros(len(trips)), groups=trips), start=1
    ):
        fit_trips = set(trips[fit_trip_idx])
        valid_trips = set(trips[valid_trip_idx])
        fit_mask = candidates["trip_id"].isin(fit_trips).to_numpy()
        valid_mask = candidates["trip_id"].isin(valid_trips).to_numpy()
        model = clone(template)
        started = time.perf_counter()
        model.fit(candidates.loc[fit_mask, feature_columns], candidates.loc[fit_mask, "label"].astype(int))
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = positive_probability(model, candidates.loc[valid_mask, feature_columns])
        inference_seconds = time.perf_counter() - started
        oof_scores[valid_mask] = scores
        obs_valid = observations.loc[observations["trip_id"].isin(valid_trips)]
        cand_valid = candidates.loc[valid_mask].reset_index(drop=True)
        for metrics in scored_slices(cand_valid, scores, obs_valid, excluded_trips):
            fold_rows.append({"model": model_name, "fold": fold,
                              "validation_trip_count": len(valid_trips),
                              "fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
                              **metrics})

    if np.isnan(oof_scores).any():
        raise RuntimeError("Some candidate rows did not receive out-of-fold scores")
    summaries = []
    for metrics in scored_slices(candidates, oof_scores, observations, excluded_trips):
        summaries.append({"model": model_name, "algorithm_family": config["family"],
                          "configuration": json.dumps(config, sort_keys=True),
                          "cv_folds": folds, "cv_trip_count": len(trips), "seed": SEED,
                          **metrics})
    return summaries, fold_rows


def evaluate_student(model_name: str, template: Pipeline, config: dict[str, Any],
                     train_candidates: pd.DataFrame, student_candidates: pd.DataFrame,
                     student_observations: pd.DataFrame, feature_columns: list[str],
                     excluded_trips: list[str]) -> tuple[list[dict[str, Any]], Pipeline]:
    model = clone(template)
    started = time.perf_counter()
    model.fit(train_candidates[feature_columns], train_candidates["label"].astype(int))
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = positive_probability(model, student_candidates[feature_columns])
    inference_seconds = time.perf_counter() - started
    rows = []
    for metrics in scored_slices(student_candidates, scores, student_observations, excluded_trips):
        rows.append({"model": model_name, "algorithm_family": config["family"],
                     "configuration": json.dumps(config, sort_keys=True), "seed": SEED,
                     "fit_seconds_full_train": fit_seconds,
                     "inference_seconds_student_test": inference_seconds,
                     "inference_ms_per_candidate": inference_seconds * 1000 / max(len(student_candidates), 1),
                     **metrics})
    return rows, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Train candidate table")
    parser.add_argument("--student-test", type=Path, required=True, help="Student-test candidate table")
    parser.add_argument("--train-obs", type=Path, required=True, help="Train observation sidecar")
    parser.add_argument("--student-test-obs", type=Path, required=True, help="Student-test observation sidecar")
    parser.add_argument("--results-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", choices=sorted(model_roster()),
                        help="Model keys to run; omit to run all five")
    parser.add_argument("--exclude-trips", nargs="+", default=OUTLIER_TRIPS,
                        help="Also report metrics excluding these trip IDs")
    args = parser.parse_args()

    train = read_table(args.train); student = read_table(args.student_test)
    train_obs = read_table(args.train_obs); student_obs = read_table(args.student_test_obs)
    features = validate_candidates(train, "train")
    student_features = validate_candidates(student, "student_test")
    if features != student_features:
        raise ValueError("Train and student-test feature columns/order must match")
    validate_observations(train_obs, "train_obs"); validate_observations(student_obs, "student_test_obs")
    validate_pair(train, train_obs, "train"); validate_pair(student, student_obs, "student_test")
    if set(train["trip_id"]) & set(student["trip_id"]):
        raise ValueError("A trip_id appears in both train and student_test")

    excluded = [normalize_trip_id(value) for value in args.exclude_trips]
    available = model_roster()
    names = args.models or list(available)
    selected = {name: available[name] for name in names}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S_") + "_".join(names)
    run_dir = args.results_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)

    cv_summaries: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    student_summaries: list[dict[str, Any]] = []
    for name, (template, config) in selected.items():
        print(f"Evaluating {name}...")
        cv_rows, fold_rows = run_cv(name, template, config, train, train_obs, features, args.folds, excluded)
        student_rows, _ = evaluate_student(name, template, config, train, student, student_obs,
                                           features, excluded)
        cv_summaries.extend(cv_rows); fold_summaries.extend(fold_rows); student_summaries.extend(student_rows)

    pd.DataFrame(cv_summaries).to_csv(run_dir / "model_comparison_cv.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(run_dir / "model_cv_folds.csv", index=False)
    pd.DataFrame(student_summaries).to_csv(run_dir / "model_comparison_student_test.csv", index=False)
    metadata = {"seed": SEED, "folds": args.folds, "models": names,
                "train_candidates": str(args.train), "student_test_candidates": str(args.student_test),
                "train_observations": str(args.train_obs), "student_test_observations": str(args.student_test_obs),
                "feature_columns": features, "outlier_trips_excluded_in_second_slice": excluded,
                "run_timestamp": datetime.now().astimezone().isoformat()}
    (run_dir / "model_run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved run outputs to {run_dir}")


if __name__ == "__main__":
    main()
