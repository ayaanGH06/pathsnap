"""Bayesian search for HistGradientBoosting on trip-grouped training CV.

The student-test partition is deliberately not loaded here. The primary
objective is observation-level Top-1 accuracy over every training observation
(candidate misses remain incorrect). The best search candidate receives the
agreed 5-fold evaluation; a separate finalization step compares it against the
baseline before selecting the export configuration.

Run from the repository root:
    python -m bayes_opt.optimize_hgb
"""
from __future__ import annotations

import itertools
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

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
FOLDS_FOR_SEARCH = 3
FINAL_FOLDS = 5
TRIAL_BUDGET = 10

# Discrete, deliberately compact search domain. The original baseline is always
# evaluated first so the search can prove whether tuning improves on it.
SPACE = {
    "max_iter": [100, 150, 200, 250],
    "learning_rate": [0.05, 0.08, 0.10, 0.15],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.0, 1.0, 5.0],
}
PARAM_NAMES = list(SPACE)
BASELINE = {
    "max_iter": 150,
    "learning_rate": 0.10,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}


def encode(config: dict[str, float | int]) -> np.ndarray:
    """Encode ordered discrete values in [0, 1] for the GP surrogate."""
    return np.array([
        SPACE[name].index(config[name]) / max(len(SPACE[name]) - 1, 1)
        for name in PARAM_NAMES
    ], dtype=float)


def decode(point: np.ndarray) -> dict[str, float | int]:
    config: dict[str, float | int] = {}
    for value, (name, choices) in zip(point, SPACE.items()):
        index = int(np.clip(np.rint(value * (len(choices) - 1)), 0, len(choices) - 1))
        config[name] = choices[index]
    return config


def all_configurations() -> list[dict[str, float | int]]:
    keys = list(SPACE)
    return [dict(zip(keys, values)) for values in itertools.product(*(SPACE[k] for k in keys))]


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    safe_sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best - 0.001  # small exploration margin
    z = improvement / safe_sigma
    ei = improvement * norm.cdf(z) + safe_sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def write_plots(frame: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(1, len(frame) + 1)
    values = frame["cv_top1_all"].to_numpy()
    incumbent = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, values, "o-", color="#2c6eaa", label="Trial CV Top-1")
    ax.step(x, incumbent, where="post", color="#d66b27", label="Best so far")
    ax.axhline(values[0], color="#666666", linestyle="--", linewidth=1,
               label="Baseline configuration")
    ax.set(xlabel="Bayesian search trial", ylabel="3-fold grouped CV Top-1 accuracy",
           title="HistGradientBoosting tuning (training trips only)")
    ax.set_ylim(max(0, values.min() - 0.02), min(1, values.max() + 0.02))
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "optimization_history.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(15, 3.8))
    for ax, name in zip(axes, PARAM_NAMES):
        values_x = pd.to_numeric(frame[name]).to_numpy()
        ax.scatter(values_x, frame["cv_top1_all"], c=x, cmap="viridis", s=48)
        ax.set_xlabel(name.replace("_", "\n"), fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("3-fold CV Top-1")
    fig.suptitle("Observed CV score by sampled hyperparameter\n(exploratory; interactions are not shown)")
    fig.tight_layout()
    fig.savefig(out_dir / "sampled_hyperparameters.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = read_table(TRAIN_PATH)
    observations = read_table(OBS_PATH)
    feature_columns = validate_candidates(train, "train")
    validate_observations(observations, "train_obs")
    validate_pair(train, observations, "train")

    template, baseline_meta = model_roster()["hist_gradient_boosting"]
    search_space = all_configurations()
    rng = np.random.default_rng(RANDOM_SEED)
    tested: set[tuple] = set()
    records: list[dict] = []
    coords: list[np.ndarray] = []
    scores: list[float] = []

    def evaluate(config: dict[str, float | int], method: str) -> None:
        key = tuple(config[name] for name in PARAM_NAMES)
        tested.add(key)
        model = template.set_params(**{f"model__{name}": value for name, value in config.items()})
        metadata = {**baseline_meta, **config, "family": "boosting"}
        started = time.perf_counter()
        summaries, folds = run_cv(
            "hist_gradient_boosting", model, metadata, train, observations,
            feature_columns, FOLDS_FOR_SEARCH, OUTLIER_TRIPS,
        )
        elapsed = time.perf_counter() - started
        all_row = next(row for row in summaries if row["evaluation_slice"] == "all_trips")
        clean_row = next(row for row in summaries if row["evaluation_slice"] == "excluding_33_54")
        fold_scores = [row["top1_accuracy"] for row in folds
                       if row["evaluation_slice"] == "all_trips"]
        record = {
            "trial": len(records) + 1,
            "acquisition": method,
            **config,
            "cv_folds": FOLDS_FOR_SEARCH,
            "cv_top1_all": all_row["top1_accuracy"],
            "cv_top1_excluding_33_54": clean_row["top1_accuracy"],
            "cv_macro_f1_all": all_row["macro_f1"],
            "cv_log_loss_all": all_row["candidate_log_loss"],
            "fold_top1_min": min(fold_scores),
            "fold_top1_max": max(fold_scores),
            "elapsed_seconds": elapsed,
        }
        records.append(record)
        coords.append(encode(config))
        scores.append(float(record["cv_top1_all"]))
        pd.DataFrame(records).to_csv(OUT_DIR / "optimization_trials.csv", index=False)
        print(f"Trial {record['trial']:02d}/{TRIAL_BUDGET}: CV Top-1="
              f"{record['cv_top1_all']:.4f}, clean-slice="
              f"{record['cv_top1_excluding_33_54']:.4f}, config={config}", flush=True)

    # Baseline first, then a small space-filling random design before EI.
    evaluate(BASELINE, "baseline")
    initial_count = min(4, TRIAL_BUDGET - 1)
    initial_pool = [c for c in search_space if tuple(c[n] for n in PARAM_NAMES) not in tested]
    rng.shuffle(initial_pool)
    for config in initial_pool[:initial_count]:
        evaluate(config, "random_initial")

    while len(records) < TRIAL_BUDGET:
        X = np.vstack(coords)
        y = np.asarray(scores)
        kernel = (ConstantKernel(1.0, (0.1, 10.0)) *
                  Matern(length_scale=np.ones(len(PARAM_NAMES)),
                         length_scale_bounds=(0.05, 10.0), nu=2.5) +
                  WhiteKernel(noise_level=0.0002, noise_level_bounds=(1e-6, 0.05)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      random_state=RANDOM_SEED, n_restarts_optimizer=1)
        gp.fit(X, y)
        remaining = [c for c in search_space
                     if tuple(c[n] for n in PARAM_NAMES) not in tested]
        candidate_X = np.vstack([encode(c) for c in remaining])
        mu, sigma = gp.predict(candidate_X, return_std=True)
        ei = expected_improvement(mu, sigma, float(y.max()))
        choice = int(np.argmax(ei))
        evaluate(remaining[choice], "expected_improvement")

    trials = pd.DataFrame(records)
    best_record = trials.loc[trials["cv_top1_all"].idxmax()]
    best_config = {name: (int(best_record[name]) if name in
                          {"max_iter", "max_leaf_nodes", "min_samples_leaf"}
                          else float(best_record[name])) for name in PARAM_NAMES}
    metadata = {
        "objective": "observation-level Top-1 accuracy across all training observations",
        "student_test_used_for_tuning": False,
        "search_cv_folds": FOLDS_FOR_SEARCH,
        "final_cv_folds": FINAL_FOLDS,
        "splitter": "GroupKFold grouped by trip_id",
        "seed": RANDOM_SEED,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "candidate_radius_m": 50,
        "feature_columns": feature_columns,
        "outlier_diagnostic_trips": OUTLIER_TRIPS,
        "trials": TRIAL_BUDGET,
        "best_config": best_config,
        "best_search_cv_top1_all": float(best_record["cv_top1_all"]),
        "baseline_search_cv_top1_all": float(trials.iloc[0]["cv_top1_all"]),
        "run_timestamp": datetime.now().astimezone().isoformat(),
    }
    (OUT_DIR / "optimization_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Evaluate the selected configuration using the agreed 5-fold trip-grouped CV.
    tuned = template.set_params(**{f"model__{name}": value for name, value in best_config.items()})
    tuned_meta = {**baseline_meta, **best_config, "family": "boosting_bayes_optimized"}
    final_summary, final_folds = run_cv(
        "hist_gradient_boosting_bayes_optimized", tuned, tuned_meta,
        train, observations, feature_columns, FINAL_FOLDS, OUTLIER_TRIPS,
    )
    pd.DataFrame(final_summary).to_csv(OUT_DIR / "best_5fold_cv.csv", index=False)
    pd.DataFrame(final_folds).to_csv(OUT_DIR / "best_5fold_folds.csv", index=False)
    write_plots(trials, OUT_DIR)
    print(f"Best config: {best_config}", flush=True)
    print(f"Saved Bayesian search and 5-fold validation outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
