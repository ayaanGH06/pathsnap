"""Assemble the shared model comparison CSV and Top-1 chart from run outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "models/model_comparison_all.csv"
OUT_PNG = RESULTS / "models/model_comparison_top1.png"

RUNS = {
    "Logistic Regression": RESULTS / "models/run_20261118_115731_logistic_regression_hist_gradient_boosting",
    "HistGradientBoosting (baseline)": RESULTS / "models/run_20261118_115731_logistic_regression_hist_gradient_boosting",
    "Random Forest": RESULTS / "models/random_forest_mlp_20261118",
    "MLP": RESULTS / "models/random_forest_mlp_20261118",
    "KNN": RESULTS / "models/knn_distance_weighted_20261118",
}


def row_for(name: str, run_dir: Path) -> dict:
    cv = pd.read_csv(run_dir / "model_comparison_cv.csv")
    student = pd.read_csv(run_dir / "model_comparison_student_test.csv")
    lookup = {
        "Logistic Regression": "logistic_regression",
        "HistGradientBoosting (baseline)": "hist_gradient_boosting",
        "Random Forest": "random_forest",
        "MLP": "mlp",
        "KNN": "knn_distance_weighted",
    }
    model = lookup[name]
    cv_model = cv.loc[cv["model"] == model]
    student_model = student.loc[student["model"] == model]
    cv_all = cv_model.loc[cv_model["evaluation_slice"] == "all_trips"].iloc[0]
    cv_clean = cv_model.loc[cv_model["evaluation_slice"] == "excluding_33_54"].iloc[0]
    test_all = student_model.loc[student_model["evaluation_slice"] == "all_trips"].iloc[0]
    return {
        "model": name,
        "cv_top1_all_trips": cv_all["top1_accuracy"],
        "cv_top1_excluding_33_54": cv_clean["top1_accuracy"],
        "cv_macro_f1_all_trips": cv_all["macro_f1"],
        "cv_macro_f1_excluding_33_54": cv_clean["macro_f1"],
        "cv_candidate_log_loss": cv_all["candidate_log_loss"],
        "student_test_top1": test_all["top1_accuracy"],
        "student_test_macro_f1": test_all["macro_f1"],
        "student_test_candidate_log_loss": test_all["candidate_log_loss"],
        "full_train_fit_seconds": test_all["fit_seconds_full_train"],
        "student_test_inference_seconds": test_all["inference_seconds_student_test"],
        "student_test_candidate_miss_rate": test_all["candidate_miss_rate"],
        "source_run": str(run_dir.relative_to(ROOT)),
    }


def main() -> None:
    rows = [row_for(name, path) for name, path in RUNS.items()]
    bayes_dir = RESULTS / "bayes/hgb_bayesopt_20261118"
    tuned_student_path = RESULTS / "bayes/selected_hgb_student_test.csv"
    tuned_cv_path = bayes_dir / "best_5fold_cv.csv"
    selection_path = bayes_dir / "selection_decision.json"
    if tuned_cv_path.exists():
        cv = pd.read_csv(tuned_cv_path)
        cv_all = cv.loc[cv["evaluation_slice"] == "all_trips"].iloc[0]
        cv_clean = cv.loc[cv["evaluation_slice"] == "excluding_33_54"].iloc[0]
        selection = {}
        if selection_path.exists():
            import json
            selection = json.loads(selection_path.read_text())
        test_fields = {}
        if (tuned_student_path.exists() and
                selection.get("selected_model") == "hist_gradient_boosting_bayes_optimized"):
            test = pd.read_csv(tuned_student_path)
            test_all = test.loc[test["evaluation_slice"] == "all_trips"].iloc[0]
            test_fields = {
                "student_test_top1": test_all["top1_accuracy"],
                "student_test_macro_f1": test_all["macro_f1"],
                "student_test_candidate_log_loss": test_all["candidate_log_loss"],
                "full_train_fit_seconds": test_all["fit_seconds_full_train"],
                "student_test_inference_seconds": test_all["inference_seconds_student_test"],
                "student_test_candidate_miss_rate": test_all["candidate_miss_rate"],
            }
        rows.append({
            "model": "HistGradientBoosting (Bayesian candidate)",
            "cv_top1_all_trips": cv_all["top1_accuracy"],
            "cv_top1_excluding_33_54": cv_clean["top1_accuracy"],
            "cv_macro_f1_all_trips": cv_all["macro_f1"],
            "cv_macro_f1_excluding_33_54": cv_clean["macro_f1"],
            "cv_candidate_log_loss": cv_all["candidate_log_loss"],
            "source_run": "results/bayes",
            **test_fields,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False)

    labels = frame["model"].tolist()
    y = range(len(labels))
    h = 0.24
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.7)))
    ax.barh([v - h for v in y], frame["cv_top1_all_trips"] * 100,
            height=h, label="5-fold CV · all trips", color="#5577aa")
    ax.barh(list(y), frame["cv_top1_excluding_33_54"] * 100,
            height=h, label="CV · excluding trips 33/54", color="#79a66b")
    if frame["student_test_top1"].notna().any():
        ax.barh([v + h for v in y], frame["student_test_top1"] * 100,
                height=h, label="Student test · 10 trips", color="#d38b45")
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_title("PathSnap candidate-ranking model comparison")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} and {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
