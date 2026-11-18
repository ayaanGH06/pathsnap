"""Assemble the shared model comparison CSV and metric figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "models"
OUT_CSV = OUT_DIR / "model_comparison_all.csv"
OUT_TOP1 = OUT_DIR / "model_comparison_top1.png"
OUT_MACRO_F1 = OUT_DIR / "model_comparison_macro_f1.png"
OUT_STUDENT = OUT_DIR / "model_comparison_student_test.png"
OUT_LOG_LOSS = OUT_DIR / "model_comparison_log_loss.png"

RUNS = {
    "Logistic Regression": RESULTS / "models/run_20261118_115731_logistic_regression_hist_gradient_boosting",
    "HistGradientBoosting (baseline)": RESULTS / "models/run_20261118_115731_logistic_regression_hist_gradient_boosting",
    "Random Forest": RESULTS / "models/random_forest_mlp_20261118",
    "MLP": RESULTS / "models/random_forest_mlp_20261118",
    "KNN": RESULTS / "models/knn_distance_weighted_20261118",
}
MODEL_KEYS = {
    "Logistic Regression": "logistic_regression",
    "HistGradientBoosting (baseline)": "hist_gradient_boosting",
    "Random Forest": "random_forest",
    "MLP": "mlp",
    "KNN": "knn_distance_weighted",
}


def row_for(name: str, run_dir: Path) -> dict:
    cv = pd.read_csv(run_dir / "model_comparison_cv.csv")
    student = pd.read_csv(run_dir / "model_comparison_student_test.csv")
    cv_model = cv.loc[cv["model"] == MODEL_KEYS[name]]
    student_model = student.loc[student["model"] == MODEL_KEYS[name]]
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


def plot_bars(frame: pd.DataFrame, series: list[tuple[str, str]], output: Path,
              title: str, xlabel: str, *, percent: bool, note: str) -> None:
    labels = frame["model"].tolist()
    y = list(range(len(labels)))
    height = min(0.22, 0.68 / len(series))
    offsets = [(i - (len(series) - 1) / 2) * height for i in range(len(series))]
    colors = ["#5577aa", "#79a66b", "#d38b45"]
    scale = 100 if percent else 1
    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.72)))
    for i, (field, label) in enumerate(series):
        values = pd.to_numeric(frame[field], errors="coerce") * scale
        ax.barh([v + offsets[i] for v in y], values, height=height * .9,
                label=label, color=colors[i % len(colors)])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=.25)
    if percent:
        ax.set_xlim(0, 100)
    else:
        values = pd.to_numeric(frame[[column for column, _ in series]].stack(), errors="coerce")
        maximum = values.max()
        ax.set_xlim(0, float(maximum) * 1.18 if pd.notna(maximum) and maximum > 0 else 1)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.17),
              ncol=len(series), fontsize=9, frameon=False)
    fig.text(.01, .01, note, ha="left", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, .88))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def add_tuned_candidate(rows: list[dict]) -> None:
    bayes_dir = RESULTS / "bayes/hgb_bayesopt_20261118"
    cv_path = bayes_dir / "best_5fold_cv.csv"
    test_path = RESULTS / "bayes/tuned_hgb_student_test.csv"
    if not cv_path.exists():
        return
    cv = pd.read_csv(cv_path)
    cv_all = cv.loc[cv["evaluation_slice"] == "all_trips"].iloc[0]
    cv_clean = cv.loc[cv["evaluation_slice"] == "excluding_33_54"].iloc[0]
    fields = {}
    if test_path.exists():
        test = pd.read_csv(test_path)
        test_all = test.loc[test["evaluation_slice"] == "all_trips"].iloc[0]
        fields = {
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
        **fields,
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_for(name, run_dir) for name, run_dir in RUNS.items()]
    add_tuned_candidate(rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False)

    plot_bars(frame, [
        ("cv_top1_all_trips", "5-fold CV · all trips"),
        ("cv_top1_excluding_33_54", "CV · excluding trips 33/54"),
        ("student_test_top1", "Student-test · 10 trips"),
    ], OUT_TOP1, "PathSnap candidate-ranking Top-1 comparison", "Top-1 accuracy (%)",
       percent=True, note="Candidate misses remain incorrect; 33/54 exclusion is diagnostic only.")
    plot_bars(frame, [
        ("cv_macro_f1_all_trips", "5-fold CV · all trips"),
        ("cv_macro_f1_excluding_33_54", "CV · excluding trips 33/54"),
        ("student_test_macro_f1", "Student-test · 10 trips"),
    ], OUT_MACRO_F1, "PathSnap candidate-ranking Macro F1 comparison", "Macro F1 (%)",
       percent=True, note="The trip 33/54 exclusion is diagnostic only; primary selection uses all trips.")
    plot_bars(frame, [
        ("student_test_top1", "Top-1 accuracy"),
        ("student_test_macro_f1", "Macro F1"),
    ], OUT_STUDENT, "Student-test model comparison (10 trips)", "Score (%)",
       percent=True, note="All models use the same held-out student-test observations.")
    plot_bars(frame, [
        ("cv_candidate_log_loss", "5-fold CV · all trips"),
        ("student_test_candidate_log_loss", "Student-test · 10 trips"),
    ], OUT_LOG_LOSS, "Candidate probability log-loss comparison", "Binary candidate log loss (lower is better)",
       percent=False, note="Tuned-candidate student-test metrics are post-selection and did not affect model choice.")

    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    for path in (OUT_TOP1, OUT_MACRO_F1, OUT_STUDENT, OUT_LOG_LOSS):
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
