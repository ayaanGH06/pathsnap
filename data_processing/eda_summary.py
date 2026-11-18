"""A13 aggregate EDA figure from committed CSVs; no GPS coordinates plotted."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path("results/data")
    inventory = pd.read_csv(root / "inventory.csv")
    quality = pd.read_csv(root / "quality.csv")
    network = pd.read_csv(root / "network.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for part, color in [("train", "#2371a8"), ("student_test", "#e89032")]:
        axes[0, 0].hist(inventory.loc[inventory.partition == part, "n_obs"],
                        bins=[0,500,1000,2000,4000,8000,12000,20000],
                        alpha=.7, label=part, color=color)
    axes[0, 0].set(title="GPS observations per trip (80 train + 10 student-test)",
                   xlabel="Observations", ylabel="Trips")
    axes[0, 0].legend()
    axes[0, 1].hist(quality.pct_1s_steps, bins=12, color="#2371a8")
    axes[0, 1].set(title="Sampling regularity (all 90 trips)",
                   xlabel="1-second steps per trip (%)", ylabel="Trips")
    axes[1, 0].hist(network.n_components, bins=15, color="#2371a8")
    axes[1, 0].set(title="Network components, undirected view (all 90)",
                   xlabel="Components per trip", ylabel="Trips")
    run = json.loads((root / "tables_run.json").read_text())
    misses = [100 * run[p]["miss_rate"] for p in ("train", "student_test")]
    # Pooled counts, from the frozen builder run and per-trip coverage checks.
    axes[1, 1].bar(["train", "student_test"], misses,
                   color=["#2371a8", "#e89032"])
    axes[1, 1].set(title="Reference candidate miss rate, frozen 50 m",
                   xlabel="Partition (80 / 10 trips)", ylabel="All observations missed (%)")
    for x,y in enumerate(misses):
        axes[1, 1].text(x, y+.25, f"{y:.2f}%", ha="center")
    axes[1, 1].set_ylim(0, 13)
    fig.suptitle("PathSnap v1: dataset and candidate coverage", fontsize=15)
    out = root / "figures" / "eda_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
