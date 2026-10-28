"""Plot saved public-safe Experiment 0 aggregates; no model or data acquisition."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/e000_20261028_greedy_collection_v3"


def main():
    summary = json.loads((RUN / "public_summary.json").read_text(encoding="utf-8"))
    conditions = sorted((c for c in summary["conditions"] if c["budget"]), key=lambda c: c["budget"])
    budgets = [c["budget"] for c in conditions]
    ranges = [100 * c["median_relative_timing_range"] for c in conditions]
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.6, 4.7), layout="constrained")
    bars = ax.bar([str(b) for b in budgets], ranges, color=["#287d60" if r <= 20 else "#bd4b42" for r in ranges], width=.62)
    ax.axhline(20, color="#333333", linestyle="--", label="Frozen criterion: at most 20%")
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in ranges], padding=5)
    ax.set(xlabel="Uniform frame budget", ylabel="Median repetition range (%)", ylim=(0, 76), title="Timing stability: four of five budgets fail")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(RUN / "figures/timing_stability.png", dpi=180)
    plt.close(fig)
    diag = json.loads((ROOT / "results/e000_20261028_cpu_stage_v4/verification.json").read_text(encoding="utf-8"))["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    x = list(range(len(diag)))
    axes[0].bar([v-.18 for v in x], [d["total_cpu_prepare_seconds"] for d in diag], .36, label="Total preparation", color="#315f8c")
    bars = axes[0].bar([v+.18 for v in x], [d["decode_selection_rgb_resize_stack"] for d in diag], .36, label="Video handling", color="#e19b43")
    axes[0].bar_label(bars, fmt="%.3f", padding=4)
    axes[0].set(xticks=x, xticklabels=[str(d["budget"]) for d in diag], xlabel="Frames", ylabel="Seconds", ylim=(0, 3.5), title="Video handling dominates CPU cost")
    axes[0].legend(frameon=False, fontsize=10)
    bars = axes[1].bar([str(d["budget"]) for d in diag], [1000*d["native_processor"] for d in diag], color="#52968b", width=.55)
    axes[1].bar_label(bars, fmt="%.1f ms", padding=4)
    axes[1].set(xlabel="Frames", ylabel="Milliseconds", ylim=(0, 60), title="Native processor is a small component")
    fig.suptitle("Separate CPU-only diagnostic; does not replace full-request timing", fontsize=12)
    fig.savefig(RUN / "figures/cpu_stage_cost.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
