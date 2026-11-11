"""Regenerate report-summary figures from published aggregate CSVs (no training)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    overall = read_csv("results/overall_comparison.csv")
    baselines = {r["experiment"]: r for r in overall if r["arm"] == "no_stage"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for y, run, label, color in [(1, "full_image", "Run 1 · whole images", "#64748b"),
                                 (0, "leaf_crop", "Run 2 · leaf crops", "#0f766e")]:
        row = baselines[run]
        point, lo, hi = (float(row[k]) for k in ("macro_f1", "ci_lo", "ci_hi"))
        ax.errorbar(point, y, xerr=[[point-lo], [hi-point]], fmt="o",
                    capsize=5, color=color, lw=2, ms=8)
        ax.text(0.685, y, f"{point:.3f} [{lo:.3f}, {hi:.3f}]", va="center",
                ha="right", fontsize=10, color=color)
    ax.axvspan(0.60, 0.70, color="#ecfdf5")
    ax.axvline(0.60, ls="--", lw=1.4, color="#0f766e")
    ax.set(yticks=[1, 0], yticklabels=["Run 1 · 3,503 images", "Run 2 · 1,902 leaf crops"],
           xlim=(0.45, 0.70), ylim=(-0.65, 1.6),
           xlabel="Image-only macro-F1 · 95% plant-bootstrap confidence interval")
    ax.set_title("The visual baseline does not pass the run 2 interpretation gate",
                 loc="left", pad=18, fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.15)
    fig.text(0.02, 0.04, "Run 2 requires the LOWER CI bound >0.60. Run 1 is shown descriptively; the rule was added afterwards.",
             fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output / "baseline_gate.svg")
    qa = ROOT / "_qa"
    qa.mkdir(exist_ok=True)
    fig.savefig(qa / "baseline_gate.png", dpi=150)
    plt.close(fig)

    status = {r["status"]: int(r["All"]) for r in read_csv("results/full_image/status_counts.csv")
              if r["status"] != "All"}
    parts = read_csv("results/full_image/part_counts.csv")
    leaf = next(r for r in parts if r["part"] == "잎")
    n_leaf = int(leaf["잎끝마름"]) + int(leaf["황화"])
    total = sum(status.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    series = [
        (["No visible symptom (N)", "Visible symptom (Y)", "Other status (E)"],
         [status["N"], status["Y"], status["E"]], ["#64748b", "#b45309", "#cbd5e1"],
         "Most images lack visible symptoms"),
        (["Leaves", "Other plant parts"], [n_leaf, total-n_leaf],
         ["#0f766e", "#94a3b8"], "Only about half show leaves")]
    for ax, (labels, counts, colors, title) in zip(axes, series):
        yy = list(range(len(labels)))
        ax.barh(yy, [100*n/total for n in counts], color=colors, height=0.5)
        ax.set(yticks=yy, yticklabels=labels, xlim=(0, 120), xticks=[0, 25, 50, 75, 100],
               xlabel="Share of run 1 images (%)")
        ax.invert_yaxis()
        for y, n in zip(yy, counts):
            ax.text(100*n/total+2, y, f"{n:,} ({100*n/total:.1f}%)", va="center", fontsize=9)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="x", alpha=0.12)
    fig.text(0.02, 0.03, "AI22_003 validation subset · n = 3,503 · annotation meanings follow the report; E is kept separate.",
             fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=2)
    fig.savefig(output / "annotation_audit.svg")
    fig.savefig(qa / "annotation_audit.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
