# MIT License, 2026
"""Generate figures from  LOSO results: Pareto frontier + per-subject bars."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY = ROOT / "experiments" / "loso_bci2b" / "summary.json"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PLT_RC = {
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.figsize": (8, 6), "figure.dpi": 150, "savefig.dpi": 300,
}


def main():
    data = json.loads(SUMMARY.read_text())
    per_subj = data["per_subject"]
    configs = data["configs"]
    cfg_names = [c["name"] for c in configs]
    subjects = sorted({r["held_out"] for r in per_subj})

    # --- Per-subject ITR vs accuracy (Pareto-style) ---
    plt.rcParams.update(PLT_RC)
    fig, ax = plt.subplots()

    colors = {n: f"C{i}" for i, n in enumerate(cfg_names)}
    # Random baseline cloud
    rand_acc = [r["results"]["random"]["accuracy_on_commits"] for r in per_subj]
    rand_itr = [r["results"]["random"]["information_transfer_rate"] for r in per_subj]
    ax.scatter(rand_acc, rand_itr, marker="x", s=44, color="0.5",
               alpha=0.7, label="random", zorder=2)

    for cn in cfg_names:
        accs = [r["results"][cn]["metrics"]["accuracy_on_commits"] for r in per_subj]
        itrs = [r["results"][cn]["metrics"]["information_transfer_rate"] for r in per_subj]
        ax.scatter(accs, itrs, s=58, color=colors[cn], alpha=0.85, label=cn, zorder=3)
        # connect mean
        ax.scatter([np.mean(accs)], [np.mean(itrs)], s=140, color=colors[cn],
                   marker="*", edgecolor="black", linewidth=1.0, zorder=4,
                   label=f"{cn} mean")

    ax.set_xlabel("Accuracy on commits")
    ax.set_ylabel("Information Transfer Rate (bits/min)")
    ax.set_title(f"BCI-IV-2b LOSO ({len(subjects)} subjects) — accuracy vs ITR")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9, ncol=2)
    fig.savefig(FIG_DIR / "loso_bci2b_acc_vs_itr.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "loso_bci2b_acc_vs_itr.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'loso_bci2b_acc_vs_itr.png'}")

    # --- Per-subject return (bars) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / (len(cfg_names) + 1)
    x = np.arange(len(subjects))
    for i, cn in enumerate(["random"] + cfg_names):
        if cn == "random":
            vals = [r["results"]["random"]["return_mean"] for r in per_subj]
            color = "0.5"
        else:
            vals = [r["results"][cn]["metrics"]["return_mean"] for r in per_subj]
            color = colors[cn]
        ax.bar(x + i * width - 0.4, vals, width=width, color=color, label=cn)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"sub{s}" for s in subjects])
    ax.set_ylabel("Episode return (held-out subject)")
    ax.set_title("BCI-IV-2b LOSO — per-subject episode return")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(FIG_DIR / "loso_bci2b_return_per_subject.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "loso_bci2b_return_per_subject.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'loso_bci2b_return_per_subject.png'}")

    # --- Aggregate table (text) ---
    print("\n=== AGGREGATE LOSO BCI-IV-2b ===")
    rows = data.get("aggregate", [])
    print(f"{'config':18s} {'return':>14s} {'acc':>14s} {'ITR':>14s} {'wrong':>10s}")
    rand = data.get("random", {})
    if rand:
        print(f"{'random':18s} {rand.get('return_mean', float('nan')):>14.3f} "
              f"{rand.get('acc_mean', float('nan')):>14.3f} {'':>14s} {'':>10s}")
    for r in rows:
        print(f"{r['config']:18s} {r['return_mean']:>+7.3f}±{r['return_std']:.2f}  "
              f"{r['acc_mean']:>7.3f}±{r['acc_std']:.3f}  "
              f"{r['itr_mean']:>7.1f}±{r['itr_std']:.1f}  "
              f"{r['wrong_rate_mean']:>10.3f}")


if __name__ == "__main__":
    main()
