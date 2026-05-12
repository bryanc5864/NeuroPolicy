# MIT License, 2026
"""Generate the  meta-correlation figure: subject decodability vs
cross-subject OPE Pearson r.

Visualises the headline meta-finding: corr( EEGNet acc,  OPE r) = +0.827
(p=0.006). Cross-subject OPE calibration quality is highly predicted by
held-out decodability.

Output: figures/m17_meta_correlation.{png,pdf}
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def main():
    m8 = json.loads((ROOT / "experiments" / "loso_bci2b" / "summary.json").read_text())
    m17 = json.loads((ROOT / "experiments" / "ope_loso_bci2b" / "summary.json").read_text())

    m8_acc = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
              for r in m8["per_subject"]}
    rows = []
    for r in m17["per_subject"]:
        sid = r["held_out"]
        if sid in m8_acc:
            rows.append({"sid": sid, "m8_acc": m8_acc[sid],
                         "ope_r": r["pearson_r"], "ope_p": r["pearson_p"]})

    if len(rows) < 3:
        print("Not enough data."); return
    accs = np.array([r["m8_acc"] for r in rows])
    rs = np.array([r["ope_r"] for r in rows])
    pearson, p_pearson = stats.pearsonr(accs, rs)
    spearman, p_spearman = stats.spearmanr(accs, rs)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    # Scatter
    sig = np.array([r["ope_p"] < 0.05 for r in rows])
    ax.scatter(accs[~sig], rs[~sig], s=120, color="C0", edgecolor="black",
                linewidth=1.2, alpha=0.7, label="OPE p ≥ 0.05",
                zorder=3)
    ax.scatter(accs[sig], rs[sig], s=140, color="C3", edgecolor="black",
                linewidth=1.5, marker="*", label="OPE p < 0.05",
                zorder=4)
    # Annotate subject IDs
    for r in rows:
        ax.annotate(f"sub{r['sid']}", (r["m8_acc"], r["ope_r"]),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=8, alpha=0.85)
    # Regression line
    slope, intercept = np.polyfit(accs, rs, 1)
    xs = np.linspace(accs.min() - 0.02, accs.max() + 0.02, 100)
    ys = slope * xs + intercept
    ax.plot(xs, ys, "--", color="grey", lw=1.5,
             label=f"linear fit (r={pearson:+.3f}, p={p_pearson:.3f})")
    # Reference lines
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.axvline(0.5, color="black", lw=0.5, alpha=0.4, linestyle=":")
    ax.fill_betweenx([-1, 1], 0.74, 0.78, color="green", alpha=0.05,
                      label="strong cluster (acc≥0.74)")
    ax.fill_betweenx([-1, 1], 0.50, 0.60, color="red", alpha=0.05,
                      label="weak cluster (acc≤0.60)")

    ax.set_xlabel("Held-out subject decodability ( EEGNet commit accuracy)")
    ax.set_ylabel("Cross-subject OPE calibration\nPearson r (V_FQE vs V_GT)")
    ax.set_xlim(0.45, 0.80)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Cross-subject OPE calibration tracks decodability\n"
                 f"Pearson r = {pearson:+.3f}, p={p_pearson:.3f} ; "
                 f"Spearman ρ = {spearman:+.3f}, p={p_spearman:.3f}",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "m17_meta_correlation.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "m17_meta_correlation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", FIG / "m17_meta_correlation.png")


if __name__ == "__main__":
    main()
