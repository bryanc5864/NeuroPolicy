# MIT License, 2026
"""Plot CVaR alpha sensitivity from experiments/cvar_alpha_ablation_bci2a/summary.json.

Two-panel figure:
  Panel A: commit accuracy vs alpha
  Panel B: wrong-commit rate and commit-rate vs alpha
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt


def main():
    summary_path = ROOT / "experiments" / "cvar_alpha_ablation_bci2a" / "summary.json"
    d = json.loads(summary_path.read_text())
    cs = d["cross_subject"]
    alphas = sorted([float(k) for k in cs.keys()])
    accs = [cs[f"{a:.2f}"]["acc_mean_of_means"] for a in alphas]
    accs_err = [cs[f"{a:.2f}"]["acc_std_of_means"] for a in alphas]
    itrs = [cs[f"{a:.2f}"]["itr_mean_of_means"] for a in alphas]
    itrs_err = [cs[f"{a:.2f}"]["itr_std_of_means"] for a in alphas]
    wrongs = [cs[f"{a:.2f}"]["wrong_mean_of_means"] for a in alphas]
    crates = [cs[f"{a:.2f}"]["commit_rate_mean_of_means"] for a in alphas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.errorbar(alphas, accs, yerr=accs_err, marker="o", capsize=4,
                  color="steelblue", linewidth=2, markersize=8, label="commit acc.")
    ax1.set_xlabel(r"CVaR $\alpha$ (eval time)")
    ax1.set_ylabel("Commit accuracy")
    ax1.set_xticks(alphas)
    ax1.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax1.set_ylim(0.80, 0.92)
    ax1.grid(alpha=0.3)
    ax1.set_title("(A) Accuracy is approximately flat\nin $\\alpha$ (cross-subject)")
    ax2t = ax1.twinx()
    ax2t.errorbar(alphas, itrs, yerr=itrs_err, marker="s", capsize=4,
                    color="firebrick", linewidth=2, markersize=7, label="ITR")
    ax2t.set_ylabel("ITR (bits/min)", color="firebrick")
    ax2t.tick_params(axis="y", colors="firebrick")
    ax2t.set_ylim(20, 45)
    ax1.legend(loc="upper left")
    ax2t.legend(loc="lower right")

    ax2.plot(alphas, wrongs, "o-", color="darkred", linewidth=2, markersize=8,
              label="wrong-commit rate")
    ax2.plot(alphas, crates, "s-", color="forestgreen", linewidth=2, markersize=8,
              label="commit rate")
    ax2.axhline(0.10, color="black", linestyle="--", linewidth=1, alpha=0.6,
                  label="CMDP $\\epsilon=0.10$")
    ax2.set_xlabel(r"CVaR $\alpha$ (eval time)")
    ax2.set_ylabel("Rate")
    ax2.set_xticks(alphas)
    ax2.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax2.set_ylim(0, 0.35)
    ax2.grid(alpha=0.3)
    ax2.set_title("(B) Lower $\\alpha$ = fewer commits,\nlower wrong-rate")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out = ROOT / "figures" / "cvar_alpha_sensitivity.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
