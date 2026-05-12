# MIT License - Bryan Cheng, 2026
"""Combined meta-correlation figure: bci2b (M17b) + bci2a (M19) cross-subject
OPE calibration vs subject decodability.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def load(loso_path, ope_path, cfg_key):
    loso = json.loads(loso_path.read_text())
    ope = json.loads(ope_path.read_text())
    accs = {r["held_out"]: r["results"][cfg_key]["metrics"]["accuracy_on_commits"]
             for r in loso["per_subject"]}
    rs = {r["held_out"]: r["pearson_r"] for r in ope["per_subject"]}
    ps = {r["held_out"]: r["pearson_p"] for r in ope["per_subject"]}
    rows = []
    for sid in sorted(rs.keys()):
        if sid in accs:
            rows.append({"sid": sid, "acc": accs[sid], "r": rs[sid], "p": ps[sid]})
    return rows


def main():
    bci2b = load(ROOT / "experiments" / "m8_loso_bci2b" / "summary.json",
                  ROOT / "experiments" / "m17_ope_loso" / "summary.json",
                  "cmdp_eps0.1")
    bci2a = load(ROOT / "experiments" / "m15_loso_bci2a" / "summary.json",
                  ROOT / "experiments" / "m19_ope_loso_bci2a" / "summary.json",
                  "cmdp_eps0.1")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for ds_rows, marker, color, label, sym in [
        (bci2b, "o", "C0", "BCI-IV-2b (2-class)", "circle"),
        (bci2a, "s", "C3", "BCI-IV-2a (4-class)", "square"),
    ]:
        accs = np.array([r["acc"] for r in ds_rows])
        rs = np.array([r["r"] for r in ds_rows])
        sig = np.array([r["p"] < 0.05 for r in ds_rows])
        # Significant
        ax.scatter(accs[sig], rs[sig], s=160, marker="*", color=color,
                    edgecolor="black", linewidth=1.5, zorder=4,
                    label=f"{label} (p<0.05)")
        # NS
        ax.scatter(accs[~sig], rs[~sig], s=110, marker=marker, color=color,
                    edgecolor="black", linewidth=1.0, alpha=0.6, zorder=3,
                    label=f"{label}")
        # Linear fit
        slope, intercept = np.polyfit(accs, rs, 1)
        xs = np.linspace(min(accs.min(), 0.20) - 0.02, max(accs.max(), 0.80) + 0.02, 100)
        pr, p_pr = stats.pearsonr(accs, rs)
        ax.plot(xs, slope * xs + intercept, "--", color=color, lw=1.5,
                 alpha=0.8, label=f"  fit r={pr:+.2f} p={p_pr:.3f}")

    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("Held-out subject decodability (LOSO commit accuracy)")
    ax.set_ylabel("Cross-subject OPE Pearson r (V_FQE vs V_GT)")
    ax.set_xlim(0.20, 0.80)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Cross-subject OPE calibration tracks decodability\n"
                 "Replicates across bci2b (M17) and bci2a (M19)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "m17_m19_meta_correlation.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "m17_m19_meta_correlation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", FIG / "m17_m19_meta_correlation.png")


if __name__ == "__main__":
    main()
