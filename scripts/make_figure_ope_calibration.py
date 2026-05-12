# MIT License, 2026
"""Regenerate the  calibration figure from the saved results.json.

Standalone so it can be re-run if the main  script crashes after writing
JSON (as happened with the deprecated 'savefig.bbox_inches' rcParam).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = ROOT / "experiments" / "ope_calibration" / "results.json"
OUT_PNG = ROOT / "figures" / "ope_calibration.png"
OUT_PDF = ROOT / "figures" / "ope_calibration.pdf"


def main():
    data = json.loads(RESULTS.read_text())
    rows = data["rows"]
    aux = data.get("aux", [])

    arr_gt = np.array([r["v_gt"] for r in rows])
    arr_fqe = np.array([r["v_fqe"] for r in rows])
    names = [r["name"] for r in rows]

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "figure.figsize": (8, 7), "figure.dpi": 150, "savefig.dpi": 300,
    })
    fig, ax = plt.subplots()
    lims = [min(arr_gt.min(), arr_fqe.min()) - 0.2,
            max(arr_gt.max(), arr_fqe.max()) + 0.2]
    ax.plot(lims, lims, "k--", alpha=0.4, label="ideal y = x", zorder=1)

    ax.scatter(arr_gt, arr_fqe, s=64, alpha=0.85,
               label=f"FQE   (r={data['pearson_fqe']:.3f}, ρ={data['spearman_fqe']:.3f})",
               zorder=3, color="C0")

    if aux:
        gt_aux = np.array([rows[r["idx"]]["v_gt"] for r in aux])
        pdis_aux = np.array([r["v_pdis"] for r in aux])
        dr_aux = np.array([r["v_dr"] for r in aux])
        ax.scatter(gt_aux, pdis_aux, marker="^", s=58, alpha=0.75,
                   label=f"PDIS (n={len(aux)})", color="C1", zorder=2)
        ax.scatter(gt_aux, dr_aux, marker="s", s=58, alpha=0.75,
                   label=f"DR    (n={len(aux)})", color="C2", zorder=2)

    # Annotate the random and best policies for orientation
    for j in (0, int(np.argmax(arr_gt))):
        ax.annotate(names[j], (arr_gt[j], arr_fqe[j]),
                    xytext=(6, -3), textcoords="offset points", fontsize=9,
                    color="0.4")
    ax.set_xlabel(r"On-policy MC ground-truth value $V_{GT}(\pi)$")
    ax.set_ylabel(r"OPE estimate $\hat V(\pi)$")
    ax.set_title(f"OPE calibration on BCI-IV-2b sub 4   "
                 f"({data['n_policies']} policies, RMSE={data['rmse_fqe']:.3f})")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
