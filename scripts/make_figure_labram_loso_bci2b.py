# MIT License, 2026
"""Figure: paired per-subject return for  LaBraM vs  EEGNet on bci2b LOSO.
Output: figures/m14_labram_vs_eegnet_bci2b.{png,pdf}
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def main():
    m8 = json.loads((ROOT / "experiments" / "loso_bci2b" / "summary.json").read_text())
    m14 = json.loads((ROOT / "experiments" / "labram_loso_bci2b" / "summary.json").read_text())
    subs = sorted([r["held_out"] for r in m14["per_subject"]])
    e8 = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["return_mean"]
           for r in m8["per_subject"]}
    e14 = {r["held_out"]: r["results"]["labram_cmdp"]["metrics"]["return_mean"]
            for r in m14["per_subject"]}
    rand = {r["held_out"]: r["results"]["random"]["return_mean"]
            for r in m14["per_subject"]}

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(subs))
    w = 0.27
    ax.bar(x - w, [rand[s] for s in subs], w, label="random", color="#aaaaaa", edgecolor="black")
    ax.bar(x, [e8[s] for s in subs], w, label="EEGNet ()", color="C0", edgecolor="black")
    ax.bar(x + w, [e14[s] for s in subs], w, label="LaBraM e2e ()", color="C3", edgecolor="black")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"sub{s}" for s in subs])
    ax.set_ylabel("Episode return (held-out subject)")
    ax.set_title("BCI-IV-2b LOSO: per-subject return\n(scalar CQL + CMDP eps=0.10; encoder differs)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "m14_labram_vs_eegnet_bci2b.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "m14_labram_vs_eegnet_bci2b.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", FIG / "m14_labram_vs_eegnet_bci2b.png")


if __name__ == "__main__":
    main()
