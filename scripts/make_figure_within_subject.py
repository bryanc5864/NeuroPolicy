# MIT License, 2026
"""Combined within-subject ablation figure: 3 datasets x 5 variants
showing the speed-accuracy frontier (acc vs ITR) across configs.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

datasets = [
    (" — bci2b sub.4 (2-class)", "cvar_cmdp_within"),
    (" — bci2a sub.3 (4-class)", "bci2a_within"),
    (" — Lee2019 sub.1 (2-class)", "lee2019_within"),
]
variants = [
    ("scalar_a1", "scalar CQL"),
    ("scalar_a1_cmdp_eps0.10", "scalar+CMDP"),
    ("cvar_a0.25", "CVaR-CQL"),
    ("cvar_a0.25_cmdp_eps0.10", "CVaR+CMDP"),
]
colors = {"scalar_a1": "#1f77b4", "scalar_a1_cmdp_eps0.10": "#ff7f0e",
          "cvar_a0.25": "#2ca02c", "cvar_a0.25_cmdp_eps0.10": "#d62728"}


fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, (title, name) in zip(axes, datasets):
    p = ROOT / "experiments" / name / "summary.json"
    if not p.exists():
        ax.set_title(f"{title}\n(no data)"); continue
    d = json.loads(p.read_text())
    rand_test = d.get("random_test", {})
    if rand_test:
        ax.scatter(rand_test["accuracy_on_commits"],
                    rand_test["information_transfer_rate"],
                    s=120, marker="X", color="#888", label="random",
                    edgecolor="black", linewidth=1, zorder=3)
    for k, label in variants:
        if k not in d: continue
        t = d[k]["test"]
        ax.scatter(t["accuracy_on_commits"], t["information_transfer_rate"],
                    s=120, marker="o", color=colors[k], label=label,
                    edgecolor="black", linewidth=1, zorder=3)
    ax.set_xlabel("Commit accuracy")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if name == datasets[0][1]:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

fig.suptitle("Within-subject speed-accuracy frontier across three MOABB datasets",
             fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "within_subject_acc_vs_itr.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG / "within_subject_acc_vs_itr.pdf", bbox_inches="tight")
plt.close(fig)
print("Wrote", FIG / "within_subject_acc_vs_itr.png")
