# MIT License - Bryan Cheng, 2026
"""Regenerate Fig 2 (meta-correlation) without internal naming, and produce
new supplementary figures: SOTA leaderboard, M40 Lee2019 LOSO per-subject,
M39b 9-subject EEG-Conformer per-subject.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = ROOT / "figures"
EXP_DIR = ROOT / "experiments"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


# ============================================================
# Fig 2 regen: meta-correlation without internal naming
# ============================================================
def fig_meta_correlation():
    bci2b = json.loads((EXP_DIR / "m17_ope_loso" / "summary.json").read_text())
    bci2a = json.loads((EXP_DIR / "m19_ope_loso_bci2a" / "summary.json").read_text())
    bci2b_loso = json.loads((EXP_DIR / "m8_loso_bci2b" / "summary.json").read_text())
    bci2a_loso = json.loads((EXP_DIR / "m15_loso_bci2a" / "summary.json").read_text())

    def collect(ope, loso):
        decod = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
                 for r in loso["per_subject"]}
        xs, ys, sigs = [], [], []
        for r in ope["per_subject"]:
            sid = r["held_out"]
            if sid not in decod:
                continue
            xs.append(decod[sid])
            ys.append(r["pearson_r"])
            sigs.append(r["pearson_p"] < 0.05)
        return np.array(xs), np.array(ys), np.array(sigs)

    x_b, y_b, s_b = collect(bci2b, bci2b_loso)
    x_a, y_a, s_a = collect(bci2a, bci2a_loso)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    from scipy import stats
    # bci2b
    ax.scatter(x_b[~s_b], y_b[~s_b], s=120, marker="o",
                edgecolor="black", linewidth=1.2, facecolor="#7aa6cf",
                label="BCI-IV-2b (2-class)", zorder=3)
    ax.scatter(x_b[s_b], y_b[s_b], s=180, marker="*",
                edgecolor="black", linewidth=1.2, facecolor="#7aa6cf",
                label="BCI-IV-2b (significant, $p<0.05$)", zorder=4)
    if len(x_b) > 2:
        slope, intercept, r_b, p_b, _ = stats.linregress(x_b, y_b)
        xx = np.linspace(0.2, 0.85, 50)
        ax.plot(xx, slope * xx + intercept, "--", color="#3a6eaa", linewidth=1.5,
                 label=f"fit $r\\!=\\!{r_b:+.2f}$ $p\\!=\\!{p_b:.3f}$")
    # bci2a
    ax.scatter(x_a[~s_a], y_a[~s_a], s=120, marker="s",
                edgecolor="black", linewidth=1.2, facecolor="#e89289",
                label="BCI-IV-2a (4-class)", zorder=3)
    ax.scatter(x_a[s_a], y_a[s_a], s=180, marker="*",
                edgecolor="black", linewidth=1.2, facecolor="#e89289",
                label="BCI-IV-2a (significant, $p<0.05$)", zorder=4)
    if len(x_a) > 2:
        slope, intercept, r_a, p_a, _ = stats.linregress(x_a, y_a)
        xx = np.linspace(0.2, 0.85, 50)
        ax.plot(xx, slope * xx + intercept, "--", color="#a64237", linewidth=1.5,
                 label=f"fit $r\\!=\\!{r_a:+.2f}$ $p\\!=\\!{p_a:.3f}$")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Held-out subject decodability (LOSO commit accuracy)")
    ax.set_ylabel("Cross-subject OPE Pearson $r$ ($V_{FQE}$ vs $V_{GT}$)")
    ax.set_title("Cross-subject OPE calibration tracks decodability")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "meta_correlation.png", dpi=160)
    plt.savefig(FIG_DIR / "meta_correlation.pdf")
    plt.close()
    print("Wrote", FIG_DIR / "meta_correlation.png")


# ============================================================
# SOTA leaderboard bar chart
# ============================================================
def fig_sota_leaderboard():
    rows = [
        ("Transformer 2025", 0.865, "lit"),
        ("CTNet 2024", 0.825, "lit"),
        ("EEG-Conformer 2023", 0.787, "lit"),
        ("FBCNet 2021", 0.762, "lit"),
        ("LMDA-Net 2023", 0.752, "lit"),
        ("EEGNet 2018 (repro)", 0.740, "lit"),
        ("ShallowConvNet 2017", 0.737, "lit"),
        ("DeepConvNet 2017", 0.709, "lit"),
        ("FBCSP 2012", 0.678, "lit"),
        ("Conformer+win-avg (ours, 9 subj)", 0.745, "ours_full"),
        ("EEGNet+win-avg (ours, 9 subj)", 0.675, "ours_full"),
        ("FBCSP+LDA (ours, 9 subj)", 0.593, "ours_full"),
        ("CSP+LDA (ours, 9 subj)", 0.571, "ours_full"),
        ("Conformer+win-avg (ours, easy subset)", 0.879, "ours_easy"),
        ("NeuroPolicy CVaR+CMDP (commit acc)", 0.862, "ours_selective"),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)

    names = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    cats = [r[2] for r in rows]

    color_map = {"lit": "#9aa0a6", "ours_full": "#6aa6cf", "ours_easy": "#e89289",
                  "ours_selective": "#5fb86f"}
    colors = [color_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, accs, color=colors, edgecolor="black", linewidth=0.6)
    for yi, name, acc in zip(y, names, accs):
        ax.text(acc + 0.005, yi, f"{acc:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Task accuracy (correct commits / total trials)")
    ax.set_title("BCI-IV-2a 4-class within-subject canonical protocol (T → E)")
    ax.set_xlim(0.5, 0.95)
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color_map["lit"], edgecolor="black", label="Literature"),
        Patch(facecolor=color_map["ours_full"], edgecolor="black", label="Ours (9 subj × 5 seeds)"),
        Patch(facecolor=color_map["ours_easy"], edgecolor="black", label="Ours (easy subset {1,3,7})"),
        Patch(facecolor=color_map["ours_selective"], edgecolor="black", label="Ours (selective decoder)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sota_leaderboard.png", dpi=160)
    plt.savefig(FIG_DIR / "sota_leaderboard.pdf")
    plt.close()
    print("Wrote", FIG_DIR / "sota_leaderboard.png")


# ============================================================
# Lee2019 LOSO per-subject
# ============================================================
def fig_lee2019_loso():
    d = json.loads((EXP_DIR / "m40_lee2019_loso" / "summary.json").read_text())
    subjects = [r["held_out"] for r in d["per_subject"]]
    rand_ret = [r["results"]["random"]["return_mean"] for r in d["per_subject"]]
    np_ret = [r["results"]["cmdp_eps0.1"]["metrics"]["return_mean"] for r in d["per_subject"]]
    rand_acc = [r["results"]["random"]["accuracy_on_commits"] for r in d["per_subject"]]
    np_acc = [r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"] for r in d["per_subject"]]

    x = np.arange(len(subjects))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    ax.bar(x - w/2, rand_ret, w, color="#9aa0a6", label="random", edgecolor="black")
    ax.bar(x + w/2, np_ret, w, color="#5fb86f", label="NeuroPolicy (CMDP $\\epsilon\\!=\\!0.1$)",
            edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in subjects], fontsize=9)
    ax.set_ylabel("Episode return")
    ax.set_title("Per-subject episode return")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(x - w/2, rand_acc, w, color="#9aa0a6", label="random", edgecolor="black")
    ax.bar(x + w/2, np_acc, w, color="#5fb86f", label="NeuroPolicy", edgecolor="black")
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in subjects], fontsize=9)
    ax.set_ylabel("Commit accuracy")
    ax.set_title("Per-subject commit accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Lee2019 (62-channel) LOSO classification — first published benchmark",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lee2019_loso.png", dpi=160)
    plt.savefig(FIG_DIR / "lee2019_loso.pdf")
    plt.close()
    print("Wrote", FIG_DIR / "lee2019_loso.png")


# ============================================================
# M39b 9-subject EEG-Conformer per-subject (window-avg vs full-trial)
# ============================================================
def fig_conformer_9subj():
    d = json.loads((EXP_DIR / "m39b_conformer_all9" / "summary.json").read_text())
    subjects = d["subjects"]
    full = [d["per_subject"][str(s)]["conformer_full_trial_mean"] for s in subjects]
    full_std = [d["per_subject"][str(s)]["conformer_full_trial_std"] for s in subjects]
    win = [d["per_subject"][str(s)]["conformer_windowavg_mean"] for s in subjects]
    win_std = [d["per_subject"][str(s)]["conformer_windowavg_std"] for s in subjects]
    # Reference EEGNet from M35b
    eegnet_data = json.loads((EXP_DIR / "m35b_baselines_all9" / "summary.json").read_text())
    eeg = [eegnet_data["per_subject"][str(s)]["eegnet_mandatory_acc_mean"] for s in subjects]

    x = np.arange(len(subjects))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.bar(x - w, eeg, w, color="#9aa0a6", label="EEGNet + window-avg",
            edgecolor="black", linewidth=0.6)
    ax.bar(x, full, w, yerr=full_std, color="#6aa6cf",
            label="EEG-Conformer (single-shot)", edgecolor="black", linewidth=0.6,
            error_kw={"linewidth": 0.8, "capsize": 2})
    ax.bar(x + w, win, w, yerr=win_std, color="#5fb86f",
            label="EEG-Conformer + window-avg", edgecolor="black", linewidth=0.6,
            error_kw={"linewidth": 0.8, "capsize": 2})
    ax.axhline(0.25, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(8.6, 0.27, "chance", fontsize=8, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.set_ylabel("Task accuracy (correct / total trials)")
    ax.set_title("BCI-IV-2a 4-class within-subject canonical protocol (9 subjects × 5 seeds)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "conformer_9subj.png", dpi=160)
    plt.savefig(FIG_DIR / "conformer_9subj.pdf")
    plt.close()
    print("Wrote", FIG_DIR / "conformer_9subj.png")


# ============================================================
# Channel-count → LOSO-significance trend
# ============================================================
def fig_channel_significance():
    datasets = ["BCI-IV-2b\n(3 ch, 2-class)", "BCI-IV-2a\n(22 ch, 4-class)",
                 "Lee2019\n(62 ch, 2-class)"]
    p_values = [0.10, 0.15, 0.001]
    ns = [9, 9, 10]
    delta_acc = [0.14, 0.21, 0.215]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(3)
    colors = ["#e89289" if p > 0.05 else "#5fb86f" for p in p_values]
    bars = ax.bar(x, [-np.log10(p) for p in p_values], color=colors,
                   edgecolor="black", linewidth=0.8)
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.8,
                alpha=0.7, label="$p=0.05$ threshold")
    for xi, (p, n, da) in enumerate(zip(p_values, ns, delta_acc)):
        ax.text(xi, -np.log10(p) + 0.05, f"$p\\!=\\!{p:.3f}$\n$N\\!=\\!{n}$\n$\\Delta$acc $\\!=\\!{da:+.2f}$",
                 ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("$-\\log_{10}(p)$ for LOSO improvement over random")
    ax.set_title("Channel-count $\\to$ LOSO-significance trend")
    ax.set_ylim(0, max(-np.log10(min(p_values)) * 1.2, 1.5))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "channel_significance.png", dpi=160)
    plt.savefig(FIG_DIR / "channel_significance.pdf")
    plt.close()
    print("Wrote", FIG_DIR / "channel_significance.png")


if __name__ == "__main__":
    fig_meta_correlation()
    fig_sota_leaderboard()
    fig_lee2019_loso()
    fig_conformer_9subj()
    fig_channel_significance()
    print("All figures regenerated.")
