# MIT License - Bryan Cheng, 2026
"""M4 figures: paired-difference forest plot + Pareto (acc, ITR) scatter.

Reads experiments/{m4_baselines,m8_loso_bci2b}/summary.json and writes:
  figures/m4_baselines_vs_neuropolicy.png/pdf
  figures/m4_pareto_acc_itr.png/pdf
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M4 = ROOT / "experiments" / "m4_baselines" / "summary.json"
M8 = ROOT / "experiments" / "m8_loso_bci2b" / "summary.json"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_per_subject_returns():
    d4 = json.loads(M4.read_text())
    d8 = json.loads(M8.read_text())

    np_ret = np.array([r["results"]["cql_a1"]["metrics"]["return_mean"]
                       for r in d8["per_subject"]])

    bases = {}
    for name in ["random", "csp_lda_fixed", "eegnet_fixed",
                 "eegnet_threshold_0.55", "eegnet_threshold_0.65",
                 "eegnet_threshold_0.75"]:
        bases[name] = np.array([r["results"][name]["return_mean"]
                                for r in d4["per_subject"]])
    aggs = {row["baseline"]: row for row in d4["aggregate"]}
    return np_ret, bases, aggs


def figure_paired_diffs(np_ret, bases):
    """Forest plot: NP - baseline, per subject + mean ± 95% CI."""
    from scipy import stats

    order = ["csp_lda_fixed", "random", "eegnet_fixed",
             "eegnet_threshold_0.55", "eegnet_threshold_0.65",
             "eegnet_threshold_0.75"]
    pretty = {
        "csp_lda_fixed": "CSP+LDA fixed",
        "random": "random",
        "eegnet_fixed": "EEGNet fixed",
        "eegnet_threshold_0.55": "EEGNet thr=0.55",
        "eegnet_threshold_0.65": "EEGNet thr=0.65",
        "eegnet_threshold_0.75": "EEGNet thr=0.75",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y_positions = np.arange(len(order))[::-1]
    for i, name in enumerate(order):
        diffs = np_ret - bases[name]
        mean = diffs.mean()
        # 95% CI from t-dist (n=9)
        sem = diffs.std(ddof=1) / np.sqrt(len(diffs))
        ci = stats.t.ppf(0.975, df=len(diffs) - 1) * sem
        try:
            _, p_one = stats.wilcoxon(diffs, alternative="greater")
        except Exception:
            p_one = float("nan")
        wins = int((diffs > 0).sum())
        y = y_positions[i]
        # Per-subject points (jittered)
        ax.scatter(diffs, np.full_like(diffs, y, dtype=float)
                   + np.random.RandomState(0).uniform(-0.15, 0.15, size=len(diffs)),
                   color="#888", alpha=0.6, s=20, zorder=2)
        # Mean ± CI
        color = "C2" if mean > 0 else "C3"
        ax.errorbar(mean, y, xerr=ci, fmt="D", color=color, ecolor=color,
                    capsize=4, markersize=10, zorder=3)
        sig = " *" if p_one < 0.05 else ""
        pretty[order[i]] = pretty[order[i]] + f"  ({wins}/9{sig})"
    ax.axvline(0, color="k", lw=0.8, alpha=0.5, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([pretty[n] for n in order])
    ax.set_xlabel("Δ Episode Return (NeuroPolicy − baseline)")
    ax.set_title("LOSO BCI-IV-2b — paired difference vs. baselines\n"
                 "(diamond = mean, error bar = 95% CI; * = Wilcoxon $p<0.05$)")
    ax.set_xlim(-1.2, 1.2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "m4_baselines_vs_neuropolicy.png", dpi=150,
                bbox_inches="tight")
    fig.savefig(FIG / "m4_baselines_vs_neuropolicy.pdf",
                bbox_inches="tight")
    plt.close(fig)


def figure_pareto(np_ret, bases, aggs):
    d4 = json.loads(M4.read_text())
    d8 = json.loads(M8.read_text())

    order = ["random", "csp_lda_fixed", "eegnet_fixed",
             "eegnet_threshold_0.55", "eegnet_threshold_0.65",
             "eegnet_threshold_0.75"]
    pretty = {
        "random": "random",
        "csp_lda_fixed": "CSP+LDA fixed",
        "eegnet_fixed": "EEGNet fixed",
        "eegnet_threshold_0.55": "EEGNet τ=0.55",
        "eegnet_threshold_0.65": "EEGNet τ=0.65",
        "eegnet_threshold_0.75": "EEGNet τ=0.75",
    }
    colors = {
        "random": "#888888",
        "csp_lda_fixed": "C0",
        "eegnet_fixed": "C1",
        "eegnet_threshold_0.55": "C3",
        "eegnet_threshold_0.65": "C4",
        "eegnet_threshold_0.75": "C5",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Per-subject scatter
    for name in order:
        accs = [r["results"][name]["accuracy_on_commits"]
                for r in d4["per_subject"]]
        itrs = [r["results"][name]["information_transfer_rate"]
                for r in d4["per_subject"]]
        ax.scatter(accs, itrs, color=colors[name], alpha=0.4, s=30,
                   label=None)

    np_accs = [r["results"]["cql_a1"]["metrics"]["accuracy_on_commits"]
               for r in d8["per_subject"]]
    np_itrs = [r["results"]["cql_a1"]["metrics"]["information_transfer_rate"]
               for r in d8["per_subject"]]
    ax.scatter(np_accs, np_itrs, color="black", alpha=0.4, s=30, marker="^",
               label=None)

    # Means as bigger markers
    for name in order:
        a = aggs[name]
        ax.scatter(a["acc_mean"], a["itr_mean"], color=colors[name], s=200,
                   marker="*", edgecolor="black", linewidth=1.0,
                   label=pretty[name])
    np_agg = [row for row in json.loads(M8.read_text())["aggregate"]
              if row["config"] == "cql_a1"][0]
    ax.scatter(np_agg["acc_mean"], np_agg["itr_mean"], color="black", s=250,
               marker="*", edgecolor="white", linewidth=1.5,
               label="NeuroPolicy (cql_a1)")

    ax.set_xlabel("Commit Accuracy")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("LOSO BCI-IV-2b — Pareto front (accuracy vs. ITR)\n"
                 "(small markers = per-subject; stars = per-method mean across 9 subjects)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "m4_pareto_acc_itr.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "m4_pareto_acc_itr.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    np_ret, bases, aggs = load_per_subject_returns()
    figure_paired_diffs(np_ret, bases)
    figure_pareto(np_ret, bases, aggs)
    print("Wrote figures:")
    print(" ", FIG / "m4_baselines_vs_neuropolicy.png")
    print(" ", FIG / "m4_pareto_acc_itr.png")


if __name__ == "__main__":
    main()
