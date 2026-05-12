# MIT License - Bryan Cheng, 2026
"""M21: FQE systematic-bias decomposition.

M20 surfaced that naive cross-subject FQE policy selection underperforms
a no-OPE default. Why? This script decomposes the FQE bias across the
combined N=18 held-out subjects (M17b bci2b + M19 bci2a) and 6 policies,
yielding 108 V_FQE estimates paired with V_GT.

We compute:
  1. Per-policy mean bias  b_p = mean_subjects (V_FQE - V_GT)
     → which policies does FQE systematically over/under-estimate?
  2. FQE shrinkage: V_FQE vs V_GT slope (mean V_FQE / mean V_GT)
     → does FQE pull toward zero (low absolute value)?
  3. Per-decodability bias: bias_high (acc>=0.5) vs bias_low (<0.5)
     → does the bias depend on subject decodability?
  4. Debias correction: subtract per-policy bias, recompute per-subject
     Pearson r and naive vs default V_GT.

Output: experiments/m21_fqe_bias/summary.json + figures/m21_fqe_bias.png
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m21_bias")


def load_corpus():
    """Concatenate M17b + M19 per-subject OPE rows, joined with M8/M15 acc."""
    bci2b_ope = json.loads((ROOT / "experiments" / "m17_ope_loso" / "summary.json").read_text())
    bci2a_ope = json.loads((ROOT / "experiments" / "m19_ope_loso_bci2a" / "summary.json").read_text())
    bci2b_loso = json.loads((ROOT / "experiments" / "m8_loso_bci2b" / "summary.json").read_text())
    bci2a_loso = json.loads((ROOT / "experiments" / "m15_loso_bci2a" / "summary.json").read_text())

    bci2b_acc = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
                 for r in bci2b_loso["per_subject"]}
    bci2a_acc = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
                 for r in bci2a_loso["per_subject"]}

    rows = []
    for r in bci2b_ope["per_subject"]:
        for p in r["policies"]:
            rows.append({"dataset": "bci2b", "sid": r["held_out"],
                          "acc": bci2b_acc.get(r["held_out"]),
                          "policy": p["name"], "v_fqe": p["v_fqe"],
                          "v_gt": p["v_gt"]})
    for r in bci2a_ope["per_subject"]:
        for p in r["policies"]:
            rows.append({"dataset": "bci2a", "sid": r["held_out"],
                          "acc": bci2a_acc.get(r["held_out"]),
                          "policy": p["name"], "v_fqe": p["v_fqe"],
                          "v_gt": p["v_gt"]})
    return rows


def per_policy_bias(rows):
    pols = sorted(set(r["policy"] for r in rows))
    out = []
    for pol in pols:
        sel = [r for r in rows if r["policy"] == pol]
        bias = np.mean([r["v_fqe"] - r["v_gt"] for r in sel])
        std = np.std([r["v_fqe"] - r["v_gt"] for r in sel], ddof=1)
        # 95% bootstrap CI
        diffs = np.array([r["v_fqe"] - r["v_gt"] for r in sel])
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                          for _ in range(10000)])
        ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
        out.append({"policy": pol, "n": len(sel), "bias_mean": float(bias),
                    "bias_std": float(std), "ci95_lo": float(ci_lo),
                    "ci95_hi": float(ci_hi)})
    return out


def shrinkage_analysis(rows):
    v_fqe = np.array([r["v_fqe"] for r in rows])
    v_gt = np.array([r["v_gt"] for r in rows])
    # Linear fit V_FQE = a * V_GT + b
    slope, intercept, r, p, se = stats.linregress(v_gt, v_fqe)
    return {"slope": float(slope), "intercept": float(intercept),
            "pearson_r": float(r), "p": float(p), "se_slope": float(se)}


def per_decodability_bias(rows, threshold=0.5):
    high = [r for r in rows if r["acc"] is not None and r["acc"] >= threshold]
    low = [r for r in rows if r["acc"] is not None and r["acc"] < threshold]
    return {
        "threshold": threshold,
        "high_n": len(high),
        "high_bias": float(np.mean([r["v_fqe"] - r["v_gt"] for r in high])),
        "low_n": len(low),
        "low_bias": float(np.mean([r["v_fqe"] - r["v_gt"] for r in low])),
    }


def debias_correction(rows, per_pol_bias):
    """Subtract per-policy mean bias from V_FQE; recompute per-subject Pearson r."""
    bias_lookup = {pb["policy"]: pb["bias_mean"] for pb in per_pol_bias}
    by_sid = {}
    for r in rows:
        key = (r["dataset"], r["sid"])
        if key not in by_sid:
            by_sid[key] = []
        by_sid[key].append({**r, "v_fqe_db": r["v_fqe"] - bias_lookup[r["policy"]]})

    # Per-subject Pearson r before/after
    rows_per_sub = []
    for (ds, sid), lst in by_sid.items():
        vgt = np.array([r["v_gt"] for r in lst])
        vfqe = np.array([r["v_fqe"] for r in lst])
        vfqe_db = np.array([r["v_fqe_db"] for r in lst])
        r_before, p_before = stats.pearsonr(vgt, vfqe)
        r_after, p_after = stats.pearsonr(vgt, vfqe_db)
        rows_per_sub.append({"dataset": ds, "sid": sid,
                              "r_before": float(r_before), "r_after": float(r_after),
                              "delta_r": float(r_after - r_before)})

    # Naive vs default V_GT after debias (per dataset)
    naive_v_after = {"bci2b": [], "bci2a": []}
    naive_v_before = {"bci2b": [], "bci2a": []}
    default_v = {"bci2b": [], "bci2a": []}
    for (ds, sid), lst in by_sid.items():
        vgts = {r["policy"]: r["v_gt"] for r in lst}
        vfqes = {r["policy"]: r["v_fqe"] for r in lst}
        vfqes_db = {r["policy"]: r["v_fqe_db"] for r in lst}
        # Naive picks
        n_before = max(vfqes, key=vfqes.get)
        n_after = max(vfqes_db, key=vfqes_db.get)
        naive_v_before[ds].append(vgts[n_before])
        naive_v_after[ds].append(vgts[n_after])
        default_v[ds].append(vgts.get("agent_T0", float("nan")))

    return {
        "per_subject_r": rows_per_sub,
        "mean_r_before": float(np.mean([r["r_before"] for r in rows_per_sub])),
        "mean_r_after": float(np.mean([r["r_after"] for r in rows_per_sub])),
        "mean_delta_r": float(np.mean([r["delta_r"] for r in rows_per_sub])),
        "naive_v_before_bci2b": float(np.mean(naive_v_before["bci2b"])),
        "naive_v_after_bci2b": float(np.mean(naive_v_after["bci2b"])),
        "default_v_bci2b": float(np.mean(default_v["bci2b"])),
        "naive_v_before_bci2a": float(np.mean(naive_v_before["bci2a"])),
        "naive_v_after_bci2a": float(np.mean(naive_v_after["bci2a"])),
        "default_v_bci2a": float(np.mean(default_v["bci2a"])),
    }


def make_figure(per_pol_bias, shr, rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Left: per-policy bias bar
    pols = [p["policy"] for p in per_pol_bias]
    means = [p["bias_mean"] for p in per_pol_bias]
    los = [p["bias_mean"] - p["ci95_lo"] for p in per_pol_bias]
    his = [p["ci95_hi"] - p["bias_mean"] for p in per_pol_bias]
    axes[0].bar(pols, means, yerr=[los, his], capsize=4,
                 color=["C0" if m < 0 else "C3" for m in means],
                 edgecolor="black")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("FQE bias  $\\bar V_{FQE} - \\bar V_{GT}$")
    axes[0].set_title("FQE shrinks toward zero\n(positive bars = FQE over-estimates V_GT)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: V_FQE vs V_GT scatter w/ identity & fit lines
    v_fqe = np.array([r["v_fqe"] for r in rows])
    v_gt = np.array([r["v_gt"] for r in rows])
    ds = np.array([r["dataset"] for r in rows])
    axes[1].scatter(v_gt[ds == "bci2b"], v_fqe[ds == "bci2b"], s=40,
                     marker="o", color="C0", label="bci2b", alpha=0.7,
                     edgecolor="black", linewidth=0.5)
    axes[1].scatter(v_gt[ds == "bci2a"], v_fqe[ds == "bci2a"], s=40,
                     marker="s", color="C3", label="bci2a", alpha=0.7,
                     edgecolor="black", linewidth=0.5)
    xs = np.linspace(v_gt.min(), v_gt.max(), 100)
    axes[1].plot(xs, xs, "--", color="black", lw=1.0, label="identity")
    axes[1].plot(xs, shr["slope"] * xs + shr["intercept"], "-", color="C2",
                  lw=1.5, label=f"fit slope={shr['slope']:.2f}, int={shr['intercept']:.2f}")
    axes[1].set_xlabel("V_GT (on-policy MC)")
    axes[1].set_ylabel("V_FQE (cross-subject FQE)")
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"Shrinkage: FQE slope = {shr['slope']:.2f} (vs identity 1.00)\n"
                       f"r = {shr['pearson_r']:+.2f}")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = ROOT / "experiments" / "m21_fqe_bias"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus()
    log.info("Loaded %d (subject, policy) pairs across %d subjects",
              len(rows), len({(r["dataset"], r["sid"]) for r in rows}))

    per_pol = per_policy_bias(rows)
    log.info("\n=== Per-policy FQE bias (V_FQE - V_GT, N=18 subjects each) ===")
    for p in per_pol:
        log.info("  %-14s n=%d bias=%+.3f ± %.3f  CI95 [%+.3f, %+.3f]",
                 p["policy"], p["n"], p["bias_mean"], p["bias_std"],
                 p["ci95_lo"], p["ci95_hi"])

    shr = shrinkage_analysis(rows)
    log.info("\n=== Shrinkage analysis (linear fit V_FQE = a*V_GT + b) ===")
    log.info("  slope = %+.3f (identity = 1.0; <1 means shrinkage toward zero)", shr["slope"])
    log.info("  intercept = %+.3f", shr["intercept"])
    log.info("  Pearson r = %+.3f (p=%.3g)", shr["pearson_r"], shr["p"])

    pdb = per_decodability_bias(rows)
    log.info("\n=== Bias by held-out decodability (threshold=%.2f) ===", pdb["threshold"])
    log.info("  high (acc >= %.2f, n=%d): mean bias = %+.3f",
             pdb["threshold"], pdb["high_n"], pdb["high_bias"])
    log.info("  low  (acc <  %.2f, n=%d): mean bias = %+.3f",
             pdb["threshold"], pdb["low_n"], pdb["low_bias"])

    db = debias_correction(rows, per_pol)
    log.info("\n=== Per-policy debias correction effect ===")
    log.info("  Mean per-subject Pearson r:  before=%+.3f → after=%+.3f (Δ=%+.3f)",
             db["mean_r_before"], db["mean_r_after"], db["mean_delta_r"])
    log.info("  bci2b naive V_GT: before=%+.3f → after=%+.3f (default=%+.3f)",
             db["naive_v_before_bci2b"], db["naive_v_after_bci2b"], db["default_v_bci2b"])
    log.info("  bci2a naive V_GT: before=%+.3f → after=%+.3f (default=%+.3f)",
             db["naive_v_before_bci2a"], db["naive_v_after_bci2a"], db["default_v_bci2a"])

    fig_path = ROOT / "figures" / "m21_fqe_bias.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(per_pol, shr, rows, fig_path)
    log.info("Wrote %s", fig_path)

    out = {"n_rows": len(rows), "n_subjects": len({(r["dataset"], r["sid"]) for r in rows}),
            "per_policy_bias": per_pol, "shrinkage": shr,
            "per_decodability_bias": pdb, "debias_correction": db}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
