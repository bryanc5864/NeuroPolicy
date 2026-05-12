# MIT License, 2026
""": Slope-corrected FQE — apply 's diagnosed correction.

 found V_FQE = 0.566 * V_GT + 0.696, i.e. cross-subject FQE shrinks
toward zero. This script applies the inverted correction
  V_FQE_corrected = (V_FQE - 0.696) / 0.566
and re-runs the  deployment evaluation. We also test a
leave-one-subject-out (LOSO) version where the slope/intercept are
re-fit on the other 17 subjects to avoid information leak.

Headline question: does correcting the diagnosed shrinkage actually
close the gap-to-oracle that  surfaced?

Output: experiments/slope_corrected_fqe/summary.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m22_slope")


def load_corpus():
    bci2b_ope = json.loads((ROOT / "experiments" / "ope_loso_bci2b" / "summary.json").read_text())
    bci2a_ope = json.loads((ROOT / "experiments" / "ope_loso_bci2a" / "summary.json").read_text())
    bci2b_loso = json.loads((ROOT / "experiments" / "loso_bci2b" / "summary.json").read_text())
    bci2a_loso = json.loads((ROOT / "experiments" / "loso_bci2a" / "summary.json").read_text())
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


def fit_correction(rows):
    """Linear fit V_FQE = a * V_GT + b on `rows`. Inversion: V_GT_hat = (V_FQE - b) / a"""
    v_gt = np.array([r["v_gt"] for r in rows])
    v_fqe = np.array([r["v_fqe"] for r in rows])
    slope, intercept, _, _, _ = stats.linregress(v_gt, v_fqe)
    return slope, intercept


def apply_correction(rows, a, b):
    out = []
    for r in rows:
        v_corr = (r["v_fqe"] - b) / a if a != 0 else r["v_fqe"]
        out.append({**r, "v_corr": v_corr})
    return out


def evaluate_naive(rows_with_corr, score_key):
    """For each subject, pick policy with max score_key; return its V_GT."""
    by_sid = {}
    for r in rows_with_corr:
        key = (r["dataset"], r["sid"])
        if key not in by_sid:
            by_sid[key] = []
        by_sid[key].append(r)
    naive_v = {"bci2b": [], "bci2a": []}
    oracle_v = {"bci2b": [], "bci2a": []}
    default_v = {"bci2b": [], "bci2a": []}
    for (ds, sid), lst in by_sid.items():
        names = [r["policy"] for r in lst]
        scores = {r["policy"]: r[score_key] for r in lst}
        vgts = {r["policy"]: r["v_gt"] for r in lst}
        naive_pick = max(names, key=lambda n: scores[n])
        oracle_pick = max(names, key=lambda n: vgts[n])
        naive_v[ds].append(vgts[naive_pick])
        oracle_v[ds].append(vgts[oracle_pick])
        default_v[ds].append(vgts.get("agent_T0", float("nan")))
    return {ds: {"naive": float(np.mean(naive_v[ds])),
                  "oracle": float(np.mean(oracle_v[ds])),
                  "default": float(np.mean(default_v[ds])),
                  "n": len(naive_v[ds])} for ds in ["bci2b", "bci2a"]}


def per_subject_loso_correction(rows):
    """Leave-one-subject-out fit, apply to held-out, repeat for all."""
    by_sid = {}
    for r in rows:
        key = (r["dataset"], r["sid"])
        if key not in by_sid:
            by_sid[key] = []
        by_sid[key].append(r)
    out = []
    fits = []
    for (ds, sid), lst in by_sid.items():
        train_rows = [r for r in rows if (r["dataset"], r["sid"]) != (ds, sid)]
        a, b = fit_correction(train_rows)
        fits.append({"held_out_dataset": ds, "held_out_sid": sid,
                      "slope": float(a), "intercept": float(b)})
        for r in lst:
            v_corr = (r["v_fqe"] - b) / a
            out.append({**r, "v_corr": v_corr})
    return out, fits


def main():
    out_dir = ROOT / "experiments" / "slope_corrected_fqe"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus()
    log.info("Loaded %d (subject, policy) pairs", len(rows))

    # === Strategy 1: full-corpus correction (information leak; oracular slope)
    a, b = fit_correction(rows)
    log.info("\n=== Full-corpus shrinkage fit (oracular) ===")
    log.info("  slope = %+.3f, intercept = %+.3f", a, b)
    rows_full = apply_correction(rows, a, b)
    res_naive = evaluate_naive(rows, "v_fqe")
    res_full = evaluate_naive(rows_full, "v_corr")

    log.info("\n=== Naive policy selection BEFORE correction ( baseline) ===")
    for ds in ["bci2b", "bci2a"]:
        log.info("  %s: oracle=%+.3f  naive=%+.3f  default=%+.3f",
                 ds, res_naive[ds]["oracle"], res_naive[ds]["naive"], res_naive[ds]["default"])

    log.info("\n=== Naive policy selection AFTER full-corpus correction ===")
    for ds in ["bci2b", "bci2a"]:
        log.info("  %s: oracle=%+.3f  naive=%+.3f  Δ vs uncorrected = %+.3f",
                 ds, res_full[ds]["oracle"], res_full[ds]["naive"],
                 res_full[ds]["naive"] - res_naive[ds]["naive"])

    # === Strategy 2: LOSO correction (no information leak)
    log.info("\n=== LOSO (no-leak) shrinkage correction ===")
    rows_loso, fits = per_subject_loso_correction(rows)
    res_loso = evaluate_naive(rows_loso, "v_corr")
    log.info("  Mean LOSO slope = %+.3f ± %.3f, mean LOSO intercept = %+.3f ± %.3f",
             np.mean([f["slope"] for f in fits]), np.std([f["slope"] for f in fits], ddof=1),
             np.mean([f["intercept"] for f in fits]), np.std([f["intercept"] for f in fits], ddof=1))
    for ds in ["bci2b", "bci2a"]:
        log.info("  %s: oracle=%+.3f  naive_LOSO=%+.3f  Δ vs uncorrected = %+.3f",
                 ds, res_loso[ds]["oracle"], res_loso[ds]["naive"],
                 res_loso[ds]["naive"] - res_naive[ds]["naive"])

    # === Per-subject Pearson r before/after
    by_sid = {}
    for r in rows_loso:
        key = (r["dataset"], r["sid"])
        if key not in by_sid:
            by_sid[key] = []
        by_sid[key].append(r)
    pearson_before = []
    pearson_after = []
    for key, lst in by_sid.items():
        vgt = np.array([r["v_gt"] for r in lst])
        vfqe = np.array([r["v_fqe"] for r in lst])
        vcorr = np.array([r["v_corr"] for r in lst])
        pearson_before.append(stats.pearsonr(vgt, vfqe)[0])
        pearson_after.append(stats.pearsonr(vgt, vcorr)[0])

    log.info("\n=== Per-subject Pearson r (slope-correction is rank-preserving for n=6 → no change expected) ===")
    log.info("  Mean r before = %+.3f, after = %+.3f, Δ = %+.3f",
             np.mean(pearson_before), np.mean(pearson_after),
             np.mean(pearson_after) - np.mean(pearson_before))

    out = {
        "n_pairs": len(rows),
        "full_corpus_fit": {"slope": float(a), "intercept": float(b)},
        "naive_before": res_naive,
        "naive_after_full": res_full,
        "naive_after_loso": res_loso,
        "loso_fits": fits,
        "pearson_before_mean": float(np.mean(pearson_before)),
        "pearson_after_mean": float(np.mean(pearson_after)),
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
