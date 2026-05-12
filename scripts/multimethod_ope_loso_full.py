# MIT License, 2026
""": Extend 's multi-method OPE benchmark to all 9 bci2b LOSO subjects.

 ran FQE/PDIS/WIS/DR on subjects {1, 4, 7} only (compute budget).
 finishes the sweep on the remaining {2, 3, 5, 6, 8, 9} so we have
N=9 paired (FQE_r, WIS_r) values per subject — enough for a Wilcoxon
signed-rank test on the WIS > FQE claim.

Strategy: import run_one_subject_multimethod from , run on the
missing 6 subjects, merge with 's saved per_subject list, save the
N=9 summary for downstream  deployment validation.

Output: experiments/multimethod_ope_full/summary.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from scipy import stats

from src.data.preprocess import preprocess_subject
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG
from multimethod_ope_3subj import run_one_subject_multimethod

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("multimethod_ope_full")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "multimethod_ope_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load  partial results
    m24_path = EXPERIMENTS_DIR / "multimethod_ope_3subj" / "summary.json"
    m24 = json.loads(m24_path.read_text())
    done = {r["held_out"] for r in m24["per_subject"]}
    log.info(" already covered subjects: %s", sorted(done))

    target_subjects = [s for s in range(1, 10) if s not in done]
    log.info(" will run on subjects: %s", target_subjects)

    log.info("Preprocessing all 9 bci2b subjects...")
    subject_tbs = {sid: preprocess_subject("bci2b", subject_id=sid)
                   for sid in range(1, 10)}
    n_classes = len(subject_tbs[1].class_labels)
    sfreq = subject_tbs[1].sfreq
    n_channels = subject_tbs[1].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))

    per_subject = list(m24["per_subject"])
    for held_out in target_subjects:
        try:
            r = run_one_subject_multimethod(held_out, subject_tbs, n_classes,
                                              sfreq, n_channels, win_samples,
                                              device, log)
            per_subject.append(r)
            (out_dir / "summary.json").write_text(json.dumps(
                {"per_subject": per_subject, "subjects": sorted([r["held_out"] for r in per_subject]),
                 "dataset": "bci2b"}, indent=2))
            log.info("  saved (%d/9 subjects so far)", len(per_subject))
        except Exception as exc:
            log.error("FAILED on held-out=%d: %s", held_out, exc)
            import traceback; traceback.print_exc()

    # === Aggregate ===
    log.info("\n===  N=%d AGGREGATE: per-method calibration ===", len(per_subject))
    methods = ["v_fqe", "v_pdis", "v_wis", "v_dr"]
    agg = {}
    for m in methods:
        rs = [r["method_metrics"][m]["pearson_r"] for r in per_subject
              if not np.isnan(r["method_metrics"][m]["pearson_r"])]
        rmses = [r["method_metrics"][m]["rmse"] for r in per_subject]
        biases = [r["method_metrics"][m]["bias"] for r in per_subject]
        agg[m] = {
            "n": len(rs),
            "pearson_r_mean": float(np.mean(rs)),
            "pearson_r_std": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            "bias_mean": float(np.mean(biases)),
            "bias_std": float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
            "per_subject_r": rs,
        }
        log.info("  %-7s  mean r = %+.3f ± %.3f   RMSE = %.3f ± %.3f   bias = %+.3f ± %.3f",
                  m, agg[m]["pearson_r_mean"], agg[m]["pearson_r_std"],
                  agg[m]["rmse_mean"], agg[m]["rmse_std"],
                  agg[m]["bias_mean"], agg[m]["bias_std"])

    # === Paired Wilcoxon: WIS vs FQE per subject ===
    log.info("\n=== Paired Wilcoxon: WIS > FQE on per-subject Pearson r (N=%d) ===",
              len(per_subject))
    fqe_rs = np.array([r["method_metrics"]["v_fqe"]["pearson_r"] for r in per_subject])
    wis_rs = np.array([r["method_metrics"]["v_wis"]["pearson_r"] for r in per_subject])
    # Some FQE rs may be NaN if std=0 — fill with 0 as conservative tie
    fqe_rs = np.where(np.isnan(fqe_rs), 0.0, fqe_rs)
    wis_rs = np.where(np.isnan(wis_rs), 0.0, wis_rs)

    diffs = wis_rs - fqe_rs
    log.info("  per-subject Δr (WIS - FQE): %s",
              [f"{d:+.3f}" for d in diffs])
    log.info("  mean Δr = %+.3f, %d/%d subjects WIS > FQE",
              diffs.mean(), int((diffs > 0).sum()), len(diffs))
    if (diffs != 0).any():
        w_stat, w_p = stats.wilcoxon(diffs, alternative="greater")
        log.info("  Wilcoxon W=%.1f, p (one-sided WIS>FQE) = %.4f", w_stat, w_p)
    else:
        w_stat, w_p = float("nan"), float("nan")

    # Same for RMSE (lower is better, test FQE > WIS i.e. WIS RMSE < FQE RMSE)
    fqe_rmses = np.array([r["method_metrics"]["v_fqe"]["rmse"] for r in per_subject])
    wis_rmses = np.array([r["method_metrics"]["v_wis"]["rmse"] for r in per_subject])
    rmse_diffs = fqe_rmses - wis_rmses
    log.info("  mean ΔRMSE (FQE - WIS) = %+.3f, %d/%d subjects WIS RMSE < FQE",
              rmse_diffs.mean(), int((rmse_diffs > 0).sum()), len(rmse_diffs))
    if (rmse_diffs != 0).any():
        w_rmse_stat, w_rmse_p = stats.wilcoxon(rmse_diffs, alternative="greater")
        log.info("  RMSE Wilcoxon W=%.1f, p (one-sided WIS RMSE < FQE) = %.4f",
                  w_rmse_stat, w_rmse_p)
    else:
        w_rmse_stat, w_rmse_p = float("nan"), float("nan")

    out = {
        "per_subject": per_subject,
        "n_subjects": len(per_subject),
        "aggregate": agg,
        "paired_wilcoxon_r": {"W": float(w_stat), "p_one_sided_wis_gt_fqe": float(w_p),
                                "diffs": diffs.tolist(),
                                "mean_diff": float(diffs.mean())},
        "paired_wilcoxon_rmse": {"W": float(w_rmse_stat), "p_one_sided_wis_lower_fqe": float(w_rmse_p),
                                   "diffs": rmse_diffs.tolist(),
                                   "mean_diff": float(rmse_diffs.mean())},
        "dataset": "bci2b",
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
