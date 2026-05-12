# MIT License - Bryan Cheng, 2026
"""M19: OPE-LOSO on bci2a (4-class) — validate cross-subject OPE
deployment rule on a second dataset.

If the M17b finding (Pearson(M8 acc, OPE r) = +0.83) replicates on
bci2a (M15 acc → bci2a OPE r), the deployment rule holds across
datasets. We run all 9 bci2a subjects with the same 6-policy panel.

Output: experiments/m19_ope_loso_bci2a/summary.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from src.data.preprocess import preprocess_subject
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG
# Reuse run_one_subject from M17 (it preprocesses encoder and uses the EEGNet pipeline)
from m17_ope_loso import run_one_subject

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m19_ope_loso_bci2a")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "m19_ope_loso_bci2a"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    target_subjects = list(range(1, 10))
    log.info("Preprocessing 9 bci2a subjects...")
    subject_tbs = {sid: preprocess_subject("bci2a", subject_id=sid)
                   for sid in target_subjects}
    n_classes = len(subject_tbs[1].class_labels)
    sfreq = subject_tbs[1].sfreq
    n_channels = subject_tbs[1].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))
    log.info("bci2a setup: %d ch, %d classes, %d samples/window",
             n_channels, n_classes, win_samples)

    per_subject = []
    for held_out in target_subjects:
        try:
            result = run_one_subject(held_out, subject_tbs, n_classes, sfreq,
                                      n_channels, win_samples, device, log)
            per_subject.append(result)
            summary_path.write_text(json.dumps(
                {"per_subject": per_subject, "subjects": target_subjects,
                 "dataset": "bci2a"}, indent=2))
            log.info("  saved (%d/%d total)", len(per_subject), len(target_subjects))
        except Exception as exc:
            log.error("FAILED on held-out=%d: %s", held_out, exc)
            import traceback; traceback.print_exc()

    # Aggregate + meta-correlation with M15
    log.info("\n=== M19 OPE-LOSO bci2a AGGREGATE (9 held-out subjects) ===")
    rs = [r["pearson_r"] for r in per_subject if not np.isnan(r["pearson_r"])]
    ps = [r["pearson_p"] for r in per_subject]
    log.info("Per-subject Pearson r: %s", [f"{r:+.3f}" for r in rs])
    log.info("Per-subject p:         %s", [f"{p:.3f}" for p in ps])
    log.info("Mean Pearson r = %.3f ± %.3f (n=%d)",
             np.mean(rs), np.std(rs, ddof=1) if len(rs) > 1 else 0.0, len(rs))
    log.info("Significant calibration (p<0.05): %d/%d",
             sum(1 for p in ps if p < 0.05), len(ps))

    # Meta-correlation: M15 EEGNet bci2a acc vs M19 OPE r
    try:
        m15_path = ROOT / "experiments" / "m15_loso_bci2a" / "summary.json"
        if m15_path.exists():
            m15 = json.loads(m15_path.read_text())
            m15_acc = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
                        for r in m15["per_subject"]}
            paired_acc = []
            paired_r = []
            for r in per_subject:
                sid = r["held_out"]
                if sid in m15_acc:
                    paired_acc.append(m15_acc[sid])
                    paired_r.append(r["pearson_r"])
            if len(paired_acc) >= 3:
                from scipy import stats
                pr, p_pr = stats.pearsonr(paired_acc, paired_r)
                sr, p_sr = stats.spearmanr(paired_acc, paired_r)
                log.info("\n=== M19 META-CORRELATION (M15 acc vs M19 OPE r) ===")
                log.info("Pearson  r = %+.3f (p=%.3f)", pr, p_pr)
                log.info("Spearman ρ = %+.3f (p=%.3f)", sr, p_sr)
                # Compare to M17b on bci2b: Pearson +0.83 (p=0.006)
                log.info("\nDeployment rule consistency:")
                log.info("  M17b bci2b: Pearson(decodability, OPE r) = +0.83 (p=0.006)")
                log.info("  M19  bci2a: Pearson(decodability, OPE r) = %+.3f (p=%.3f)", pr, p_pr)
    except Exception as exc:
        log.warning("Meta-correlation failed: %s", exc)


if __name__ == "__main__":
    main()
