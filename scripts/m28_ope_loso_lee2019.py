# MIT License - Bryan Cheng, 2026
"""M28: OPE-LOSO on Lee2019 — third-dataset replication of the deployment rule.

W2, W11 critique: meta-correlation rule (M17b/M19) shown only on two
MOABB BCI Competition IV datasets. Lee2019 (62-ch, 54-subject MI
corpus) is the canonical third dataset. We run a 5-subject LOSO
subset (subjects 1, 5, 10, 15, 20 — index into MOABB Lee2019), each
held out against a pool of 4 training subjects.

If r(decodability, OPE_r) > 0 on Lee2019, the deployment rule
replicates across all three motor-imagery datasets and across
channel counts {3, 22, 62} and class counts {2, 4}.

Output: experiments/m28_ope_loso_lee2019/summary.json
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
from scipy import stats

from src.data.preprocess import preprocess_subject
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG
from m17_ope_loso import run_one_subject

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m28_lee2019")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "m28_ope_loso_lee2019"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5-subject subset, spanning the Lee2019 subject index range
    target_subjects = [1, 5, 10, 15, 20]
    log.info("Lee2019 LOSO subset: %s", target_subjects)

    log.info("Preprocessing Lee2019 subjects %s...", target_subjects)
    subject_tbs = {}
    for sid in target_subjects:
        try:
            tb = preprocess_subject("lee2019", subject_id=sid)
            subject_tbs[sid] = tb
            log.info("  sub %d: X=%s n_trials=%d", sid, tb.X.shape, tb.n_trials)
        except Exception as exc:
            log.error("FAILED preprocessing sub %d: %s", sid, exc)

    if len(subject_tbs) < 3:
        log.error("Need ≥3 subjects to do LOSO meta-correlation; got %d. Abort.",
                  len(subject_tbs))
        return

    sids = sorted(subject_tbs.keys())
    n_classes = len(subject_tbs[sids[0]].class_labels)
    sfreq = subject_tbs[sids[0]].sfreq
    n_channels = subject_tbs[sids[0]].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))
    log.info("Lee2019 setup: %d ch, %d classes, %d samples/window",
             n_channels, n_classes, win_samples)

    per_subject = []
    t_start = time.time()
    for held_out in sids:
        try:
            result = run_one_subject(held_out, subject_tbs, n_classes, sfreq,
                                      n_channels, win_samples, device, log)
            per_subject.append(result)
            (out_dir / "summary.json").write_text(json.dumps(
                {"per_subject": per_subject, "subjects": sids,
                 "dataset": "lee2019"}, indent=2))
            log.info("  saved (%d/%d total, %.1fs elapsed)",
                     len(per_subject), len(sids), time.time() - t_start)
        except Exception as exc:
            log.error("FAILED on held-out=%d: %s", held_out, exc)
            import traceback; traceback.print_exc()

    # Aggregate
    log.info("\n=== M28 OPE-LOSO Lee2019 AGGREGATE (%d subjects) ===", len(per_subject))
    rs = [r["pearson_r"] for r in per_subject if not np.isnan(r["pearson_r"])]
    ps = [r["pearson_p"] for r in per_subject]
    log.info("Per-subject Pearson r: %s", [f"{r:+.3f}" for r in rs])
    log.info("Mean Pearson r = %.3f ± %.3f (n=%d)",
             np.mean(rs), np.std(rs, ddof=1) if len(rs) > 1 else 0.0, len(rs))
    log.info("Significant calibration (p<0.05): %d/%d",
             sum(1 for p in ps if p < 0.05), len(ps))

    # Meta-correlation: encoder val_acc (proxy for decodability) vs OPE r
    encoder_accs = {r["held_out"]: r["encoder_val_acc"] for r in per_subject}
    log.info("\n=== M28 META-CORRELATION (encoder val acc vs OPE r) ===")
    paired_acc, paired_r = [], []
    for r in per_subject:
        sid = r["held_out"]
        paired_acc.append(encoder_accs[sid])
        paired_r.append(r["pearson_r"])
    if len(paired_acc) >= 3:
        pr, p_pr = stats.pearsonr(paired_acc, paired_r)
        sr, p_sr = stats.spearmanr(paired_acc, paired_r)
        log.info("Pearson  r = %+.3f (p=%.3f)", pr, p_pr)
        log.info("Spearman ρ = %+.3f (p=%.3f)", sr, p_sr)
        log.info("\nDeployment rule consistency:")
        log.info("  M17b bci2b:    Pearson(decodability, OPE r) = +0.83 (p=0.006)")
        log.info("  M19  bci2a:    Pearson(decodability, OPE r) = +0.61 (p=0.08)")
        log.info("  M28  lee2019:  Pearson(decodability, OPE r) = %+.3f (p=%.3f)",
                 pr, p_pr)

    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
