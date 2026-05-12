# MIT License, 2026
""": Extend 's 5-subject Lee2019 LOSO to 8 subjects.

W2/W11 power:  reported mean OPE r = +0.803 on 5 Lee2019 subjects.
 adds subjects 25, 30, 35 so that the third-dataset replication
has N=8 — enough for a paired-bootstrap CI on the channel-count
→ OPE-quality relationship.

Reuses 's saved per_subject list; only the new 3 subjects are run.

Output: experiments/ope_loso_lee2019_extended/summary.json
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
from ope_loso_bci2b import run_one_subject

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("ope_loso_lee2019_extended")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "ope_loso_lee2019_extended"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load  partial results
    m28_path = EXPERIMENTS_DIR / "ope_loso_lee2019" / "summary.json"
    m28 = json.loads(m28_path.read_text())
    done = {r["held_out"] for r in m28["per_subject"]}
    log.info(" covered: %s", sorted(done))

    # Full target subset and missing
    target_subjects = sorted(set(list(done) + [25, 30, 35]))
    missing = [s for s in target_subjects if s not in done]
    log.info(" target subjects: %s (running %s)", target_subjects, missing)

    log.info("Preprocessing Lee2019 subjects %s (all of  target)...", target_subjects)
    subject_tbs = {}
    for sid in target_subjects:
        try:
            tb = preprocess_subject("lee2019", subject_id=sid)
            subject_tbs[sid] = tb
            log.info("  sub %d: X=%s n_trials=%d", sid, tb.X.shape, tb.n_trials)
        except Exception as exc:
            log.error("FAILED preprocessing sub %d: %s", sid, exc)

    sids = sorted(subject_tbs.keys())
    n_classes = len(subject_tbs[sids[0]].class_labels)
    sfreq = subject_tbs[sids[0]].sfreq
    n_channels = subject_tbs[sids[0]].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))
    log.info("Lee2019 setup: %d ch, %d classes, %d samples/window",
              n_channels, n_classes, win_samples)

    # Reuse  results
    per_subject = list(m28["per_subject"])
    # Now run new subjects
    t_start = time.time()
    for held_out in missing:
        if held_out not in subject_tbs:
            log.warning("Skipping sub %d (preprocessing failed)", held_out); continue
        try:
            result = run_one_subject(held_out, subject_tbs, n_classes, sfreq,
                                      n_channels, win_samples, device, log)
            per_subject.append(result)
            (out_dir / "summary.json").write_text(json.dumps(
                {"per_subject": per_subject,
                 "subjects": sorted([r["held_out"] for r in per_subject]),
                 "dataset": "lee2019"}, indent=2))
            log.info("  saved (%d/%d total, %.1fs elapsed)",
                      len(per_subject), len(target_subjects), time.time() - t_start)
        except Exception as exc:
            log.error("FAILED on held-out=%d: %s", held_out, exc)
            import traceback; traceback.print_exc()

    # Aggregate
    log.info("\n===  OPE-LOSO Lee2019 AGGREGATE (%d subjects) ===", len(per_subject))
    rs = [r["pearson_r"] for r in per_subject if not np.isnan(r["pearson_r"])]
    ps = [r["pearson_p"] for r in per_subject]
    log.info("Per-subject Pearson r: %s", [f"{r:+.3f}" for r in rs])
    log.info("Mean Pearson r = %.3f ± %.3f (n=%d)",
              np.mean(rs), np.std(rs, ddof=1) if len(rs) > 1 else 0.0, len(rs))
    log.info("Significant calibration (p<0.05): %d/%d",
              sum(1 for p in ps if p < 0.05), len(ps))

    # Meta-correlation
    encoder_accs = {r["held_out"]: r["encoder_val_acc"] for r in per_subject}
    paired_acc = [encoder_accs[r["held_out"]] for r in per_subject]
    paired_r = [r["pearson_r"] for r in per_subject]
    if len(paired_acc) >= 3:
        pr, p_pr = stats.pearsonr(paired_acc, paired_r)
        sr, p_sr = stats.spearmanr(paired_acc, paired_r)
        log.info("\n===  META-CORRELATION (Lee2019 N=%d) ===", len(per_subject))
        log.info("Pearson  r = %+.3f (p=%.3f)", pr, p_pr)
        log.info("Spearman ρ = %+.3f (p=%.3f)", sr, p_sr)

    # Bootstrap CI on mean OPE r
    rng = np.random.default_rng(42)
    B = 10000
    rs_arr = np.array(rs)
    boot_means = np.array([rng.choice(rs_arr, size=len(rs_arr), replace=True).mean()
                            for _ in range(B)])
    ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])
    log.info("\n=== Bootstrap CI on mean OPE r (B=%d) ===", B)
    log.info("Mean = %.3f, 95%% CI = [%.3f, %.3f]", rs_arr.mean(), ci_lo, ci_hi)
    log.info("P(mean > 0)          = %.4f", float((boot_means > 0).mean()))
    log.info("P(mean > 0.5)        = %.4f", float((boot_means > 0.5).mean()))

    # Channel-count cross-dataset table
    log.info("\n=== Channel-count → OPE-quality (across 3 datasets) ===")
    log.info("  bci2b (3 ch, , N=9): mean r = +0.171")
    log.info("  bci2a (22 ch, , N=9): mean r = +0.459")
    log.info("  lee2019 (62 ch, , N=%d): mean r = %.3f", len(rs_arr), rs_arr.mean())

    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
