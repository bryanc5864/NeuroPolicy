# MIT License, 2026
""": OPE-LOSO calibration on remaining 6 bci2b subjects.

 covered subjects 1, 4, 7. This script covers 2, 3, 5, 6, 8, 9 to
give the full N=9 cross-subject OPE calibration sweep. Same protocol
as  (6-policy panel, 500-iter FQE per target).

Output appended to experiments/ope_loso_bci2b/summary.json (merged).
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

from src.data.preprocess import preprocess_subject
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG

# Reuse run_one_subject from 
from ope_loso_bci2b import run_one_subject

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("ope_loso_bci2b_full")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "ope_loso_bci2b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subjects already done in 
    existing_subjects = {1, 4, 7}
    all_subjects = list(range(1, 10))
    new_subjects = [s for s in all_subjects if s not in existing_subjects]

    log.info("Running OPE-LOSO on remaining %d subjects: %s",
             len(new_subjects), new_subjects)
    log.info("Preprocessing all 9 bci2b subjects...")
    subject_tbs = {sid: preprocess_subject("bci2b", subject_id=sid) for sid in all_subjects}
    n_classes = len(subject_tbs[1].class_labels)
    sfreq = subject_tbs[1].sfreq
    n_channels = subject_tbs[1].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))

    # Load existing  results
    existing_path = out_dir / "summary.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        per_subject = existing.get("per_subject", [])
        log.info("Loaded %d existing  results", len(per_subject))
    else:
        per_subject = []

    for held_out in new_subjects:
        try:
            result = run_one_subject(held_out, subject_tbs, n_classes, sfreq,
                                      n_channels, win_samples, device, log)
            per_subject.append(result)
            existing_path.write_text(json.dumps(
                {"per_subject": per_subject, "subjects": all_subjects,
                 "dataset": "bci2b"}, indent=2))
            log.info("  saved (%d / %d total)", len(per_subject), len(all_subjects))
        except Exception as exc:
            log.error("FAILED on held-out=%d: %s", held_out, exc)
            import traceback; traceback.print_exc()

    # Aggregate
    log.info("\n===  FULL OPE-LOSO AGGREGATE (9 held-out subjects) ===")
    rs = [r["pearson_r"] for r in per_subject if not np.isnan(r["pearson_r"])]
    rmses = [r["rmse"] for r in per_subject]
    ps = [r["pearson_p"] for r in per_subject]
    log.info("Pearson r per subject: %s",
             [f"{r:.3f}" for r in [r2["pearson_r"] for r2 in per_subject]])
    log.info("Per-subject p: %s", [f"{p:.3f}" for p in ps])
    log.info("Mean Pearson r = %.3f ± %.3f (across %d held-out subjects)",
             np.mean(rs), np.std(rs, ddof=1) if len(rs) > 1 else 0.0, len(rs))
    log.info("Mean RMSE = %.3f", np.mean(rmses))
    log.info("Subjects with significant calibration (p < 0.05): %d/%d",
             sum(1 for p in ps if p < 0.05), len(ps))


if __name__ == "__main__":
    main()
