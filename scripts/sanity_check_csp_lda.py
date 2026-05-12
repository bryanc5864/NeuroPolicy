# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""sanity check: within-subject CSP+LDA on a small subset of subjects.

Per RESEARCH_PLAN.md , we verify the data pipeline is sound by
reproducing classical CSP+LDA accuracies. We run on BCI-IV-2a (4-class) and
BCI-IV-2b (2-class), 3 subjects each (smoke-test scope), and report mean
accuracy along with literature reference ranges.

Output: experiments/sanity_check/csp_lda_results.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.preprocess import preprocess_subject
from src.evaluation.csp_lda import csp_lda_cv
from src.utils.config import EXPERIMENTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("sanity_check")

# Reference ranges (literature, not target — sanity bounds).
REF_RANGES = {
    "bci2a": (0.55, 0.85),    # 4-class CSP+LDA, within-subject
    "bci2b": (0.65, 0.85),    # 2-class CSP+LDA, within-subject
}

PLAN = {
    "bci2a": [1, 3, 7],   # three subjects: covers high/middle/low MOABB-reported accuracy
    "bci2b": [2, 4, 9],
}


def main():
    out_dir = EXPERIMENTS_DIR / "sanity_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "csp_lda_results.json"

    results: list[dict] = []
    for dataset_key, subjects in PLAN.items():
        for sid in subjects:
            t0 = time.time()
            log.info("Preprocessing %s subject %d", dataset_key, sid)
            tb = preprocess_subject(dataset_key, sid)
            t_pre = time.time() - t0

            t0 = time.time()
            res = csp_lda_cv(tb, n_components=6, n_splits=5, seed=0)
            t_cv = time.time() - t0

            in_range = REF_RANGES[dataset_key][0] <= res.mean_acc <= REF_RANGES[dataset_key][1]
            log.info(
                "[%s sub %d] CSP+LDA acc = %.3f ± %.3f  (pre=%.1fs, cv=%.1fs)  range_ok=%s",
                dataset_key, sid, res.mean_acc, res.std_acc, t_pre, t_cv, in_range,
            )
            results.append({
                **res.asdict(),
                "n_trials": int(tb.n_trials),
                "n_channels": int(tb.n_channels),
                "n_times": int(tb.n_times),
                "preprocess_seconds": t_pre,
                "cv_seconds": t_cv,
                "ref_range": REF_RANGES[dataset_key],
                "in_ref_range": bool(in_range),
            })

    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_path)

    # Print summary table
    by_ds: dict[str, list[float]] = {}
    for r in results:
        by_ds.setdefault(r["dataset_key"], []).append(r["mean_acc"])
    print("\n===  sanity summary ===")
    print(f"{'dataset':10s} {'subj_mean':>10s} {'subj_std':>10s} {'subjects':>10s} {'ref_range':>14s}")
    for ds, accs in by_ds.items():
        ref = REF_RANGES[ds]
        print(f"{ds:10s} {np.mean(accs):>10.3f} {np.std(accs):>10.3f} {len(accs):>10d}  ({ref[0]:.2f}, {ref[1]:.2f})")


if __name__ == "__main__":
    main()
