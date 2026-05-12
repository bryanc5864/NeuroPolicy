# MIT License - Bryan Cheng, 2026
"""M20: Validate the OPE deployment rule actionably.

The M17b/M19 finding is that subject decodability predicts cross-subject
OPE calibration quality (Pearson r ≈ 0.6-0.8). This script tests whether
*using* this finding to gate FQE-based policy selection actually
improves outcomes vs naive FQE-based selection.

Three policy-selection strategies, evaluated per subject:
  - oracle:   pick policy with highest V_GT (upper bound)
  - naive:    pick policy with highest V_FQE (uses OPE blindly)
  - gated:    if held-out decodability >= threshold, pick by V_FQE;
              else fall back to default (e.g., agent_T0)
  - default:  always use agent_T0 (no OPE)

Headline metric: mean V_GT of the selected policy across subjects.

If gated > naive on average, the deployment rule is actionable.

Output: experiments/m20_deployment_validation/summary.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m20_deployment")


def evaluate_strategy(per_subject, m_acc, threshold, default_policy="agent_T0"):
    """Return per-subject and mean V_GT(selected_policy) under each strategy."""
    rows = []
    for r in per_subject:
        sid = r["held_out"]
        policies = r["policies"]
        names = [p["name"] for p in policies]
        vgts = {p["name"]: p["v_gt"] for p in policies}
        vfqes = {p["name"]: p["v_fqe"] for p in policies}
        acc = m_acc.get(sid, float("nan"))

        # Oracle
        oracle_pick = max(names, key=lambda n: vgts[n])
        # Naive: highest V_FQE
        naive_pick = max(names, key=lambda n: vfqes[n])
        # Gated: trust FQE if decodability ≥ threshold; else default
        gated_pick = naive_pick if acc >= threshold else default_policy
        # Default
        default_pick = default_policy

        rows.append({
            "sid": sid,
            "decodability": float(acc),
            "oracle_pick": oracle_pick, "v_oracle": vgts[oracle_pick],
            "naive_pick": naive_pick, "v_naive": vgts[naive_pick],
            "gated_pick": gated_pick, "v_gated": vgts[gated_pick],
            "default_pick": default_pick, "v_default": vgts[default_pick],
            "v_random": vgts.get("random", float("nan")),
        })
    return rows


def main():
    out_dir = ROOT / "experiments" / "m20_deployment_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bci2b"

    if dataset == "bci2b":
        m17 = json.loads((ROOT / "experiments" / "m17_ope_loso" / "summary.json").read_text())
        m8 = json.loads((ROOT / "experiments" / "m8_loso_bci2b" / "summary.json").read_text())
        m8_acc = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
                  for r in m8["per_subject"]}
    else:
        m17 = json.loads((ROOT / "experiments" / "m19_ope_loso_bci2a" / "summary.json").read_text())
        m8 = json.loads((ROOT / "experiments" / "m15_loso_bci2a" / "summary.json").read_text())
        m8_acc = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
                  for r in m8["per_subject"]}

    log.info("\n=== M20 %s deployment-rule validation ===", dataset)
    log.info("Sweeping decodability thresholds...\n")

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.74]
    summaries = []

    # Compute strategy values without gating first (oracle, naive, default)
    rows_default = evaluate_strategy(m17["per_subject"], m8_acc, threshold=-1.0)
    v_oracle = np.mean([r["v_oracle"] for r in rows_default])
    v_naive = np.mean([r["v_naive"] for r in rows_default])
    v_default = np.mean([r["v_default"] for r in rows_default])
    v_random_mean = np.mean([r["v_random"] for r in rows_default])

    log.info("Mean V_GT across N=%d subjects:", len(rows_default))
    log.info("  oracle (best by V_GT):     %+.3f", v_oracle)
    log.info("  naive  (best by V_FQE):    %+.3f", v_naive)
    log.info("  default (always agent_T0): %+.3f", v_default)
    log.info("  random:                    %+.3f", v_random_mean)
    log.info("")
    log.info("Gated strategies — pick by V_FQE if decodability >= threshold; else fall back to agent_T0:")

    for thr in thresholds:
        rows = evaluate_strategy(m17["per_subject"], m8_acc, threshold=thr)
        v_gated = np.mean([r["v_gated"] for r in rows])
        n_used_fqe = sum(1 for r in rows if r["gated_pick"] == r["naive_pick"])
        log.info("  threshold=%.2f: V_gated=%+.3f, used FQE on %d/%d subjects",
                 thr, v_gated, n_used_fqe, len(rows))
        summaries.append({"threshold": thr, "v_gated": float(v_gated),
                           "n_used_fqe": int(n_used_fqe), "rows": rows})

    # Find optimal threshold
    best = max(summaries, key=lambda s: s["v_gated"])
    log.info("\n🎯 Best threshold: %.2f → V_gated=%+.3f (vs naive=%+.3f, default=%+.3f)",
             best["threshold"], best["v_gated"], v_naive, v_default)
    log.info("   Improvement gated vs naive: %+.3f", best["v_gated"] - v_naive)
    log.info("   Improvement gated vs default: %+.3f", best["v_gated"] - v_default)
    log.info("   Gap to oracle: %.3f", v_oracle - best["v_gated"])

    # Per-subject breakdown at best threshold
    log.info("\nPer-subject V_GT at best threshold (%.2f):", best["threshold"])
    log.info("  sub | acc   | naive_pick | naive_vgt | gated_pick | gated_vgt | oracle_vgt")
    for r in best["rows"]:
        log.info(f"  {r['sid']:>3d} | {r['decodability']:.3f} | "
                 f"{r['naive_pick']:<13s}| {r['v_naive']:+.3f}    | "
                 f"{r['gated_pick']:<13s}| {r['v_gated']:+.3f}    | {r['v_oracle']:+.3f}")

    out = {
        "dataset": dataset, "n_subjects": len(rows_default),
        "v_oracle": float(v_oracle), "v_naive": float(v_naive),
        "v_default": float(v_default), "v_random": float(v_random_mean),
        "thresholds": summaries,
        "best_threshold": best["threshold"], "best_v_gated": best["v_gated"],
        "gain_gated_vs_naive": float(best["v_gated"] - v_naive),
        "gain_gated_vs_default": float(best["v_gated"] - v_default),
    }
    (out_dir / f"summary_{dataset}.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_dir / f"summary_{dataset}.json")


if __name__ == "__main__":
    main()
