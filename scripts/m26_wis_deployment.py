# MIT License - Bryan Cheng, 2026
"""M26: Does WIS make naive cross-subject OPE deployment viable?

M20 found naive FQE deployment is harmful relative to default-T0:
  V_FQE_naive = -1.36 vs V_default = -1.33 on bci2b.
M22 proved no monotone affine FQE correction can fix this.
M24 found WIS dominates FQE on calibration; M25 confirmed at N=9.

This script directly tests whether WIS — a structurally distinct
estimator — makes the naive cross-subject deployment claim positive,
i.e. whether V_WIS_naive ≥ V_default. If yes, WIS solves the M20
problem that the M22 impossibility proved was unsolvable for FQE.

Compares 5 strategies on the M25 corpus (N=9 bci2b LOSO subjects):
  - oracle:      pick policy with highest V_GT (upper bound)
  - naive_FQE:   pick policy with highest V_FQE   (M20's failing strategy)
  - naive_WIS:   pick policy with highest V_WIS   (M26's candidate fix)
  - naive_DR:    pick policy with highest V_DR    (control)
  - default:     always agent_T0 (no OPE)

Headline metrics: mean V_GT(selected) per strategy + paired-bootstrap
P(WIS_naive >= default) and P(WIS_naive > FQE_naive).

Output: experiments/m26_wis_deployment/summary.json
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
log = logging.getLogger("m26_wis_deployment")

RNG = np.random.default_rng(42)
B = 10000


def evaluate_strategies(per_subject, default_policy="agent_T0"):
    rows = []
    for r in per_subject:
        sid = r["held_out"]
        policies = r["policies"]
        names = [p["name"] for p in policies]
        vgts = {p["name"]: p["v_gt"] for p in policies}
        vfqes = {p["name"]: p["v_fqe"] for p in policies}
        vwiss = {p["name"]: p["v_wis"] for p in policies}
        vdrs = {p["name"]: p["v_dr"] for p in policies}

        oracle_pick = max(names, key=lambda n: vgts[n])
        naive_fqe = max(names, key=lambda n: vfqes[n])
        naive_wis = max(names, key=lambda n: vwiss[n])
        naive_dr = max(names, key=lambda n: vdrs[n])
        default_pick = default_policy

        rows.append({
            "sid": sid,
            "oracle_pick": oracle_pick, "v_oracle": vgts[oracle_pick],
            "naive_fqe_pick": naive_fqe, "v_naive_fqe": vgts[naive_fqe],
            "naive_wis_pick": naive_wis, "v_naive_wis": vgts[naive_wis],
            "naive_dr_pick": naive_dr, "v_naive_dr": vgts[naive_dr],
            "default_pick": default_pick, "v_default": vgts[default_policy],
        })
    return rows


def boot_means_and_diffs(rows, B=10000):
    n = len(rows)
    keys = ["v_oracle", "v_naive_fqe", "v_naive_wis", "v_naive_dr", "v_default"]
    arrs = {k: np.array([r[k] for r in rows]) for k in keys}

    boot_means = {k: [] for k in keys}
    # Probabilities of interest
    P_wis_geq_default = []
    P_wis_geq_fqe = []
    P_wis_geq_dr = []
    P_fqe_lt_default = []
    for _ in range(B):
        idx = RNG.integers(0, n, size=n)
        for k in keys:
            boot_means[k].append(arrs[k][idx].mean())
        P_wis_geq_default.append(arrs["v_naive_wis"][idx].mean() >= arrs["v_default"][idx].mean())
        P_wis_geq_fqe.append(arrs["v_naive_wis"][idx].mean() >= arrs["v_naive_fqe"][idx].mean())
        P_wis_geq_dr.append(arrs["v_naive_wis"][idx].mean() >= arrs["v_naive_dr"][idx].mean())
        P_fqe_lt_default.append(arrs["v_naive_fqe"][idx].mean() < arrs["v_default"][idx].mean())

    out = {}
    for k in keys:
        a = np.array(boot_means[k])
        out[k] = {
            "mean": float(arrs[k].mean()),
            "ci95_lo": float(np.quantile(a, 0.025)),
            "ci95_hi": float(np.quantile(a, 0.975)),
        }
    out["P_wis_geq_default"] = float(np.mean(P_wis_geq_default))
    out["P_wis_geq_fqe"] = float(np.mean(P_wis_geq_fqe))
    out["P_wis_geq_dr"] = float(np.mean(P_wis_geq_dr))
    out["P_fqe_lt_default"] = float(np.mean(P_fqe_lt_default))
    return out


def main():
    out_dir = ROOT / "experiments" / "m26_wis_deployment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load M25 (full 9-subject) corpus
    m25_path = ROOT / "experiments" / "m25_multimethod_full" / "summary.json"
    if not m25_path.exists():
        log.error("M25 corpus not found at %s — run scripts/m25_multimethod_full.py first.",
                   m25_path)
        return
    m25 = json.loads(m25_path.read_text())
    per_subject = m25["per_subject"]
    log.info("Loaded M25 corpus: %d held-out subjects", len(per_subject))

    rows = evaluate_strategies(per_subject)

    log.info("\n=== M26 deployment outcomes per subject (V_GT of selected policy) ===")
    log.info("  sid | oracle | naive_FQE | naive_WIS | naive_DR | default")
    for r in rows:
        log.info("  %3d | %+.3f  | %+.3f    | %+.3f    | %+.3f   | %+.3f",
                 r["sid"], r["v_oracle"], r["v_naive_fqe"], r["v_naive_wis"],
                 r["v_naive_dr"], r["v_default"])

    boot = boot_means_and_diffs(rows, B=B)
    log.info("\n=== Strategy means (95%% paired bootstrap CI, B=%d, N=%d subjects) ===", B, len(rows))
    for k in ["v_oracle", "v_naive_fqe", "v_naive_wis", "v_naive_dr", "v_default"]:
        d = boot[k]
        log.info("  %-13s = %+.3f  [%+.3f, %+.3f]", k, d["mean"], d["ci95_lo"], d["ci95_hi"])

    log.info("\n=== Headline probability claims (paired bootstrap, B=%d) ===", B)
    log.info("  P(WIS naive >= default)  = %.4f  ← M22 impossibility resolved if ≥ 0.95",
             boot["P_wis_geq_default"])
    log.info("  P(WIS naive >= FQE naive)= %.4f", boot["P_wis_geq_fqe"])
    log.info("  P(WIS naive >= DR naive) = %.4f", boot["P_wis_geq_dr"])
    log.info("  P(FQE naive  <  default) = %.4f  (M20's harmful claim)",
             boot["P_fqe_lt_default"])

    out = {
        "n_subjects": len(rows),
        "per_subject": rows,
        "bootstrap": boot,
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    log.info("\nWrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
