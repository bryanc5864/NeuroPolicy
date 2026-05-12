# MIT License - Bryan Cheng, 2026
"""M32: Does WIS work as a deployment selector on a stochastic-only panel?

M26 found WIS-naive deployment collapses to deterministic argmax
(agent_T0) for all 9 subjects --- the deterministic-target IS pathology.
The hypothesis here: WIS is a sound estimator when targets are
NON-degenerate. Restricting the panel to stochastic policies only
(T0.5, T1.0, T5.0, mix0.3 --- all assign positive probability to
every action) should let WIS make non-trivial selections.

Strategies on stochastic-only panel:
  - oracle_stoch:  best V_GT over {T0.5, T1.0, T5.0, mix}
  - naive_FQE_stoch
  - naive_WIS_stoch
  - naive_DR_stoch
  - default_T0  (still report as benchmark; technically not in panel)

We compare on the M25 N=9 corpus + paired bootstrap.

Output: experiments/m32_wis_stochastic/summary.json
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m32_wis_stoch")

RNG = np.random.default_rng(42)
B = 10000

STOCH_POLICIES = {"agent_T0.5", "agent_T1.0", "agent_T5.0", "agent_mix0.3"}


def evaluate(per_subject):
    rows = []
    for r in per_subject:
        sid = r["held_out"]
        policies = r["policies"]
        stoch = [p for p in policies if p["name"] in STOCH_POLICIES]
        if len(stoch) < 2:
            log.warning("Skipping sub %d: only %d stoch policies",
                         sid, len(stoch))
            continue
        names = [p["name"] for p in stoch]
        vgts = {p["name"]: p["v_gt"] for p in stoch}
        vfqes = {p["name"]: p["v_fqe"] for p in stoch}
        vwiss = {p["name"]: p["v_wis"] for p in stoch}
        vdrs = {p["name"]: p["v_dr"] for p in stoch}

        oracle_stoch = max(names, key=lambda n: vgts[n])
        naive_fqe_stoch = max(names, key=lambda n: vfqes[n])
        naive_wis_stoch = max(names, key=lambda n: vwiss[n])
        naive_dr_stoch = max(names, key=lambda n: vdrs[n])

        # Default policies for reference
        all_policies = {p["name"]: p["v_gt"] for p in policies}
        v_default_t0 = all_policies.get("agent_T0", float("nan"))

        rows.append({
            "sid": sid,
            "oracle_stoch_pick": oracle_stoch, "v_oracle_stoch": vgts[oracle_stoch],
            "naive_fqe_stoch_pick": naive_fqe_stoch, "v_naive_fqe_stoch": vgts[naive_fqe_stoch],
            "naive_wis_stoch_pick": naive_wis_stoch, "v_naive_wis_stoch": vgts[naive_wis_stoch],
            "naive_dr_stoch_pick": naive_dr_stoch, "v_naive_dr_stoch": vgts[naive_dr_stoch],
            "v_default_t0": v_default_t0,
        })
    return rows


def boot(rows):
    n = len(rows)
    keys = ["v_oracle_stoch", "v_naive_fqe_stoch", "v_naive_wis_stoch",
             "v_naive_dr_stoch", "v_default_t0"]
    arrs = {k: np.array([r[k] for r in rows]) for k in keys}

    boot_means = {k: [] for k in keys}
    P_wis_gt_default = []
    P_wis_gt_fqe = []
    P_wis_gt_oracle_minus = []  # P(naive_wis ≥ oracle_stoch − 0.05)
    P_wis_eq_oracle = []
    for _ in range(B):
        idx = RNG.integers(0, n, size=n)
        for k in keys:
            boot_means[k].append(arrs[k][idx].mean())
        P_wis_gt_default.append(arrs["v_naive_wis_stoch"][idx].mean() >= arrs["v_default_t0"][idx].mean())
        P_wis_gt_fqe.append(arrs["v_naive_wis_stoch"][idx].mean() >= arrs["v_naive_fqe_stoch"][idx].mean())
        P_wis_eq_oracle.append((arrs["v_naive_wis_stoch"][idx] == arrs["v_oracle_stoch"][idx]).mean())

    out = {k: {"mean": float(arrs[k].mean()),
                "ci95_lo": float(np.quantile(boot_means[k], 0.025)),
                "ci95_hi": float(np.quantile(boot_means[k], 0.975))} for k in keys}
    out["P_wis_geq_default"] = float(np.mean(P_wis_gt_default))
    out["P_wis_geq_fqe"] = float(np.mean(P_wis_gt_fqe))
    out["mean_frac_wis_eq_oracle"] = float(np.mean(P_wis_eq_oracle))
    return out


def main():
    out_dir = ROOT / "experiments" / "m32_wis_stochastic"
    out_dir.mkdir(parents=True, exist_ok=True)

    m25 = json.loads((ROOT / "experiments" / "m25_multimethod_full" / "summary.json").read_text())
    per_subject = m25["per_subject"]
    log.info("Loaded M25 corpus: %d subjects", len(per_subject))

    rows = evaluate(per_subject)
    log.info("\n=== Stochastic-only panel (%s) — N=%d subjects ===",
              ", ".join(sorted(STOCH_POLICIES)), len(rows))
    log.info("  sid | oracle_pick | oracle_v | naive_FQE_pick | naive_FQE_v | naive_WIS_pick | naive_WIS_v | naive_DR_v | default_T0")
    for r in rows:
        log.info("  %3d | %-12s | %+.3f   | %-12s | %+.3f      | %-12s | %+.3f      | %+.3f     | %+.3f",
                 r["sid"], r["oracle_stoch_pick"], r["v_oracle_stoch"],
                 r["naive_fqe_stoch_pick"], r["v_naive_fqe_stoch"],
                 r["naive_wis_stoch_pick"], r["v_naive_wis_stoch"],
                 r["v_naive_dr_stoch"], r["v_default_t0"])

    bres = boot(rows)
    log.info("\n=== Strategy means (95%% paired bootstrap, B=%d, N=%d) ===", B, len(rows))
    for k in ["v_oracle_stoch", "v_naive_fqe_stoch", "v_naive_wis_stoch",
               "v_naive_dr_stoch", "v_default_t0"]:
        log.info("  %-23s = %+.3f  [%+.3f, %+.3f]", k, bres[k]["mean"],
                  bres[k]["ci95_lo"], bres[k]["ci95_hi"])

    log.info("\n=== Headline probability claims ===")
    log.info("  P(WIS_naive_stoch >= default_T0) = %.4f", bres["P_wis_geq_default"])
    log.info("  P(WIS_naive_stoch >= FQE_naive_stoch) = %.4f", bres["P_wis_geq_fqe"])
    log.info("  Mean fraction subjects WIS == stochastic-oracle = %.3f", bres["mean_frac_wis_eq_oracle"])

    out = {"n_subjects": len(rows), "stoch_policies": sorted(STOCH_POLICIES),
            "per_subject": rows, "bootstrap": bres}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
