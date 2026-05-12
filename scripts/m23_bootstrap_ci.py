# MIT License - Bryan Cheng, 2026
"""M23: Bootstrap CIs on the meta-correlation (M17b/M19) and deployment-
rule (M20) headline statistics.

We perform paired-subject bootstrap (resample 9 subjects with replacement
per dataset, B=10000) and report 95% CIs on:
  - Pearson r(decodability, OPE r)  [M17b/M19 meta-correlation]
  - Mean V_GT under naive / gated / default selection [M20]
  - Gain V_gated - V_naive (no-regret claim)

Output: experiments/m23_bootstrap_ci/summary.json
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
log = logging.getLogger("m23_bootstrap")
RNG = np.random.default_rng(42)
B = 10000


def load_meta(dataset):
    if dataset == "bci2b":
        ope = json.loads((ROOT / "experiments" / "m17_ope_loso" / "summary.json").read_text())
        loso = json.loads((ROOT / "experiments" / "m8_loso_bci2b" / "summary.json").read_text())
    else:
        ope = json.loads((ROOT / "experiments" / "m19_ope_loso_bci2a" / "summary.json").read_text())
        loso = json.loads((ROOT / "experiments" / "m15_loso_bci2a" / "summary.json").read_text())
    accs = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
             for r in loso["per_subject"]}
    rows = []
    for r in ope["per_subject"]:
        sid = r["held_out"]
        if sid not in accs:
            continue
        rows.append({"sid": sid, "acc": accs[sid], "ope_r": r["pearson_r"],
                      "policies": r["policies"]})
    return rows


def boot_pearson(accs, rs, n=B):
    n_sub = len(accs)
    boot_r = []
    for _ in range(n):
        idx = RNG.integers(0, n_sub, size=n_sub)
        a, r = accs[idx], rs[idx]
        if np.std(a) > 0 and np.std(r) > 0:
            boot_r.append(stats.pearsonr(a, r)[0])
        else:
            boot_r.append(0.0)
    return np.array(boot_r)


def deploy_value(rows, score_key, threshold=None, default_policy="agent_T0"):
    """Per subject pick policy by score; if threshold given and acc<threshold use default."""
    vs = []
    for r in rows:
        names = [p["name"] for p in r["policies"]]
        scores = {p["name"]: p[score_key] for p in r["policies"]}
        vgts = {p["name"]: p["v_gt"] for p in r["policies"]}
        if threshold is not None and r["acc"] < threshold:
            pick = default_policy
        else:
            pick = max(names, key=lambda n: scores[n])
        vs.append(vgts.get(pick, float("nan")))
    return np.array(vs)


def boot_strategy_means(rows, n=B):
    """Bootstrap mean V_GT for naive / gated / default / oracle, paired by subject."""
    n_sub = len(rows)
    naive = deploy_value(rows, "v_fqe")
    gated = deploy_value(rows, "v_fqe", threshold=0.50)
    default_ = np.array([next(p["v_gt"] for p in r["policies"] if p["name"] == "agent_T0")
                         for r in rows])
    oracle = np.array([max(p["v_gt"] for p in r["policies"]) for r in rows])

    boot_naive, boot_gated, boot_default, boot_oracle = [], [], [], []
    boot_gain = []  # V_gated - V_naive paired diff
    for _ in range(n):
        idx = RNG.integers(0, n_sub, size=n_sub)
        boot_naive.append(naive[idx].mean())
        boot_gated.append(gated[idx].mean())
        boot_default.append(default_[idx].mean())
        boot_oracle.append(oracle[idx].mean())
        boot_gain.append((gated[idx] - naive[idx]).mean())
    return {
        "naive_mean": float(naive.mean()), "naive_ci": tuple(np.quantile(boot_naive, [0.025, 0.975])),
        "gated_mean": float(gated.mean()), "gated_ci": tuple(np.quantile(boot_gated, [0.025, 0.975])),
        "default_mean": float(default_.mean()), "default_ci": tuple(np.quantile(boot_default, [0.025, 0.975])),
        "oracle_mean": float(oracle.mean()), "oracle_ci": tuple(np.quantile(boot_oracle, [0.025, 0.975])),
        "gain_gated_minus_naive_mean": float((gated - naive).mean()),
        "gain_gated_minus_naive_ci": tuple(np.quantile(boot_gain, [0.025, 0.975])),
        "n_subjects": n_sub,
    }


def fmt_ci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]"


def main():
    out_dir = ROOT / "experiments" / "m23_bootstrap_ci"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    log.info("\n=== M23 Bootstrap CIs (B=%d) ===\n", B)

    for ds in ["bci2b", "bci2a"]:
        rows = load_meta(ds)
        accs = np.array([r["acc"] for r in rows])
        rs_ope = np.array([r["ope_r"] for r in rows])
        # Meta-correlation
        r_obs, p_obs = stats.pearsonr(accs, rs_ope)
        boot_r = boot_pearson(accs, rs_ope)
        ci_lo, ci_hi = np.quantile(boot_r, [0.025, 0.975])
        frac_pos = float((boot_r > 0).mean())
        log.info("[%s] meta-correlation r(decodability, OPE r):", ds)
        log.info("  observed r = %+.3f (analytical p=%.3g, n=%d)", r_obs, p_obs, len(rows))
        log.info("  bootstrap 95%% CI %s, P(r>0) = %.4f", fmt_ci(ci_lo, ci_hi), frac_pos)

        ds_out = {"meta": {"r_obs": float(r_obs), "p_obs": float(p_obs),
                            "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
                            "P_r_gt_0": frac_pos, "n_subjects": len(rows)}}

        # Deployment-rule strategies
        ds_out["deploy"] = boot_strategy_means(rows)
        d = ds_out["deploy"]
        log.info("[%s] deployment-rule strategy means [95%% CI]:", ds)
        log.info("  oracle  = %+.3f %s", d["oracle_mean"], fmt_ci(*d["oracle_ci"]))
        log.info("  gated   = %+.3f %s", d["gated_mean"], fmt_ci(*d["gated_ci"]))
        log.info("  default = %+.3f %s", d["default_mean"], fmt_ci(*d["default_ci"]))
        log.info("  naive   = %+.3f %s", d["naive_mean"], fmt_ci(*d["naive_ci"]))
        log.info("  gain (gated - naive) = %+.4f %s",
                 d["gain_gated_minus_naive_mean"], fmt_ci(*d["gain_gated_minus_naive_ci"]))
        log.info("    P(gain >= 0) [no-regret rate, paired bootstrap]: see below")
        # Re-compute the no-regret rate explicitly
        n_sub = len(rows)
        naive = deploy_value(rows, "v_fqe")
        gated = deploy_value(rows, "v_fqe", threshold=0.50)
        no_regret = []
        for _ in range(B):
            idx = RNG.integers(0, n_sub, size=n_sub)
            no_regret.append(((gated[idx] - naive[idx]).mean()) >= 0)
        nr_rate = float(np.mean(no_regret))
        log.info("    P(gated - naive >= 0) = %.4f", nr_rate)
        ds_out["deploy"]["no_regret_rate"] = nr_rate

        out[ds] = ds_out
        log.info("")

    # Combined N=18 deployment claim
    rows_all = load_meta("bci2b") + load_meta("bci2a")
    naive_all = deploy_value(rows_all, "v_fqe")
    gated_all = deploy_value(rows_all, "v_fqe", threshold=0.50)
    default_all = np.array([next(p["v_gt"] for p in r["policies"] if p["name"] == "agent_T0")
                             for r in rows_all])
    n_all = len(rows_all)
    boot_naive_lt_default = []
    boot_gated_geq_naive = []
    for _ in range(B):
        idx = RNG.integers(0, n_all, size=n_all)
        boot_naive_lt_default.append(naive_all[idx].mean() < default_all[idx].mean())
        boot_gated_geq_naive.append(gated_all[idx].mean() >= naive_all[idx].mean())
    log.info("=== Combined N=%d cross-dataset claims ===", n_all)
    log.info("  P(naive < default) = %.4f  [supports 'naive cross-subject FQE is harmful']",
             float(np.mean(boot_naive_lt_default)))
    log.info("  P(gated >= naive)  = %.4f  [supports 'gated is no-regret']",
             float(np.mean(boot_gated_geq_naive)))

    out["combined"] = {
        "n": n_all,
        "P_naive_lt_default": float(np.mean(boot_naive_lt_default)),
        "P_gated_geq_naive": float(np.mean(boot_gated_geq_naive)),
    }

    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    log.info("\nWrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
