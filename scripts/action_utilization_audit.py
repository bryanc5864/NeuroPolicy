# MIT License, 2026
""": Action-utilization analysis across all trained agents.

W1, W9 critique: does the rich {Commit, Defer, Recal, Abstain} action
space matter? This script aggregates `outcome_counts` from every saved
experiment to quantify what fraction of actions each agent actually uses.

Sources (all single-agent variants across datasets):
  -   bci2b sub 4 within (scalar CQL)
  -   bci2b sub 4 within (4 variants: scalar/scalar+CMDP/CVaR/CVaR+CMDP)
  -  bci2a sub 3 within (same 4 variants)
  -  Lee2019 sub 1 within (same 4 variants)
  -   bci2b LOSO per-subject (9 × 3 configs)
  -  bci2a LOSO per-subject (9 × 3 configs)

Output: experiments/action_utilization/summary.json + a per-action
share table with mean over agents.
"""
from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("action_utilization")


def harvest_outcome_counts(summary_path: Path, label: str):
    """Recursively descend `outcome_counts` dicts; emit (variant, split, counts) per occurrence."""
    try:
        d = json.loads(summary_path.read_text())
    except Exception:
        return []
    rows = []

    def _emit(prefix, obj):
        if isinstance(obj, dict):
            if "outcome_counts" in obj:
                rows.append({"source": label, "variant": prefix,
                              "counts": obj["outcome_counts"]})
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    _emit(f"{prefix}/{k}", v)
        elif isinstance(obj, list):
            for i, x in enumerate(obj):
                _emit(f"{prefix}[{i}]", x)

    _emit("", d)
    return rows


def normalise(counts: dict) -> dict:
    """Return action shares + raw totals, ignoring `recal_repeated_no_op` (env-internal)."""
    keep = {k: v for k, v in counts.items() if k != "recal_repeated_no_op"}
    total = sum(keep.values())
    if total == 0:
        return None
    return {k: v / total for k, v in keep.items()} | {"_total": total}


def main():
    out_dir = ROOT / "experiments" / "action_utilization"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map experiment dir → label
    targets = [
        ("cvar_cmdp_within/summary.json", "M9_bci2b_within_sub4"),
        ("bci2a_within/summary.json", "M10_bci2a_within_sub3"),
        ("lee2019_within/summary.json", "M11_lee2019_within_sub1"),
        ("cvar_alpha_sweep/summary.json", "M12_cvar_sweep_bci2b_sub4"),
        ("loso_bci2b/summary.json", "M8_loso_bci2b"),
        ("loso_bci2a/summary.json", "M15_loso_bci2a"),
        ("labram_finetune_within/summary.json", "M13b_labram_bci2a_sub3"),
        ("labram_loso_bci2b/summary.json", "M14_labram_loso_bci2b"),
        ("labram_loso_bci2a/summary.json", "M16_labram_loso_bci2a"),
    ]

    all_rows = []
    for rel, label in targets:
        p = ROOT / "experiments" / rel
        if not p.exists():
            log.warning("missing %s", p); continue
        rows = harvest_outcome_counts(p, label)
        all_rows.extend(rows)
        log.info("%-30s → %d rows", label, len(rows))

    # Filter to trained-agent rows only (drop random_*)
    trained = [r for r in all_rows
                if "random" not in r["variant"].lower()
                and r["counts"].get("defer", 0) + r["counts"].get("correct_commit", 0) > 0]
    log.info("\nTotal rows: %d  →  trained-agent rows: %d", len(all_rows), len(trained))

    # Per-row shares
    shares = []
    for r in trained:
        sh = normalise(r["counts"])
        if sh is None:
            continue
        shares.append({"source": r["source"], "variant": r["variant"], **sh})

    # Aggregate per source × per action
    action_keys = {"correct_commit", "wrong_commit", "defer", "abstain",
                   "abstain_timeout", "recal"}
    agg = defaultdict(lambda: defaultdict(list))
    for s in shares:
        for k in action_keys:
            agg[s["source"]][k].append(s.get(k, 0.0))

    log.info("\n=== Action utilisation share (mean over agents in each source) ===")
    header = f"  {'source':<32s} {'n':>3s}  {'commit_correct':>15s} {'commit_wrong':>13s} {'defer':>8s} {'abstain':>8s} {'abstain_TO':>10s} {'recal':>7s}"
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))
    rollup = {}
    for src, d in agg.items():
        n = len(d.get("defer", []))
        row = {k: float(np.mean(v)) if v else 0.0 for k, v in d.items()}
        row["n_agents"] = n
        rollup[src] = row
        log.info(f"  {src:<32s} {n:>3d}  "
                  f"{row['correct_commit']:>15.3f} {row['wrong_commit']:>13.3f} "
                  f"{row['defer']:>8.3f} {row['abstain']:>8.3f} "
                  f"{row['abstain_timeout']:>10.3f} {row['recal']:>7.3f}")

    # Overall trained-agent average
    overall = {k: float(np.mean([s.get(k, 0.0) for s in shares])) for k in action_keys}
    overall["n_agents"] = len(shares)
    log.info("\n=== OVERALL TRAINED-AGENT AVERAGE (N=%d agent×split) ===", len(shares))
    log.info("  commit_correct = %.3f", overall["correct_commit"])
    log.info("  commit_wrong   = %.3f", overall["wrong_commit"])
    log.info("  defer          = %.3f", overall["defer"])
    log.info("  abstain        = %.3f", overall["abstain"])
    log.info("  abstain_TO     = %.3f", overall["abstain_timeout"])
    log.info("  recal          = %.3f", overall["recal"])

    # Fraction of agents that use each action AT ALL (>0%)
    log.info("\n=== Fraction of agents using each action AT LEAST ONCE ===")
    used = {}
    for k in action_keys:
        used[k] = float(np.mean([(s.get(k, 0.0) > 0.0) for s in shares]))
        log.info("  %-14s: %.2f", k, used[k])

    # Recal/Abstain analysis specifically
    n_used_recal = sum(s.get("recal", 0.0) > 0 for s in shares)
    n_used_abstain = sum(s.get("abstain", 0.0) > 0 for s in shares)
    n_used_abstain_TO = sum(s.get("abstain_timeout", 0.0) > 0 for s in shares)
    log.info("\n=== Sharp findings ===")
    log.info("  Recal usage: %d/%d agents (%.1f%%)", n_used_recal, len(shares),
              100*n_used_recal/len(shares))
    log.info("  Explicit Abstain usage: %d/%d agents (%.1f%%)",
              n_used_abstain, len(shares), 100*n_used_abstain/len(shares))
    log.info("  Abstain-by-timeout usage: %d/%d agents (%.1f%%)",
              n_used_abstain_TO, len(shares), 100*n_used_abstain_TO/len(shares))

    summary = {"per_source": rollup, "overall": overall, "used": used,
                "n_rows": len(shares), "shares": shares}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("\nWrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
