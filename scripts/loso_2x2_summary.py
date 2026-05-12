# MIT License - Bryan Cheng, 2026
"""Generate the 2x2 LOSO summary: {bci2a, bci2b} × {EEGNet, LaBraM}.

Outputs a markdown table that serves as the headline cross-subject
result for the paper.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_loso(name, dataset, encoder, agg_key, cfg_key):
    path = ROOT / "experiments" / name / "summary.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    rets = []; accs = []; itrs = []; rand_rets = []
    for r in d["per_subject"]:
        m = r["results"][cfg_key]["metrics"] if "metrics" in r["results"][cfg_key] else r["results"][cfg_key]
        rets.append(m["return_mean"])
        accs.append(m["accuracy_on_commits"])
        itrs.append(m["information_transfer_rate"])
        rand_rets.append(r["results"]["random"]["return_mean"])
    rets = np.array(rets); accs = np.array(accs); itrs = np.array(itrs); rand_rets = np.array(rand_rets)
    deltas = rets - rand_rets
    n_wins = int((deltas > 0).sum())
    try:
        from scipy import stats
        _, p_one = stats.wilcoxon(deltas, alternative="greater")
    except Exception:
        p_one = float("nan")
    return {
        "dataset": dataset, "encoder": encoder,
        "ret_mean": float(rets.mean()), "ret_std": float(rets.std(ddof=1)),
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
        "itr_mean": float(itrs.mean()), "itr_std": float(itrs.std(ddof=1)),
        "delta_ret_mean": float(deltas.mean()), "n_wins": n_wins, "n_total": len(deltas),
        "wilcoxon_p_one_sided": float(p_one),
    }


def main():
    rows = [
        load_loso("m8_loso_bci2b", "bci2b (2-cls, N=9)", "EEGNet (M8)",
                   "agg", "cmdp_eps0.1"),
        load_loso("m14_labram_loso_bci2b", "bci2b (2-cls, N=9)", "LaBraM (M14)",
                   "agg", "labram_cmdp"),
        load_loso("m15_loso_bci2a", "bci2a (4-cls, N=9)", "EEGNet (M15)",
                   "agg", "cmdp_eps0.1"),
        load_loso("m16_labram_loso_bci2a", "bci2a (4-cls, N=9)", "LaBraM (M16)",
                   "agg", "labram_cmdp"),
    ]

    print(f"{'Dataset':<26} {'Encoder':<14} {'mean ret':>9} {'mean acc':>9} {'mean ITR':>9} {'Δret':>7} {'wins':>5} {'p (1s)':>7}")
    print("-" * 90)
    for r in rows:
        if r is None:
            print("(not found)"); continue
        print(f"{r['dataset']:<26} {r['encoder']:<14} {r['ret_mean']:+.3f}    {r['acc_mean']:.3f}    {r['itr_mean']:5.2f}     "
              f"{r['delta_ret_mean']:+.3f}  {r['n_wins']}/{r['n_total']}  {r['wilcoxon_p_one_sided']:.3f}")
    print()
    print("Markdown table:")
    print()
    print("| Dataset | Encoder | Mean ret | Mean acc | Mean ITR | Δret | Wins | Wilcoxon p |")
    print("|---|---|---:|---:|---:|---:|:---:|:---:|")
    for r in rows:
        if r is None: continue
        print(f"| {r['dataset']} | {r['encoder']} | {r['ret_mean']:+.3f}±{r['ret_std']:.2f} | "
              f"{r['acc_mean']:.3f}±{r['acc_std']:.3f} | {r['itr_mean']:.1f}±{r['itr_std']:.1f} | "
              f"{r['delta_ret_mean']:+.3f} | {r['n_wins']}/{r['n_total']} | {r['wilcoxon_p_one_sided']:.3f} |")

    out_path = ROOT / "experiments" / "loso_2x2_summary.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
