# MIT License - Bryan Cheng, 2026
"""Compare M14 LaBraM LOSO vs M8 EEGNet LOSO (apples-to-apples).

Both use scalar CQL + CMDP eps=0.10, 9 LOSO subjects on bci2b. The only
difference is the encoder.

Output: console summary + saves paired-difference summary to
experiments/m14_labram_loso_bci2b/m14_vs_m8_compare.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    m14 = json.loads((ROOT / "experiments" / "m14_labram_loso_bci2b" / "summary.json").read_text())
    m8 = json.loads((ROOT / "experiments" / "m8_loso_bci2b" / "summary.json").read_text())

    # Extract per-subject metrics for cmdp_eps0.1 (M8) and labram_cmdp (M14)
    m14_per = {r["held_out"]: r["results"]["labram_cmdp"]["metrics"] for r in m14["per_subject"]}
    m8_per = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"] for r in m8["per_subject"]}
    m14_random = {r["held_out"]: r["results"]["random"] for r in m14["per_subject"]}
    m8_random = {r["held_out"]: r["results"]["random"] for r in m8["per_subject"]}

    rows = []
    print(f"{'sub':>3} {'M8 EEGNet':>30} {'M14 LaBraM':>30} {'d_ret':>8} {'d_acc':>8} {'verdict':>10}")
    print("-" * 110)
    deltas_ret = []
    deltas_acc = []
    deltas_itr = []
    n_wins = 0
    for sid in sorted(m14_per.keys()):
        e = m8_per[sid]
        l = m14_per[sid]
        d_ret = l["return_mean"] - e["return_mean"]
        d_acc = l["accuracy_on_commits"] - e["accuracy_on_commits"]
        d_itr = l["information_transfer_rate"] - e["information_transfer_rate"]
        verdict = "WIN" if d_ret > 0 else ("LOSS" if d_ret < 0 else "TIE")
        if d_ret > 0:
            n_wins += 1
        print(f"{sid:>3} ret={e['return_mean']:+.3f} acc={e['accuracy_on_commits']:.3f} ITR={e['information_transfer_rate']:.2f}  "
              f"  ret={l['return_mean']:+.3f} acc={l['accuracy_on_commits']:.3f} ITR={l['information_transfer_rate']:.2f}   "
              f"{d_ret:+.3f} {d_acc:+.3f}  {verdict}")
        deltas_ret.append(d_ret); deltas_acc.append(d_acc); deltas_itr.append(d_itr)
        rows.append({"subject_id": int(sid),
                     "eegnet_metrics": {k: float(e[k]) for k in
                         ["return_mean", "accuracy_on_commits", "information_transfer_rate", "wrong_commit_rate"]},
                     "labram_metrics": {k: float(l[k]) for k in
                         ["return_mean", "accuracy_on_commits", "information_transfer_rate", "wrong_commit_rate"]},
                     "delta_return": float(d_ret),
                     "delta_accuracy": float(d_acc),
                     "delta_itr": float(d_itr)})
    print("-" * 110)
    print(f"Mean Δret  = {np.mean(deltas_ret):+.3f} ± {np.std(deltas_ret, ddof=1):.3f}")
    print(f"Mean Δacc  = {np.mean(deltas_acc):+.3f} ± {np.std(deltas_acc, ddof=1):.3f}")
    print(f"Mean Δitr  = {np.mean(deltas_itr):+.3f} ± {np.std(deltas_itr, ddof=1):.3f}")
    print(f"Wins (LaBraM > EEGNet on return): {n_wins}/{len(deltas_ret)}")

    try:
        from scipy import stats
        w, p_one = stats.wilcoxon(deltas_ret, alternative="greater")
        w2, p_two = stats.wilcoxon(deltas_ret, alternative="two-sided")
        print(f"Paired Wilcoxon (one-sided LaBraM > EEGNet): p = {p_one:.3f}")
        print(f"Paired Wilcoxon (two-sided):                 p = {p_two:.3f}")
    except Exception as e:
        print(f"Wilcoxon failed: {e}")
        p_one = p_two = float("nan")

    out = {"per_subject": rows,
           "mean_delta_return": float(np.mean(deltas_ret)),
           "std_delta_return": float(np.std(deltas_ret, ddof=1)),
           "mean_delta_accuracy": float(np.mean(deltas_acc)),
           "mean_delta_itr": float(np.mean(deltas_itr)),
           "n_wins": int(n_wins),
           "n_subjects": len(deltas_ret),
           "wilcoxon_one_sided_p": float(p_one),
           "wilcoxon_two_sided_p": float(p_two)}
    out_path = ROOT / "experiments" / "m14_labram_loso_bci2b" / "m14_vs_m8_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
