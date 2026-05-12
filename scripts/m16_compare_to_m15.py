# MIT License - Bryan Cheng, 2026
"""Compare M16 LaBraM LOSO vs M15 EEGNet LOSO on bci2a (paired test)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    m15 = json.loads((ROOT / "experiments" / "m15_loso_bci2a" / "summary.json").read_text())
    m16 = json.loads((ROOT / "experiments" / "m16_labram_loso_bci2a" / "summary.json").read_text())

    e_per = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"] for r in m15["per_subject"]}
    l_per = {r["held_out"]: r["results"]["labram_cmdp"]["metrics"] for r in m16["per_subject"]}

    print(f"{'sub':>3}   {'EEGNet ret':>11}/{'acc':>5}/{'ITR':>5}  {'LaBraM ret':>11}/{'acc':>5}/{'ITR':>5}  {'dret':>8} {'dacc':>8} {'verdict':>8}")
    print("-" * 110)
    deltas_ret = []; deltas_acc = []; deltas_itr = []; n_wins = 0
    rows = []
    for sid in sorted(l_per.keys()):
        e = e_per[sid]; l = l_per[sid]
        d_ret = l["return_mean"] - e["return_mean"]
        d_acc = l["accuracy_on_commits"] - e["accuracy_on_commits"]
        d_itr = l["information_transfer_rate"] - e["information_transfer_rate"]
        verdict = "WIN" if d_ret > 0 else ("LOSS" if d_ret < 0 else "TIE")
        if d_ret > 0:
            n_wins += 1
        print(f"{sid:>3}   {e['return_mean']:+.3f}/{e['accuracy_on_commits']:.3f}/{e['information_transfer_rate']:5.2f}    "
              f"{l['return_mean']:+.3f}/{l['accuracy_on_commits']:.3f}/{l['information_transfer_rate']:5.2f}    "
              f"{d_ret:+.3f}  {d_acc:+.3f}  {verdict:>5}")
        deltas_ret.append(d_ret); deltas_acc.append(d_acc); deltas_itr.append(d_itr)
        rows.append({"subject_id": int(sid),
                     "eegnet_metrics": {k: float(e[k]) for k in
                         ["return_mean", "accuracy_on_commits", "information_transfer_rate", "wrong_commit_rate"]},
                     "labram_metrics": {k: float(l[k]) for k in
                         ["return_mean", "accuracy_on_commits", "information_transfer_rate", "wrong_commit_rate"]},
                     "delta_return": float(d_ret), "delta_accuracy": float(d_acc),
                     "delta_itr": float(d_itr)})
    print("-" * 110)
    print(f"Mean Δret  = {np.mean(deltas_ret):+.3f} ± {np.std(deltas_ret, ddof=1):.3f}")
    print(f"Mean Δacc  = {np.mean(deltas_acc):+.3f} ± {np.std(deltas_acc, ddof=1):.3f}")
    print(f"Mean Δitr  = {np.mean(deltas_itr):+.3f} ± {np.std(deltas_itr, ddof=1):.3f}")
    print(f"Wins (LaBraM > EEGNet on return): {n_wins}/{len(deltas_ret)}")

    try:
        from scipy import stats
        _, p_one = stats.wilcoxon(deltas_ret, alternative="greater")
        _, p_two = stats.wilcoxon(deltas_ret, alternative="two-sided")
        print(f"Paired Wilcoxon (LaBraM > EEGNet, 1-sided): p = {p_one:.3f}")
        print(f"Paired Wilcoxon (two-sided):                 p = {p_two:.3f}")
    except Exception as e:
        print(f"Wilcoxon failed: {e}")
        p_one = p_two = float("nan")

    out = {"per_subject": rows,
           "mean_delta_return": float(np.mean(deltas_ret)),
           "std_delta_return": float(np.std(deltas_ret, ddof=1)),
           "mean_delta_accuracy": float(np.mean(deltas_acc)),
           "mean_delta_itr": float(np.mean(deltas_itr)),
           "n_wins": int(n_wins), "n_subjects": len(deltas_ret),
           "wilcoxon_one_sided_p": float(p_one),
           "wilcoxon_two_sided_p": float(p_two)}
    out_path = ROOT / "experiments" / "m16_labram_loso_bci2a" / "m16_vs_m15_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
