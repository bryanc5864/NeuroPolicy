# MIT License, 2026
"""Generate the time-to-commit characterization figure.

Reads experiments/neuropolicy_canonical_bci2a_conformer_9subj/summary.json
(or any NeuroPolicy summary that contains decision_seconds + commit_correct
per-seed entries) and produces:

  figures/time_to_commit.png      - 2-panel figure
    Panel A: histogram of commit times within trial (0..4s)
    Panel B: commit accuracy as a function of decision time bin

This is task #79 and complements §results characterisation of when the
selective decoder commits within a trial.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt


def load_decisions(summary_path: Path, config: str = "cvar_a0.25_cmdp_eps0.10"):
    """Pool decision_seconds + commit_correct across subjects and seeds."""
    d = json.loads(summary_path.read_text())
    all_secs = []
    all_correct = []
    per_subject = d.get("per_subject", {})
    for sid, sd in per_subject.items():
        seeds = sd.get("per_seed", {})
        for seed_key, sr in seeds.items():
            block = sr.get(config, {})
            test = block.get("test", block)  # canonical_bci2a uses .test, threshold uses flat
            ds = test.get("decision_seconds")
            cc = test.get("commit_correct")
            if ds is not None:
                all_secs.extend([float(x) for x in ds])
                all_correct.extend([int(x) for x in (cc or [-1] * len(ds))])
    return np.array(all_secs), np.array(all_correct)


def main():
    summary_path = ROOT / "experiments" / "neuropolicy_canonical_bci2a_conformer_9subj" / "summary.json"
    if not summary_path.exists():
        # Fallback to the 3-subject panel summary.
        summary_path = ROOT / "experiments" / "neuropolicy_canonical_bci2a_conformer" / "summary.json"
    if not summary_path.exists():
        print(f"No summary found. Need decision_seconds in experiment output first.")
        return

    secs, correct = load_decisions(summary_path)
    if len(secs) == 0:
        print("Summary exists but has no decision_seconds entries. Re-run experiment "
              "with the updated policy_eval.py to populate them.")
        return

    print(f"Loaded {len(secs)} commit decisions from {summary_path.name}")
    print(f"  mean decision time: {secs.mean():.2f}s, median: {np.median(secs):.2f}s")
    print(f"  decision time range: [{secs.min():.2f}, {secs.max():.2f}]s")
    print(f"  commit accuracy: {correct.mean():.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: histogram of commit times.
    ax1.hist(secs, bins=np.linspace(0, secs.max() + 0.1, 16),
              color="steelblue", edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Decision time within trial (s)")
    ax1.set_ylabel("# commit decisions")
    ax1.set_title(f"(A) Commit time distribution\n(N={len(secs)}, median={np.median(secs):.2f}s)")
    ax1.axvline(np.median(secs), color="firebrick", linestyle="--", linewidth=1.5,
                  label=f"median {np.median(secs):.2f}s")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # Panel B: commit accuracy vs decision time bin.
    bins = np.linspace(0, secs.max() + 0.1, 7)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    accs, ns = [], []
    for i in range(len(bins) - 1):
        m = (secs >= bins[i]) & (secs < bins[i + 1])
        if m.sum() >= 5:
            accs.append(correct[m].mean())
            ns.append(int(m.sum()))
        else:
            accs.append(np.nan); ns.append(int(m.sum()))
    ax2.bar(bin_centers, accs, width=(bins[1] - bins[0]) * 0.85,
              color="seagreen", edgecolor="black", alpha=0.85)
    for x, a, n in zip(bin_centers, accs, ns):
        if not np.isnan(a):
            ax2.text(x, a + 0.01, f"n={n}", ha="center", fontsize=8)
    ax2.set_xlabel("Decision time within trial (s)")
    ax2.set_ylabel("Commit accuracy")
    ax2.set_title(f"(B) Accuracy by decision time\n(overall {correct.mean():.3f})")
    ax2.set_ylim(0, 1.0)
    ax2.axhline(0.25, color="grey", linestyle=":", linewidth=1, label="chance (1/4)")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = ROOT / "figures" / "time_to_commit.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
