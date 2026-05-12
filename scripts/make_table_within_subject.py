# MIT License, 2026
"""Generate combined within-subject ablation table across  (bci2b),
 (bci2a),  (Lee2019). Outputs a markdown table to stdout.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict | None:
    p = ROOT / "experiments" / name / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    m9 = load("cvar_cmdp_within")
    m10 = load("bci2a_within")
    m11 = load("lee2019_within")
    print(f"# Within-subject ablation across three datasets")
    print()
    print("| Dataset | Subject | Variant | Test return | Test acc | Test ITR | Wrong rate |")
    print("|---|---|---|---:|---:|---:|---:|")
    for name, d, label in [
        ("bci2b sub 4 (2-class)", m9, ""),
        ("bci2a sub 3 (4-class)", m10, ""),
        ("Lee2019 sub 1 (2-class)", m11, ""),
    ]:
        if d is None:
            print(f"| {name} | — | — | (not yet run) | | | |")
            continue
        rand_test = d.get("random_test", {})
        print(f"| {name} | random | random | {rand_test.get('return_mean', '—'):+.3f} | {rand_test.get('accuracy_on_commits', '—'):.3f} | {rand_test.get('information_transfer_rate', '—'):.1f} | — |")
        for k in ["scalar_a1", "scalar_a1_cmdp_eps0.10", "cvar_a0.25", "cvar_a0.25_cmdp_eps0.10"]:
            if k in d:
                t = d[k]["test"]
                print(f"| {name} | {k} | {label} | {t['return_mean']:+.3f} | {t['accuracy_on_commits']:.3f} | {t['information_transfer_rate']:.1f} | {t['wrong_commit_rate']:.3f} |")
    print()
    print("Best per dataset (max test return):")
    for name, d in [("bci2b sub 4 (2-class)", m9), ("bci2a sub 3 (4-class)", m10), ("Lee2019 sub 1 (2-class)", m11)]:
        if d is None: continue
        best_k, best_r = None, -1e9
        for k in ["scalar_a1", "scalar_a1_cmdp_eps0.10", "cvar_a0.25", "cvar_a0.25_cmdp_eps0.10"]:
            if k in d:
                r = d[k]["test"]["return_mean"]
                if r > best_r:
                    best_k, best_r = k, r
        print(f"  {name}: {best_k} (return {best_r:+.3f})")


if __name__ == "__main__":
    main()
