# MIT License, 2026
""": SOTA-comparison table builder for bci2a 4-class within-subject.

Aggregates:
  *   : single-seed cvar+cmdp on bci2a sub 3 (random split, 70/15/15)
  *   : multi-seed × 3 subjects × 4 configs (random split)
  *   : within-subject mandatory baselines (FBCSP/CSP/EEGNet-mandatory,
            canonical session-T → session-E protocol)
  *   : multi-seed × 3 subjects × 2 configs (canonical protocol)
  * Literature: FBCSP (Ang 2012), EEGNet (Lawhern 2018), ShallowConvNet
                (Schirrmeister 2017), EEG-Conformer (Song 2023), CTNet
                (Zhao 2024), Transformer-2025 (Nat Sci Reports).

Produces:
  * experiments/sota_leaderboard/summary.json (machine-readable)
  * experiments/sota_leaderboard/sota_table.md (human-readable Markdown)

The literature numbers are mean accuracy across 9 subjects (canonical
session-T → session-E split). Our numbers from  /  are mean ± std
across 5 seeds × 3 subjects on the same canonical split, directly
comparable.  /  use random splits and are flagged as "non-canonical".
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m36_sota")

# Published SOTA numbers on bci2a 4-class within-subject (canonical split,
# session T train -> session E test, 9 subjects, mean accuracy).
LITERATURE = [
    {"name": "FBCSP (Ang et al. 2012)",                "acc": 0.6775, "kappa": 0.569, "n_subj": 9, "year": 2012},
    {"name": "DeepConvNet (Schirrmeister 2017)",       "acc": 0.7090, "kappa": None,  "n_subj": 9, "year": 2017},
    {"name": "ShallowConvNet (Schirrmeister 2017)",    "acc": 0.7370, "kappa": None,  "n_subj": 9, "year": 2017},
    {"name": "EEGNet (Lawhern 2018, reproduced)",      "acc": 0.7400, "kappa": None,  "n_subj": 9, "year": 2018},
    {"name": "LMDA-Net (Miao 2023)",                   "acc": 0.7520, "kappa": 0.670, "n_subj": 9, "year": 2023},
    {"name": "FBCNet (Mane 2021)",                     "acc": 0.7620, "kappa": None,  "n_subj": 9, "year": 2021},
    {"name": "EEG-Conformer (Song 2023)",              "acc": 0.7866, "kappa": 0.720, "n_subj": 9, "year": 2023},
    {"name": "CTNet (Zhao 2024)",                      "acc": 0.8252, "kappa": None,  "n_subj": 9, "year": 2024},
    {"name": "Transformer (Nat. Sci. Reports 2025)",   "acc": 0.8646, "kappa": None,  "n_subj": 9, "year": 2025},
]


def safe_load(path):
    p = ROOT / "experiments" / path
    if not p.exists():
        log.warning("Missing %s — entry skipped", p)
        return None
    return json.loads(p.read_text())


def main():
    out_dir = ROOT / "experiments" / "sota_leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    m10 = safe_load("bci2a_within/summary.json")
    m34 = safe_load("multiseed_within_bci2a/summary.json")
    m35 = safe_load("within_subject_baselines_3subj/summary.json")
    m38 = safe_load("neuropolicy_canonical_bci2a/summary.json")
    m39 = safe_load("conformer_canonical_bci2a_3subj/summary.json")
    m39b = safe_load("conformer_canonical_bci2a_all9/summary.json")

    rows = []

    # === Literature ===
    for lit in LITERATURE:
        rows.append({"name": lit["name"], "protocol": "canonical (T→E)",
                      "task_acc": lit["acc"], "task_acc_std": None,
                      "commit_rate": 1.0, "itr_bpm": None, "n_subj": lit["n_subj"],
                      "n_seeds": 1, "selective": False, "source": "literature"})

    # === : our baselines on canonical protocol ===
    if m35 is not None and "aggregate" in m35:
        agg = m35["aggregate"]
        rows.append({"name": "CSP+LDA (ours, canonical, 3 subj)", "protocol": "canonical (T→E)",
                      "task_acc": agg["csp_lda_mean"], "task_acc_std": float(np.std(agg["csp_lda_per_subject"], ddof=1)),
                      "commit_rate": 1.0, "itr_bpm": None,
                      "n_subj": len(m35["subjects"]), "n_seeds": 1,
                      "selective": False, "source": ""})
        rows.append({"name": "FBCSP+LDA (ours, canonical, 3 subj)", "protocol": "canonical (T→E)",
                      "task_acc": agg["fbcsp_lda_mean"], "task_acc_std": float(np.std(agg["fbcsp_lda_per_subject"], ddof=1)),
                      "commit_rate": 1.0, "itr_bpm": None,
                      "n_subj": len(m35["subjects"]), "n_seeds": 1,
                      "selective": False, "source": ""})
        rows.append({"name": "EEGNet-mandatory (ours, canonical, 3 subj × 5 seeds)", "protocol": "canonical (T→E)",
                      "task_acc": agg["eegnet_mandatory_mean"], "task_acc_std": float(np.std(agg["eegnet_mandatory_per_subject"], ddof=1)),
                      "commit_rate": 1.0, "itr_bpm": None,
                      "n_subj": len(m35["subjects"]), "n_seeds": 5,
                      "selective": False, "source": ""})

    # ===  / : EEG-Conformer (Song 2023) — SOTA push ===
    for tag, src in [(" (3 subj)", m39), (" (9 subj)", m39b)]:
        if src is None or "aggregate" not in src:
            continue
        agg = src["aggregate"]
        n_subj = len(src["subjects"])
        n_seeds = len(src["seeds"])
        rows.append({
            "name": f"EEG-Conformer full-trial (ours, canonical, {n_subj} subj × {n_seeds} seeds) [{tag}]",
            "protocol": "canonical (T→E)",
            "task_acc": agg["conformer_full_trial_mean"],
            "task_acc_std": agg["conformer_full_trial_std_subjects"],
            "commit_rate": 1.0, "itr_bpm": None,
            "n_subj": n_subj, "n_seeds": n_seeds,
            "selective": False, "source": tag,
        })
        rows.append({
            "name": f"EEG-Conformer + window-avg (ours, canonical, {n_subj} subj × {n_seeds} seeds) [{tag}]",
            "protocol": "canonical (T→E)",
            "task_acc": agg["conformer_windowavg_mean"],
            "task_acc_std": agg["conformer_windowavg_std_subjects"],
            "commit_rate": 1.0, "itr_bpm": None,
            "n_subj": n_subj, "n_seeds": n_seeds,
            "selective": False, "source": tag,
        })

    # === : NeuroPolicy on canonical protocol ===
    if m38 is not None and "cross_subject" in m38:
        for name in m38["configs"]:
            cs = m38["cross_subject"][name]
            # Aggregate std across seeds for the task_acc
            stds = []
            for sid in m38["subjects"]:
                stds.append(m38["per_subject"][str(sid)]["aggregate"][name]["task_accuracy_std"])
            rows.append({
                "name": f"NeuroPolicy {name} (, canonical, 3 subj × 5 seeds)",
                "protocol": "canonical (T→E)",
                "task_acc": cs["task_accuracy_mean_of_means"],
                "task_acc_std": float(np.mean(stds)),
                "commit_acc": cs["accuracy_on_commits_mean_of_means"],
                "commit_rate": None,  # filled below
                "itr_bpm": cs["itr_mean_of_means"],
                "n_subj": len(m38["subjects"]), "n_seeds": len(m38["seeds"]),
                "selective": True, "source": ""})
            # Add commit-rate from per-subject means
            rates = [m38["per_subject"][str(sid)]["aggregate"][name]["commit_rate_mean"]
                      for sid in m38["subjects"]]
            rows[-1]["commit_rate"] = float(np.mean(rates))

    # === : NeuroPolicy on random protocol (non-canonical, less comparable) ===
    if m34 is not None and "cross_subject" in m34:
        for name in m34["configs"]:
            cs = m34["cross_subject"][name]
            stds = []
            for sid in m34["subjects"]:
                stds.append(m34["per_subject"][str(sid)]["aggregate"][name]["task_accuracy_std"])
            rates = [m34["per_subject"][str(sid)]["aggregate"][name]["commit_rate_mean"]
                      for sid in m34["subjects"]]
            rows.append({
                "name": f"NeuroPolicy {name} (, RANDOM 70/15/15, 3 subj × 5 seeds)",
                "protocol": "non-canonical random",
                "task_acc": cs["task_accuracy_mean_of_means"],
                "task_acc_std": float(np.mean(stds)),
                "commit_acc": cs["accuracy_on_commits_mean_of_means"],
                "commit_rate": float(np.mean(rates)),
                "itr_bpm": cs["itr_mean_of_means"],
                "n_subj": len(m34["subjects"]), "n_seeds": len(m34["seeds"]),
                "selective": True, "source": ""})

    # === : single-seed cvar reference ===
    if m10 is not None and "cvar_a0.25" in m10:
        t = m10["cvar_a0.25"]["test"]
        rows.append({"name": "NeuroPolicy cvar_a0.25 (, RANDOM 70/15/15, sub 3, seed 0)",
                      "protocol": "non-canonical random",
                      "task_acc": t["n_correct_commits"] / m10["n_test_episodes"],
                      "task_acc_std": None,
                      "commit_acc": t["accuracy_on_commits"],
                      "commit_rate": t["n_commits"] / m10["n_test_episodes"],
                      "itr_bpm": t["information_transfer_rate"],
                      "n_subj": 1, "n_seeds": 1,
                      "selective": True, "source": ""})

    # === Sort & write ===
    rows_sorted = sorted(rows, key=lambda r: -(r["task_acc"] or 0))
    out = {"rows": rows_sorted,
            "literature_sources": LITERATURE,
            "notes": ("Canonical protocol = session 0 (T, 288 trials) train+val, "
                      "session 1 (E, 288 trials) test. The 9 subjects in literature numbers "
                      "may not match the 3 subjects used in our / — comparison is "
                      "indicative of regime, not strictly head-to-head.")}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))

    # Markdown table
    md_lines = [
        "# SOTA Comparison — bci2a 4-class within-subject",
        "",
        "Sorted by task accuracy (correct commits / total trials).",
        "",
        "| Method | Protocol | Task Acc | Commit Acc | Commit Rate | ITR (b/m) | n_subj | n_seeds | Source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        ta = f"{r['task_acc']:.3f}"
        if r.get("task_acc_std") is not None:
            ta += f" ± {r['task_acc_std']:.3f}"
        ca = f"{r.get('commit_acc', 1.0):.3f}" if r.get("commit_acc") is not None else f"{r['task_acc']:.3f}"
        cr = f"{r['commit_rate']:.2f}" if r["commit_rate"] is not None else "—"
        itr = f"{r['itr_bpm']:.1f}" if r["itr_bpm"] is not None else "—"
        md_lines.append(
            f"| {r['name']} | {r['protocol']} | {ta} | {ca} | {cr} | {itr} | {r['n_subj']} | {r['n_seeds']} | {r['source']} |"
        )
    md_lines += [
        "",
        "**Notes:**",
        "- *Task Acc* = correct commits / total trials. This is what classification baselines report (every trial gets a forced prediction). For selective decoders (NeuroPolicy), deferred/abstained trials count as 0 by this metric.",
        "- *Commit Acc* = correct commits / committed trials. Only meaningful for selective decoders; for mandatory baselines, commit acc = task acc.",
        "- *Commit Rate* = committed trials / total trials. Selective decoders trade commit rate for commit accuracy and ITR.",
        "- *Protocol*: \"canonical (T→E)\" = standard BCI Competition IV split (session 0 train + session 1 test); \"non-canonical random\" = 70/15/15 random across both sessions (3-8% accuracy inflation per published comparisons).",
        "- Literature numbers report mean across 9 subjects on canonical protocol. Our / numbers are mean across 3 subjects (1, 3, 7) × 5 seeds on the same canonical protocol — indicative of regime, not strictly subject-matched.",
    ]
    (out_dir / "sota_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    log.info("Wrote SOTA table to %s and %s",
              out_dir / "summary.json", out_dir / "sota_table.md")
    log.info("Sorted preview:")
    for r in rows_sorted[:8]:
        log.info("  %.3f  %s", r["task_acc"], r["name"][:60])


if __name__ == "__main__":
    main()
