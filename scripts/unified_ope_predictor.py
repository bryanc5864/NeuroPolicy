# MIT License, 2026
""": Unified OPE-quality predictor — multivariate cross-dataset regression.

Combines per-subject OPE r from  (bci2b N=9),  (bci2a N=9),
and  (Lee2019 N=8) for a 3-dataset corpus of N=26 (subject, dataset)
pairs. Regresses per-subject Pearson r on (log channel count, held-out
decodability) using OLS + LOOCV. Identifies the dominant predictor
and yields a deployable formula:
    OPE_r ≈ a · log2(channels) + b · decodability + c

Cross-validation: leave-one-subject-out RMSE, leave-one-dataset-out
generalisation.

Output: experiments/unified_ope_predictor/summary.json
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
log = logging.getLogger("unified_ope_predictor")

DATASETS = [
    ("bci2b", 3,  "ope_loso_bci2b/summary.json",       "loso_bci2b/summary.json"),
    ("bci2a", 22, "ope_loso_bci2a/summary.json", "loso_bci2a/summary.json"),
    ("lee2019", 62, "ope_loso_lee2019_extended/summary.json", None),  # use encoder_val_acc for Lee2019
]


def load_corpus():
    rows = []
    for ds, n_ch, ope_rel, loso_rel in DATASETS:
        ope = json.loads((ROOT / "experiments" / ope_rel).read_text())
        if loso_rel:
            loso = json.loads((ROOT / "experiments" / loso_rel).read_text())
            decod = {r["held_out"]: r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
                      for r in loso["per_subject"]}
        else:
            # For Lee2019, use encoder_val_acc as decodability proxy
            decod = {r["held_out"]: r["encoder_val_acc"]
                      for r in ope["per_subject"]}
        for r in ope["per_subject"]:
            sid = r["held_out"]
            if sid not in decod:
                continue
            rows.append({
                "dataset": ds, "sid": sid, "n_channels": n_ch,
                "log2_channels": float(np.log2(n_ch)),
                "decodability": float(decod[sid]),
                "ope_r": float(r["pearson_r"]),
            })
    return rows


def ols_fit(X, y):
    """Plain OLS with intercept. X: (N, k), y: (N,). Returns (beta, residuals, r2)."""
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, y_hat, r2


def loocv_rmse(X, y):
    n = X.shape[0]
    sse = 0.0
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        beta, _, _ = ols_fit(X[mask], y[mask])
        X1_i = np.concatenate(([1.0], X[i]))
        preds[i] = float(X1_i @ beta)
        sse += (preds[i] - y[i]) ** 2
    return float(np.sqrt(sse / n)), preds


def lodo_rmse(rows, X_cols):
    """Leave-one-dataset-out generalisation."""
    datasets = sorted({r["dataset"] for r in rows})
    results = {}
    for held_ds in datasets:
        train = [r for r in rows if r["dataset"] != held_ds]
        test = [r for r in rows if r["dataset"] == held_ds]
        X_tr = np.array([[r[c] for c in X_cols] for r in train])
        y_tr = np.array([r["ope_r"] for r in train])
        X_te = np.array([[r[c] for c in X_cols] for r in test])
        y_te = np.array([r["ope_r"] for r in test])
        beta, _, _ = ols_fit(X_tr, y_tr)
        y_hat = np.column_stack([np.ones(X_te.shape[0]), X_te]) @ beta
        rmse = float(np.sqrt(np.mean((y_hat - y_te) ** 2)))
        if len(y_te) >= 2 and np.std(y_te) > 0 and np.std(y_hat) > 0:
            r_pred, p_pred = stats.pearsonr(y_te, y_hat)
        else:
            r_pred, p_pred = float("nan"), float("nan")
        results[held_ds] = {"rmse": rmse, "n_test": len(y_te),
                              "pred_actual_r": float(r_pred),
                              "pred_actual_p": float(p_pred),
                              "beta": beta.tolist()}
    return results


def main():
    out_dir = ROOT / "experiments" / "unified_ope_predictor"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus()
    log.info("Loaded %d (subject, dataset) rows", len(rows))
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        log.info("  %-10s: n=%d, mean OPE r=%+.3f, mean decod=%.3f",
                  ds, len(sub),
                  np.mean([r["ope_r"] for r in sub]),
                  np.mean([r["decodability"] for r in sub]))

    # Univariate fits
    y = np.array([r["ope_r"] for r in rows])
    log.info("\n=== Univariate predictors ===")
    for col in ["log2_channels", "decodability", "n_channels"]:
        x = np.array([r[col] for r in rows])
        if np.std(x) > 0:
            r_p, p_p = stats.pearsonr(x, y)
            sr, p_sr = stats.spearmanr(x, y)
            beta, _, r2 = ols_fit(x[:, None], y)
            log.info("  %-15s: Pearson r=%+.3f p=%.4f, Spearman ρ=%+.3f, OLS R²=%.3f, β=%+.4f",
                      col, r_p, p_p, sr, r2, float(beta[1]))

    # Multivariate fits
    log.info("\n=== Multivariate OLS ===")
    for cols in [["log2_channels", "decodability"],
                  ["log2_channels"],
                  ["decodability"]]:
        X = np.array([[r[c] for c in cols] for r in rows])
        beta, y_hat, r2 = ols_fit(X, y)
        loocv, preds = loocv_rmse(X, y)
        log.info("  X=%s: β=%s, in-sample R²=%.3f, LOOCV RMSE=%.3f",
                  cols, [f"{b:+.4f}" for b in beta], r2, loocv)

    # Leave-one-dataset-out: does model trained on 2 datasets predict the 3rd?
    log.info("\n=== Leave-one-dataset-out ===")
    lodo = lodo_rmse(rows, ["log2_channels", "decodability"])
    for ds, res in lodo.items():
        log.info("  Train on others, test on %-8s: RMSE=%.3f (n=%d), r(pred, actual)=%+.3f p=%.3f",
                  ds, res["rmse"], res["n_test"],
                  res["pred_actual_r"], res["pred_actual_p"])

    # Final full-corpus fit
    X_full = np.array([[r["log2_channels"], r["decodability"]] for r in rows])
    beta_full, _, r2_full = ols_fit(X_full, y)
    log.info("\n=== Final deployable formula ===")
    log.info("  Predicted OPE r ≈ %+.4f + %+.4f · log2(channels) + %+.4f · decodability",
              beta_full[0], beta_full[1], beta_full[2])
    log.info("  In-sample R² = %.3f (N=%d)", r2_full, len(rows))

    # Worked examples
    log.info("\n=== Worked examples ===")
    for n_ch, decod, label in [(3, 0.65, "bci2b avg subject"),
                                  (22, 0.55, "bci2a avg subject"),
                                  (62, 0.75, "Lee2019 avg subject"),
                                  (128, 0.70, "hypothetical 128-ch high decodability")]:
        pred = beta_full[0] + beta_full[1] * np.log2(n_ch) + beta_full[2] * decod
        log.info("  %d ch, decod=%.2f (%s): predicted OPE r = %+.3f",
                  n_ch, decod, label, pred)

    out = {
        "n_rows": len(rows),
        "rows": rows,
        "univariate": {
            "log2_channels": {"pearson_r": float(stats.pearsonr(
                np.array([r["log2_channels"] for r in rows]), y)[0])},
            "decodability": {"pearson_r": float(stats.pearsonr(
                np.array([r["decodability"] for r in rows]), y)[0])},
        },
        "full_corpus_beta": beta_full.tolist(),
        "full_corpus_r2": float(r2_full),
        "lodo": lodo,
        "formula_text": (f"OPE_r ≈ {beta_full[0]:+.4f} "
                          f"+ {beta_full[1]:+.4f}·log2(channels) "
                          f"+ {beta_full[2]:+.4f}·decodability"),
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    log.info("\nWrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
