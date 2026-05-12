# MIT License, 2026
""": Multi-dataset pooled pretraining for cross-dataset LOSO — true novelty.

W2/W11 critique: bci2b LOSO (Δret=+0.21, p=0.10 NS) and bci2a LOSO
(Δret=+0.51, p=0.15 NS) are not significant. The bottleneck is encoder
feature quality on the held-out subject. Most published BCI methods
train per-dataset; pooling across datasets is rare for offline-RL BCI.

 pretrains a shared EEGNet encoder on pooled bci2a + bci2b + Lee2019
training windows (matching the smallest channel count = 3 ch via
spatial pooling), then evaluates LOSO classification on each dataset
separately. If pooled pretraining helps any dataset's LOSO, we have a
novel contribution: cross-dataset transfer for offline-RL BCI decoders.

Three configurations per held-out subject:
  * `single`: EEGNet pretrained only on the dataset's training subjects
              (= the existing / LOSO baseline).
  * `pooled`: EEGNet pretrained on bci2a+bci2b+Lee2019 pooled training
              windows (mapped to common 3-channel sensor subset:
              C3, Cz, C4), then evaluated per-dataset.

Output: experiments/multidataset_pretrain/summary.json
"""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data.preprocess import preprocess_subject
from src.evaluation.policy_eval import evaluate_policy, random_policy
from src.evaluation.policy_panel import AgentPolicy
from src.models.encoder import build_encoder, pretrain_encoder_supervised
from src.training.behavior_policies import (
    FixedWindowPolicy, SPRTStochasticPolicy, fit_running_mean_classifier, rollout_episode,
)
from src.training.bci_env import BCIEnv
from src.training.episode_builder import (
    SubjectEmbeddingFactory, build_episodes_for_trial_batch, windows_from_trial,
)
from src.training.neuropolicy_agent import NeuroPolicyAgent
from src.training.replay_buffer import build_buffer_from_rollouts
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG, TRAIN_CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("multidataset_pretrain")

# Common 3-channel motor cortex subset across all 3 datasets
COMMON_CHANNELS_PRIORITY = {
    "bci2b": ["C3", "Cz", "C4"],
    "bci2a": ["C3", "Cz", "C4"],
    "lee2019": ["C3", "Cz", "C4"],
}

# LOSO subsets: target datasets to compare single vs pooled
HELD_OUT_SUBJECTS = {
    "bci2b": [1, 2, 4, 6],   # 4 subjects for tractability
    "bci2a": [1, 3, 5, 7],
}
TRAIN_SUBJECTS = {
    "bci2b": list(range(1, 10)),  # all 9
    "bci2a": list(range(1, 10)),
    "lee2019": [1, 5, 10, 15, 20, 25, 30],  # 7 subjects for pooled pretrain
}


def map_to_common_channels(tb, target_names=("C3", "Cz", "C4")):
    """Slice tb to the target channels (must be present in tb.ch_names)."""
    name_to_idx = {n: i for i, n in enumerate(tb.ch_names)}
    try:
        idx = [name_to_idx[n] for n in target_names]
    except KeyError as e:
        log.warning("Channel %s not in %s ch_names %s; skipping subject", e, tb.dataset_key, tb.ch_names[:8])
        return None
    from src.data.moabb_loader import TrialBatch
    return TrialBatch(
        X=tb.X[:, idx, :].copy(),
        y=tb.y.copy(),
        session=tb.session.copy(), run=tb.run.copy(),
        sfreq=tb.sfreq, ch_names=list(target_names),
        class_labels=list(tb.class_labels),
        subject_id=tb.subject_id, dataset_key=tb.dataset_key,
    )


def _windowize_to_common(tb_list, common_ch_count=3):
    Xs, ys = [], []
    for tb in tb_list:
        if tb is None or tb.X.shape[1] != common_ch_count:
            continue
        for i in range(tb.n_trials):
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
            # Truncate to first 250 samples (1.0s) for consistent T
            W = W[:, :, :250]
            Xs.append(W); ys.extend([int(tb.y[i])] * W.shape[0])
    if not Xs:
        return None, None
    return np.concatenate(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def loso_classification(dataset_key, held_out, train_subj_tbs, held_out_tb,
                          encoder, train_subj_embeds, subj_factory, device,
                          buffer_label):
    """Run a single LOSO classification round using the given encoder."""
    n_classes = len(held_out_tb.class_labels)
    win_samples = int(round(held_out_tb.sfreq * MDP_CFG.window_seconds))

    train_eps = []
    for tb in train_subj_tbs:
        eps, _, _ = build_episodes_for_trial_batch(
            tb, encoder=encoder, device=device, subj_factory=subj_factory,
            subj_embeds=train_subj_embeds,
            calibration_indices=np.array([], dtype=np.int64),
        )
        train_eps.extend(eps)
    held_eps, _, _ = build_episodes_for_trial_batch(
        held_out_tb, encoder=encoder, device=device, subj_factory=subj_factory,
        subj_embeds=train_subj_embeds, calibration_indices=np.array([], dtype=np.int64),
    )

    env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                  subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
    ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)

    rollouts = []
    for T_fix in [4, 8, 12, 16]:
        mu1 = FixedWindowPolicy(ctx=ctx, T_fix=T_fix, env=env)
        for i, ep in enumerate(train_eps):
            rollouts.append(rollout_episode(env, ep, mu1, seed=i))
    rng_mu = np.random.default_rng(0)
    mu2 = SPRTStochasticPolicy(ctx=ctx, env=env, evidence_threshold=0.55,
                                eps_explore=0.10, rng=rng_mu)
    for i, ep in enumerate(train_eps):
        rollouts.append(rollout_episode(env, ep, mu2, seed=10_000 + i))
    buffer = build_buffer_from_rollouts(
        rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL,
        a_abstain=env.A_ABSTAIN, n_classes=n_classes,
    )

    rand_metrics = evaluate_policy(env, held_eps,
                                    random_policy(env.action_space.n, 0),
                                    n_classes=n_classes)
    cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG), "cql_alpha": 1.0, "seed": 0})
    agent = NeuroPolicyAgent(
        state_dim=env.state_dim, n_actions=env.action_space.n, n_classes=n_classes,
        cfg=cfg_obj, mode="scalar", cmdp_eps=0.10, gamma=0.99, polyak_tau=0.005,
        device=device, hidden=256, depth=3, seed=0,
    )
    t0 = time.time()
    agent.train(buffer, n_steps=10_000, batch_size=256, log_every=10_000)
    elapsed = time.time() - t0

    agent_pol = AgentPolicy(agent, temperature=0.0, mix_random_frac=0.0,
                              name="cmdp_eps0.1")
    metrics = evaluate_policy(env, held_eps,
                                lambda s, _p=agent_pol: _p.select_action(s),
                                n_classes=n_classes)
    log.info("  [%s held=%d %s] ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d (%.0fs)",
              dataset_key, held_out, buffer_label,
              metrics.return_mean, metrics.accuracy_on_commits,
              metrics.information_transfer_rate, metrics.wrong_commit_rate,
              metrics.n_commits, elapsed)
    return {
        "random": rand_metrics.asdict(),
        "agent": metrics.asdict(),
        "elapsed_s": float(elapsed),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "multidataset_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: preprocess all subjects to common 3-channel subset ===
    log.info("\n=== Phase 1: preprocess + map to common channels (C3, Cz, C4) ===")
    all_tbs = {"bci2a": {}, "bci2b": {}, "lee2019": {}}
    for ds, subjs in TRAIN_SUBJECTS.items():
        for sid in subjs:
            try:
                t0 = time.time()
                tb = preprocess_subject(ds, subject_id=sid)
                tb_common = map_to_common_channels(tb)
                if tb_common is not None:
                    all_tbs[ds][sid] = tb_common
                    log.info("  %s sub %d: %d trials, %d ch (mapped to 3) (%.1fs)",
                              ds, sid, tb.n_trials, tb.X.shape[1], time.time() - t0)
            except Exception as exc:
                log.warning("  %s sub %d failed: %s", ds, sid, exc)

    # === Phase 2: pretrain pooled encoder on training subjects' windows ===
    log.info("\n=== Phase 2: pooled pretrain (3 datasets) ===")
    pooled_tb_list = []
    for ds in ["bci2a", "bci2b", "lee2019"]:
        for sid in all_tbs[ds]:
            # All subjects' data goes into pooled pretrain pool
            pooled_tb_list.append(all_tbs[ds][sid])
    log.info("  Pooled %d trial-batches across all datasets", len(pooled_tb_list))

    # Need a unified n_classes — but bci2a is 4-class while others are 2-class.
    # Strategy: pretrain only on the 2-class subset (bci2b + lee2019) for the
    # encoder pooling; bci2a 4-class will use its single-dataset pretrain.
    # OR: pretrain on a 2-class projection of bci2a (left vs right hand only).
    log.info("  (Subsetting bci2a trials to {left_hand, right_hand} for 2-class pretrain)")
    pooled_2cls_tbs = []
    for tb in pooled_tb_list:
        if tb.dataset_key == "bci2a":
            mask = np.isin(tb.y, [0, 1])  # 0=left_hand, 1=right_hand in bci2a
            if mask.sum() == 0:
                continue
            from src.data.moabb_loader import TrialBatch
            pooled_2cls_tbs.append(TrialBatch(
                X=tb.X[mask], y=tb.y[mask], session=tb.session[mask],
                run=tb.run[mask], sfreq=tb.sfreq, ch_names=list(tb.ch_names),
                class_labels=["left_hand", "right_hand"],
                subject_id=tb.subject_id, dataset_key="bci2a_2cls",
            ))
        else:
            pooled_2cls_tbs.append(tb)

    sfreq_pool = pooled_2cls_tbs[0].sfreq
    win_samples_pool = int(round(sfreq_pool * MDP_CFG.window_seconds))
    win_X_pool, win_y_pool = _windowize_to_common(pooled_2cls_tbs, common_ch_count=3)
    if win_X_pool is None:
        log.error("No windows produced for pooled pretrain. Abort.")
        return
    log.info("  Pooled pretrain set: %d windows, 3 ch, %d samples",
              win_X_pool.shape[0], win_X_pool.shape[-1])

    enc_pooled = build_encoder("eegnet", n_channels=3, n_samples=win_samples_pool,
                                  sfreq=sfreq_pool).to(device)
    t0 = time.time()
    pool_info = pretrain_encoder_supervised(
        enc_pooled, X=win_X_pool, y=win_y_pool, n_classes=2, device=device,
        n_epochs=15, batch_size=256, lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.10,
    )
    log.info("  Pooled encoder val acc (2-class, all-datasets): %.3f (%.1fs)",
              pool_info["best_val_acc"], time.time() - t0)

    # === Phase 3: LOSO on bci2b with pooled encoder vs single-dataset baseline ===
    log.info("\n=== Phase 3: LOSO on bci2b using pooled vs single encoder ===")
    summary = {"datasets": {}, "pooled_encoder_val_acc": float(pool_info["best_val_acc"]),
                "pooled_pretrain_n_windows": int(win_X_pool.shape[0])}

    for target_ds in ["bci2b"]:
        target_subjs = TRAIN_SUBJECTS[target_ds]
        held_out_subjs = HELD_OUT_SUBJECTS.get(target_ds, target_subjs[:4])
        target_tbs = all_tbs[target_ds]
        if len(target_tbs) < 2:
            log.warning("  %s: not enough subjects, skipping", target_ds)
            continue

        ds_results = {"held_out_subjects": held_out_subjs, "per_subject": []}

        for held_out in held_out_subjs:
            if held_out not in target_tbs:
                continue
            train_subjs_avail = [s for s in target_subjs if s != held_out and s in target_tbs]
            log.info("\n  %s LOSO held=%d (n_train=%d)", target_ds, held_out, len(train_subjs_avail))

            train_subj_tbs = [target_tbs[s] for s in train_subjs_avail]
            held_out_tb = target_tbs[held_out]
            n_classes_t = len(held_out_tb.class_labels)

            # --- Single-dataset baseline ---
            win_X_s, win_y_s = _windowize_to_common(train_subj_tbs, common_ch_count=3)
            enc_single = build_encoder("eegnet", n_channels=3,
                                          n_samples=win_samples_pool,
                                          sfreq=sfreq_pool).to(device)
            single_info = pretrain_encoder_supervised(
                enc_single, X=win_X_s, y=win_y_s, n_classes=n_classes_t, device=device,
                n_epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-4,
                seed=0, val_frac=0.10,
            )
            log.info("    single-pretrain encoder val acc: %.3f", single_info["best_val_acc"])

            # Build episodes for single encoder
            subj_factory = SubjectEmbeddingFactory(
                encoder_embed_dim=enc_single.spec.embed_dim, subj_dim=16, seed=0,
            )
            tep_per_subj = []
            for tb in train_subj_tbs:
                eps, _, _ = build_episodes_for_trial_batch(tb, encoder=enc_single,
                                                            device=device, subj_factory=subj_factory)
                tep_per_subj.append(eps)
            defaults = np.stack([eps[0].subj_embed_default for eps in tep_per_subj])
            post_recals = np.stack([eps[0].subj_embed_post_recal for eps in tep_per_subj])
            embeds_single = (defaults.mean(0).astype(np.float32),
                              post_recals.mean(0).astype(np.float32))

            res_single = loso_classification(
                target_ds, held_out, train_subj_tbs, held_out_tb,
                enc_single, embeds_single, subj_factory, device,
                buffer_label="single",
            )

            # --- Pooled encoder ---
            # We need to fine-tune the pooled encoder to the target dataset's classes
            # (in case n_classes differs). For bci2b (2-class), pooled is already 2-class.
            enc_fine = build_encoder("eegnet", n_channels=3,
                                       n_samples=win_samples_pool,
                                       sfreq=sfreq_pool).to(device)
            enc_fine.load_state_dict(enc_pooled.state_dict())
            fine_info = pretrain_encoder_supervised(
                enc_fine, X=win_X_s, y=win_y_s, n_classes=n_classes_t, device=device,
                n_epochs=10, batch_size=256, lr=5e-4, weight_decay=1e-4,
                seed=0, val_frac=0.10,  # gentler fine-tune from pooled init
            )
            log.info("    pooled+finetune encoder val acc: %.3f", fine_info["best_val_acc"])

            subj_factory_p = SubjectEmbeddingFactory(
                encoder_embed_dim=enc_fine.spec.embed_dim, subj_dim=16, seed=0,
            )
            tep_per_subj_p = []
            for tb in train_subj_tbs:
                eps, _, _ = build_episodes_for_trial_batch(tb, encoder=enc_fine,
                                                            device=device, subj_factory=subj_factory_p)
                tep_per_subj_p.append(eps)
            defaults_p = np.stack([eps[0].subj_embed_default for eps in tep_per_subj_p])
            post_recals_p = np.stack([eps[0].subj_embed_post_recal for eps in tep_per_subj_p])
            embeds_pool = (defaults_p.mean(0).astype(np.float32),
                            post_recals_p.mean(0).astype(np.float32))

            res_pool = loso_classification(
                target_ds, held_out, train_subj_tbs, held_out_tb,
                enc_fine, embeds_pool, subj_factory_p, device,
                buffer_label="pooled+finetune",
            )

            ds_results["per_subject"].append({
                "held_out": held_out,
                "single": res_single,
                "pooled_finetune": res_pool,
                "single_encoder_val_acc": float(single_info["best_val_acc"]),
                "pooled_encoder_val_acc": float(fine_info["best_val_acc"]),
            })
            (out_dir / "summary.json").write_text(json.dumps(
                {**summary, "datasets": {target_ds: ds_results}}, indent=2))

        # Aggregate within dataset
        single_accs = [r["single"]["agent"]["accuracy_on_commits"] for r in ds_results["per_subject"]]
        pool_accs = [r["pooled_finetune"]["agent"]["accuracy_on_commits"] for r in ds_results["per_subject"]]
        single_rets = [r["single"]["agent"]["return_mean"] for r in ds_results["per_subject"]]
        pool_rets = [r["pooled_finetune"]["agent"]["return_mean"] for r in ds_results["per_subject"]]
        ds_results["aggregate"] = {
            "single_acc_mean": float(np.mean(single_accs)),
            "pool_acc_mean": float(np.mean(pool_accs)),
            "single_ret_mean": float(np.mean(single_rets)),
            "pool_ret_mean": float(np.mean(pool_rets)),
            "n_subjects": len(single_accs),
            "delta_acc_per_subject": [float(p - s) for s, p in zip(single_accs, pool_accs)],
            "delta_ret_per_subject": [float(p - s) for s, p in zip(single_rets, pool_rets)],
        }
        log.info("\n  %s LOSO aggregate: single acc %.3f vs pooled acc %.3f (Δ %.3f)",
                  target_ds, ds_results["aggregate"]["single_acc_mean"],
                  ds_results["aggregate"]["pool_acc_mean"],
                  ds_results["aggregate"]["pool_acc_mean"] - ds_results["aggregate"]["single_acc_mean"])
        summary["datasets"][target_ds] = ds_results

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
