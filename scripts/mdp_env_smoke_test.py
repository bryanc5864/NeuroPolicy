# MIT License, 2026
"""Smoke test : BCI-as-MDP env + episode builder + behavior policies.

End-to-end pipeline check on bci2b sub 4:
  1. Preprocess subject -> TrialBatch.
  2. Build EEGNet encoder + SubjectEmbeddingFactory.
  3. Encode all windows -> TrialEpisodes.
  4. Construct BCIEnv.
  5. Roll out FixedWindowPolicy (μ₁) and SPRTStochasticPolicy (μ₂) on 20
     episodes; report basic statistics.
"""
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data.preprocess import preprocess_subject
from src.models.encoder import build_encoder
from src.training.behavior_policies import (
    FixedWindowPolicy,
    SPRTStochasticPolicy,
    fit_running_mean_classifier,
    rollout_episode,
)
from src.training.bci_env import BCIEnv
from src.training.episode_builder import (
    SubjectEmbeddingFactory,
    build_episodes_for_trial_batch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m3_smoke")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    tb = preprocess_subject("bci2b", subject_id=4)
    log.info("Loaded bci2b sub 4: X=%s y=%s n_classes=%d sfreq=%.1f",
             tb.X.shape, tb.y.shape, len(tb.class_labels), tb.sfreq)

    # 1) Build EEGNet encoder for this dataset's geometry
    enc = build_encoder("eegnet", n_channels=tb.n_channels,
                        n_samples=int(round(tb.sfreq * 1.0)),  # 1.0s window
                        sfreq=tb.sfreq).to(device)
    enc.eval()
    log.info("Encoder %s; embed_dim=%d  params=%d",
             enc.spec.name, enc.spec.embed_dim,
             sum(p.numel() for p in enc.parameters()))

    # 2) Subject-embedding factory
    subj_factory = SubjectEmbeddingFactory(
        encoder_embed_dim=enc.spec.embed_dim, subj_dim=16, seed=0,
    )

    # 3) Build episodes (excludes 10% calibration trials)
    episodes, all_feats, cal_idx = build_episodes_for_trial_batch(
        tb, encoder=enc, device=device, subj_factory=subj_factory,
    )
    log.info("Built %d episodes; %d cal trials reserved; embed_dim=%d  T_per_trial≈%d",
             len(episodes), len(cal_idx), episodes[0].embed_dim, episodes[0].T)
    log.info("Class distribution in episodes: %s",
             dict(Counter([e.label for e in episodes])))

    # 4) Construct env
    env = BCIEnv(
        n_classes=len(tb.class_labels),
        embed_dim=episodes[0].embed_dim,
        subj_dim=episodes[0].subj_dim,
        max_T=episodes[0].T + 4,
    )
    log.info("Env action_space=%s observation_space=%s state_dim=%d",
             env.action_space, env.observation_space, env.state_dim)
    log.info("Action ids: DEFER=%d, RECAL=%d, ABSTAIN=%d",
             env.A_DEFER, env.A_RECAL, env.A_ABSTAIN)

    # 5) Fit logging classifier on the first 80% of episodes (the rest we use for rollouts)
    n_train = int(len(episodes) * 0.8)
    train_episodes = episodes[:n_train]
    rollout_eps = episodes[n_train:n_train + 20]
    ctx = fit_running_mean_classifier(train_episodes, n_classes=len(tb.class_labels))
    train_acc = float(ctx.classifier.score(
        np.stack([e.window_features.mean(0) for e in train_episodes]),
        np.array([e.label for e in train_episodes]),
    ))
    log.info("Logging-policy classifier train acc on running-mean: %.3f", train_acc)

    # 6) μ₁: FixedWindowPolicy with T_fix=8 strides ≈ 2 s
    mu1 = FixedWindowPolicy(ctx=ctx, T_fix=8, env=env)
    rolls = [rollout_episode(env, ep, mu1, seed=i) for i, ep in enumerate(rollout_eps)]
    out1 = Counter(t.info["outcome"] for r in rolls for t in r if "outcome" in t.info)
    log.info("μ₁ outcomes (%d eps, %d steps): %s",
             len(rolls), sum(len(r) for r in rolls), dict(out1))

    # 7) μ₂: SPRTStochasticPolicy
    mu2 = SPRTStochasticPolicy(ctx=ctx, env=env, evidence_threshold=0.55, eps_explore=0.10,
                               rng=np.random.default_rng(0))
    rolls2 = [rollout_episode(env, ep, mu2, seed=100 + i) for i, ep in enumerate(rollout_eps)]
    out2 = Counter(t.info["outcome"] for r in rolls2 for t in r if "outcome" in t.info)
    log.info("μ₂ outcomes (%d eps, %d steps): %s",
             len(rolls2), sum(len(r) for r in rolls2), dict(out2))

    # 8) Report rewards
    def total_returns(rolls):
        return np.array([sum(t.reward for t in r) for r in rolls])
    R1, R2 = total_returns(rolls), total_returns(rolls2)
    log.info("μ₁ mean episode return: %.3f ± %.3f", R1.mean(), R1.std())
    log.info("μ₂ mean episode return: %.3f ± %.3f", R2.mean(), R2.std())

    # 9) Verify a single env reset/step/terminate cycle works cleanly
    obs, info = env.reset(options={"episode": rollout_eps[0]}, seed=0)
    assert obs.shape == (env.state_dim,)
    assert obs.dtype == np.float32
    next_obs, r, term, trunc, info = env.step(int(env.A_DEFER))
    assert obs.shape == next_obs.shape
    log.info("Env cycle OK. Smoke complete.")


if __name__ == "__main__":
    main()
