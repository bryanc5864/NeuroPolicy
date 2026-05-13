# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""Encoder abstraction for windowed EEG -> embedding.

Provides:
    BaseEncoder        : Abstract Module returning (B, embed_dim) per window.
    EEGNetEncoder      : From-scratch EEGNet-style backbone (Lawhern 2018).
    LaBraMEncoderStub  : Placeholder that loads LaBraM checkpoint *if* the
                         official repo is available on PYTHONPATH; otherwise
                         raises a clear error. Full integration is -stretch.

The MDP (src/training/bci_env.py) and the RL agent (src/training/neuropolicy_agent.py)
talk to BaseEncoder, so swapping encoders is a one-line change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EncoderSpec:
    """Information the rest of the pipeline reads from an encoder."""
    embed_dim: int            # output embedding dim per window
    n_channels: int           # expected input channels
    n_samples: int            # expected window samples
    sfreq: float              # expected sampling rate (Hz)
    name: str                 # short identifier for logging


class BaseEncoder(nn.Module, ABC):
    """Abstract: takes (B, C, T) -> (B, embed_dim)."""

    @property
    @abstractmethod
    def spec(self) -> EncoderSpec: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def freeze(self) -> "BaseEncoder":
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self


# ---------------------------------------------------------------------------
# EEGNet-style encoder
# ---------------------------------------------------------------------------
class EEGNetEncoder(BaseEncoder):
    """Compact EEGNet-style encoder per Lawhern et al. 2018.

    Architecture:
        Conv2d(1, F1, (1, kt))           # temporal filter
        DepthwiseConv2d(F1, D*F1, (C, 1)) # spatial filter
        ELU + AvgPool(1, 4)
        SeparableConv2d(D*F1, F2, (1, kt2))
        ELU + AvgPool(1, 8)
        Flatten -> embed_dim

    For C=22 (bci2a) or C=3 (bci2b) or C=62 (lee2019), the model accepts
    any channel count via construction; embed_dim depends on T.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sfreq: float,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kt: int = 64,    # temporal kernel length (~ sfreq/4 for 250 Hz)
        kt2: int = 16,   # second-stage kernel
        dropout: float = 0.25,
    ):
        super().__init__()
        F2 = F2 or D * F1
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sfreq = sfreq

        # Block 1: temporal + spatial
        self.temporal = nn.Conv2d(1, F1, (1, kt), padding=(0, kt // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv
        self.sep_dw = nn.Conv2d(D * F1, D * F1, (1, kt2), padding=(0, kt2 // 2),
                                groups=D * F1, bias=False)
        self.sep_pw = nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Compute embed_dim from a forward dry-run
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            z = self._forward_features(dummy)
            self._embed_dim = z.shape[1]

    @property
    def spec(self) -> EncoderSpec:
        return EncoderSpec(
            embed_dim=self._embed_dim,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            sfreq=self.sfreq,
            name="EEGNet",
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.temporal(x)
        x = self.bn1(x)
        x = self.spatial(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_dw(x)
        x = self.sep_pw(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, C, T) or (B, 1, C, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self._forward_features(x)


# ---------------------------------------------------------------------------
# LaBraM encoder (foundation-model integration via braindecode 0.9 + HF Hub)
# ---------------------------------------------------------------------------
class LaBraMEncoder(BaseEncoder):
    """Pretrained LaBraM (Jiang et al. 2024) wrapped as our BaseEncoder.

    The pretrained checkpoint hosted on HuggingFace
    (`braindecode/Labram-Braindecode/braindecode_labram_base.pt`)
    was trained with a 64-entry channel mapping (the first 64 entries of
    the LABRAM_CHANNEL_ORDER table — FP1 through PZ) and 8 patches of 200
    samples (= 8 s @ 200 Hz). braindecode 0.9's Labram class defaults to
    a 128-entry mapping and arbitrary `n_times`; we surgically copy the
    first 65 position-embedding rows (CLS + first 64 channels) and the
    first 2 temporal-embedding rows (CLS + first patch) from the
    pretrained checkpoint into our model. Channels mapping into indices
    >= 64 (e.g. P2, POz on bci2a) get init'd position embeddings rather
    than pretrained ones — acceptable for a small minority of channels.

    Output: (B, embed_dim=200) per window — the CLS token after the
    transformer blocks.
    """
    LABRAM_PRETRAINED_URL = (
        "https://huggingface.co/braindecode/Labram-Braindecode/"
        "resolve/main/braindecode_labram_base.pt"
    )
    PATCH_SAMPLES = 200          # one patch = 1 s @ 200 Hz

    def __init__(self, n_channels: int, n_samples: int, sfreq: float,
                 ch_names: list[str], load_pretrained: bool = True):
        super().__init__()
        from braindecode.models import Labram

        self.n_channels = n_channels
        self.n_samples = n_samples            # input n_samples (at our sfreq)
        self.sfreq = sfreq
        self.ch_names = list(ch_names)
        if n_channels != len(ch_names):
            raise ValueError(
                f"n_channels={n_channels} != len(ch_names)={len(ch_names)}"
            )
        # Compute the n_times we feed into LaBraM (at 200 Hz). Must be a
        # multiple of PATCH_SAMPLES (=200). Our input may need on-the-fly
        # resampling from sfreq to 200 Hz.
        seconds = n_samples / sfreq
        labram_n_times = int(round(seconds * 200.0))
        # Round up to nearest multiple of PATCH_SAMPLES
        rem = labram_n_times % self.PATCH_SAMPLES
        if rem != 0:
            labram_n_times += self.PATCH_SAMPLES - rem
        self.labram_n_times = labram_n_times

        chs_info = [
            {"ch_name": n, "kind": 2, "logno": i + 1, "scanno": i + 1, "cal": 1.0,
             "range": 1.0, "unit_mul": 0, "unit": 107, "coord_frame": 4,
             "loc": [0] * 12, "coil_type": 1, "ch_id": 0}
            for i, n in enumerate(ch_names)
        ]
        # n_outputs irrelevant — we only use forward_features (CLS token).
        self.labram = Labram(
            n_times=labram_n_times, n_chans=n_channels, n_outputs=2,
            on_unknown_chs="ignore", chs_info=chs_info, sfreq=200.0,
        )
        if load_pretrained:
            self._load_pretrained()

        self._spec = EncoderSpec(
            embed_dim=200,
            n_channels=n_channels, n_samples=n_samples, sfreq=sfreq,
            name=f"LaBraM-base{'-pretrained' if load_pretrained else '-init'}",
        )

    def _load_pretrained(self) -> None:
        state = torch.hub.load_state_dict_from_url(
            self.LABRAM_PRETRAINED_URL, progress=False
        )
        own = self.labram.state_dict()
        # Direct-shape-match keys
        compatible = {k: v for k, v in state.items()
                      if k in own and own[k].shape == v.shape}
        # Surgical shape-mismatch keys
        # position_embedding: ckpt (1, 65, 200), our (1, 129, 200)
        if "position_embedding" in state and "position_embedding" in own:
            pe_ckpt = state["position_embedding"]
            pe_own = own["position_embedding"].clone()
            n_copy = min(pe_ckpt.shape[1], pe_own.shape[1])
            pe_own[:, :n_copy, :] = pe_ckpt[:, :n_copy, :]
            compatible["position_embedding"] = pe_own
        # temporal_embedding: ckpt (1, 9, 200), our (1, n_patches+1, 200)
        if "temporal_embedding" in state and "temporal_embedding" in own:
            te_ckpt = state["temporal_embedding"]
            te_own = own["temporal_embedding"].clone()
            n_copy = min(te_ckpt.shape[1], te_own.shape[1])
            te_own[:, :n_copy, :] = te_ckpt[:, :n_copy, :]
            compatible["temporal_embedding"] = te_own
        missing, unexpected = self.labram.load_state_dict(compatible, strict=False)
        # Sanity log via attrs (caller can inspect)
        self._n_loaded = len(compatible)
        self._n_total_ckpt = len(state)
        self._n_missing = len(missing)
        self._n_unexpected = len(unexpected)

    @property
    def spec(self) -> EncoderSpec:
        return self._spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) at self.sfreq. Resample on-the-fly to 200 Hz if needed.
        if x.shape[-1] != self.labram_n_times:
            # Linear interpolation along the time axis to labram_n_times samples.
            x = F.interpolate(x, size=self.labram_n_times,
                               mode="linear", align_corners=False)
        out = self.labram(x, return_features=True)
        return out["cls_token"]

    def unfreeze(self) -> "LaBraMEncoder":
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True
        self.train()
        return self


# Backwards-compat alias for existing import paths.
LaBraMEncoderStub = LaBraMEncoder


# ---------------------------------------------------------------------------
# EEG-Conformer encoder (Song et al. 2023, IEEE TNSRE) — 78.66% published on bci2a 4-class.
# ---------------------------------------------------------------------------
class EEGConformerEncoder(BaseEncoder):
    """EEG-Conformer encoder per Song et al. 2023 (IEEE TNSRE 31:710-719).

    Three modules:
      1. Convolutional patch embedding: temporal conv (kernel (1,25)) +
         spatial conv (kernel (C,1)) + BN + ELU + avg-pool (kernel (1,75),
         stride (1,15)) + dropout + 1x1 projection.
      2. Transformer encoder: 6 self-attention layers, emb=40, heads=10,
         FFN expansion 4x, dropout 0.5.
      3. Classifier head (optional, not part of encoder): two Linear layers.

    Reference: github.com/eeyhsong/EEG-Conformer (file conformer.py).

    For T=1000 samples (4 s @ 250 Hz, the original setting), the post-pool
    token count is 61, flatten dim 2440. For shorter windows the token count
    scales as floor((T - 25 - 75 + 1) / 15) + 1 (≥ 1 required).
    """

    def __init__(self, n_channels: int, n_samples: int, sfreq: float,
                  emb: int = 40, heads: int = 10, depth: int = 6,
                  ffn_expansion: int = 4, dropout: float = 0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sfreq = sfreq
        self._emb = emb

        # 1. Patch embedding
        self.conv_temporal = nn.Conv2d(1, emb, kernel_size=(1, 25), stride=(1, 1))
        self.conv_spatial = nn.Conv2d(emb, emb, kernel_size=(n_channels, 1), stride=(1, 1))
        self.bn_embed = nn.BatchNorm2d(emb)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop_embed = nn.Dropout(dropout)
        self.proj = nn.Conv2d(emb, emb, kernel_size=(1, 1))

        # 2. Transformer (PyTorch built-in: pre-norm, 6 layers, 10 heads, FFN 4x).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads,
            dim_feedforward=emb * ffn_expansion,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Compute embed_dim (flattened) by a dry-run.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            tokens = self._forward_tokens(dummy)
            self._n_tokens = tokens.shape[1]
            self._embed_dim = self._n_tokens * emb

    @property
    def spec(self) -> EncoderSpec:
        return EncoderSpec(
            embed_dim=self._embed_dim,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            sfreq=self.sfreq,
            name=f"EEGConformer(tokens={self._n_tokens},emb={self._emb})",
        )

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T) → tokens (B, n_tokens, emb)
        x = self.conv_temporal(x)        # (B, emb, C, T - 24)
        x = self.conv_spatial(x)         # (B, emb, 1, T - 24)
        x = self.bn_embed(x)
        x = F.elu(x)
        x = self.pool(x)                 # (B, emb, 1, n_tokens)
        x = self.drop_embed(x)
        x = self.proj(x)                 # (B, emb, 1, n_tokens)
        x = x.squeeze(2)                 # (B, emb, n_tokens)
        x = x.permute(0, 2, 1).contiguous()  # (B, n_tokens, emb)
        x = self.transformer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        tokens = self._forward_tokens(x)
        return tokens.flatten(start_dim=1)


# ---------------------------------------------------------------------------
# CTNet encoder (Zhao et al., Scientific Reports 2024, DOI 10.1038/s41598-024-71118-7).
# Reference implementation: braindecode.models.CTNet (1.4.0+).
# Published 4-class subject-specific bci2a accuracy: 82.52%.
# ---------------------------------------------------------------------------
class CTNetEncoder(BaseEncoder):
    """CTNet (Zhao 2024) feature extractor — drops the final classifier head.

    Architecture follows the paper's defaults: CNN (kernel 64, F1=20 temporal
    filters, depth multiplier 2, two avg-pool stages of size 8) + 6-layer
    transformer (embed_dim 40, 4 heads). The forward returns the flattened
    feature vector that the paper's final classifier consumes, i.e.
    (cnn + transformer) residual then flatten.

    Embedding dim depends on n_times: at 1 s @ 250 Hz this is 120 (3 tokens
    of dim 40); at 4 s @ 250 Hz it would be 480.
    """

    def __init__(self, n_channels: int, n_samples: int, sfreq: float):
        super().__init__()
        from braindecode.models import CTNet
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sfreq = sfreq
        self.ctnet = CTNet(n_outputs=2, n_chans=n_channels,
                          n_times=n_samples, sfreq=sfreq)
        # Compute embed_dim with a dry-run.
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_samples)
            emb = self._forward_features(dummy)
            self._embed_dim = int(emb.shape[-1])
            self._n_tokens = self.ctnet.embed_dim and (self._embed_dim // self.ctnet.embed_dim)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        import math
        # CTNet's ensuredim() prepends a feature dim; safe to call.
        x = self.ctnet.ensuredim(x)
        cnn = self.ctnet.cnn(x)
        cnn = cnn * math.sqrt(self.ctnet.embed_dim)
        cnn_p = self.ctnet.position(cnn)
        trans = self.ctnet.trans(cnn_p)
        features = cnn_p + trans
        return self.ctnet.flatten(features)

    @property
    def spec(self) -> EncoderSpec:
        return EncoderSpec(
            embed_dim=self._embed_dim,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            sfreq=self.sfreq,
            name=f"CTNet(tokens={self._n_tokens},emb={self.ctnet.embed_dim})",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self._forward_features(x)


def build_encoder(name: str, n_channels: int, n_samples: int, sfreq: float,
                  ch_names: list[str] | None = None,
                  load_pretrained: bool = True) -> BaseEncoder:
    """Factory for encoder construction by name."""
    name = name.lower()
    if name in {"eegnet", "eegnet8"}:
        return EEGNetEncoder(n_channels=n_channels, n_samples=n_samples, sfreq=sfreq)
    if name in {"conformer", "eeg-conformer", "eegconformer"}:
        return EEGConformerEncoder(n_channels=n_channels, n_samples=n_samples, sfreq=sfreq)
    if name in {"ctnet"}:
        return CTNetEncoder(n_channels=n_channels, n_samples=n_samples, sfreq=sfreq)
    if name in {"ms_bandmamba", "msbandmamba", "ms-bandmamba"}:
        from .ms_bandmamba import MSBandMambaEncoder
        return MSBandMambaEncoder(n_channels=n_channels, n_samples=n_samples, sfreq=sfreq)
    if name in {"labram", "labram-base"}:
        if ch_names is None:
            raise ValueError("LaBraM requires ch_names")
        return LaBraMEncoder(n_channels=n_channels, n_samples=n_samples,
                              sfreq=sfreq, ch_names=ch_names,
                              load_pretrained=load_pretrained)
    raise ValueError(f"Unknown encoder name: {name}")


# ---------------------------------------------------------------------------
# Supervised encoder pretraining (EEGNet stand-in for LaBraM).
# ---------------------------------------------------------------------------
def pretrain_encoder_supervised(
    encoder: BaseEncoder,
    X: "np.ndarray",         # noqa: F821
    y: "np.ndarray",         # noqa: F821
    n_classes: int,
    device: torch.device,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    val_frac: float = 0.15,
) -> dict:
    """Train encoder + linear head on (X, y), then return the trained encoder.

    Uses a held-out fraction of (X, y) as a validation set for early stopping
    on val accuracy. The encoder is left in eval() mode on return.

    NOTE: This trains the encoder on full-trial inputs (typically 4 s @ 250 Hz).
    For the MDP we then encode shorter rolling windows (1.0 s) — the encoder
    architecture is convolutional, so it accepts arbitrary T given the same
    n_channels. In practice though, the encoder's pool sizes were chosen for
    a specific T at construction. We re-init the encoder for the window size
    used in the MDP and pretrain at THAT size to keep things consistent.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_val = int(round(val_frac * X.shape[0]))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]

    head = nn.Linear(encoder.spec.embed_dim, n_classes).to(device)
    encoder.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X.astype("float32"))
    yt = torch.from_numpy(y.astype("int64"))
    Xt_tr = Xt[tr_idx].to(device); yt_tr = yt[tr_idx].to(device)
    Xt_va = Xt[val_idx].to(device); yt_va = yt[val_idx].to(device)

    best_val = -1.0
    best_state = None
    for epoch in range(n_epochs):
        encoder.train()
        head.train()
        p = torch.randperm(Xt_tr.shape[0], device=device)
        for i in range(0, Xt_tr.shape[0], batch_size):
            idx = p[i:i + batch_size]
            logits = head(encoder(Xt_tr[idx]))
            loss = crit(logits, yt_tr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        encoder.eval()
        head.eval()
        with torch.no_grad():
            preds = head(encoder(Xt_va)).argmax(1)
            val_acc = float((preds == yt_va).float().mean())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}

    if best_state is not None:
        encoder.load_state_dict(best_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return {"best_val_acc": best_val, "n_train": int(len(tr_idx)), "n_val": int(len(val_idx))}


# ---------------------------------------------------------------------------
# Published-recipe Conformer pretraining (Song et al. 2023, IEEE TNSRE)
# ---------------------------------------------------------------------------
def pretrain_encoder_conformer_recipe(
    encoder: BaseEncoder,
    X,
    y,
    n_classes: int,
    device: torch.device,
    n_epochs: int = 250,
    batch_size: int = 72,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 0,
    val_frac: float = 0.15,
    sr_ns: int = 8,
    sr_max_shift: int = 25,
) -> dict:
    """Extended Conformer training recipe from Song et al. 2023 (IEEE TNSRE).

    Differences vs ``pretrain_encoder_supervised``:
      * 250 epochs (vs 20-30) — the paper reports convergence around epoch 250.
      * Adam(beta1=0.5, beta2=0.999) — the paper's published betas (default Adam
        uses beta1=0.9, which differs).
      * Shift & Rotate (S&R) augmentation: each training trial is split into
        Ns=8 segments along time, each segment is independently rotated by a
        random shift in [0, sr_max_shift] samples. Implemented via roll-along-time
        per segment.
      * Validation-best-state retention with early-stopping budget.

    See: github.com/eeyhsong/EEG-Conformer; Section IV "Implementation Details".
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_val = int(round(val_frac * X.shape[0]))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]

    head = nn.Linear(encoder.spec.embed_dim, n_classes).to(device)
    encoder.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                            lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X.astype("float32"))
    yt = torch.from_numpy(y.astype("int64"))
    Xt_tr = Xt[tr_idx].to(device); yt_tr = yt[tr_idx].to(device)
    Xt_va = Xt[val_idx].to(device); yt_va = yt[val_idx].to(device)
    T = Xt_tr.shape[-1]
    seg_len = T // sr_ns  # last segment may be slightly longer; we pad to even chunks

    def _shift_and_rotate(batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, C, T). Split into sr_ns segments along T, rotate each by
        # a random shift drawn uniformly in [0, sr_max_shift], concatenate back.
        if seg_len <= 1:
            return batch
        out = batch.clone()
        cuts = [seg_len * i for i in range(sr_ns)] + [T]
        for i in range(sr_ns):
            a, b = cuts[i], cuts[i + 1]
            shift = int(torch.randint(0, sr_max_shift + 1, ()).item())
            if shift > 0 and (b - a) > 1:
                out[:, :, a:b] = torch.roll(batch[:, :, a:b], shifts=shift, dims=-1)
        return out

    best_val = -1.0
    best_state = None
    for epoch in range(n_epochs):
        encoder.train()
        head.train()
        p = torch.randperm(Xt_tr.shape[0], device=device)
        for i in range(0, Xt_tr.shape[0], batch_size):
            idx = p[i:i + batch_size]
            xb = _shift_and_rotate(Xt_tr[idx])
            logits = head(encoder(xb))
            loss = crit(logits, yt_tr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        # Eval every epoch but only save state when improved.
        encoder.eval()
        head.eval()
        with torch.no_grad():
            preds = head(encoder(Xt_va)).argmax(1)
            val_acc = float((preds == yt_va).float().mean())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}

    if best_state is not None:
        encoder.load_state_dict(best_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return {"best_val_acc": best_val, "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)), "n_epochs": n_epochs, "recipe": "conformer_song2023"}
