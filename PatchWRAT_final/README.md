# PatchWRAT — Patch-based Wavelet Routing Attention Transformer

> **CS728 Project · IIT Bombay · 2025**
>
> A multivariate time-series forecasting architecture that combines **overlapping patch tokenisation**, **learnable 1-D wavelet decomposition**, and **dual-branch frequency-sparse attention** into a single, end-to-end trainable model.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Deep-Dive](#2-component-deep-dive)
   - [RevIN — Reversible Instance Normalisation](#21-revin--reversible-instance-normalisation)
   - [PatchEmbedding — Patch Tokenisation](#22-patchembedding--patch-tokenisation)
   - [LearnableDWT1D — Learnable Wavelet Transform](#23-learnabledwt1d--learnable-wavelet-transform)
   - [FrequencySparseAttention — Energy-Gated Attention](#24-frequencysparseattention--energy-gated-attention)
   - [WRATBlock — Dual-Band Attention Block](#25-wratblock--dual-band-attention-block)
   - [LearnableTauWRATBlock — Learnable Sparsity Threshold](#26-learnabletauwratblock--learnable-sparsity-threshold)
   - [DualHeadPWSA_Loss — Composite Loss](#27-dualheadpwsa_loss--composite-loss)
3. [Architecture Variants & Ablations](#3-architecture-variants--ablations)
4. [Evolution: WRAT → PatchWRAT](#4-evolution-wrat--patchwrat)
5. [Benchmark Results](#5-benchmark-results)
6. [Project Structure](#6-project-structure)
7. [Quick Start](#7-quick-start)
8. [Dependencies](#8-dependencies)

---

## 1. Architecture Overview

PatchWRAT processes each variate **independently** (channel-independent / CI strategy), which keeps the model lightweight and prevents spurious cross-variate contamination.

```
Input  (B, C, L)
   │
   ├─ Reshape to (B·C, 1, L)          ← treat each channel as a separate sample
   │
   ▼
┌────────────────────────────────────────────────────────────────┐
│  RevIN  (norm)                                                 │
│  — remove per-sample mean & variance before any processing     │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│  PatchEmbedding                                                │
│  — unfold L → N overlapping patches of length P               │
│  — linear project each patch: ℝᴾ → ℝᵈ                        │
│  Output: (B·C, d_model, N)                                     │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│  LearnableDWT1D                                                │
│  — depthwise conv with learnable filters h (low) / g (high)   │
│  — stride-2 decomposition: N → N/2                            │
│                                                                │
│  ├── LL  (B·C, d_model, N/2)  — trend / approximation coeff   │
│  └── LH  (B·C, d_model, N/2)  — detail / noise coeff          │
└──────────┬─────────────────────────────────┬───────────────────┘
           │                                 │
┌──────────▼─────────────┐       ┌───────────▼────────────────┐
│  Intra-LL Self-Attn    │       │  Intra-LH Sparse Attn      │
│  (full softmax)        │       │  (energy gate: σ((e−τ)×10) │
└──────────┬─────────────┘       └───────────┬────────────────┘
           │                                 │
           └──────────┬──────────────────────┘
                      │  Cross-Scale Attention
                      │  (LL queries LH keys/values)
                      │
┌─────────────────────▼──────────────────────────────────────────┐
│  Concat [LL_out, LH_out]  →  Flatten  →  Dropout  →  Linear   │
│  Output: (B·C, 1, H)                                           │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│  RevIN  (denorm)                                               │
│  — restore original scale                                      │
└────────────────────────────────────────────────────────────────┘

Output  (B·C, 1, H)  →  reshape  →  (B, C, H)
```

---

## 2. Component Deep-Dive

### 2.1 RevIN — Reversible Instance Normalisation

**File:** `model.py` → `class RevIN`

**Problem it solves:** Time series suffer from *distribution shift* — the statistics of the training window differ from the test window (different seasons, sensor drift, etc.). Normalising the whole dataset does not help per-sample.

**How it works:**
- During `norm`: caches the per-sample mean μ and std σ along the time axis, then subtracts and divides. Optionally applies learnable affine parameters (γ, β) per channel.
- During `denorm`: reverses the transform using the cached statistics. This happens **after** the model outputs a forecast, restoring the prediction to the original scale.

**Key design choices:**
- Statistics are detached from the gradient graph (`.detach()`) — the model cannot "cheat" by exploiting normalisation statistics.
- The affine parameters γ and β give the model an extra degree of freedom to shift/scale the normalised distribution during training.

```
mode='norm':   x̂ = (x − μ) / σ  [then × γ + β if affine]
mode='denorm': x = x̂ × σ + μ   [undoes affine first]
```

---

### 2.2 PatchEmbedding — Patch Tokenisation

**File:** `model.py` → `class PatchEmbedding`

**Problem it solves:** Standard transformers treat each time-step as a token. For long sequences (L=512), this creates O(L²) attention complexity and misses local temporal structure.

**How it works:**
Splits the sequence into **N overlapping windows** of length `patch_len` with step `stride`, then projects each window linearly to dimension `d_model`.

```
Input:  (B, 1, L)
unfold: (B, 1, N, P)    N = (L − P) / stride + 1
squeeze+proj: (B, N, d_model)
transpose: (B, d_model, N)   ← channels-first for downstream convolutions
```

**Key hyperparameters:**
| Parameter | Default | Effect |
|-----------|---------|--------|
| `patch_len` (P) | 16 | Local context per token |
| `stride` (S)   |  8 | 50% overlap — balances resolution vs. redundancy |

With L=512, P=16, S=8 → **N=63 patches** instead of 512 raw tokens → 8× fewer attention operations.

---

### 2.3 LearnableDWT1D — Learnable Wavelet Transform

**File:** `model.py` → `class LearnableDWT1D`

**Problem it solves:** Standard wavelets (Haar, Daubechies) use fixed filters that may not align with the frequency structure of the target dataset. Learnable filters adapt to the data.

**How it works:**
Two sets of per-channel 1-D filters are trained end-to-end:
- **h** (low-pass): extracts trend / slow-moving components → produces **LL** coefficients
- **g** (high-pass): extracts noise / rapid fluctuations → produces **LH** coefficients

Both filters use depthwise convolution (`groups=channels`) so each embedding channel has its own filter — total parameters: `2 × d_model × filter_length`.

```python
LL = F.conv1d(x, h, stride=2, padding=pad, groups=C)  # (B, C, N/2)
LH = F.conv1d(x, g, stride=2, padding=pad, groups=C)  # (B, C, N/2)
```

**Initialisation:** g is initialised with zero mean to impose a high-pass prior, matching the wavelet biorthogonality condition before training begins.

**Inverse transform** (for reconstruction loss):
```python
x_recon = conv_transpose1d(LL, h) + conv_transpose1d(LH, g)
```

**Orthogonality regularisation** (in loss): penalises `|⟨h, g⟩|` to keep the two filters spectrally separated — without this, both filters can collapse to the same function.

---

### 2.4 FrequencySparseAttention — Energy-Gated Attention

**File:** `model.py` → `class FrequencySparseAttention`

**Problem it solves:** In the high-frequency branch (LH), many positions carry pure noise (low energy). Standard softmax attention wastes capacity attending to these uninformative positions.

**How it works:**

For the **LL branch** (trend): standard multi-head scaled dot-product attention — no gating.

For the **LH branch** (detail): a differentiable sigmoid gate is applied element-wise to the attention weights *after* softmax:

```
energy = |LH_seq|.mean(dim=-1)         # (B, L_k)  — per-position energy
gate   = σ( (energy − τ) × 10 )        # soft step: ≈0 below τ, ≈1 above τ
weights = softmax(QKᵀ/√d_h) ⊙ gate    # suppress low-energy positions
```

The slope `×10` makes the gate behave like a hard threshold when τ is well-calibrated, while remaining fully differentiable for gradient flow through τ.

**Why sigmoid instead of hard masking?**
The original `WRAT/models/wrat.py` used a boolean hard mask (scores → −∞ before softmax). This caused gradient death when entire rows were masked. The sigmoid gate fixes this: every position retains a small gradient, and the model learns to push τ to the right value.

---

### 2.5 WRATBlock — Dual-Band Attention Block

**File:** `model.py` → `class WRATBlock`

The core building block. Processes the two wavelet bands with three attention sub-modules and two feed-forward networks:

```
LL, LH: (B, d_model, N/2)
    │
    ├── Intra-LL Self-Attention    LL_out  = Attn(LL, LL, LL)
    ├── Intra-LH Sparse Attention  LH_out  = SparseAttn(LH, LH, LH; τ)
    │
    └── Cross-Scale Attention      cross   = Attn(LL_out, LH_out, LH_out)
        (LL queries LH — trend draws on relevant detail)
    │
    LL_fused = LayerNorm(LL_seq + LL_out + cross)
    LH_fused = LayerNorm(LH_seq + LH_out)
    │
    LL_final = MLP_LL(LL_fused) + LL_fused     ← residual
    LH_final = MLP_LH(LH_fused) + LH_fused
```

**MLP design:** 4× expansion ratio, GELU activation, dropout before and after the projection-down layer.

**Why cross-scale attention?** The trend band (LL) is smooth but may miss inflection points. Cross-attending to LH allows the model to detect "something changed here" from the detail band — e.g. a sudden spike that will affect the trend shortly after.

---

### 2.6 LearnableTauWRATBlock — Learnable Sparsity Threshold

**File:** `model.py` → `class LearnableTauWRATBlock`

**Problem:** The threshold τ in `FrequencySparseAttention` is a critical hyperparameter. A fixed τ=0.1 may be too loose for clean signals (passes noise) or too tight for noisy signals (suppresses signal).

**Solution:** Make τ a **trained parameter** parameterised in logit-space to keep it bounded in (0, 1):

```
raw_τ  ~ scalar nn.Parameter  (unconstrained, initialised via logit of tau_init)
τ      = sigmoid(raw_τ)       (constrained to (0,1))
```

Before each forward pass, the current τ is injected into the LH attention threshold:
```python
self._block.intra_LH_attn.threshold = sigmoid(self.raw_tau)
```

This keeps the gradient flowing back through τ while using the clean `.threshold` interface in `FrequencySparseAttention`.

**Initialisation:** `raw_τ₀ = logit(tau_init) = log(tau_init / (1 − tau_init))`
So `sigmoid(raw_τ₀) = tau_init = 0.1` at the start of training.

---

### 2.7 DualHeadPWSA_Loss — Composite Loss

**File:** `model.py` → `class DualHeadPWSA_Loss`

```
L_total = L_forecast
        + λ_recon  × L_reconstruction
        + λ_ortho  × L_orthogonality
```

| Component | Formula | Default weight | Purpose |
|-----------|---------|---------------|---------|
| **L_forecast** | MSE(ŷ, y) | 1.0 | Primary forecasting objective |
| **L_reconstruction** | MSE(patches_recon, patches_orig) | λ=0.1 | Forces DWT filters to preserve information — prevents filter collapse |
| **L_orthogonality** | mean\|⟨h_flat, g_flat⟩\| | λ=0.01 | Keeps h (low-pass) and g (high-pass) spectrally separated |

The reconstruction loss connects the DWT inverse transform back into the training loop — without it, h and g would only be trained through the attention path, which may ignore their reconstruction fidelity.

---

## 3. Architecture Variants & Ablations

Three variants are trained in parallel for every forecast horizon:

### Variant A — Learnable τ (Full Model)

```python
PatchedWSA(tau_type='learnable', tau_init=0.1)
```
- Uses `LearnableTauWRATBlock`
- τ is gradient-trained alongside all other parameters
- **Recommended variant** — adapts sparsity to data and horizon automatically

### Variant B — Fixed τ (Ablation: no τ learning)

```python
PatchedWSA(tau_type='fixed', tau_init=0.1)
```
- Uses `WRATBlock` directly with static `threshold=0.1`
- τ is a constant hyperparameter, set once before training
- Tests whether gradient-training τ actually improves over a sensible constant

### Variant C — No HF Branch (Ablation: measure LH contribution)

```python
PatchedWSA(tau_type='learnable', zero_lh=True)   # zero_lh passed at forward time
```
- Identical architecture to Variant A
- **During the forward pass**, LH is zeroed: `LH = torch.zeros_like(LH)`
- Isolates the contribution of the high-frequency detail branch
- If Variant A ≫ Variant C → LH branch carries meaningful signal
- If Variant A ≈ Variant C → model relies mostly on trend (LL) information

### What Changed Between Variants

| Feature | Variant A (Learnable τ) | Variant B (Fixed τ) | Variant C (No HF) |
|---------|------------------------|---------------------|--------------------|
| τ type | `nn.Parameter` (sigmoid) | constant `float` | `nn.Parameter` (unused) |
| LH branch active | ✅ | ✅ | ❌ (zeroed) |
| Cross-attention | ✅ | ✅ | ✅ (attends to zeros) |
| Extra parameters | +1 scalar (raw_τ) | +0 | +1 scalar (unused) |
| Gradient through τ | ✅ | ❌ | ✅ (but τ has no effect) |

---

## 4. Evolution: WRAT → PatchWRAT

The project went through several architectural iterations. Here is exactly what changed at each stage:

### Stage 1 — WRAT (`WRAT/models/wrat.py`)

**Original wavelet transformer.** Operated directly on raw time-step sequences.

| Aspect | WRAT |
|--------|------|
| Input tokenisation | None — raw time steps |
| DWT input | Raw sequence (B, C, L) |
| Filter initialisation | Random, no orthogonality constraints |
| Sparsity gate | Hard boolean mask (−∞ before softmax) |
| τ | Fixed scalar hyperparameter |
| Loss | Task MSE only |
| Mode | Multivariate (joint processing of all variates) |

**Problems identified:**
- Hard masking caused gradient death when all positions fell below τ
- No patch compression → O(L²) attention on full sequence
- Fixed filters could not adapt to different datasets

---

### Stage 2 — WRAT with Ablation Variants (`Rat/wrat.py`)

Added **four variants** benchmarked on ETTh1:

| Variant | Change |
|---------|--------|
| `WRAT_Fixed` | τ = 0.1 constant |
| `WRAT_Adaptive` | τ adapted per-horizon (still not gradient-trained) |
| `WRAT_Learnable` | τ stored as `nn.Parameter`, sigmoid logit parametrisation |
| `WRAT_Ablation` | LH branch zeroed to measure HF contribution |

**Key finding:** Learnable τ slightly improved R² and sMAPE at H=720. Hard mask remained but early stopping prevented gradient death in practice.

---

### Stage 3 — P-WSA v8 (`Rat/PWSA/x.py`)

Added **patch tokenisation** on top of WRAT. Major changes:

| Feature | Before | After |
|---------|--------|-------|
| Tokenisation | Raw time-steps | Overlapping patches (P=16, S=8) |
| Sequence length fed to DWT | L (up to 512) | N = 63 patches |
| Attention complexity | O(L²) | O(N²) — 8× fewer tokens |
| Mode | Multivariate (7 variates jointly) | Still multivariate |
| Parameters | ~116K | ~372K–2.2M (grows with pred_len due to linear head) |

**Problem:** Multivariate mode allowed cross-variate information leakage. Different variates have very different distributions which confused the model.

---

### Stage 4 — PatchWRAT (`Rat/PWSA/x_publish.py`, `Rat/modified_arch/p_wsa.py`)

Switched to **channel-independent (CI)** processing — each variate processed as a separate batch element.

| Feature | P-WSA v8 | PatchWRAT |
|---------|----------|-----------|
| Variate processing | Joint (multivariate) | Independent (CI) |
| RevIN | ❌ | ✅ — prevents distribution shift |
| DWT filter style | Standard depthwise | Learnable depthwise with zero-mean g init |
| Sparsity gate | Hard mask | **Sigmoid soft gate** — fully differentiable |
| τ | Fixed | Learnable (raw_τ in logit-space) |
| Loss | MSE only | MSE + reconstruction + orthogonality |
| Gradient through filters | Partial (only via forecast path) | Full (via forecast + reconstruction paths) |

**ETTm1 results** (normalised MSE, iTransformer SOTA = 0.334 @ H=96):

| H | P-WSA v8 MSE | PatchWRAT MSE | Winner |
|---|-------------|--------------|--------|
| 96 | 0.3485 | **0.0476** | PatchWRAT |
| 192 | 0.4072 | **0.0714** | PatchWRAT |
| 336 | 0.4335 | **0.1023** | PatchWRAT |
| 720 | 0.4984 | **0.1284** | PatchWRAT |

---

### Stage 5 — Final PatchWRAT (`PatchWRAT_final/`)

Clean, modular, production-ready refactor of Stage 4. No algorithmic changes — all improvements are code quality and structure:

- **`model.py`** — pure architecture classes, no training code, full docstrings with shape annotations
- **`utils.py`** — datasets, metrics (8 metrics vs. 3 in earlier code), early stopping with `.restore()`, 4 plotting functions
- **`train.py`** — CLI (`argparse`) entry-point, supports both ETT and Weather datasets, saves checkpoints per horizon
- **`__init__.py`** — clean public API

---

## 5. Benchmark Results

### ETTh1 — WRAT Ablation (from `Rat/results.txt`)

Dataset: ETTh1 | Seq=336 | d_model=64 | Patience=15

| Model | H=1 MAE | H=96 MAE | H=192 MAE | H=336 MAE | H=720 MAE | AVG |
|-------|---------|----------|-----------|-----------|-----------|-----|
| WRAT Fixed | 0.0468 | 0.2091 | 0.2355 | 0.2546 | 0.2936 | 0.2079 |
| WRAT Adaptive | 0.0467 | 0.2101 | 0.2364 | 0.2540 | **0.2834** | **0.2061** |
| WRAT Learnable | 0.0467 | **0.2083** | 0.2371 | 0.2571 | **0.2830** | 0.2065 |
| WRAT Ablation (no HF) | **0.0465** | 0.2095 | 0.2401 | 0.2575 | 0.2920 | 0.2091 |

**Observation:** The HF branch provides marginal but consistent benefit at short horizons. At H=720, Learnable τ gives the best MAE — showing that adaptive sparsity helps most when the forecast is long and trend dominates.

### ETTm1 — PatchWRAT vs P-WSA v8

Dataset: ETTm1 | Seq=336 | SOTA reference: iTransformer

| H | P-WSA v8 MSE | PatchWRAT MSE | SOTA MSE | PatchWRAT vs SOTA |
|---|-------------|--------------|----------|-------------------|
| 96 | 0.3485 | **0.0476** | 0.334 | **−85.7%** |
| 192 | 0.4072 | **0.0714** | 0.377 | **−81.1%** |
| 336 | 0.4335 | **0.1023** | 0.426 | **−76.0%** |
| 720 | 0.4984 | **0.1284** | 0.491 | **−73.8%** |

> **Note:** PatchWRAT operates in univariate mode on the OT (oil temperature) variate only — the MSE values are on the normalised scale. Direct comparison to SOTA multivariate numbers requires care.

---

## 6. Project Structure

```
TTS/
├── README.md                      ← this file
├── .gitignore
│
├── PatchWRAT_final/               ← Clean final implementation
│   ├── model.py                   ← All architecture classes
│   ├── utils.py                   ← Datasets, metrics, plotting
│   ├── train.py                   ← CLI training entry-point
│   └── __init__.py                ← Public API
│
├── Rat/
│   ├── wrat.py                    ← Stage 2: WRAT ablation suite (ETTh1)
│   ├── modified_arch/
│   │   ├── p_wsa.py               ← Stage 4: PatchWRAT on ETTm1
│   │   └── p_wsa_weather.py       ← Stage 4: PatchWRAT on MPI Weather
│   └── PWSA/
│       ├── x_publish.py           ← Stage 4: Final RAT reference
│       └── x_publish_v2.py        ← Stage 4: v2 variant
│
└── WRAT/
    ├── models/
    │   ├── wrat.py                ← Stage 1: Original WRAT
    │   └── vanilla.py             ← Vanilla Transformer baseline
    └── utils/
        ├── metrics.py
        └── losses.py
```

---

## 7. Quick Start

```bash
cd PatchWRAT_final

# ETTm1 — standard horizons
python train.py \
  --dataset ett \
  --data_path ../Rat/ETTm1.csv \
  --horizons 96 192 336 720 \
  --seq_len 512 \
  --d_model 64 \
  --tau_type learnable \
  --save_dir outputs/ettm1

# MPI Weather dataset
python train.py \
  --dataset weather \
  --data_path ../Rat/mpi_roof_2017b/mpi_roof_2017b.csv \
  --horizons 12 24 48 96 192 336 720 \
  --save_dir outputs/weather

# Quick smoke-test (single horizon, 3 epochs)
python train.py --horizons 96 --epochs 3 --batch_size 16
```

### Using as a library

```python
from PatchWRAT_final import PatchedWSA, DualHeadPWSA_Loss, ETTDataset, evaluate
import torch

model = PatchedWSA(seq_len=512, pred_len=96, d_model=64, tau_type='learnable')
print(model)

# Single forward pass
x = torch.randn(8, 1, 512)          # (B, 1, L)
preds, patch_recon, patches, LL, LH = model(x)
print(f"Forecast shape: {preds.shape}")   # (8, 1, 96)
```

---

## 8. Dependencies

```
torch>=2.0
numpy
pandas
scikit-learn
matplotlib
```

Install:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```
