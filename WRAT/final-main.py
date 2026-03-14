# ============================================================
# WMRT Benchmark v2 — Three τ Variants + Ablation Study
# ============================================================
# Paste your class definitions above this block:
#   LearnableDWT1D, FrequencySparseAttention, WMRTBlock,
#   WaveletTransformerLoss, VanillaTransformerBaseline, PositionalEncoding
#
# NEW in v2:
#   1. Three WMRT variants run simultaneously:
#        - Fixed τ=0.1  (original baseline)
#        - Adaptive τ   (τ decays from 0.5→0.05 over epochs, scheduled)
#        - Learnable τ  (τ is a nn.Parameter, optimised end-to-end)
#   2. Sparsity ablation: LH zeroed out → confirms attention contribution
#   3. R² and Pearson r computed on INVERSE-TRANSFORMED scale (honest)
#   4. MAPE / sMAPE also computed on original scale
# ============================================================

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

# ─────────────────────────────────────────────
# PATCH: Learnable-τ wrapper for WMRTBlock
# Wraps an existing WMRTBlock and overrides its sparsity_tau
# with a sigmoid-bounded nn.Parameter so τ ∈ (0, 1).
# ─────────────────────────────────────────────
class LearnableTauWMRTBlock(nn.Module):
    """Wraps WMRTBlock and replaces fixed τ with a learnable parameter."""
    def __init__(self, d_model, num_heads, tau_init=0.1):
        super().__init__()
        # Store raw logit; τ = sigmoid(raw_tau) keeps it in (0,1)
        self.raw_tau = nn.Parameter(torch.tensor(
            math.log(tau_init / (1 - tau_init))   # inverse sigmoid
        ))
        self._block = WMRTBlock(d_model, num_heads, sparsity_tau=tau_init)

    @property
    def tau(self):
        return torch.sigmoid(self.raw_tau).item()

    def forward(self, LL, LH):
        # Inject current τ into block before forward pass
        self._block.sparsity_tau = torch.sigmoid(self.raw_tau)
        return self._block(LL, LH)

    def parameters(self, recurse=True):
        yield self.raw_tau
        yield from self._block.parameters(recurse=recurse)


# ─────────────────────────────────────────────
# 1. ETTm1 Dataset
# ─────────────────────────────────────────────
class ETTm1Dataset(Dataset):
    def __init__(self, seq_len=128, split='train', file_path='ETTm1.csv', target_col='OT'):
        self.seq_len = seq_len

        df   = pd.read_csv(file_path)
        data = df[target_col].values.reshape(-1, 1)

        train_end = 12 * 30 * 24 * 4       # 17,280
        val_end   = train_end + 4 * 30 * 24 * 4  # 23,040

        if split == 'train':
            raw = data[:train_end]
        elif split == 'val':
            raw = data[train_end:val_end]
        else:
            raw = data[val_end:]

        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])          # fit ONLY on train
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)
        self._scaler_mean  = self.scaler.mean_[0]
        self._scaler_scale = self.scaler.scale_[0]

    def inverse(self, x_norm):
        """Inverse-transform a numpy array from normalised to original scale."""
        return x_norm * self._scaler_scale + self._scaler_mean

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx          : idx + self.seq_len]
        y = self.data[idx + 1      : idx + self.seq_len + 1]
        return x.t(), y.t()


# ─────────────────────────────────────────────
# 2. Evaluation — honest metrics
#    Regression errors  : normalised scale  (fair model comparison)
#    R², Pearson, MAPE  : ORIGINAL scale    (honest interpretation)
# ─────────────────────────────────────────────
def evaluate(model_fn, loader, device, inv_fn=None):
    all_p, all_t = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            preds  = model_fn(bx)
            L      = min(preds.shape[-1], by.shape[-1])
            all_p.append(preds[..., :L].cpu())
            all_t.append(by[...,   :L].cpu())

    p_norm = torch.cat(all_p, dim=0).flatten().numpy()
    t_norm = torch.cat(all_t, dim=0).flatten().numpy()

    # ── Normalised-scale regression errors ──────────────────────
    err  = p_norm - t_norm
    mae  = np.abs(err).mean()
    mse  = (err ** 2).mean()
    rmse = mse ** 0.5
    maxe = np.abs(err).max()

    # ── Original-scale metrics (inverse transform) ───────────────
    if inv_fn is not None:
        p_orig = inv_fn(p_norm)
        t_orig = inv_fn(t_norm)
    else:
        p_orig, t_orig = p_norm, t_norm

    eps   = 1e-8
    err_o = p_orig - t_orig
    # MAPE: exclude timesteps where |target| < 1% of its std to avoid /0 explosion
    t_std  = t_orig.std() + eps
    valid  = np.abs(t_orig) > 0.01 * t_std
    mape   = (np.abs(err_o[valid]) / np.abs(t_orig[valid])).mean() * 100 if valid.sum() > 0 else float('nan')
    smape  = (2 * np.abs(err_o) / (np.abs(p_orig) + np.abs(t_orig) + eps)).mean() * 100

    ss_res = (err_o ** 2).sum()
    ss_tot = ((t_orig - t_orig.mean()) ** 2).sum()
    r2     = 1 - ss_res / (ss_tot + eps)

    corr   = np.corrcoef(p_orig, t_orig)[0, 1] if len(p_orig) > 1 else 0.0

    dp, dt = np.diff(p_norm), np.diff(t_norm)
    dir_acc = (np.sign(dp) == np.sign(dt)).mean() * 100

    return dict(mae=mae, mse=mse, rmse=rmse, maxe=maxe,
                mape=mape, smape=smape, r2=r2, corr=corr, dir_acc=dir_acc)


# ─────────────────────────────────────────────
# 3. Hyper-parameters
# ─────────────────────────────────────────────
SEQ_LEN      = 128
BATCH_SIZE   = 64
D_MODEL      = 16
NUM_HEADS    = 4
EPOCHS       = 30
LR           = 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TAU_FIXED    = 0.1
TAU_ADAPT_START = 0.5
TAU_ADAPT_END   = 0.05
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────
# 4. DataLoaders
# ─────────────────────────────────────────────
train_ds = ETTm1Dataset(seq_len=SEQ_LEN, split='train')
val_ds   = ETTm1Dataset(seq_len=SEQ_LEN, split='val')
test_ds  = ETTm1Dataset(seq_len=SEQ_LEN, split='test')

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

inv_fn = test_ds.inverse      # inverse transform for honest metrics
print(f"Train {len(train_ds):,} | Val {len(val_ds):,} | Test {len(test_ds):,}")


# ─────────────────────────────────────────────
# 5. Build all models
# ─────────────────────────────────────────────
def make_wmrt(tau_type='fixed', tau_val=TAU_FIXED):
    dwt   = LearnableDWT1D(1, D_MODEL).to(DEVICE)
    if tau_type == 'learnable':
        block = LearnableTauWMRTBlock(D_MODEL, NUM_HEADS, tau_init=tau_val).to(DEVICE)
    else:
        block = WMRTBlock(D_MODEL, NUM_HEADS, sparsity_tau=tau_val).to(DEVICE)
    scaler_conv = nn.Conv1d(1, 1, kernel_size=1).to(DEVICE)
    params = list(dwt.parameters()) + list(block.parameters()) + list(scaler_conv.parameters())
    opt    = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    return dwt, block, scaler_conv, opt

# ── Three τ variants ──────────────────────────
dwt_fixed,  block_fixed,  sc_fixed,  opt_fixed  = make_wmrt('fixed',     TAU_FIXED)
dwt_adapt,  block_adapt,  sc_adapt,  opt_adapt  = make_wmrt('fixed',     TAU_ADAPT_START)  # tau updated per-epoch
dwt_learn,  block_learn,  sc_learn,  opt_learn  = make_wmrt('learnable', TAU_FIXED)

# ── Vanilla baseline ──────────────────────────
vanilla = VanillaTransformerBaseline(
    in_channels=1, d_model=D_MODEL, num_heads=NUM_HEADS, seq_len=SEQ_LEN).to(DEVICE)
opt_van = optim.AdamW(vanilla.parameters(), lr=LR)

# ── Ablation: WMRT with LH zeroed (no HF attention) ──
dwt_abl, block_abl, sc_abl, opt_abl = make_wmrt('fixed', TAU_FIXED)

crit_wmrt = WaveletTransformerLoss(lambda_recon=1.0, lambda_ortho=0.1)
crit_mse  = nn.MSELoss()

def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())

print(f"\nParams — Fixed τ: {count_params(dwt_fixed, block_fixed, sc_fixed):,} | "
      f"Adaptive τ: {count_params(dwt_adapt, block_adapt, sc_adapt):,} | "
      f"Learnable τ: {count_params(dwt_learn, block_learn, sc_learn):,} | "
      f"Vanilla: {count_params(vanilla):,} | Ablation: {count_params(dwt_abl, block_abl, sc_abl):,}")


# ─────────────────────────────────────────────
# 6. Adaptive τ schedule
# ─────────────────────────────────────────────
def get_adaptive_tau(epoch, total=EPOCHS,
                     tau_start=TAU_ADAPT_START, tau_end=TAU_ADAPT_END):
    """Cosine annealing from tau_start → tau_end."""
    progress = (epoch - 1) / max(total - 1, 1)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return tau_end + (tau_start - tau_end) * cosine


# ─────────────────────────────────────────────
# 7. Generic WMRT train step
# ─────────────────────────────────────────────
def train_wmrt_epoch(dwt, block, sc, opt, loader, tau_override=None, zero_lh=False):
    dwt.train(); block.train(); sc.train()
    loss_sum = sparsity_sum = n = 0

    # Update adaptive τ if provided
    if tau_override is not None and hasattr(block, 'sparsity_tau'):
        block.sparsity_tau = tau_override

    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()

        LL, LH = dwt(bx)
        if zero_lh:
            LH = torch.zeros_like(LH)   # ablation: kill HF path

        LL_o, LH_o = block(LL, LH)
        raw   = dwt.inverse(LL_o, LH_o)
        preds = sc(raw)
        xr    = dwt.inverse(LL, LH)

        L = min(preds.shape[-1], by.shape[-1])
        total_loss, task_loss, *_ = crit_wmrt(
            preds[..., :L], by[..., :L], bx, xr, dwt)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(dwt.parameters()) + list(block.parameters()) + list(sc.parameters()), 1.0)
        opt.step()

        # Sparsity: fraction of HF coefficients below τ
        current_tau = (block.tau if isinstance(block, LearnableTauWMRTBlock)
                       else (tau_override if tau_override is not None
                             else getattr(block, 'sparsity_tau', TAU_FIXED)))
        if isinstance(current_tau, torch.Tensor):
            current_tau = current_tau.item()

        energy = torch.abs(LH).mean(dim=1)
        sparsity_sum += (~(energy > current_tau)).float().mean().item() * 100
        loss_sum += task_loss.item()
        n += 1

    return loss_sum / n, sparsity_sum / n


def make_eval_fn(dwt, block, sc, zero_lh=False):
    def fn(bx):
        dwt.eval(); block.eval(); sc.eval()
        LL, LH = dwt(bx)
        if zero_lh:
            LH = torch.zeros_like(LH)
        LL_o, LH_o = block(LL, LH)
        return sc(dwt.inverse(LL_o, LH_o))
    return fn


# ─────────────────────────────────────────────
# 8. Training loop — all 5 models simultaneously
# ─────────────────────────────────────────────
keys = ['fixed', 'adaptive', 'learnable', 'vanilla', 'ablation']
hist = {k: {'train': [], 'val': [], 'sparsity': [], 'tau': []} for k in keys}

print(f"\n{'='*95}")
print(f"{'Ep':>3} | "
      f"{'Fixed τ':>9}({'tr':>5}/{'vl':>6}) | "
      f"{'Adapt τ':>9}({'tr':>5}/{'vl':>6}) | "
      f"{'Learn τ':>9}({'tr':>5}/{'vl':>6}) | "
      f"{'Vanilla':>9}({'tr':>5}/{'vl':>6}) | "
      f"{'Ablation':>9}({'vl':>6})")
print('='*95)

for epoch in range(1, EPOCHS + 1):

    tau_adapt = get_adaptive_tau(epoch)

    # ── Train all models ──────────────────────────────────────────
    tr_fixed,  sp_fixed  = train_wmrt_epoch(dwt_fixed, block_fixed, sc_fixed,
                                             opt_fixed,  train_loader)
    tr_adapt,  sp_adapt  = train_wmrt_epoch(dwt_adapt, block_adapt, sc_adapt,
                                             opt_adapt,  train_loader,
                                             tau_override=tau_adapt)
    tr_learn,  sp_learn  = train_wmrt_epoch(dwt_learn, block_learn, sc_learn,
                                             opt_learn,  train_loader)
    tr_abl,    sp_abl    = train_wmrt_epoch(dwt_abl,   block_abl,   sc_abl,
                                             opt_abl,    train_loader, zero_lh=True)

    vanilla.train()
    v_loss = v_n = 0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt_van.zero_grad()
        preds = vanilla(bx)
        L = min(preds.shape[-1], by.shape[-1])
        loss = crit_mse(preds[..., :L], by[..., :L])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vanilla.parameters(), 1.0)
        opt_van.step()
        v_loss += loss.item(); v_n += 1
    tr_van = v_loss / v_n

    # ── Validate all models ───────────────────────────────────────
    vanilla.eval()
    val_fixed  = evaluate(make_eval_fn(dwt_fixed, block_fixed, sc_fixed),       val_loader, DEVICE, inv_fn)
    val_adapt  = evaluate(make_eval_fn(dwt_adapt, block_adapt, sc_adapt),       val_loader, DEVICE, inv_fn)
    val_learn  = evaluate(make_eval_fn(dwt_learn, block_learn, sc_learn),       val_loader, DEVICE, inv_fn)
    val_van    = evaluate(lambda bx: vanilla(bx),                                val_loader, DEVICE, inv_fn)
    val_abl    = evaluate(make_eval_fn(dwt_abl, block_abl, sc_abl, zero_lh=True), val_loader, DEVICE, inv_fn)

    # ── Record ────────────────────────────────────────────────────
    for k, tr, vl, sp, tau in [
        ('fixed',     tr_fixed, val_fixed['mse'],  sp_fixed,  TAU_FIXED),
        ('adaptive',  tr_adapt, val_adapt['mse'],  sp_adapt,  tau_adapt),
        ('learnable', tr_learn, val_learn['mse'],  sp_learn,  block_learn.tau),
        ('vanilla',   tr_van,   val_van['mse'],    0.0,       None),
        ('ablation',  tr_abl,   val_abl['mse'],    sp_abl,    TAU_FIXED),
    ]:
        hist[k]['train'].append(tr)
        hist[k]['val'].append(vl)
        hist[k]['sparsity'].append(sp)
        hist[k]['tau'].append(tau)

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>3} | "
              f"τ={TAU_FIXED:.2f} {tr_fixed:>6.4f}/{val_fixed['mse']:>7.5f} | "
              f"τ={tau_adapt:.3f} {tr_adapt:>6.4f}/{val_adapt['mse']:>7.5f} | "
              f"τ={block_learn.tau:.3f} {tr_learn:>6.4f}/{val_learn['mse']:>7.5f} | "
              f"       {tr_van:>6.4f}/{val_van['mse']:>7.5f} | "
              f"abl    {val_abl['mse']:>7.5f}")


# ─────────────────────────────────────────────
# 9. Final Test Evaluation
# ─────────────────────────────────────────────
vanilla.eval()

# NOTE: use plain ASCII keys to avoid any Unicode/encoding edge-cases on Windows
results = {
    'Fixed_tau':    evaluate(make_eval_fn(dwt_fixed, block_fixed, sc_fixed),               test_loader, DEVICE, inv_fn),
    'Adaptive_tau': evaluate(make_eval_fn(dwt_adapt, block_adapt, sc_adapt),               test_loader, DEVICE, inv_fn),
    'Learnable_tau':evaluate(make_eval_fn(dwt_learn, block_learn, sc_learn),               test_loader, DEVICE, inv_fn),
    'Vanilla':      evaluate(lambda bx: vanilla(bx),                                       test_loader, DEVICE, inv_fn),
    'Ablation_LH0': evaluate(make_eval_fn(dwt_abl,   block_abl,   sc_abl, zero_lh=True),  test_loader, DEVICE, inv_fn),
}

# Human-readable display names (same order)
DISPLAY_NAMES = {
    'Fixed_tau':    'Fixed tau',
    'Adaptive_tau': 'Adaptive tau',
    'Learnable_tau':'Learnable tau',
    'Vanilla':      'Vanilla',
    'Ablation_LH0': 'Ablation(LH=0)',
}

# ── Print table ──────────────────────────────
metrics_info = [
    ('mae',     'MAE (lower)',        False, '.5f'),
    ('mse',     'MSE (lower)',        False, '.5f'),
    ('rmse',    'RMSE (lower)',       False, '.5f'),
    ('maxe',    'Max Err (lower)',    False, '.4f'),
    ('mape',    'MAPE% (lower)',      False, '.2f'),
    ('smape',   'sMAPE% (lower)',     False, '.2f'),
    ('r2',      'R2 (higher/orig)',   True,  '.4f'),
    ('corr',    'Pearson (higher)',   True,  '.4f'),
    ('dir_acc', 'DirAcc% (higher)',   True,  '.2f'),
]

col_keys  = list(results.keys())
col_w     = 15
sep       = '=' * (18 + col_w * len(col_keys))

print(f"\n{sep}")
print(f"{'TEST SET — FULL BENCHMARK  (R2/Pearson/MAPE on ORIGINAL scale)':^{len(sep)}}")
print(sep)
header = f"{'Metric':<18}" + "".join(f"{DISPLAY_NAMES[k]:>{col_w}}" for k in col_keys)
print(header)
print('-' * len(sep))

for metric_key, label, higher_is_better, fmt in metrics_info:
    vals = [results[k][metric_key] for k in col_keys]
    best = max(vals) if higher_is_better else min(vals)
    row  = f"{label:<18}"
    for v in vals:
        cell = f"{v:{fmt}}" + ('*' if abs(v - best) < 1e-9 else ' ')
        row += f"{cell:>{col_w}}"
    print(row)

print('-' * len(sep))
print("* = best in row  |  R2, Pearson, MAPE computed on ORIGINAL (inverse-transformed) scale")

# ── Ablation interpretation ──────────────────
abl_dir = results['Ablation_LH0']['dir_acc']
fix_dir = results['Fixed_tau']['dir_acc']
delta   = fix_dir - abl_dir
print(f"\n[Ablation] Zeroing LH (HF path) changes DirAcc by {delta:+.1f}pp "
      f"({fix_dir:.1f}% -> {abl_dir:.1f}%).")
if delta > 5:
    print("  CONFIRMED: HF attention path contributes meaningfully — not just trend isolation.")
elif delta > 0:
    print("  MARGINAL: HF attention adds a small but positive contribution.")
else:
    print("  WARNING: HF attention path has no positive impact — gain is from DWT trend (LL) alone.")

# ── Learnable tau final value ────────────────
print(f"\n[Learnable tau] Final learned tau = {block_learn.tau:.4f}  "
      f"(initialised at {TAU_FIXED:.4f})")


# ─────────────────────────────────────────────
# 10. Plots  (all keys are plain ASCII)
# ─────────────────────────────────────────────
COLORS = {
    'Fixed_tau':    '#4f8ef7',
    'Adaptive_tau': '#f7a24f',
    'Learnable_tau':'#6dbf67',
    'Vanilla':      '#e05c5c',
    'Ablation_LH0': '#a78bfa',
}
LINE_STYLES = {
    'Fixed_tau':    '-',
    'Adaptive_tau': '--',
    'Learnable_tau':'-.',
    'Vanilla':      ':',
    'Ablation_LH0': (0, (3,1,1,1)),
}
# Subset used in history plots (ablation has no hist entry for train)
HIST_KEYS = {
    'Fixed_tau':    'fixed',
    'Adaptive_tau': 'adaptive',
    'Learnable_tau':'learnable',
    'Vanilla':      'vanilla',
}

ep = range(1, EPOCHS + 1)
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("WMRT v2 — Fixed / Adaptive / Learnable tau + Ablation | ETTm1",
             fontsize=14, fontweight='bold')

# Row 0: Train MSE, Val MSE, tau evolution
ax = axes[0, 0]
for rkey, hk in HIST_KEYS.items():
    ax.plot(ep, hist[hk]['train'], label=DISPLAY_NAMES[rkey],
            color=COLORS[rkey], linestyle=LINE_STYLES[rkey], linewidth=1.6)
ax.set_title('Training MSE'); ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[0, 1]
for rkey, hk in HIST_KEYS.items():
    ax.plot(ep, hist[hk]['val'], label=DISPLAY_NAMES[rkey],
            color=COLORS[rkey], linestyle=LINE_STYLES[rkey], linewidth=1.6)
ax.set_title('Validation MSE'); ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[0, 2]
ax.plot(ep, hist['adaptive']['tau'],  label='Adaptive tau (scheduled)',
        color=COLORS['Adaptive_tau'], linestyle='--', linewidth=1.6)
ax.plot(ep, hist['learnable']['tau'], label='Learnable tau (trained)',
        color=COLORS['Learnable_tau'], linestyle='-.', linewidth=1.6)
ax.axhline(TAU_FIXED, color=COLORS['Fixed_tau'], linestyle=':', linewidth=1.4,
           label=f'Fixed tau={TAU_FIXED}')
ax.set_title('Tau Evolution Over Training'); ax.set_xlabel('Epoch')
ax.set_ylabel('tau value'); ax.legend(fontsize=8); ax.grid(alpha=.3)

# Row 1: Sparsity, lower-is-better bars, higher-is-better bars
ax = axes[1, 0]
for rkey, hk in [('Fixed_tau','fixed'),('Adaptive_tau','adaptive'),('Learnable_tau','learnable')]:
    ax.plot(ep, hist[hk]['sparsity'], label=DISPLAY_NAMES[rkey],
            color=COLORS[rkey], linestyle=LINE_STYLES[rkey], linewidth=1.6)
ax.axhline(50, color='gray', linestyle=':', label='50% baseline', linewidth=1)
ax.set_title('High-Freq Sparsity %'); ax.set_xlabel('Epoch')
ax.set_ylabel('%'); ax.legend(fontsize=8); ax.grid(alpha=.3)

lower_keys   = ['mae', 'rmse', 'smape']
lower_labels = ['MAE', 'RMSE', 'sMAPE%']
higher_keys  = ['r2', 'corr', 'dir_acc']
higher_labels= ['R2', 'Pearson r', 'DirAcc%']
x_pos = np.arange(len(lower_keys))
bar_w = 0.15

ax = axes[1, 1]
for i, (rkey, res) in enumerate(results.items()):
    vals = [res[k] for k in lower_keys]
    ax.bar(x_pos + i*bar_w, vals, bar_w,
           label=DISPLAY_NAMES[rkey], color=COLORS[rkey], alpha=0.85)
ax.set_xticks(x_pos + bar_w*2); ax.set_xticklabels(lower_labels, fontsize=9)
ax.set_title('Test Metrics (lower = better)'); ax.legend(fontsize=7); ax.grid(alpha=.3, axis='y')

ax = axes[1, 2]
x_pos2 = np.arange(len(higher_keys))
for i, (rkey, res) in enumerate(results.items()):
    vals = [res[k] for k in higher_keys]
    ax.bar(x_pos2 + i*bar_w, vals, bar_w,
           label=DISPLAY_NAMES[rkey], color=COLORS[rkey], alpha=0.85)
ax.set_xticks(x_pos2 + bar_w*2); ax.set_xticklabels(higher_labels, fontsize=9)
ax.set_title('Test Metrics (higher = better)'); ax.legend(fontsize=7); ax.grid(alpha=.3, axis='y')

# Row 2: Qualitative forecasts — 3 test samples
sample_indices = [0, 10, 20]
for col_idx, sample_idx in enumerate(sample_indices):
    ax = axes[2, col_idx]
    ds_iter = iter(test_loader)
    for _ in range(sample_idx + 1):
        sbx, sby = next(ds_iter)
    sbx, sby = sbx.to(DEVICE), sby.to(DEVICE)

    t_np = sby[0, 0].cpu().numpy()
    ax.plot(t_np, label='Ground Truth', color='black', alpha=0.75, linewidth=1.5)

    with torch.no_grad():
        forecast_models = [
            ('Fixed_tau',    make_eval_fn(dwt_fixed, block_fixed, sc_fixed)),
            ('Adaptive_tau', make_eval_fn(dwt_adapt, block_adapt, sc_adapt)),
            ('Learnable_tau',make_eval_fn(dwt_learn, block_learn, sc_learn)),
            ('Vanilla',      lambda bx: vanilla(bx)),
        ]
        for rkey, fn in forecast_models:
            pred = fn(sbx)[0, 0, :len(t_np)].cpu().numpy()
            ax.plot(pred, label=DISPLAY_NAMES[rkey], color=COLORS[rkey],
                    linestyle=LINE_STYLES[rkey], alpha=0.85, linewidth=1.3)

    ax.set_title(f'Forecast — Test Sample {sample_idx}')
    ax.legend(fontsize=7); ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig('wmrt_v2_benchmark.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → wmrt_v2_benchmark.png")