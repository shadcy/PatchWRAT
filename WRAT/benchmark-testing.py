# ============================================================
# WMRT Full Baseline Comparison — Publication-Grade Benchmark
# ============================================================
# Baselines implemented from scratch (no external deps):
#   1. DLinear       — Zeng et al. 2023 "Are Transformers Effective for TSF?"
#   2. NLinear       — same paper, normalised variant
#   3. PatchTST      — Nie et al. 2023, patched transformer
#   4. iTransformer  — Liu et al. 2024, inverted attention
#   5. TimesNet      — Wu et al. 2023, 2D temporal variation
#   6. Vanilla Transformer (existing baseline)
#
# Forecast horizons tested: 1, 96, 192, 336, 720
# Datasets: ETTm1 (extend to ETTh1/ETTm2 by changing file_path)
#
# PASTE YOUR CLASS DEFINITIONS ABOVE THIS BLOCK:
#   LearnableDWT1D, FrequencySparseAttention, WRATBlock,
#   WaveletTransformerLoss, VanillaTransformerBaseline,
#   PositionalEncoding, LearnableTauWRATBlock
# ============================================================

import math, time, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models import LearnableDWT1D, FrequencySparseAttention, WRATBlock, VanillaTransformerBaseline, PositionalEncoding, LearnableDWT1D
from utils import WaveletTransformerLoss
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# BASELINE MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

# ── 1. DLinear ───────────────────────────────────────────────
class DLinear(nn.Module):
    """
    Zeng et al. 2023. Decomposes series into trend+residual,
    applies separate linear layers to each, sums predictions.
    """
    def __init__(self, seq_len, pred_len, in_channels=1):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        kernel = 25
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1,
                                padding=kernel//2, count_include_pad=False)
        self.linear_trend    = nn.Linear(seq_len, pred_len)
        self.linear_residual = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, C, L)
        trend    = self.avg(x)
        if trend.shape[-1] != x.shape[-1]:
            trend = trend[..., :x.shape[-1]]
        residual = x - trend
        return self.linear_trend(trend) + self.linear_residual(residual)


# ── 2. NLinear ───────────────────────────────────────────────
class NLinear(nn.Module):
    """
    Zeng et al. 2023. Subtracts last value (normalises),
    applies linear, adds last value back.
    """
    def __init__(self, seq_len, pred_len, in_channels=1):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        last = x[..., -1:]          # (B, C, 1)
        out  = self.linear(x - last)
        return out + last


# ── 3. PatchTST ──────────────────────────────────────────────
class PatchTST(nn.Module):
    """
    Nie et al. 2023. Splits time series into patches,
    applies transformer encoder per channel (channel-independent).
    """
    def __init__(self, seq_len, pred_len, in_channels=1,
                 patch_len=16, stride=8, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.patch_len   = patch_len
        self.stride      = stride
        self.pred_len    = pred_len
        self.in_channels = in_channels

        n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head     = nn.Linear(n_patches * d_model, pred_len)
        self.dropout  = nn.Dropout(dropout)
        self.n_patches = n_patches
        self.d_model   = d_model

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        # unfold into patches: (B*C, n_patches, patch_len)
        x_p = x.reshape(B * C, 1, L)
        x_p = x_p.unfold(-1, self.patch_len, self.stride)  # (B*C, 1, n_p, pl)
        x_p = x_p.squeeze(1)                                # (B*C, n_p, pl)
        x_p = self.patch_embed(x_p) + self.pos_embed        # (B*C, n_p, d)
        x_p = self.dropout(x_p)
        out  = self.encoder(x_p)                             # (B*C, n_p, d)
        out  = out.reshape(B * C, -1)                        # (B*C, n_p*d)
        out  = self.head(out).reshape(B, C, self.pred_len)   # (B, C, pred_len)
        return out


# ── 4. iTransformer ──────────────────────────────────────────
class iTransformer(nn.Module):
    """
    Liu et al. 2024. Inverts the attention — attends across
    variates rather than time. For univariate we apply across
    the time dimension with inverted token representation.
    """
    def __init__(self, seq_len, pred_len, in_channels=1,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        # Embed each variate's full time series as a single token
        self.embed   = nn.Linear(seq_len, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.project = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L) — each channel is a token
        tok = self.dropout(self.embed(x))   # (B, C, d_model)
        out = self.encoder(tok)             # (B, C, d_model)
        return self.project(out)            # (B, C, pred_len)


# ── 5. TimesNet (lightweight version) ────────────────────────
class TimesBlock(nn.Module):
    """
    Simplified TimesNet block: reshape 1-D series to 2-D via
    dominant period, apply 2-D conv, reshape back.
    """
    def __init__(self, seq_len, d_model, top_k=3, conv_channels=32):
        super().__init__()
        self.top_k   = top_k
        self.seq_len = seq_len
        self.conv    = nn.Sequential(
            nn.Conv2d(d_model, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(conv_channels, d_model, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, D = x.shape
        # FFT to find dominant periods
        fft_vals = torch.abs(torch.fft.rfft(x.mean(-1), dim=-1))
        fft_vals[:, 0] = 0   # remove DC
        top_periods = torch.topk(fft_vals, self.top_k, dim=-1).indices + 1

        out = torch.zeros_like(x)
        for b in range(B):
            for period in top_periods[b]:
                p = period.item()
                if p <= 1:
                    continue
                T = math.ceil(L / p)
                pad_len = T * p - L
                xi = F.pad(x[b].T.unsqueeze(0), (0, pad_len))  # (1, D, T*p)
                xi = xi.reshape(1, D, T, p)                     # (1, D, T, p)
                xi = self.conv(xi)                               # (1, D, T, p)
                xi = xi.reshape(1, D, T * p)[..., :L]           # (1, D, L)
                out[b] += xi.squeeze(0).T
        out = out / self.top_k
        return self.norm(x + out)


class TimesNet(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels=1,
                 d_model=32, n_layers=2, top_k=3, dropout=0.1):
        super().__init__()
        self.embed   = nn.Linear(in_channels, d_model)
        self.blocks  = nn.ModuleList([
            TimesBlock(seq_len, d_model, top_k=top_k) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(d_model, in_channels)
        self.proj    = nn.Linear(seq_len, pred_len)
        self.pred_len = pred_len

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        x_in = x.permute(0, 2, 1)          # (B, L, C)
        x_in = self.dropout(self.embed(x_in))  # (B, L, d_model)
        for blk in self.blocks:
            x_in = blk(x_in)
        out = self.head(x_in)               # (B, L, C)
        out = out.permute(0, 2, 1)          # (B, C, L)
        out = self.proj(out)                # (B, C, pred_len)
        return out


# ─────────────────────────────────────────────────────────────
# DATASET — multi-horizon support
# ─────────────────────────────────────────────────────────────
class ETTDataset(Dataset):
    def __init__(self, seq_len=96, pred_len=96, split='train',
                 file_path='ETTm1.csv', target_col='OT'):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        df   = pd.read_csv(file_path)
        data = df[target_col].values.reshape(-1, 1)

        train_end = 12 * 30 * 24 * 4
        val_end   = train_end + 4 * 30 * 24 * 4

        slices = {'train': data[:train_end],
                  'val':   data[train_end:val_end],
                  'test':  data[val_end:]}
        raw = slices[split]

        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)

    def inverse(self, x):
        return x * self.scaler.scale_[0] + self.scaler.mean_[0]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx):
        x = self.data[idx          : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x.t(), y.t()


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(model_fn, loader, device, inv_fn=None):
    all_p, all_t = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred   = model_fn(bx)
            L      = min(pred.shape[-1], by.shape[-1])
            all_p.append(pred[..., :L].cpu())
            all_t.append(by[...,   :L].cpu())

    p = torch.cat(all_p, 0).flatten().numpy()
    t = torch.cat(all_t, 0).flatten().numpy()

    err  = p - t
    mae  = np.abs(err).mean()
    mse  = (err**2).mean()
    rmse = mse**0.5

    if inv_fn is not None:
        p_o, t_o = inv_fn(p), inv_fn(t)
    else:
        p_o, t_o = p, t

    eps    = 1e-8
    t_std  = t_o.std() + eps
    valid  = np.abs(t_o) > 0.01 * t_std
    smape  = (2*np.abs(p_o-t_o)/(np.abs(p_o)+np.abs(t_o)+eps)).mean()*100
    ss_res = ((p_o - t_o)**2).sum()
    ss_tot = ((t_o - t_o.mean())**2).sum()
    r2     = 1 - ss_res / (ss_tot + eps)
    corr   = float(np.corrcoef(p_o, t_o)[0,1]) if len(p_o)>1 else 0.
    dp, dt = np.diff(p), np.diff(t)
    dir_acc = (np.sign(dp)==np.sign(dt)).mean()*100

    return dict(mae=mae, mse=mse, rmse=rmse, smape=smape,
                r2=r2, corr=corr, dir_acc=dir_acc)


# ─────────────────────────────────────────────────────────────
# WMRT helpers (reuse from v2)
# ─────────────────────────────────────────────────────────────
def make_wmrt_adaptive(d_model, num_heads, seq_len, device, tau_init=0.1):
    dwt   = LearnableDWT1D(1, d_model).to(device)
    block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init).to(device)
    sc    = nn.Conv1d(1, 1, kernel_size=1).to(device)
    return dwt, block, sc

def wmrt_forward(dwt, block, sc, bx, zero_lh=False):
    LL, LH = dwt(bx)
    if zero_lh: LH = torch.zeros_like(LH)
    LL_o, LH_o = block(LL, LH)
    return sc(dwt.inverse(LL_o, LH_o))

def get_adaptive_tau(epoch, total, tau_start=0.5, tau_end=0.05):
    p = (epoch-1)/max(total-1,1)
    return tau_end + (tau_start-tau_end)*0.5*(1+math.cos(math.pi*p))


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 96
HORIZONS   = [1, 96, 192, 336, 720]
BATCH_SIZE = 64
D_MODEL    = 64       # raised to match baseline scale
NUM_HEADS  = 4
EPOCHS     = 30
LR         = 1e-3
FILE_PATH  = 'ETTm1.csv'

print(f"Device: {DEVICE}")
print(f"Horizons: {HORIZONS}  |  d_model={D_MODEL}  |  epochs={EPOCHS}\n")

crit_mse  = nn.MSELoss()
crit_wmrt = WaveletTransformerLoss(lambda_recon=1.0, lambda_ortho=0.1)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP — over all horizons
# ─────────────────────────────────────────────────────────────
# results[horizon][model_name] = metrics_dict
all_results   = {}
all_histories = {}

for PRED_LEN in HORIZONS:
    print(f"\n{'='*70}")
    print(f"  HORIZON = {PRED_LEN}")
    print(f"{'='*70}")

    # ── DataLoaders ──────────────────────────────────────────
    train_ds = ETTDataset(SEQ_LEN, PRED_LEN, 'train', FILE_PATH)
    val_ds   = ETTDataset(SEQ_LEN, PRED_LEN, 'val',   FILE_PATH)
    test_ds  = ETTDataset(SEQ_LEN, PRED_LEN, 'test',  FILE_PATH)
    inv_fn   = test_ds.inverse

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, drop_last=True)
    print(f"  Samples — Train:{len(train_ds):,} Val:{len(val_ds):,} Test:{len(test_ds):,}")

    # ── Instantiate all models ───────────────────────────────
    models = {
        'WMRT_Adaptive': None,   # handled separately
        'DLinear':    DLinear(SEQ_LEN, PRED_LEN).to(DEVICE),
        'NLinear':    NLinear(SEQ_LEN, PRED_LEN).to(DEVICE),
        'PatchTST':   PatchTST(SEQ_LEN, PRED_LEN, d_model=D_MODEL,
                               n_heads=NUM_HEADS).to(DEVICE),
        'iTransformer': iTransformer(SEQ_LEN, PRED_LEN, d_model=D_MODEL,
                                     n_heads=NUM_HEADS).to(DEVICE),
        'TimesNet':   TimesNet(SEQ_LEN, PRED_LEN, d_model=D_MODEL//2).to(DEVICE),
        'Vanilla':    VanillaTransformerBaseline(
                          in_channels=1, d_model=D_MODEL,
                          num_heads=NUM_HEADS, seq_len=SEQ_LEN).to(DEVICE),
    }

    # WMRT Adaptive τ
    dwt, wmrt_block, wmrt_sc = make_wmrt_adaptive(D_MODEL, NUM_HEADS, SEQ_LEN, DEVICE)

    # Print param counts (once, at first horizon)
    if PRED_LEN == HORIZONS[0]:
        wmrt_p = sum(p.numel() for m in [dwt,wmrt_block,wmrt_sc] for p in m.parameters())
        print(f"\n  Param counts:")
        print(f"    WMRT_Adaptive : {wmrt_p:,}")
        for name, m in models.items():
            if m is not None:
                print(f"    {name:<14}: {sum(p.numel() for p in m.parameters()):,}")

    # Optimisers
    opt_wmrt = optim.AdamW(
        list(dwt.parameters())+list(wmrt_block.parameters())+list(wmrt_sc.parameters()),
        lr=LR, weight_decay=1e-4)
    opts = {name: optim.AdamW(m.parameters(), lr=LR)
            for name, m in models.items() if m is not None}

    # Training history
    history = {name: {'train':[], 'val':[]} for name in list(models.keys())+['WMRT_Adaptive']}

    # ── Training ─────────────────────────────────────────────
    for epoch in range(1, EPOCHS+1):
        tau = get_adaptive_tau(epoch, EPOCHS)

        # WMRT Adaptive
        dwt.train(); wmrt_block.train(); wmrt_sc.train()
        wmrt_block.sparsity_tau = tau
        w_loss = w_n = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt_wmrt.zero_grad()
            LL, LH      = dwt(bx)
            LL_o, LH_o  = wmrt_block(LL, LH)
            preds       = wmrt_sc(dwt.inverse(LL_o, LH_o))
            xr          = dwt.inverse(LL, LH)
            L           = min(preds.shape[-1], by.shape[-1])
            loss, tl, *_ = crit_wmrt(preds[..., :L], by[..., :L], bx, xr, dwt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(dwt.parameters())+list(wmrt_block.parameters())+list(wmrt_sc.parameters()), 1.0)
            opt_wmrt.step()
            w_loss += tl.item(); w_n += 1
        history['WMRT_Adaptive']['train'].append(w_loss/w_n)

        # Standard models
        for name, model in models.items():
            if model is None: continue
            model.train()
            t_loss = t_n = 0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opts[name].zero_grad()
                pred = model(bx)
                L    = min(pred.shape[-1], by.shape[-1])
                loss = crit_mse(pred[..., :L], by[..., :L])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opts[name].step()
                t_loss += loss.item(); t_n += 1
            history[name]['train'].append(t_loss/t_n)

        # Validation
        dwt.eval(); wmrt_block.eval(); wmrt_sc.eval()
        def _wmrt_fn(bx):
            LL, LH = dwt(bx); LL_o,LH_o = wmrt_block(LL,LH)
            return wmrt_sc(dwt.inverse(LL_o,LH_o))
        v = evaluate(_wmrt_fn, val_loader, DEVICE, inv_fn)
        history['WMRT_Adaptive']['val'].append(v['mse'])

        for name, model in models.items():
            if model is None: continue
            model.eval()
            v = evaluate(lambda bx, m=model: m(bx), val_loader, DEVICE, inv_fn)
            history[name]['val'].append(v['mse'])

        if epoch % 10 == 0 or epoch == 1:
            wmrt_v = history['WMRT_Adaptive']['val'][-1]
            dlin_v = history['DLinear']['val'][-1]
            ptst_v = history['PatchTST']['val'][-1]
            print(f"  ep{epoch:>3}  WMRT:{wmrt_v:.5f}  DLinear:{dlin_v:.5f}  PatchTST:{ptst_v:.5f}")

    # ── Test evaluation ──────────────────────────────────────
    dwt.eval(); wmrt_block.eval(); wmrt_sc.eval()
    horizon_results = {}
    horizon_results['WMRT_Adaptive'] = evaluate(_wmrt_fn, test_loader, DEVICE, inv_fn)
    for name, model in models.items():
        if model is None: continue
        model.eval()
        horizon_results[name] = evaluate(lambda bx, m=model: m(bx), test_loader, DEVICE, inv_fn)

    all_results[PRED_LEN]   = horizon_results
    all_histories[PRED_LEN] = history

    # Print horizon summary
    model_names = ['WMRT_Adaptive','DLinear','NLinear','PatchTST','iTransformer','TimesNet','Vanilla']
    print(f"\n  Horizon {PRED_LEN} Test Results:")
    print(f"  {'Model':<16} {'MAE':>8} {'MSE':>8} {'RMSE':>8} {'sMAPE%':>8} {'R2':>7} {'DirAcc%':>9}")
    print(f"  {'-'*66}")
    for name in model_names:
        r = horizon_results[name]
        marker = ' <--' if name == 'WMRT_Adaptive' else ''
        print(f"  {name:<16} {r['mae']:>8.4f} {r['mse']:>8.4f} {r['rmse']:>8.4f} "
              f"{r['smape']:>8.2f} {r['r2']:>7.4f} {r['dir_acc']:>9.2f}{marker}")


# ─────────────────────────────────────────────────────────────
# SUMMARY TABLE — MAE and DirAcc across all horizons
# ─────────────────────────────────────────────────────────────
model_names = ['WMRT_Adaptive','DLinear','NLinear','PatchTST','iTransformer','TimesNet','Vanilla']

print(f"\n\n{'='*90}")
print(f"{'FULL BENCHMARK SUMMARY — MAE across all horizons (lower is better)':^90}")
print(f"{'='*90}")
header = f"{'Model':<16}" + "".join(f"H={h:>4}" for h in HORIZONS) + "   AVG"
print(header); print('-'*90)
for name in model_names:
    vals = [all_results[h][name]['mae'] for h in HORIZONS]
    avg  = np.mean(vals)
    row  = f"{name:<16}" + "".join(f"{v:>8.4f}" for v in vals) + f"  {avg:>6.4f}"
    print(row)

print(f"\n{'='*90}")
print(f"{'FULL BENCHMARK SUMMARY — DirAcc% across all horizons (higher is better)':^90}")
print(f"{'='*90}")
print(header.replace('MAE','Dir')); print('-'*90)
for name in model_names:
    vals = [all_results[h][name]['dir_acc'] for h in HORIZONS]
    avg  = np.mean(vals)
    row  = f"{name:<16}" + "".join(f"{v:>8.2f}" for v in vals) + f"  {avg:>6.2f}"
    print(row)

print(f"\n{'='*90}")
print(f"{'FULL BENCHMARK SUMMARY — R2 across all horizons (higher is better)':^90}")
print(f"{'='*90}")
print(header.replace('MAE','R2 ')); print('-'*90)
for name in model_names:
    vals = [all_results[h][name]['r2'] for h in HORIZONS]
    avg  = np.mean(vals)
    row  = f"{name:<16}" + "".join(f"{v:>8.4f}" for v in vals) + f"  {avg:>6.4f}"
    print(row)

# Win count
print(f"\n{'='*90}")
print("WIN COUNT (how many horizon/metric combinations each model wins)")
print('-'*90)
win_counts = {n: 0 for n in model_names}
for h in HORIZONS:
    for metric, higher in [('mae',False),('mse',False),('rmse',False),
                            ('smape',False),('r2',True),('dir_acc',True)]:
        vals  = {n: all_results[h][n][metric] for n in model_names}
        best  = max(vals.values()) if higher else min(vals.values())
        winner = [n for n,v in vals.items() if abs(v-best)<1e-9]
        for w in winner: win_counts[w] += 1
total_comps = len(HORIZONS) * 6
for name in sorted(model_names, key=lambda n: -win_counts[n]):
    bar = '█' * win_counts[name]
    print(f"  {name:<16} {win_counts[name]:>3}/{total_comps}  {bar}")


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
COLORS = {
    'WMRT_Adaptive': '#2563eb',
    'DLinear':       '#16a34a',
    'NLinear':       '#65a30d',
    'PatchTST':      '#dc2626',
    'iTransformer':  '#9333ea',
    'TimesNet':      '#ea580c',
    'Vanilla':       '#94a3b8',
}
MARKERS = {n: m for n,m in zip(model_names, ['o','s','D','^','v','P','x'])}

fig = plt.figure(figsize=(20, 16))
fig.suptitle("WMRT vs All Baselines — ETTm1 Multi-Horizon Benchmark",
             fontsize=15, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

# Row 0: MAE / DirAcc / R2 vs horizon
for ax_col, (metric, label, higher) in enumerate([
        ('mae',     'MAE (lower=better)',     False),
        ('dir_acc', 'DirAcc% (higher=better)',True),
        ('r2',      'R2 (higher=better)',     True),
]):
    ax = fig.add_subplot(gs[0, ax_col])
    for name in model_names:
        vals = [all_results[h][name][metric] for h in HORIZONS]
        lw   = 2.5 if name == 'WMRT_Adaptive' else 1.4
        zo   = 5   if name == 'WMRT_Adaptive' else 2
        ax.plot(HORIZONS, vals, marker=MARKERS[name], color=COLORS[name],
                linewidth=lw, markersize=6, label=name, zorder=zo)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Forecast Horizon')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

# Row 1: Training curves for H=96 and H=336
for ax_col, h in enumerate([96, 336, 720]):
    ax = fig.add_subplot(gs[1, ax_col])
    if h in all_histories:
        ep = range(1, EPOCHS+1)
        for name in model_names:
            hist = all_histories[h][name]['val']
            lw   = 2.2 if name == 'WMRT_Adaptive' else 1.2
            ax.plot(ep, hist, color=COLORS[name], linewidth=lw,
                    label=name, alpha=0.9)
    ax.set_title(f'Val MSE — H={h}', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

# Row 2: Bar chart — MAE and DirAcc at H=96, 336, 720
plot_horizons = [96, 336, 720]
for ax_col, h in enumerate(plot_horizons):
    ax = fig.add_subplot(gs[2, ax_col])
    names_short = [n.replace('_Adaptive','') for n in model_names]
    mae_vals    = [all_results[h][n]['mae']     for n in model_names]
    dir_vals    = [all_results[h][n]['dir_acc'] for n in model_names]
    x   = np.arange(len(model_names))
    w   = 0.4
    ax2 = ax.twinx()
    bars1 = ax.bar(x - w/2, mae_vals, w, color=[COLORS[n] for n in model_names],
                   alpha=0.75, label='MAE')
    bars2 = ax2.bar(x + w/2, dir_vals, w, color=[COLORS[n] for n in model_names],
                    alpha=0.4, label='DirAcc%')
    ax.set_xticks(x)
    ax.set_xticklabels(names_short, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('MAE', fontsize=8)
    ax2.set_ylabel('DirAcc%', fontsize=8)
    ax.set_title(f'MAE & DirAcc — H={h}', fontsize=10)
    # Highlight WMRT bar
    wmrt_idx = model_names.index('WMRT_Adaptive')
    bars1[wmrt_idx].set_edgecolor('black'); bars1[wmrt_idx].set_linewidth(2)
    ax.grid(alpha=0.2, axis='y')

plt.savefig('wmrt_full_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved -> wmrt_full_baseline.png")
print("\nDone. Check win counts above to assess publishability.")