# ============================================================
# run_predict.py  —  WRAT Model Prediction Test (ETTh1)
# ============================================================
# Variants : Fixed tau ablation across 5 values
# Horizons : 1, 96, 192, 336, 720
# Metrics  : MAE, MSE, RMSE, sMAPE, R2, Pearson, DirAcc%
# ============================================================

import math, warnings, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SECTION 1 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None
        self.std  = None

    def normalize(self, x):
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std  = x.std(dim=-1, keepdim=True) + self.eps
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class LearnableDWT1D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_length=4):
        super().__init__()
        self.h = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.filter_length = filter_length
        self.padding_val   = (filter_length - 2) // 2
        with torch.no_grad():
            self.g -= self.g.mean(dim=-1, keepdim=True)

    def forward(self, x):
        LL = F.conv1d(x, self.h, stride=2, padding=self.padding_val)
        LH = F.conv1d(x, self.g, stride=2, padding=self.padding_val)
        return LL, LH

    def inverse(self, LL, LH):
        x_recon_L = F.conv_transpose1d(LL, self.h, stride=2, padding=self.padding_val)
        x_recon_H = F.conv_transpose1d(LH, self.g, stride=2, padding=self.padding_val)
        min_len = min(x_recon_L.shape[-1], x_recon_H.shape[-1])
        return x_recon_L[..., :min_len] + x_recon_H[..., :min_len]


class FrequencySparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, threshold=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model   = d_model
        self.threshold = threshold
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q_x, k_x, v_x, energy_coeffs=None):
        B, L, D = q_x.shape
        H   = self.num_heads
        D_h = D // H
        Q = self.q_proj(q_x).view(B, L, H, D_h).transpose(1, 2)
        K = self.k_proj(k_x).view(B, -1, H, D_h).transpose(1, 2)
        V = self.v_proj(v_x).view(B, -1, H, D_h).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D_h ** 0.5)
        if energy_coeffs is not None:
            energy = torch.abs(energy_coeffs).mean(dim=-1)            
            energy = energy / (energy.max(dim=-1, keepdim=True).values + 1e-8)
            gate   = (energy > self.threshold).float() * 0.9 + 0.1   # [0.1, 1.0]
            scores = scores * gate.view(B, 1, 1, -1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class WRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_tau=0.1):
        super().__init__()
        self.sparsity_tau  = sparsity_tau
        self.intra_LL_attn = FrequencySparseAttention(d_model, num_heads)
        self.intra_LH_attn = FrequencySparseAttention(d_model, num_heads, threshold=sparsity_tau)
        self.cross_attn    = FrequencySparseAttention(d_model, num_heads)
        self.mlp_LL = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.mlp_LH = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)

    def forward(self, LL, LH):
        LL_seq    = LL.transpose(1, 2)
        LH_seq    = LH.transpose(1, 2)
        LL_out    = self.intra_LL_attn(LL_seq, LL_seq, LL_seq)
        LH_out    = self.intra_LH_attn(LH_seq, LH_seq, LH_seq, energy_coeffs=LH_seq)
        cross_out = self.cross_attn(LL_out, LH_out, LH_out)
        LL_fused  = self.norm1(LL_seq + LL_out + cross_out)
        LH_fused  = self.norm2(LH_seq + LH_out)
        LL_final  = self.mlp_LL(LL_fused) + LL_fused
        LH_final  = self.mlp_LH(LH_fused) + LH_fused
        return LL_final.transpose(1, 2), LH_final.transpose(1, 2)


class WaveletTransformerLoss(nn.Module):
    def __init__(self, lambda_recon=0.1, lambda_ortho=0.1, lambda_dir=0.05):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho
        self.lambda_dir   = lambda_dir

    def forward(self, preds, targets, x_orig, x_recon, dwt_layer):
        task_loss  = F.mse_loss(preds, targets)
        recon_loss = F.mse_loss(x_recon, x_orig) if x_recon is not None else 0.0
        h_flat     = dwt_layer.h.view(dwt_layer.h.shape[0], -1)
        g_flat     = dwt_layer.g.view(dwt_layer.g.shape[0], -1)
        ortho_loss = (h_flat * g_flat).sum().abs()
        if preds.shape[-1] > 1:
            dp = torch.diff(preds,   dim=-1)
            dt = torch.diff(targets, dim=-1)
            dir_loss = F.mse_loss(torch.tanh(dp * 10), torch.tanh(dt * 10))
        else:
            dir_loss = torch.tensor(0.0, device=preds.device)
        total = (task_loss
                 + self.lambda_recon * recon_loss
                 + self.lambda_ortho * ortho_loss
                 + self.lambda_dir   * dir_loss)
        return total, task_loss, recon_loss, ortho_loss


class WRATModel(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, pred_len, tau_init=0.1):
        super().__init__()
        self.revin = RevIN()
        self.dwt   = LearnableDWT1D(1, d_model)
        self.block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init)
        self.proj  = nn.Linear(seq_len, pred_len)

    def forward(self, x, zero_lh=False):
        x_norm       = self.revin.normalize(x)
        LL, LH       = self.dwt(x_norm)
        if zero_lh:
            LH = torch.zeros_like(LH)
        LL_o, LH_o   = self.block(LL, LH)
        recon        = self.dwt.inverse(LL_o, LH_o)
        out          = self.proj(recon)
        return self.revin.denormalize(out)

    def forward_with_recon(self, x, zero_lh=False):
        x_norm       = self.revin.normalize(x)
        LL, LH       = self.dwt(x_norm)
        x_recon_raw  = self.dwt.inverse(LL.clone(), LH.clone())
        if zero_lh:
            LH = torch.zeros_like(LH)
        LL_o, LH_o   = self.block(LL, LH)
        recon        = self.dwt.inverse(LL_o, LH_o)
        out          = self.proj(recon)
        out          = self.revin.denormalize(out)
        return out, x_recon_raw, x_norm


# ══════════════════════════════════════════════════════════════
# SECTION 2 — DATASET 
# ══════════════════════════════════════════════════════════════

class ETThDataset(Dataset):
    def __init__(self, seq_len=96, pred_len=96, split='train',
                 file_path='ETTh1.csv', target_col='OT'):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        df   = pd.read_csv(file_path)
        data = df[target_col].values.reshape(-1, 1)

        train_end = 12 * 30 * 24        # 8,640
        val_end   = train_end + 4 * 30 * 24    # 11,520

        raw = {'train': data[:train_end],
               'val':   data[train_end:val_end],
               'test':  data[val_end:]}[split]

        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])
        self.data = torch.tensor(
            self.scaler.transform(raw), dtype=torch.float32)

    def inverse(self, x_norm):
        return x_norm * self.scaler.scale_[0] + self.scaler.mean_[0]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx):
        x = self.data[idx                : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x.t(), y.t() 


# ══════════════════════════════════════════════════════════════
# SECTION 3 — EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(model, loader, device, inv_fn=None, zero_lh=False):
    all_p, all_t = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred   = model(bx, zero_lh=zero_lh)
            L      = min(pred.shape[-1], by.shape[-1])
            all_p.append(pred[..., :L].cpu())
            all_t.append(by[...,   :L].cpu())

    p = torch.cat(all_p, 0).flatten().numpy()
    t = torch.cat(all_t, 0).flatten().numpy()

    err  = p - t
    mae  = float(np.abs(err).mean())
    mse  = float((err**2).mean())
    rmse = float(mse**0.5)

    p_o = inv_fn(p) if inv_fn else p
    t_o = inv_fn(t) if inv_fn else t
    eps    = 1e-8
    smape  = float((2*np.abs(p_o-t_o)/(np.abs(p_o)+np.abs(t_o)+eps)).mean()*100)
    ss_res = float(((p_o-t_o)**2).sum())
    ss_tot = float(((t_o-t_o.mean())**2).sum())
    r2     = float(1 - ss_res/(ss_tot+eps))
    corr   = float(np.corrcoef(p_o, t_o)[0,1]) if len(p_o)>1 else 0.0
    dp, dt = np.diff(p), np.diff(t)
    dir_acc = float((np.sign(dp)==np.sign(dt)).mean()*100)

    return dict(mae=mae, mse=mse, rmse=rmse, smape=smape,
                r2=r2, corr=corr, dir_acc=dir_acc)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — CONFIG (Updated for 5 Tau values)
# ══════════════════════════════════════════════════════════════

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE_PATH   = 'ETTh1.csv'
SEQ_LEN     = 96
HORIZONS    = [1, 96, 192, 336, 720]
BATCH_SIZE  = 64
D_MODEL     = 64
NUM_HEADS   = 4
EPOCHS      = 60
LR          = 1e-3
PATIENCE    = 15   
WARMUP_EP   = 5    

# The 5 Fixed Tau Values we want to test
TAU_VALUES  = [0.01, 0.05, 0.1, 0.2, 0.5]
MODEL_NAMES = [f'Tau_{t}' for t in TAU_VALUES]

crit_wmrt   = WaveletTransformerLoss(lambda_recon=0.1, lambda_ortho=0.1, lambda_dir=0.05)

print(f"Device  : {DEVICE}")
print(f"File    : {FILE_PATH}")
print(f"Horizons: {HORIZONS}")
print(f"Tau Vals: {TAU_VALUES}")
print(f"Epochs  : up to {EPOCHS} (patience={PATIENCE}, warmup={WARMUP_EP})")
print(f"d_model={D_MODEL}  |  batch={BATCH_SIZE}\n")


def make_model(tau_init, pred_len):
    m   = WRATModel(D_MODEL, NUM_HEADS, SEQ_LEN, pred_len, tau_init=tau_init).to(DEVICE)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    def lr_lambda(ep):
        if ep < WARMUP_EP:
            return (ep + 1) / WARMUP_EP          
        progress = (ep - WARMUP_EP) / max(EPOCHS - WARMUP_EP, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
    sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return m, opt, sch


def train_one_epoch(model, opt, loader):
    model.train()
    total_loss = n = 0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        preds, x_recon, x_norm = model.forward_with_recon(bx)
        L    = min(preds.shape[-1], by.shape[-1])
        loss, tl, *_ = crit_wmrt(preds[..., :L], by[..., :L],
                                   x_norm, x_recon, model.dwt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += tl.item(); n += 1
    return total_loss / n


# ══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN TRAINING + EVALUATION LOOP
# ══════════════════════════════════════════════════════════════

all_results   = {}
all_histories = {}

for PRED_LEN in HORIZONS:
    print(f"\n{'='*68}")
    print(f"  HORIZON = {PRED_LEN} steps")
    print(f"{'='*68}")

    train_ds = ETThDataset(SEQ_LEN, PRED_LEN, 'train', FILE_PATH)
    val_ds   = ETThDataset(SEQ_LEN, PRED_LEN, 'val',   FILE_PATH)
    test_ds  = ETThDataset(SEQ_LEN, PRED_LEN, 'test',  FILE_PATH)
    inv_fn   = test_ds.inverse

    trl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True)
    vll = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, drop_last=True)
    tel = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, drop_last=True)
    print(f"  Train:{len(train_ds):,}  Val:{len(val_ds):,}  Test:{len(test_ds):,}")

    models_info = []
    for tau, name in zip(TAU_VALUES, MODEL_NAMES):
        m, opt, sch = make_model(tau_init=tau, pred_len=PRED_LEN)
        models_info.append((name, m, opt, sch))

    if PRED_LEN == HORIZONS[0]:
        wp = sum(p.numel() for p in models_info[0][1].parameters())
        print(f"\n  WRAT parameters per variant: {wp:,}\n")

    hist       = {name: {'train': [], 'val': []} for name in MODEL_NAMES}
    best_val   = {name: float('inf') for name in MODEL_NAMES}
    best_ckpt  = {name: None         for name in MODEL_NAMES}
    no_improve = {name: 0            for name in MODEL_NAMES}
    stopped_at = {name: EPOCHS       for name in MODEL_NAMES}

    header_str = "  Ep " + "".join(f"{n:>12}" for n in MODEL_NAMES)
    print(header_str)
    print(f"  {'-'* (5 + 12 * len(MODEL_NAMES))}")

    for epoch in range(1, EPOCHS+1):
        for name, m, opt, sch in models_info:
            if no_improve[name] >= PATIENCE:
                hist[name]['val'].append(hist[name]['val'][-1])
                continue
            
            tl = train_one_epoch(m, opt, trl)
            sch.step()
            hist[name]['train'].append(tl)

            v = evaluate(m, vll, DEVICE, inv_fn)
            monitor = v['mae']
            hist[name]['val'].append(v['mse'])
            
            if monitor < best_val[name] - 1e-5:
                best_val[name]   = monitor
                best_ckpt[name]  = copy.deepcopy(m.state_dict())
                no_improve[name] = 0
            else:
                no_improve[name] += 1
                if no_improve[name] >= PATIENCE:
                    stopped_at[name] = epoch
                    print(f"  → {name} early stop @ epoch {epoch} "
                          f"(best val MAE={best_val[name]:.5f})")

        if epoch % 10 == 0 or epoch == 1:
            val_str = " ".join(f"{hist[n]['val'][-1]:>11.5f}" for n in MODEL_NAMES)
            print(f"  {epoch:>2} {val_str}")

        if all(no_improve[n] >= PATIENCE for n in MODEL_NAMES):
            print(f"  All models stopped. Moving to next horizon.")
            break

    # Restore best checkpoints
    for name, m, _, _ in models_info:
        if best_ckpt[name] is not None:
            m.load_state_dict(best_ckpt[name])

    hr = {}
    for name, m, _, _ in models_info:
        hr[name] = evaluate(m, tel, DEVICE, inv_fn)

    all_results[PRED_LEN]   = hr
    all_histories[PRED_LEN] = hist

    print(f"\n  Test results — Horizon {PRED_LEN}:")
    print(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'sMAPE%':>8} {'R2':>7} {'DirAcc%':>9} {'StopEp':>7}")
    print(f"  {'-'*65}")
    for name, m, _, _ in models_info:
        r = hr[name]
        print(f"  {name:<12} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
              f"{r['smape']:>8.2f} {r['r2']:>7.4f} {r['dir_acc']:>9.2f} "
              f"{stopped_at[name]:>7}")


# ══════════════════════════════════════════════════════════════
# SECTION 6 — SUMMARY TABLES
# ══════════════════════════════════════════════════════════════

def summary_table(metric, label, higher):
    print(f"\n{'='*76}")
    print(f"  {label}  ({'higher' if higher else 'lower'} is better)")
    print(f"{'='*76}")
    header = f"  {'Model':<12}" + "".join(f"  H={h:<6}" for h in HORIZONS) + "    AVG"
    print(header)
    print(f"  {'-'*72}")
    for name in MODEL_NAMES:
        vals = [all_results[h][name][metric] for h in HORIZONS]
        avg  = np.mean(vals)
        row  = f"  {name:<12}" + "".join(f"  {v:>7.4f}" for v in vals) + f"  {avg:>7.4f}"
        print(row)

summary_table('mae',     'MAE',     False)
summary_table('rmse',    'RMSE',    False)
summary_table('smape',   'sMAPE%',  False)
summary_table('r2',      'R2',      True)
summary_table('dir_acc', 'DirAcc%', True)


# ══════════════════════════════════════════════════════════════
# SECTION 7 — PLOTS
# ══════════════════════════════════════════════════════════════

# Generate styling maps dynamically for the 5 tau models
COLORS = dict(zip(MODEL_NAMES, ['#1d4ed8', '#059669', '#d97706', '#dc2626', '#7c3aed']))
LINES  = dict(zip(MODEL_NAMES, ['-', '--', '-.', ':', '-']))
MKR    = dict(zip(MODEL_NAMES, ['o', 's', '^', 'd', 'x']))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("WRAT Fixed Tau Ablation — ETTh1 Multi-Horizon Benchmark",
             fontsize=14, fontweight='bold')

for col, (metric, label) in enumerate([
        ('mae',     'MAE (lower = better)'),
        ('dir_acc', 'DirAcc% (higher = better)'),
        ('r2',      'R2 (higher = better)'),
]):
    ax = axes[0, col]
    for name in MODEL_NAMES:
        vals = [all_results[h][name][metric] for h in HORIZONS]
        ax.plot(HORIZONS, vals, marker=MKR[name], color=COLORS[name],
                linewidth=2.0, markersize=6, label=name, linestyle=LINES[name])
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Horizon')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

for col, h in enumerate([96, 336, 720]):
    ax = axes[1, col]
    if h in all_histories:
        for name in MODEL_NAMES:
            vals = all_histories[h][name]['val']
            ep_x = range(1, len(vals)+1)
            ax.plot(ep_x, vals, color=COLORS[name], linewidth=2.0,
                    label=name, linestyle=LINES[name], alpha=0.9)
    ax.set_title(f'Val MSE — H={h}', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('wrat_tau_ablation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved -> wrat_tau_ablation.png")
print("\nAll done.")