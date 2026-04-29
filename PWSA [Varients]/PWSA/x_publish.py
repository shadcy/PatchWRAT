# =============================================================================
# benchmark_wrat_multivariate.py  —  Wavelet Multiresolution Transformer (WMRT)
# =============================================================================
# Goal: Evaluate multivariate PatchWRAT on ETTm1 with rigorous ablation studies.
# Features:
#   • Multivariate time-series forecasting (Channel Independence approach)
#   • Full Ablation Suite (w/o RevIN, w/o Ortho Loss, w/o Sparse Attn)
#   • Automated test-set visualizations (plots/ directory)
#   • Same data splits (60/20/20) and horizons [96, 192, 336, 720]
#
# Usage:
#   python benchmark_wrat_multivariate.py
#   (ETTm1.csv must be in the same directory)
# =============================================================================

from __future__ import annotations
import math, warnings, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)

CSV_PATH     = r'C:\Users\Asus\Desktop\TTS\Rat\ETTm1.csv'
SEQ_LEN      = 336
HORIZONS     = [96, 192, 336, 720]
BATCH_SIZE   = 64
EPOCHS       = 30
PATIENCE     = 10
LR           = 3e-4

_temp_df     = pd.read_csv(CSV_PATH, nrows=0, encoding='unicode_escape')
CHANNELS     = _temp_df.columns.tolist()[1:]  # skip date column
NUM_CHANNELS = len(CHANNELS)

print(f"[ENV]  Device : {DEVICE} | PyTorch {torch.__version__}")
print(f"[DATA] {CSV_PATH}  seq={SEQ_LEN}  horizons={HORIZONS}")
print("\n[WMRT Framework] Initializing Wavelet Multiresolution Transformer ablation suite...")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _splits(n):
    te = max(1, min(int(n * 0.60), n - 2))
    ve = max(te + 1, min(int(n * 0.80), n - 1))
    return te, ve

class ETTDataset(Dataset):
    """Loads ETTm1.csv returning all 7 variates for multivariate forecasting."""
    def __init__(self, seq_len, pred_len, split='train'):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        df   = pd.read_csv(CSV_PATH, encoding='unicode_escape')
        
        # Grab the 7 features
        raw_all = df[CHANNELS].values.astype(np.float32)

        n       = len(raw_all)
        te, ve  = _splits(n)
        seg     = {'train': raw_all[:te], 'val': raw_all[te:ve], 'test': raw_all[ve:]}[split]
        sc      = StandardScaler().fit(raw_all[:te])
        self.data = torch.tensor(sc.transform(seg), dtype=torch.float32)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, i):
        x = self.data[i           : i + self.seq_len]
        y = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]
        return x.t(), y.t()   # (C, T)

def get_loaders(pred_len):
    kw = dict(num_workers=0, pin_memory=False)
    tr = DataLoader(ETTDataset(SEQ_LEN, pred_len, 'train'), BATCH_SIZE, shuffle=True,  drop_last=True,  **kw)
    va = DataLoader(ETTDataset(SEQ_LEN, pred_len, 'val'),   BATCH_SIZE, shuffle=False, drop_last=True,  **kw)
    te = DataLoader(ETTDataset(SEQ_LEN, pred_len, 'test'),  BATCH_SIZE, shuffle=False, drop_last=False, **kw)
    return tr, va, te

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best       = None
        self.early_stop = False
        self.state      = None

    def __call__(self, val_loss, model):
        improved = self.best is None or val_loss < self.best - self.min_delta
        if improved:
            self.best    = val_loss
            self.counter = 0
            self.state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        preds, _, _, _ = model(bx)
        all_p.append(preds.cpu())
        all_t.append(by.cpu())
    p = torch.cat(all_p).flatten().numpy()
    t = torch.cat(all_t).flatten().numpy()
    e = p - t
    return dict(mse=float((e**2).mean()), mae=float(np.abs(e).mean()),
                dir_acc=float((np.sign(np.diff(p)) == np.sign(np.diff(t))).mean() * 100))

def make_scheduler(opt, epochs=EPOCHS):
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5)
    cos    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 5), eta_min=LR * 0.02)
    return warmup, cos

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(model, loader, horizon, num_samples=4, out_dir="plots"):
    """Plots multivariate predictions vs ground truth for specific samples."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    bx, by = next(iter(loader))
    bx, by = bx.to(DEVICE), by.to(DEVICE)
    
    with torch.no_grad():
        preds, _, _, _ = model(bx)

    bx = bx.cpu().numpy()
    by = by.cpu().numpy()
    preds = preds.cpu().numpy()

    t_hist = np.arange(SEQ_LEN)
    t_pred = np.arange(SEQ_LEN, SEQ_LEN + horizon)

    for i in range(min(num_samples, bx.shape[0])):
        fig, axes = plt.subplots(NUM_CHANNELS, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f"WMRT Prediction vs Ground Truth | Horizon: {horizon} | Sample: {i+1}", fontsize=16)

        for c, ax in enumerate(axes):
            ax.plot(t_hist, bx[i, c, :], label='History (Input)', color='dimgray', linewidth=1)
            ax.plot(t_pred, by[i, c, :], label='Ground Truth', color='blue', linewidth=1.5)
            ax.plot(t_pred, preds[i, c, :], label='PatchWRAT Pred', color='red', linestyle='--', linewidth=1.5)
            ax.set_ylabel(CHANNELS[c], fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if c == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

        plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.85, 0.97]) # adjust for legend
        
        filepath = os.path.join(out_dir, f'wrat_pred_H{horizon}_sample{i+1}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"    [Visuals] Saved {num_samples} multivariate plots to '{out_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# MULTIVARIATE PATCHWRAT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(1, num_features, 1))
        self.b   = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, mode):
        # x shape: (B, C, L)
        if mode == 'norm':
            self.mean  = x.mean(-1, keepdim=True).detach()
            self.stdev = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            return ((x - self.mean) / self.stdev) * self.w + self.b
        return ((x - self.b) / self.w) * self.stdev + self.mean

class LearnableDWT1D(nn.Module):
    def __init__(self, in_ch, out_ch, filter_len=4):
        super().__init__()
        self.h   = nn.Parameter(torch.randn(out_ch, in_ch, filter_len) * 0.1)
        self.g   = nn.Parameter(torch.randn(out_ch, in_ch, filter_len) * 0.1)
        self.pad = (filter_len - 2) // 2
        with torch.no_grad():
            self.g -= self.g.mean(dim=-1, keepdim=True)

    def forward(self, x):
        return (F.conv1d(x, self.h, stride=2, padding=self.pad),
                F.conv1d(x, self.g, stride=2, padding=self.pad))

    def inverse(self, LL, LH):
        rL = F.conv_transpose1d(LL, self.h, stride=2, padding=self.pad)
        rH = F.conv_transpose1d(LH, self.g, stride=2, padding=self.pad)
        n  = min(rL.shape[-1], rH.shape[-1])
        return rL[..., :n] + rH[..., :n]

class FreqSparseAttn(nn.Module):
    def __init__(self, d_model, num_heads, threshold=0.1):
        super().__init__()
        self.H         = num_heads
        self.threshold = threshold
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, q_x, k_x, v_x, energy=None, use_sparse=True):
        B, L, D = q_x.shape; Dh = D // self.H
        Q = self.q(q_x).view(B, L, self.H, Dh).transpose(1, 2)
        K = self.k(k_x).view(B, -1, self.H, Dh).transpose(1, 2)
        V = self.v(v_x).view(B, -1, self.H, Dh).transpose(1, 2)
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        
        if use_sparse and energy is not None:
            mask = torch.abs(energy).mean(-1).view(B, 1, 1, -1) > self.threshold
            sc   = sc.masked_fill(~mask, float('-inf'))
            
        w = torch.nan_to_num(F.softmax(sc, dim=-1), nan=0.0)
        o = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.o(o)

class WRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, tau=0.1, dropout=0.2):
        super().__init__()
        self.raw_tau     = nn.Parameter(torch.tensor(math.log(max(tau, 1e-6) / max(1.0 - tau, 1e-6))))
        self.ll_attn     = FreqSparseAttn(d_model, num_heads)
        self.lh_attn     = FreqSparseAttn(d_model, num_heads, threshold=tau)
        self.cross_attn  = FreqSparseAttn(d_model, num_heads)
        self.mlp_ll      = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.mlp_lh      = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.n1          = nn.LayerNorm(d_model)
        self.n2          = nn.LayerNorm(d_model)

    def forward(self, LL, LH, use_sparse=True):
        tau = torch.sigmoid(self.raw_tau).item()
        self.lh_attn.threshold = tau
        ll_s = LL.transpose(1, 2); lh_s = LH.transpose(1, 2)
        
        ll_o = self.ll_attn(ll_s, ll_s, ll_s, use_sparse=False) # LL is always dense
        lh_o = self.lh_attn(lh_s, lh_s, lh_s, energy=lh_s, use_sparse=use_sparse)
        cr_o = self.cross_attn(ll_o, lh_o, lh_o, use_sparse=False)
        
        ll_f = self.n1(ll_s + ll_o + cr_o);  ll_f = self.mlp_ll(ll_f) + ll_f
        lh_f = self.n2(lh_s + lh_o);         lh_f = self.mlp_lh(lh_f) + lh_f
        return ll_f.transpose(1, 2), lh_f.transpose(1, 2)

class PatchWRAT(nn.Module):
    def __init__(self, seq_len, pred_len, num_channels=7, d_model=32, num_heads=4, 
                 dropout=0.3, use_revin=True, use_sparse=False):
        super().__init__()
        self.use_revin  = use_revin
        self.use_sparse = use_sparse
        self.pred_len   = pred_len
        self.num_ch     = num_channels

        if self.use_revin:
            self.revin = RevIN(num_channels)
            
        self.dwt  = LearnableDWT1D(1, d_model)
        self.wrat = WRATBlock(d_model, num_heads, dropout=dropout)
        
        l_half    = seq_len // 2
        flat_dim  = l_half * d_model * 2
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, pred_len)
        )
        self.sc   = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        B, C, L = x.shape
        
        if self.use_revin:
            x = self.revin(x, 'norm')

        # Channel Independence: merge Batch and Channel dims
        x_ci = x.reshape(B * C, 1, L)
        
        LL, LH     = self.dwt(x_ci)
        LL_o, LH_o = self.wrat(LL, LH, use_sparse=self.use_sparse)
        
        fused      = torch.cat([LL_o, LH_o], dim=1)
        preds      = self.head(fused).unsqueeze(1) # (B*C, 1, pred_len)
        
        # Reshape back to multivariate dims
        preds = preds.reshape(B, C, self.pred_len)
        xr    = self.sc(self.dwt.inverse(LL_o, LH_o)).reshape(B, C, -1)
        LL    = LL.reshape(B, C, -1)
        LH    = LH.reshape(B, C, -1)

        if self.use_revin:
            preds = self.revin(preds, 'denorm')
            xr    = self.revin(xr, 'denorm')

        return preds, xr, LL, LH

def wrat_loss(preds, targets, x_orig, x_recon, dwt, lam_r=1.0, lam_o=0.1):
    task  = F.mse_loss(preds, targets)
    n     = min(x_orig.shape[-1], x_recon.shape[-1])
    recon = F.mse_loss(x_recon[..., :n], x_orig[..., :n])
    
    if lam_o > 0.0:
        h_f   = dwt.h.view(dwt.h.shape[0], -1)
        g_f   = dwt.g.view(dwt.g.shape[0], -1)
        ortho = (h_f * g_f).sum().abs()
    else:
        ortho = 0.0
        
    return task + lam_r * recon + lam_o * ortho, task.item()

def train_wrat(model, loader, opt, lam_o):
    model.train()
    losses = []
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        preds, xr, _, _ = model(bx)
        loss, mse = wrat_loss(preds, by, bx, xr, model.dwt, lam_o=lam_o)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(mse)
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(name, model_params, lam_o, tr_loader, va_loader, te_loader):
    torch.manual_seed(SEED)
    model = PatchWRAT(**model_params).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warmup, cos = make_scheduler(opt, EPOCHS)
    es    = EarlyStopping(PATIENCE)

    print(f"\n  [{name}] Training initiated...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_wrat(model, tr_loader, opt, lam_o=lam_o)
        val_m   = evaluate(model, va_loader, DEVICE)['mse']
        if epoch <= 5: warmup.step()
        else:          cos.step()
        improved = es(val_m, model)
        
        # Suppressing excessive logs for cleanliness, print every 10 or best
        if epoch % 10 == 0 or improved:
            flag = '★' if improved else ''
            print(f"    ep{epoch:>3} | train={tr_loss:.4f} | val={val_m:.4f} {flag}")
            
        if es.early_stop:
            print(f"    Early stop @ ep{epoch} | best_val={es.best:.4f}")
            break

    model.load_state_dict(es.state)
    return evaluate(model, te_loader, DEVICE), model

# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Ensure ETTm1.csv is mapped properly.")

ablation_results = {}

for PRED_LEN in HORIZONS:
    print(f"\n{'='*70}")
    print(f" HORIZON = {PRED_LEN}  |  seq={SEQ_LEN}  | Multivariate Channel Independence")
    print(f"{'='*70}")

    tr_loader, va_loader, te_loader = get_loaders(PRED_LEN)

    configs = {
        'w/o Sparse':   {'params': dict(seq_len=SEQ_LEN, pred_len=PRED_LEN, num_channels=NUM_CHANNELS, d_model=32, num_heads=4, use_revin=True,  use_sparse=False), 'lam_o': 0.1},
    }

    ablation_results[PRED_LEN] = {}
    
    for name, cfg in configs.items():
        res, trained_model = run_ablation(name, cfg['params'], cfg['lam_o'], tr_loader, va_loader, te_loader)
        ablation_results[PRED_LEN][name] = res
        
        # Plot predictions ONLY for the w/o Sparse variant to save time/space
        if name == 'w/o Sparse':
            plot_predictions(trained_model, te_loader, PRED_LEN, num_samples=4)

    # Print Ablation Summary Table for the current Horizon
    print(f"\n  ┌─ Horizon {PRED_LEN} Ablation Summary {'─'*30}┐")
    print(f"  │  {'Variant':<15} {'MSE':>8}  {'MAE':>8}  {'DirAcc':>8}          │")
    print(f"  │  {'─'*53}  │")
    for name in configs.keys():
        r = ablation_results[PRED_LEN][name]
        print(f"  │  {name:<15} {r['mse']:>8.4f}  {r['mae']:>8.4f}  {r['dir_acc']:>7.1f}%          │")
    print(f"  └{'─'*57}┘")

print(f"\n{'='*70}\n[WMRT Framework] Benchmark complete. Plots saved in local directory.\n{'='*70}")