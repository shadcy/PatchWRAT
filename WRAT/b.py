# ============================================================
# run_benchmark.py  —  WMRT Full Publication Benchmark
# ============================================================
# Single file. No modifications needed. Just run:
#   python run_benchmark.py
#
# What this tests:
#   WMRT variants : Fixed tau, Adaptive tau, Learnable tau, Ablation (LH=0)
#   Baselines     : DLinear, NLinear, PatchTST, iTransformer, TimesNet, Vanilla
#   Horizons      : 1, 96, 192, 336, 720
#   Dataset       : ETTm1.csv  (must be in same folder)
#   Metrics       : MAE, MSE, RMSE, sMAPE, R2, Pearson, DirAcc%
# ============================================================

import math, warnings
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

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SECTION 1 — YOUR MODEL CLASSES  (pasted verbatim)
# ══════════════════════════════════════════════════════════════

class LearnableDWT1D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_length=4):
        super().__init__()
        self.h = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.filter_length = filter_length
        self.padding_val = (filter_length - 2) // 2
        with torch.no_grad():
            self.g -= self.g.mean(dim=-1, keepdim=True)

    def forward(self, x):
        # This specific padding calculation is designed to ensure that the output sequence length is exactly half of the input sequence length
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
        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q_x, k_x, v_x, energy_coeffs=None):
        B, L, D = q_x.shape
        H  = self.num_heads
        D_h = D // H
        Q = self.q_proj(q_x).view(B, L, H, D_h).transpose(1, 2)
        K = self.k_proj(k_x).view(B, -1, H, D_h).transpose(1, 2)
        V = self.v_proj(v_x).view(B, -1, H, D_h).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D_h ** 0.5)
        if energy_coeffs is not None:
            energy = torch.abs(energy_coeffs).mean(dim=-1)
            mask   = energy > self.threshold
            mask   = mask.view(B, 1, 1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class WRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_tau=0.1):
        super().__init__()
        self.sparsity_tau   = sparsity_tau
        self.intra_LL_attn  = FrequencySparseAttention(d_model, num_heads)
        self.intra_LH_attn  = FrequencySparseAttention(d_model, num_heads, threshold=sparsity_tau)
        self.cross_attn     = FrequencySparseAttention(d_model, num_heads)
        self.mlp_LL = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.mlp_LH = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)

    def forward(self, LL, LH):
        LL_seq = LL.transpose(1, 2)
        LH_seq = LH.transpose(1, 2)
        LL_out    = self.intra_LL_attn(LL_seq, LL_seq, LL_seq)
        LH_out    = self.intra_LH_attn(LH_seq, LH_seq, LH_seq, energy_coeffs=LH_seq)
        cross_out = self.cross_attn(LL_out, LH_out, LH_out)
        LL_fused  = self.norm1(LL_seq + LL_out + cross_out)
        LH_fused  = self.norm2(LH_seq + LH_out)
        LL_final  = self.mlp_LL(LL_fused) + LL_fused
        LH_final  = self.mlp_LH(LH_fused) + LH_fused
        return LL_final.transpose(1, 2), LH_final.transpose(1, 2)


class WaveletTransformerLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_ortho=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho

    def forward(self, preds, targets, x_orig, x_recon, dwt_layer):
        task_loss  = F.mse_loss(preds, targets)
        recon_loss = F.mse_loss(x_recon, x_orig) if x_recon is not None else 0.0
        # Orthogonality: penalise dot product between h and g filters
        h_flat = dwt_layer.h.view(dwt_layer.h.shape[0], -1)
        g_flat = dwt_layer.g.view(dwt_layer.g.shape[0], -1)
        ortho_loss = (h_flat * g_flat).sum().abs()
        total = task_loss + self.lambda_recon * recon_loss + self.lambda_ortho * ortho_loss
        return total, task_loss, recon_loss, ortho_loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class VanillaTransformerBaseline(nn.Module):
    def __init__(self, in_channels=1, d_model=16, num_heads=4, seq_len=128):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=2, stride=2)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.output_head = nn.Sequential(
            nn.ConvTranspose1d(d_model, in_channels, kernel_size=2, stride=2),
            nn.Conv1d(in_channels, in_channels, kernel_size=1))

    def forward(self, x):
        x_emb  = self.patch_embed(x)
        x_seq  = self.pos_encoder(x_emb.transpose(1, 2))
        out    = self.transformer_encoder(x_seq).transpose(1, 2)
        return self.output_head(out)


# ── Learnable-tau wrapper ────────────────────────────────────
class LearnableTauWRATBlock(nn.Module):
    """WRATBlock with tau as a trainable sigmoid-bounded parameter."""
    def __init__(self, d_model, num_heads, tau_init=0.1):
        super().__init__()
        self.raw_tau = nn.Parameter(torch.tensor(
            math.log(tau_init / (1.0 - tau_init))))   # inverse sigmoid
        self._block  = WRATBlock(d_model, num_heads, sparsity_tau=tau_init)

    @property
    def tau(self):
        return torch.sigmoid(self.raw_tau).item()

    def forward(self, LL, LH):
        self._block.intra_LH_attn.threshold = torch.sigmoid(self.raw_tau).item()
        return self._block(LL, LH)


# ══════════════════════════════════════════════════════════════
# SECTION 2 — BASELINE MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

class DLinear(nn.Module):
    """Zeng et al. 2023 — decomposition + two linear layers."""
    def __init__(self, seq_len, pred_len):
        super().__init__()
        kernel = 25
        self.avg  = nn.AvgPool1d(kernel_size=kernel, stride=1,
                                  padding=kernel//2, count_include_pad=False)
        self.lt   = nn.Linear(seq_len, pred_len)   # trend
        self.lr   = nn.Linear(seq_len, pred_len)   # residual

    def forward(self, x):                           # x: (B, C, L)
        trend    = self.avg(x)[..., :x.shape[-1]]
        residual = x - trend
        return self.lt(trend) + self.lr(residual)


class NLinear(nn.Module):
    """Zeng et al. 2023 — subtract last value, linear, add back."""
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        last = x[..., -1:]
        return self.linear(x - last) + last


class PatchTST(nn.Module):
    """Nie et al. 2023 — patch-based channel-independent transformer."""
    def __init__(self, seq_len, pred_len, d_model=64, n_heads=4,
                 n_layers=2, patch_len=16, stride=8, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.pred_len  = pred_len
        n_patches      = (seq_len - patch_len) // stride + 1
        self.embed     = nn.Linear(patch_len, d_model)
        self.pos       = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                  dim_feedforward=d_model*4, dropout=dropout,
                  batch_first=True, norm_first=True)
        self.encoder   = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head      = nn.Linear(n_patches * d_model, pred_len)
        self.drop      = nn.Dropout(dropout)
        self.n_patches = n_patches
        self.d_model   = d_model

    def forward(self, x):
        B, C, L = x.shape
        xp = x.reshape(B*C, 1, L).unfold(-1, self.patch_len, self.stride).squeeze(1)
        xp = self.drop(self.embed(xp) + self.pos)
        out = self.encoder(xp).reshape(B*C, -1)
        return self.head(out).reshape(B, C, self.pred_len)


class iTransformer(nn.Module):
    """Liu et al. 2024 — inverted attention (variates as tokens)."""
    def __init__(self, seq_len, pred_len, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.embed   = nn.Linear(seq_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                  dim_feedforward=d_model*4, dropout=dropout,
                  batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.project = nn.Linear(d_model, pred_len)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):                           # x: (B, C, L)
        tok = self.drop(self.embed(x))              # (B, C, d_model)
        return self.project(self.encoder(tok))      # (B, C, pred_len)


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, top_k=3):
        super().__init__()
        self.top_k   = top_k
        self.seq_len = seq_len
        self.conv    = nn.Sequential(
            nn.Conv2d(d_model, d_model*2, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(d_model*2, d_model, kernel_size=3, padding=1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                           # x: (B, L, D)
        B, L, D = x.shape
        fft_v = torch.abs(torch.fft.rfft(x.mean(-1), dim=-1))
        fft_v[:, 0] = 0
        top_p = torch.topk(fft_v, self.top_k, dim=-1).indices + 1
        out   = torch.zeros_like(x)
        for b in range(B):
            for period in top_p[b]:
                p = period.item()
                if p <= 1: continue
                T       = math.ceil(L / p)
                pad_len = T * p - L
                xi      = F.pad(x[b].T.unsqueeze(0), (0, pad_len))  # (1, D, T*p)
                xi      = self.conv(xi.reshape(1, D, T, p))          # (1, D, T, p)
                out[b] += xi.reshape(1, D, T*p)[..., :L].squeeze(0).T
        return self.norm(x + out / self.top_k)


class TimesNet(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=32, n_layers=2, top_k=3, dropout=0.1):
        super().__init__()
        self.embed  = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([TimesBlock(seq_len, d_model, top_k) for _ in range(n_layers)])
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Linear(d_model, 1)
        self.proj   = nn.Linear(seq_len, pred_len)

    def forward(self, x):                           # x: (B, C, L)
        B, C, L = x.shape
        xi = self.drop(self.embed(x.permute(0, 2, 1)))  # (B, L, d_model)
        for blk in self.blocks:
            xi = blk(xi)
        out = self.head(xi).permute(0, 2, 1)            # (B, C, L)
        return self.proj(out)                            # (B, C, pred_len)


# ══════════════════════════════════════════════════════════════
# SECTION 3 — DATASET
# ══════════════════════════════════════════════════════════════

class ETTDataset(Dataset):
    """
    ETTm1 dataset with proper train/val/test splits and no leakage.
    Scaler fitted on train only. Supports any pred_len.

    Split boundaries (standard):
      Train : [0,       17280)
      Val   : [17280,   23040)
      Test  : [23040,   end  )
    """
    def __init__(self, seq_len=96, pred_len=96, split='train',
                 file_path='ETTm1.csv', target_col='OT'):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        df   = pd.read_csv(file_path)
        data = df[target_col].values.reshape(-1, 1)

        train_end = 12 * 30 * 24 * 4   # 17,280
        val_end   = train_end + 4 * 30 * 24 * 4  # 23,040

        raw = {'train': data[:train_end],
               'val':   data[train_end:val_end],
               'test':  data[val_end:]}[split]

        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])            # fit ONLY on train
        self.data = torch.tensor(
            self.scaler.transform(raw), dtype=torch.float32)

    def inverse(self, x_norm):
        return x_norm * self.scaler.scale_[0] + self.scaler.mean_[0]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx):
        x = self.data[idx                    : idx + self.seq_len]
        y = self.data[idx + self.seq_len     : idx + self.seq_len + self.pred_len]
        return x.t(), y.t()   # (1, seq_len), (1, pred_len)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION
# ══════════════════════════════════════════════════════════════

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
    mae  = float(np.abs(err).mean())
    mse  = float((err**2).mean())
    rmse = float(mse**0.5)

    # Original-scale metrics
    p_o = inv_fn(p) if inv_fn else p
    t_o = inv_fn(t) if inv_fn else t
    err_o  = p_o - t_o
    eps    = 1e-8
    t_std  = t_o.std() + eps
    smape  = float((2*np.abs(err_o)/(np.abs(p_o)+np.abs(t_o)+eps)).mean()*100)
    ss_res = float(((p_o-t_o)**2).sum())
    ss_tot = float(((t_o-t_o.mean())**2).sum())
    r2     = float(1 - ss_res/(ss_tot+eps))
    corr   = float(np.corrcoef(p_o, t_o)[0,1]) if len(p_o)>1 else 0.0
    dp, dt = np.diff(p), np.diff(t)
    dir_acc = float((np.sign(dp)==np.sign(dt)).mean()*100)

    return dict(mae=mae, mse=mse, rmse=rmse, smape=smape,
                r2=r2, corr=corr, dir_acc=dir_acc)


# ══════════════════════════════════════════════════════════════
# SECTION 5 — CONFIG
# ══════════════════════════════════════════════════════════════

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE_PATH  = 'ETTm1.csv'
SEQ_LEN    = 96
HORIZONS   = [1, 96, 192, 336, 720]
BATCH_SIZE = 64
D_MODEL    = 64
NUM_HEADS  = 4
EPOCHS     = 30
LR         = 1e-3
TAU_FIXED  = 0.1
TAU_START  = 0.5
TAU_END    = 0.05

crit_wmrt = WaveletTransformerLoss(lambda_recon=1.0, lambda_ortho=0.1)
crit_mse  = nn.MSELoss()

print(f"Device  : {DEVICE}")
print(f"Horizons: {HORIZONS}")
print(f"Epochs  : {EPOCHS}  |  d_model={D_MODEL}  |  batch={BATCH_SIZE}\n")


def get_tau(epoch, total):
    p = (epoch-1)/max(total-1,1)
    return TAU_END + (TAU_START-TAU_END)*0.5*(1+math.cos(math.pi*p))


def make_wmrt(d_model, num_heads, tau_type='fixed', tau_init=TAU_FIXED):
    dwt = LearnableDWT1D(1, d_model).to(DEVICE)
    if tau_type == 'learnable':
        block = LearnableTauWRATBlock(d_model, num_heads, tau_init).to(DEVICE)
    else:
        block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init).to(DEVICE)
    sc = nn.Conv1d(1, 1, kernel_size=1).to(DEVICE)
    opt = optim.AdamW(
        list(dwt.parameters())+list(block.parameters())+list(sc.parameters()),
        lr=LR, weight_decay=1e-4)
    return dwt, block, sc, opt


def wmrt_pred(dwt, block, sc, bx, zero_lh=False):
    LL, LH = dwt(bx)
    if zero_lh: LH = torch.zeros_like(LH)
    LL_o, LH_o = block(LL, LH)
    return sc(dwt.inverse(LL_o, LH_o))


def train_wmrt(dwt, block, sc, opt, loader, tau_override=None, zero_lh=False):
    dwt.train(); block.train(); sc.train()
    if tau_override is not None and hasattr(block, 'sparsity_tau'):
        block.sparsity_tau = tau_override
    total_loss = n = 0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        LL, LH = dwt(bx)
        if zero_lh: LH = torch.zeros_like(LH)
        LL_o, LH_o = block(LL, LH)
        preds = sc(dwt.inverse(LL_o, LH_o))
        xr    = dwt.inverse(LL, LH)
        L     = min(preds.shape[-1], by.shape[-1])
        loss, tl, *_ = crit_wmrt(preds[..., :L], by[..., :L], bx, xr, dwt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(dwt.parameters())+list(block.parameters())+list(sc.parameters()), 1.0)
        opt.step()
        total_loss += tl.item(); n += 1
    return total_loss / n


def train_std(model, opt, loader):
    model.train()
    total_loss = n = 0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        pred = model(bx)
        L    = min(pred.shape[-1], by.shape[-1])
        loss = crit_mse(pred[..., :L], by[..., :L])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); n += 1
    return total_loss / n


# ══════════════════════════════════════════════════════════════
# SECTION 6 — MAIN TRAINING + EVALUATION LOOP
# ══════════════════════════════════════════════════════════════

all_results   = {}   # all_results[horizon][model_name] = metrics
all_histories = {}   # all_histories[horizon][model_name] = {'train':[], 'val':[]}

MODEL_NAMES = [
    'WRAT_Fixed', 'WRAT_Adaptive', 'WRAT_Learnable', 'WRAT_Ablation',
    'DLinear', 'NLinear', 'PatchTST', 'iTransformer', 'TimesNet', 'Vanilla'
]

for PRED_LEN in HORIZONS:
    print(f"\n{'='*72}")
    print(f"  HORIZON = {PRED_LEN} steps")
    print(f"{'='*72}")

    # ── Data ────────────────────────────────────────────────
    train_ds = ETTDataset(SEQ_LEN, PRED_LEN, 'train', FILE_PATH)
    val_ds   = ETTDataset(SEQ_LEN, PRED_LEN, 'val',   FILE_PATH)
    test_ds  = ETTDataset(SEQ_LEN, PRED_LEN, 'test',  FILE_PATH)
    inv_fn   = test_ds.inverse

    trl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True)
    vll = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, drop_last=True)
    tel = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, drop_last=True)
    print(f"  Train:{len(train_ds):,}  Val:{len(val_ds):,}  Test:{len(test_ds):,}")

    # ── Instantiate models ───────────────────────────────────
    # WRAT variants
    dwt_fix, blk_fix, sc_fix, opt_fix = make_wmrt(D_MODEL, NUM_HEADS, 'fixed')
    dwt_adp, blk_adp, sc_adp, opt_adp = make_wmrt(D_MODEL, NUM_HEADS, 'fixed', TAU_START)
    dwt_lrn, blk_lrn, sc_lrn, opt_lrn = make_wmrt(D_MODEL, NUM_HEADS, 'learnable')
    dwt_abl, blk_abl, sc_abl, opt_abl = make_wmrt(D_MODEL, NUM_HEADS, 'fixed')

    # Standard baselines
    std_models = {
        'DLinear':      DLinear(SEQ_LEN, PRED_LEN).to(DEVICE),
        'NLinear':      NLinear(SEQ_LEN, PRED_LEN).to(DEVICE),
        'PatchTST':     PatchTST(SEQ_LEN, PRED_LEN, d_model=D_MODEL, n_heads=NUM_HEADS).to(DEVICE),
        'iTransformer': iTransformer(SEQ_LEN, PRED_LEN, d_model=D_MODEL, n_heads=NUM_HEADS).to(DEVICE),
        'TimesNet':     TimesNet(SEQ_LEN, PRED_LEN, d_model=D_MODEL//2).to(DEVICE),
        'Vanilla':      VanillaTransformerBaseline(1, D_MODEL, NUM_HEADS, SEQ_LEN).to(DEVICE),
    }
    std_opts = {name: optim.AdamW(m.parameters(), lr=LR) for name, m in std_models.items()}

    # Print param counts once
    if PRED_LEN == HORIZONS[0]:
        wp = sum(p.numel() for m in [dwt_fix,blk_fix,sc_fix] for p in m.parameters())
        print(f"\n  Parameters:")
        print(f"    WRAT variants : {wp:,}")
        for name, m in std_models.items():
            print(f"    {name:<14}: {sum(p.numel() for p in m.parameters()):,}")
        print()

    # ── History ──────────────────────────────────────────────
    hist = {name: {'train':[], 'val':[]} for name in MODEL_NAMES}

    # ── Training ─────────────────────────────────────────────
    print(f"  {'Ep':>3}  {'WRAT_Fix':>10} {'WRAT_Adp':>10} {'WRAT_Lrn':>10} "
          f"{'DLinear':>10} {'PatchTST':>10} {'Vanilla':>10}")
    print(f"  {'-'*65}")

    for epoch in range(1, EPOCHS+1):
        tau = get_tau(epoch, EPOCHS)

        # Train WRAT variants
        tl_fix = train_wmrt(dwt_fix, blk_fix, sc_fix, opt_fix, trl)
        tl_adp = train_wmrt(dwt_adp, blk_adp, sc_adp, opt_adp, trl, tau_override=tau)
        tl_lrn = train_wmrt(dwt_lrn, blk_lrn, sc_lrn, opt_lrn, trl)
        tl_abl = train_wmrt(dwt_abl, blk_abl, sc_abl, opt_abl, trl, zero_lh=True)

        # Train standard baselines
        tl_std = {name: train_std(m, std_opts[name], trl) for name, m in std_models.items()}

        # Record train losses
        for name, tl in zip(['WRAT_Fixed','WRAT_Adaptive','WRAT_Learnable','WRAT_Ablation'],
                             [tl_fix, tl_adp, tl_lrn, tl_abl]):
            hist[name]['train'].append(tl)
        for name, tl in tl_std.items():
            hist[name]['train'].append(tl)

        # Validate
        def _fn_fix(bx): return wmrt_pred(dwt_fix, blk_fix, sc_fix, bx)
        def _fn_adp(bx): return wmrt_pred(dwt_adp, blk_adp, sc_adp, bx)
        def _fn_lrn(bx): return wmrt_pred(dwt_lrn, blk_lrn, sc_lrn, bx)
        def _fn_abl(bx): return wmrt_pred(dwt_abl, blk_abl, sc_abl, bx, zero_lh=True)

        for name, fn in zip(['WRAT_Fixed','WRAT_Adaptive','WRAT_Learnable','WRAT_Ablation'],
                             [_fn_fix, _fn_adp, _fn_lrn, _fn_abl]):
            dwt_fix.eval(); blk_fix.eval(); sc_fix.eval()
            dwt_adp.eval(); blk_adp.eval(); sc_adp.eval()
            dwt_lrn.eval(); blk_lrn.eval(); sc_lrn.eval()
            dwt_abl.eval(); blk_abl.eval(); sc_abl.eval()
            v = evaluate(fn, vll, DEVICE, inv_fn)
            hist[name]['val'].append(v['mse'])

        for name, m in std_models.items():
            m.eval()
            v = evaluate(lambda bx, _m=m: _m(bx), vll, DEVICE, inv_fn)
            hist[name]['val'].append(v['mse'])
            m.train()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>3}  "
                  f"{hist['WRAT_Fixed']['val'][-1]:>10.5f} "
                  f"{hist['WRAT_Adaptive']['val'][-1]:>10.5f} "
                  f"{hist['WRAT_Learnable']['val'][-1]:>10.5f} "
                  f"{hist['DLinear']['val'][-1]:>10.5f} "
                  f"{hist['PatchTST']['val'][-1]:>10.5f} "
                  f"{hist['Vanilla']['val'][-1]:>10.5f}")

    # ── Test ─────────────────────────────────────────────────
    dwt_fix.eval(); blk_fix.eval(); sc_fix.eval()
    dwt_adp.eval(); blk_adp.eval(); sc_adp.eval()
    dwt_lrn.eval(); blk_lrn.eval(); sc_lrn.eval()
    dwt_abl.eval(); blk_abl.eval(); sc_abl.eval()
    for m in std_models.values(): m.eval()

    hr = {}
    hr['WRAT_Fixed']     = evaluate(_fn_fix, tel, DEVICE, inv_fn)
    hr['WRAT_Adaptive']  = evaluate(_fn_adp, tel, DEVICE, inv_fn)
    hr['WRAT_Learnable'] = evaluate(_fn_lrn, tel, DEVICE, inv_fn)
    hr['WRAT_Ablation']  = evaluate(_fn_abl, tel, DEVICE, inv_fn)
    for name, m in std_models.items():
        hr[name] = evaluate(lambda bx, _m=m: _m(bx), tel, DEVICE, inv_fn)

    all_results[PRED_LEN]   = hr
    all_histories[PRED_LEN] = hist

    # Per-horizon summary
    print(f"\n  Test results — Horizon {PRED_LEN}:")
    print(f"  {'Model':<16} {'MAE':>8} {'RMSE':>8} {'sMAPE%':>8} {'R2':>7} {'DirAcc%':>9}")
    print(f"  {'-'*58}")
    for name in MODEL_NAMES:
        r = hr[name]
        tag = ' *' if name.startswith('WRAT') and not name.endswith('Ablation') else ''
        print(f"  {name:<16} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
              f"{r['smape']:>8.2f} {r['r2']:>7.4f} {r['dir_acc']:>9.2f}{tag}")


# ══════════════════════════════════════════════════════════════
# SECTION 7 — SUMMARY TABLES
# ══════════════════════════════════════════════════════════════

def summary_table(metric, label, higher):
    w = 9
    print(f"\n{'='*80}")
    print(f"  {label} across all horizons  ({'higher' if higher else 'lower'} is better)")
    print(f"{'='*80}")
    header = f"  {'Model':<16}" + "".join(f"H={h:>{w-2}}" for h in HORIZONS) + f"  {'AVG':>{w-1}}"
    print(header); print(f"  {'-'*76}")
    for name in MODEL_NAMES:
        vals = [all_results[h][name][metric] for h in HORIZONS]
        avg  = np.mean(vals)
        best_val = max(vals+[avg]) if higher else min(vals+[avg])
        row  = f"  {name:<16}"
        for v in vals:
            cell = f"{v:.4f}"
            row += f"{'* '+cell if abs(v-best_val)<1e-9 else '  '+cell:>{w}}"
        row += f"  {avg:.4f}"
        print(row)

summary_table('mae',     'MAE',      False)
summary_table('rmse',    'RMSE',     False)
summary_table('smape',   'sMAPE%',   False)
summary_table('r2',      'R2',       True)
summary_table('dir_acc', 'DirAcc%',  True)

# Win count
print(f"\n{'='*60}")
print(f"  WIN COUNT  (out of {len(HORIZONS)*6} metric-horizon combos)")
print(f"{'='*60}")
win_counts = {n: 0 for n in MODEL_NAMES}
for h in HORIZONS:
    for metric, higher in [('mae',False),('mse',False),('rmse',False),
                            ('smape',False),('r2',True),('dir_acc',True)]:
        vals = {n: all_results[h][n][metric] for n in MODEL_NAMES}
        best = max(vals.values()) if higher else min(vals.values())
        for n, v in vals.items():
            if abs(v - best) < 1e-9:
                win_counts[n] += 1
total = len(HORIZONS) * 6
for name in sorted(MODEL_NAMES, key=lambda n: -win_counts[n]):
    bar = '█' * win_counts[name]
    tag = '  <-- WRAT' if name.startswith('WRAT') and not name.endswith('Ablation') else ''
    print(f"  {name:<18} {win_counts[name]:>3}/{total}  {bar}{tag}")

# Ablation delta
print(f"\n{'='*60}")
print("  ABLATION — HF attention contribution (DirAcc delta)")
print(f"{'='*60}")
for h in HORIZONS:
    fix = all_results[h]['WRAT_Fixed']['dir_acc']
    abl = all_results[h]['WRAT_Ablation']['dir_acc']
    print(f"  H={h:<4}  Fixed={fix:.2f}%  Ablation={abl:.2f}%  delta={fix-abl:+.2f}pp")

print(f"\n  Learnable tau final values per horizon:")
# (re-use last trained blk_lrn — only valid for last horizon, shown for reference)
print(f"  Last horizon tau = {blk_lrn.tau:.4f}" if hasattr(blk_lrn, 'tau') else "  N/A")


# ══════════════════════════════════════════════════════════════
# SECTION 8 — PLOTS
# ══════════════════════════════════════════════════════════════

COLORS = {
    'WRAT_Fixed':     '#1d4ed8',
    'WRAT_Adaptive':  '#2563eb',
    'WRAT_Learnable': '#60a5fa',
    'WRAT_Ablation':  '#93c5fd',
    'DLinear':        '#16a34a',
    'NLinear':        '#4ade80',
    'PatchTST':       '#dc2626',
    'iTransformer':   '#9333ea',
    'TimesNet':       '#ea580c',
    'Vanilla':        '#94a3b8',
}
LINES = {
    'WRAT_Fixed':'-','WRAT_Adaptive':'--','WRAT_Learnable':'-.','WRAT_Ablation':':',
    'DLinear':'-','NLinear':'--','PatchTST':'-.','iTransformer':':','TimesNet':'-','Vanilla':'--'
}
MKR = {n: m for n, m in zip(MODEL_NAMES, ['o','o','o','o','s','D','^','v','P','x'])}

fig = plt.figure(figsize=(22, 16))
fig.suptitle("WRAT vs All Baselines — ETTm1 Multi-Horizon Benchmark",
             fontsize=15, fontweight='bold', y=0.99)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)

# Row 0 — metric vs horizon lines
for col, (metric, label) in enumerate([
        ('mae',     'MAE (lower = better)'),
        ('dir_acc', 'DirAcc% (higher = better)'),
        ('r2',      'R2 (higher = better)'),
]):
    ax = fig.add_subplot(gs[0, col])
    for name in MODEL_NAMES:
        vals = [all_results[h][name][metric] for h in HORIZONS]
        lw   = 2.5 if name.startswith('WRAT') and not name.endswith('Ablation') else 1.3
        ax.plot(HORIZONS, vals, marker=MKR[name], color=COLORS[name],
                linewidth=lw, markersize=5, label=name, linestyle=LINES[name])
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Horizon')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

# Row 1 — val MSE training curves for H=96, 336, 720
for col, h in enumerate([96, 336, 720]):
    ax = fig.add_subplot(gs[1, col])
    ep = range(1, EPOCHS+1)
    if h in all_histories:
        for name in MODEL_NAMES:
            lw = 2.0 if name.startswith('WRAT') and not name.endswith('Ablation') else 1.1
            ax.plot(ep, all_histories[h][name]['val'],
                    color=COLORS[name], linewidth=lw, label=name,
                    linestyle=LINES[name], alpha=0.9)
    ax.set_title(f'Val MSE Curves — H={h}', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

# Row 2 — bar charts MAE + DirAcc at H=96, 336, 720
for col, h in enumerate([96, 336, 720]):
    ax  = fig.add_subplot(gs[2, col])
    ax2 = ax.twinx()
    x   = np.arange(len(MODEL_NAMES))
    w   = 0.4
    mae_v = [all_results[h][n]['mae']     for n in MODEL_NAMES]
    dir_v = [all_results[h][n]['dir_acc'] for n in MODEL_NAMES]
    b1 = ax.bar(x - w/2, mae_v, w, color=[COLORS[n] for n in MODEL_NAMES], alpha=0.85)
    b2 = ax2.bar(x + w/2, dir_v, w, color=[COLORS[n] for n in MODEL_NAMES], alpha=0.4)
    # Bold outline on best WRAT
    best_idx = MODEL_NAMES.index('WRAT_Adaptive')
    b1[best_idx].set_edgecolor('black'); b1[best_idx].set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('WRAT_','W_').replace('Trans','T') for n in MODEL_NAMES],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('MAE', fontsize=8)
    ax2.set_ylabel('DirAcc%', fontsize=8)
    ax.set_title(f'MAE (solid) & DirAcc% (faded) — H={h}', fontsize=9)
    ax.grid(alpha=0.2, axis='y')

plt.savefig('wrat_full_benchmark.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved -> wrat_full_benchmark.png")
print("\nAll done.")