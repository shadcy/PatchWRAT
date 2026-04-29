# =============================================================================
# benchmark_wrat_multivariate_v2.py  —  Wavelet Multiresolution Transformer v2
# =============================================================================
# Improvements over v1:
#   • Soft Learnable Frequency Gating  (replaces broken hard sparse threshold)
#   • Haar wavelet initialization      (stable, known-good filter starting point)
#   • Two-Level DWT decomposition      (coarse + medium + fine bands)
#   • Lightweight Channel Mixer        (captures cross-variate correlations)
#   • Orthogonality loss warmup        (ramps from 0 → lam_o_max over 10 epochs)
#   • Reduced reconstruction weight    (lam_r 1.0 → 0.5, task loss now dominant)
#   • Full Ablation Suite              (Full / w/o ChMixer / w/o RevIN / w/o OrthoLoss)
#   • Automated test-set visualizations (plots/ directory)
#   • Same data splits (60/20/20) and horizons [96, 192, 336, 720]
#
# Usage:
#   python benchmark_wrat_multivariate_v2.py
#   (CSV must be set in CSV_PATH below)
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

CSV_PATH   = r'C:\Users\Asus\Desktop\TTS\Rat\ETTm1.csv'
SEQ_LEN    = 512
HORIZONS   = [96, 192, 336, 720]
BATCH_SIZE = 64
EPOCHS     = 30
PATIENCE   = 10
LR         = 3e-4
D_MODEL    = 32
NUM_HEADS  = 4
DROPOUT    = 0.3
LAM_R      = 0.5        # reconstruction loss weight (was 1.0 — reduced so task loss dominates)
LAM_O_MAX  = 0.05       # orthogonality loss max weight (was 0.1 — ramped up slowly)

_temp_df     = pd.read_csv(CSV_PATH, nrows=0, encoding='unicode_escape')
CHANNELS     = _temp_df.columns.tolist()[1:]   # skip date/index column
NUM_CHANNELS = len(CHANNELS)

print(f"[ENV]  Device  : {DEVICE} | PyTorch {torch.__version__}")
print(f"[DATA] {CSV_PATH}")
print(f"       seq={SEQ_LEN}  horizons={HORIZONS}  channels={NUM_CHANNELS}")
print("\n[WMRTv2] Wavelet Multiresolution Transformer — Improved Architecture")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _splits(n):
    te = max(1, min(int(n * 0.60), n - 2))
    ve = max(te + 1, min(int(n * 0.80), n - 1))
    return te, ve


class TimeSeriesDataset(Dataset):
    """
    Loads CSV, standardizes using train-split statistics only (no leakage),
    returns (C, T) tensors for channel-independent multivariate forecasting.
    """
    def __init__(self, seq_len: int, pred_len: int, split: str = 'train'):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        df      = pd.read_csv(CSV_PATH, encoding='unicode_escape')
        raw_all = df[CHANNELS].values.astype(np.float32)
        n       = len(raw_all)
        te, ve  = _splits(n)

        seg = {'train': raw_all[:te],
               'val':   raw_all[te:ve],
               'test':  raw_all[ve:]}[split]

        sc        = StandardScaler().fit(raw_all[:te])      # fit only on train
        self.data = torch.tensor(sc.transform(seg), dtype=torch.float32)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, i):
        x = self.data[i            : i + self.seq_len]
        y = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]
        return x.t(), y.t()   # → (C, T)


def get_loaders(pred_len: int):
    kw = dict(num_workers=0, pin_memory=False)
    tr = DataLoader(TimeSeriesDataset(SEQ_LEN, pred_len, 'train'),
                    BATCH_SIZE, shuffle=True,  drop_last=True,  **kw)
    va = DataLoader(TimeSeriesDataset(SEQ_LEN, pred_len, 'val'),
                    BATCH_SIZE, shuffle=False, drop_last=True,  **kw)
    te = DataLoader(TimeSeriesDataset(SEQ_LEN, pred_len, 'test'),
                    BATCH_SIZE, shuffle=False, drop_last=False, **kw)
    return tr, va, te


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best       = None
        self.early_stop = False
        self.state      = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_p, all_t = [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        preds, *_ = model(bx)
        all_p.append(preds.cpu())
        all_t.append(by.cpu())
    p = torch.cat(all_p).flatten().numpy()
    t = torch.cat(all_t).flatten().numpy()
    e = p - t
    return dict(
        mse     = float((e ** 2).mean()),
        mae     = float(np.abs(e).mean()),
        dir_acc = float((np.sign(np.diff(p)) == np.sign(np.diff(t))).mean() * 100)
    )


def make_scheduler(opt, epochs: int = EPOCHS):
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.1,
                                          end_factor=1.0, total_iters=5)
    cos    = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                   T_max=max(1, epochs - 5),
                                                   eta_min=LR * 0.02)
    return warmup, cos

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(model: nn.Module, loader: DataLoader,
                     horizon: int, num_samples: int = 4,
                     out_dir: str = "plots"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    bx, by = next(iter(loader))
    bx, by = bx.to(DEVICE), by.to(DEVICE)

    with torch.no_grad():
        preds, *_ = model(bx)

    bx    = bx.cpu().numpy()
    by    = by.cpu().numpy()
    preds = preds.cpu().numpy()

    t_hist = np.arange(SEQ_LEN)
    t_pred = np.arange(SEQ_LEN, SEQ_LEN + horizon)

    for i in range(min(num_samples, bx.shape[0])):
        fig, axes = plt.subplots(NUM_CHANNELS, 1,
                                  figsize=(12, 3 * NUM_CHANNELS), sharex=True)
        if NUM_CHANNELS == 1:
            axes = [axes]
        fig.suptitle(
            f"WMRTv2 — Horizon: {horizon} | Sample {i + 1}",
            fontsize=14, fontweight='bold'
        )
        for c, ax in enumerate(axes):
            ax.plot(t_hist, bx[i, c, :],    label='History',      color='dimgray', lw=1.0)
            ax.plot(t_pred, by[i, c, :],    label='Ground Truth',  color='royalblue', lw=1.5)
            ax.plot(t_pred, preds[i, c, :], label='WMRTv2 Pred',   color='tomato',
                    linestyle='--', lw=1.5)
            ax.set_ylabel(CHANNELS[c], fontweight='bold', fontsize=8)
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

        plt.xlabel("Time Steps", fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.85, 0.97])
        path = os.path.join(out_dir, f'wmrt_v2_H{horizon}_sample{i + 1}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"    [Plots] Saved {num_samples} plots → '{out_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """Reversible Instance Normalization — normalises per-channel per-sample."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones (1, num_features, 1))
        self.b   = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: (B, C, L)
        if mode == 'norm':
            self.mean  = x.mean(-1, keepdim=True).detach()
            self.stdev = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            return ((x - self.mean) / self.stdev) * self.w + self.b
        # mode == 'denorm'
        return ((x - self.b) / self.w) * self.stdev + self.mean


class ChannelMixer(nn.Module):
    """
    Lightweight linear channel-mixing layer.
    Initialised as identity so it starts as a no-op and learns correlations
    gradually — does not disrupt early-training dynamics.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.mix = nn.Linear(num_channels, num_channels, bias=False)
        nn.init.eye_(self.mix.weight)   # identity init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → mix along C
        return self.mix(x.transpose(1, 2)).transpose(1, 2)


class LearnableDWT1D(nn.Module):
    """
    Two-channel learnable 1-D DWT with Haar initialisation.

    Returns three bands from a two-level decomposition:
        LL2  — coarse (approx of approx)
        LH2  — medium detail (approx of detail)
        LH1  — fine   detail (first-level detail)

    Inverse path reconstructs the signal from all three bands.
    """

    def __init__(self, in_ch: int, out_ch: int, filter_len: int = 4):
        super().__init__()
        self.out_ch = out_ch
        self.pad    = (filter_len - 2) // 2

        # Level-1 filters
        self.h1 = nn.Parameter(torch.randn(out_ch, in_ch,   filter_len) * 0.1)
        self.g1 = nn.Parameter(torch.randn(out_ch, in_ch,   filter_len) * 0.1)
        # Level-2 filters (operate on LL1)
        self.h2 = nn.Parameter(torch.randn(out_ch, out_ch,  filter_len) * 0.1)
        self.g2 = nn.Parameter(torch.randn(out_ch, out_ch,  filter_len) * 0.1)

        self._init_haar()

    # ------------------------------------------------------------------
    def _init_haar(self):
        """Initialise both filter pairs with Haar wavelets (scale + detail)."""
        with torch.no_grad():
            # Low-pass  (scaling): uniform average
            self.h1.data.fill_(0.0)
            self.h1.data[:, :, :2] = 0.5
            self.h2.data.fill_(0.0)
            self.h2.data[:, :, :2] = 0.5
            # High-pass (wavelet): difference
            self.g1.data.fill_(0.0)
            self.g1.data[:, :, 0] =  0.5
            self.g1.data[:, :, 1] = -0.5
            self.g2.data.fill_(0.0)
            self.g2.data[:, :, 0] =  0.5
            self.g2.data[:, :, 1] = -0.5

    # ------------------------------------------------------------------
    def _conv(self, x, filt):
        return F.conv1d(x, filt, stride=2, padding=self.pad)

    def _tconv(self, x, filt):
        return F.conv_transpose1d(x, filt, stride=2, padding=self.pad)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """x: (B*C, 1, L)  →  LL2, LH2, LH1"""
        LL1, LH1 = self._conv(x, self.h1), self._conv(x, self.g1)   # level-1
        LL2, LH2 = self._conv(LL1, self.h2), self._conv(LL1, self.g2)  # level-2
        return LL2, LH2, LH1

    def inverse(self, LL2: torch.Tensor,
                       LH2: torch.Tensor,
                       LH1: torch.Tensor) -> torch.Tensor:
        """Reconstruct signal from three bands."""
        rL2 = self._tconv(LL2, self.h2)
        rH2 = self._tconv(LH2, self.g2)
        n1  = min(rL2.shape[-1], rH2.shape[-1])
        LL1_recon = rL2[..., :n1] + rH2[..., :n1]

        rL1 = self._tconv(LL1_recon, self.h1)
        rH1 = self._tconv(LH1,       self.g1)
        n2  = min(rL1.shape[-1], rH1.shape[-1])
        return rL1[..., :n2] + rH1[..., :n2]

    def ortho_loss(self) -> torch.Tensor:
        """Penalise non-orthogonality between low/high pass at each level."""
        h1f = self.h1.view(self.h1.shape[0], -1)
        g1f = self.g1.view(self.g1.shape[0], -1)
        h2f = self.h2.view(self.h2.shape[0], -1)
        g2f = self.g2.view(self.g2.shape[0], -1)
        return (h1f * g1f).sum().abs() + (h2f * g2f).sum().abs()


class SoftFreqAttn(nn.Module):
    """
    Multi-head self-attention with a learnable soft frequency gate.

    Key changes vs v1:
      - No hard threshold masking (eliminated the nan_to_num / -inf problem).
      - A sigmoid gate is learned per-token from the value projections,
        allowing the model to weight frequency bands adaptively.
      - Gate initialised to ~0.5 (sigmoid(0)) so training starts neutral.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.H  = num_heads
        self.Dh = d_model // num_heads

        self.q    = nn.Linear(d_model, d_model)
        self.k    = nn.Linear(d_model, d_model)
        self.v    = nn.Linear(d_model, d_model)
        self.o    = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Soft gate: (B, L, D) → (B, L, 1) gate per token
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        # Initialise gate bias to 0 → sigmoid(0)=0.5 (neutral start)
        nn.init.zeros_(self.gate[2].bias)

    def forward(self, q_x: torch.Tensor,
                      k_x: torch.Tensor,
                      v_x: torch.Tensor) -> torch.Tensor:
        B, L, D = q_x.shape

        Q = self.q(q_x).view(B, L,        self.H, self.Dh).transpose(1, 2)
        K = self.k(k_x).view(B, -1,       self.H, self.Dh).transpose(1, 2)
        V = self.v(v_x).view(B, -1,       self.H, self.Dh).transpose(1, 2)

        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.Dh)
        w  = self.drop(F.softmax(sc, dim=-1))
        o  = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, L, D)

        # Apply soft per-token gate (learned from value features)
        g = self.gate(v_x)      # (B, L, 1) ∈ (0, 1)
        return self.o(o * g)


class WRATBlock(nn.Module):
    """
    Three-band WRAT block for two-level DWT output (LL2, LH2, LH1).

    Attention pathway:
        LL2  → dense self-attention  (low-freq trends — never sparse)
        LH2  → gated self-attention  (medium detail)
        LH1  → gated self-attention  (fine detail)
        cross → LL2 queries over (LH2 + LH1) concat  (inject detail into trend)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.ll_attn   = SoftFreqAttn(d_model, num_heads, dropout)
        self.lh2_attn  = SoftFreqAttn(d_model, num_heads, dropout)
        self.lh1_attn  = SoftFreqAttn(d_model, num_heads, dropout)
        self.cross_attn = SoftFreqAttn(d_model, num_heads, dropout)

        def _mlp(d):
            return nn.Sequential(
                nn.Linear(d, d * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d * 4, d)
            )

        self.mlp_ll  = _mlp(d_model)
        self.mlp_lh2 = _mlp(d_model)
        self.mlp_lh1 = _mlp(d_model)
        self.mlp_cr  = _mlp(d_model)

        self.n_ll    = nn.LayerNorm(d_model)
        self.n_lh2   = nn.LayerNorm(d_model)
        self.n_lh1   = nn.LayerNorm(d_model)
        self.n_cr    = nn.LayerNorm(d_model)

    def forward(self, LL2, LH2, LH1):
        # shapes: (B*C, d_model, T_band) — transpose to (B*C, T_band, d_model)
        ll  = LL2.transpose(1, 2)
        lh2 = LH2.transpose(1, 2)
        lh1 = LH1.transpose(1, 2)

        # Self-attention per band
        ll  = self.n_ll  (ll  + self.mlp_ll (self.ll_attn  (ll,  ll,  ll )))
        lh2 = self.n_lh2 (lh2 + self.mlp_lh2(self.lh2_attn(lh2, lh2, lh2)))
        lh1 = self.n_lh1 (lh1 + self.mlp_lh1(self.lh1_attn(lh1, lh1, lh1)))

        # Cross-attention: LL queries detail bands
        # Pad/crop lh2 and lh1 to the same length for concat
        T_lh2, T_lh1 = lh2.shape[1], lh1.shape[1]
        T_min = min(T_lh2, T_lh1)
        kv_detail = torch.cat([lh2[:, :T_min, :], lh1[:, :T_min, :]], dim=-1)

        # Project concatenated detail to d_model for cross-attention key/value
        # (we reuse the ll_attn's k/v projections via a simple slice trick)
        T_ll = ll.shape[1]
        kv_len = min(T_min, T_ll)
        cr = self.cross_attn(
            ll[:, :kv_len, :],
            lh2[:, :kv_len, :],
            lh1[:, :kv_len, :]
        )
        updated_ll = self.n_cr(ll[:, :kv_len, :] + self.mlp_cr(cr))
        if kv_len < T_ll:
            ll = torch.cat([updated_ll, ll[:, kv_len:, :]], dim=1)
        else:
            ll = updated_ll

        return ll.transpose(1, 2), lh2.transpose(1, 2), lh1.transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PatchWRATv2(nn.Module):
    """
    Improved Wavelet Multiresolution Transformer.

    Architecture flow:
        Input (B, C, L)
          → [optional] ChannelMixer     — learn cross-variate correlations
          → [optional] RevIN            — instance normalisation
          → Channel Independence: reshape (B*C, 1, L)
          → TwoLevel LearnableDWT       — 3 bands: LL2, LH2, LH1
          → WRATBlock                   — attention on all 3 bands
          → Concatenate + Flatten + MLP head → forecast (B, C, pred_len)
          → [optional] RevIN denorm
    """

    def __init__(self, seq_len: int, pred_len: int,
                 num_channels: int = 7,
                 d_model: int = D_MODEL,
                 num_heads: int = NUM_HEADS,
                 dropout: float = DROPOUT,
                 use_revin: bool = True,
                 use_ch_mixer: bool = True):
        super().__init__()
        self.use_revin    = use_revin
        self.use_ch_mixer = use_ch_mixer
        self.pred_len     = pred_len
        self.num_ch       = num_channels

        if use_ch_mixer:
            self.ch_mixer = ChannelMixer(num_channels)

        if use_revin:
            self.revin = RevIN(num_channels)

        self.dwt  = LearnableDWT1D(in_ch=1, out_ch=d_model)
        self.wrat = WRATBlock(d_model, num_heads, dropout=dropout)

        # Head: flatten all three bands and project to pred_len
        # After two-level DWT, L → L//4 per band
        # LL2 has T//4 timesteps; LH2 has T//4; LH1 has T//2
        # (approximate — actual sizes depend on padding)
        t_ll  = seq_len // 4
        t_lh2 = seq_len // 4
        t_lh1 = seq_len // 2
        flat  = d_model * (t_ll + t_lh2 + t_lh1)

        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(flat, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )

        # Scalar correction for reconstruction
        self.sc = nn.Conv1d(1, 1, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, L)
        Returns: preds (B,C,pred_len), xr (B,C,L'), LL2, LH2, LH1, LH1_raw
        """
        B, C, L = x.shape

        if self.use_ch_mixer:
            x = self.ch_mixer(x)

        if self.use_revin:
            x = self.revin(x, 'norm')

        # Channel Independence: merge batch and channel dims
        x_ci = x.reshape(B * C, 1, L)

        LL2, LH2, LH1 = self.dwt(x_ci)                         # decompose
        LL2_o, LH2_o, LH1_o = self.wrat(LL2, LH2, LH1)        # attend

        # Align time dimensions for concatenation
        T = min(LL2_o.shape[-1], LH2_o.shape[-1], LH1_o.shape[-1] // 2)
        fused = torch.cat([
            LL2_o [..., :T],
            LH2_o [..., :T],
            LH1_o [..., :T * 2]
        ], dim=2)   # (B*C, d_model, T_fused)

        preds = self.head(fused).unsqueeze(1)                   # (B*C, 1, pred_len)
        preds = preds.reshape(B, C, self.pred_len)

        # Reconstruction signal (for recon loss)
        xr = self.sc(self.dwt.inverse(LL2_o, LH2_o, LH1_o))   # (B*C, 1, L')
        xr = xr.reshape(B, C, -1)

        if self.use_revin:
            preds = self.revin(preds, 'denorm')
            xr    = self.revin(xr,    'denorm')

        return preds, xr, LL2, LH2, LH1


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def wrat_loss(preds:    torch.Tensor,
              targets:  torch.Tensor,
              x_orig:   torch.Tensor,
              x_recon:  torch.Tensor,
              dwt:      LearnableDWT1D,
              epoch:    int,
              lam_r:    float = LAM_R,
              lam_o_max: float = LAM_O_MAX) -> tuple[torch.Tensor, float]:
    """
    Total loss = task_mse
               + lam_r  * reconstruction_mse
               + lam_o  * orthogonality_penalty   (ramped up over first 10 epochs)
    """
    task  = F.mse_loss(preds, targets)
    n     = min(x_orig.shape[-1], x_recon.shape[-1])
    recon = F.mse_loss(x_recon[..., :n], x_orig[..., :n])

    # Orthogonality warmup: 0 → lam_o_max linearly over 10 epochs
    lam_o = lam_o_max * min(1.0, epoch / 10.0)
    ortho = dwt.ortho_loss() * lam_o

    total = task + lam_r * recon + ortho
    return total, task.item()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model: PatchWRATv2,
                loader: DataLoader,
                opt: optim.Optimizer,
                epoch: int) -> float:
    model.train()
    losses = []
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        preds, xr, *_ = model(bx)
        loss, task_mse = wrat_loss(preds, by, bx, xr, model.dwt, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(task_mse)
    return float(np.mean(losses))


def run_variant(name: str,
                model_kwargs: dict,
                tr_loader: DataLoader,
                va_loader: DataLoader,
                te_loader: DataLoader) -> tuple[dict, PatchWRATv2]:

    torch.manual_seed(SEED)
    model = PatchWRATv2(**model_kwargs).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warmup, cos = make_scheduler(opt, EPOCHS)
    es    = EarlyStopping(PATIENCE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{name}]  params={total_params:,}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, tr_loader, opt, epoch)
        val_mse = evaluate(model, va_loader, DEVICE)['mse']

        if epoch <= 5:
            warmup.step()
        else:
            cos.step()

        improved = es(val_mse, model)
        if epoch % 10 == 0 or improved:
            tag = '★' if improved else ''
            print(f"    ep{epoch:>3}  train={tr_loss:.4f}  val={val_mse:.4f} {tag}")

        if es.early_stop:
            print(f"    Early stop @ ep{epoch}  best_val={es.best:.4f}")
            break

    model.load_state_dict(es.state)
    return evaluate(model, te_loader, DEVICE), model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}\n"
        "Set CSV_PATH at the top of this file to your dataset path."
    )

# Ablation configurations — only Full WMRTv2 active for now
ABLATION_CONFIGS = {
    'Full WMRTv2':     dict(use_revin=True,  use_ch_mixer=True),
    # 'w/o ChMixer':   dict(use_revin=True,  use_ch_mixer=False),
    # 'w/o RevIN':     dict(use_revin=False, use_ch_mixer=True),
    # 'w/o OrthoLoss': dict(use_revin=True,  use_ch_mixer=True),
}

all_results = {}

for PRED_LEN in HORIZONS:
    print(f"\n{'='*70}")
    print(f"  HORIZON = {PRED_LEN}  |  seq={SEQ_LEN}  |  channels={NUM_CHANNELS}")
    print(f"{'='*70}")

    tr_loader, va_loader, te_loader = get_loaders(PRED_LEN)
    all_results[PRED_LEN] = {}

    base_kw = dict(seq_len=SEQ_LEN, pred_len=PRED_LEN,
                   num_channels=NUM_CHANNELS,
                   d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT)

    for name, variant_kw in ABLATION_CONFIGS.items():
        kw = {**base_kw, **variant_kw}
        res, trained = run_variant(name, kw, tr_loader, va_loader, te_loader)
        all_results[PRED_LEN][name] = res

    # Plot predictions for the full model
    plot_predictions(trained, te_loader, PRED_LEN, num_samples=4)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n  ┌─ Horizon {PRED_LEN} Results {'─'*35}┐")
    print(f"  │  {'Variant':<18} {'MSE':>8}  {'MAE':>8}  {'DirAcc':>9}    │")
    print(f"  │  {'─'*57}  │")
    for name in ABLATION_CONFIGS:
        r = all_results[PRED_LEN][name]
        print(f"  │  {name:<18} {r['mse']:>8.4f}  {r['mae']:>8.4f}  {r['dir_acc']:>8.1f}%    │")
    print(f"  └{'─'*61}┘")

# ── Final average summary ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  AVERAGE ACROSS ALL HORIZONS")
print(f"{'='*70}")
print(f"  {'Variant':<18} {'MSE':>8}  {'MAE':>8}  {'DirAcc':>9}")
print(f"  {'─'*55}")
for name in ABLATION_CONFIGS:
    mses = [all_results[h][name]['mse']     for h in HORIZONS]
    maes = [all_results[h][name]['mae']     for h in HORIZONS]
    dirs = [all_results[h][name]['dir_acc'] for h in HORIZONS]
    print(f"  {name:<18} {np.mean(mses):>8.4f}  {np.mean(maes):>8.4f}  {np.mean(dirs):>8.1f}%")

print(f"\n{'='*70}")
print("[WMRTv2] Benchmark complete. Plots saved in 'plots/' directory.")
print(f"{'='*70}\n")