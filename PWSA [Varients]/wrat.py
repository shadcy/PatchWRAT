# =============================================================================
# run_pwsa_complete.py  —  P-WSA++ Complete Integrated Architecture
# =============================================================================
#
# WHAT IS IN THIS FILE (everything in one place)
# -----------------------------------------------
# Base:        RevIN, PatchEmbedding, LearnableDWT, WRATBlock (FSA + GLU)
#
# Upgrade 1:   FITSBypass          — low-frequency interpolation bypass
# Upgrade 2:   SelectiveSSM        — Mamba-style scan replaces MLP lanes
# Upgrade 3:   CrossBandAttention  — LL <-> LH bidirectional cross-attention
# Upgrade 4:   EvidentialHead      — Normal-Inverse-Gamma uncertainty output
# Upgrade 5:   VariateGraphMixer   — learned inter-variate dependency graph
#
# Quality:     InfoConstrainedDWT  — orthogonality + noise + MI filter losses
#              SignalQualityScorer — differentiable ACF + AR1 + log-var-ratio
#              SSLLHPredictor      — GRU pretext task + ACF quality signal
#              SignalQualityLoss   — full loss with all quality terms
#
# Fusion fix:  detail_alpha gate   — starts near 0, opens only when q > noise floor
#              Warmup detach       — LH frozen for first 3 epochs
#              Separate LR         — LH branch gets 3x higher learning rate
#              FITS on residual    — avoids double-counting low-frequency signal
#
# seq_len:     336 (was 128)
#              ETTm1 daily cycle = 96 steps. At seq_len=128, the model saw
#              only 1.3 daily cycles — not enough lag context. At 336 it
#              sees 3.5 daily cycles. Attention cost: 42²=1764 vs 16²=256
#              patch pairs, but all are parallelised. Patch count: 16 → 42.
#              head_dim: 512 → 1344. Extra params: ~160k per head (tiny).
#
# HOW TO RUN
# ----------
#   python run_pwsa_complete.py
#   Requires: ETTm1.csv in the same directory
#   Outputs:  plots/ folder
#
# =============================================================================

import os, math, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

warnings.filterwarnings('ignore')
for d in ['plots/learning_curves', 'plots/predictions', 'plots/benchmarks', 'plots/quality']:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"[ENV] Device: {DEVICE} | PyTorch {torch.__version__}")

# =============================================================================
# SECTION 1 — BASE MODULES
# =============================================================================

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(num_features))
        self.b = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean  = x.mean(-1, keepdim=True).detach()
            self.stdev = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.stdev
            x = x * self.w.unsqueeze(-1) + self.b.unsqueeze(-1)
        elif mode == 'denorm':
            x = (x - self.b.unsqueeze(-1)) / (self.w.unsqueeze(-1) + self.eps**2)
            x = x * self.stdev + self.mean
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj      = nn.Linear(patch_len, d_model)

    def forward(self, x):                                   # (B, 1, L)
        pad = x[..., -1:].repeat(1, 1, self.stride)
        x   = torch.cat([x, pad], dim=-1)
        x   = x.unfold(-1, self.patch_len, self.stride).squeeze(1)  # (B, P, patch_len)
        return self.proj(x).transpose(1, 2)                # (B, D, P)


class EnvelopeExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
    def forward(self, x): return F.gelu(self.c(torch.abs(x)))


class FrequencySparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, threshold=0.1):
        super().__init__()
        self.H   = num_heads
        self.tau = torch.tensor(threshold) if isinstance(threshold, float) else threshold
        self.q   = nn.Linear(d_model, d_model)
        self.k   = nn.Linear(d_model, d_model)
        self.v   = nn.Linear(d_model, d_model)
        self.o   = nn.Linear(d_model, d_model)

    def forward(self, q_x, k_x, v_x, ec=None):
        B, L, D = q_x.shape; H = self.H; Dh = D // H
        Q = self.q(q_x).view(B,L,H,Dh).transpose(1,2)
        K = self.k(k_x).view(B,-1,H,Dh).transpose(1,2)
        V = self.v(v_x).view(B,-1,H,Dh).transpose(1,2)
        w = torch.nan_to_num(F.softmax(Q@K.transpose(-2,-1)/Dh**0.5, dim=-1), nan=0.0)
        if ec is not None:
            gate = torch.sigmoid((ec.abs().mean(-1) - self.tau.to(ec.device))*10).view(B,1,1,-1)
            w = w * gate
        return self.o((w@V).transpose(1,2).contiguous().view(B,L,D))


# =============================================================================
# SECTION 2 — UPGRADE 1: FITS FREQUENCY BYPASS
# =============================================================================

class FITSBypass(nn.Module):
    """Low-frequency interpolation bypass via truncated FFT."""
    def __init__(self, seq_len, pred_len, cut_freq=None):
        super().__init__()
        self.cut = cut_freq or max(seq_len // 4, 1)
        self.dec = nn.Linear(self.cut * 2, pred_len)
        nn.init.xavier_uniform_(self.dec.weight, gain=0.1)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):                        # (B, 1, L)
        f = torch.fft.rfft(x, dim=-1)[..., :self.cut]
        return self.dec(torch.cat([f.real, f.imag], -1).squeeze(1)).unsqueeze(1)


# =============================================================================
# SECTION 3 — UPGRADE 2: MAMBA-STYLE SELECTIVE SSM
# =============================================================================

class SelectiveSSM(nn.Module):
    """Pure-PyTorch selective state-space scan (O(L·D·N))."""
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_state  = d_state
        self.in_proj  = nn.Linear(d_model, d_model*2, bias=False)
        self.x_proj   = nn.Linear(d_model, d_state*2+d_model, bias=False)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)
        self.A_log    = nn.Parameter(torch.arange(1,d_state+1,dtype=torch.float).log().unsqueeze(0).repeat(d_model,1))
        self.D        = nn.Parameter(torch.ones(d_model))
        self.out      = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x_in):                     # (B, L, D)
        B, L, D = x_in.shape; N = self.d_state
        xz = self.in_proj(x_in); x, z = xz.chunk(2, dim=-1)
        delta_raw, B_ssm, C_ssm = self.x_proj(x).split([D, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))
        A     = -torch.exp(self.A_log.float())
        dA    = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        dB    = torch.einsum('bld,bln->bldn', delta, B_ssm)
        h     = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys    = []
        for t in range(L):
            h = dA[:,t]*h + dB[:,t]*x[:,t].unsqueeze(-1)
            ys.append(torch.einsum('bdn,bn->bd', h, C_ssm[:,t]))
        y = torch.stack(ys, 1) + x*self.D
        return self.norm(self.drop(self.out(y*F.silu(z))) + x_in)


# =============================================================================
# SECTION 4 — UPGRADE 3: CROSS-BAND ATTENTION
# =============================================================================

class CrossBandAttention(nn.Module):
    """Bidirectional LL<->LH cross-attention with near-zero init gates."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.a1 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.a2 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.g1 = nn.Parameter(torch.tensor(-3.0))  # starts near 0
        self.g2 = nn.Parameter(torch.tensor(-3.0))

    def forward(self, LL, LH):                   # (B, D, T) -> (B, D, T)
        Lt = LL.transpose(1,2); Ht = LH.transpose(1,2)
        c1,_ = self.a1(Lt, Ht, Ht)
        c2,_ = self.a2(Ht, Lt, Lt)
        return (self.n1(Lt + torch.sigmoid(self.g1)*c1)).transpose(1,2), \
               (self.n2(Ht + torch.sigmoid(self.g2)*c2)).transpose(1,2)



# =============================================================================
# SECTION 4b — VARIATE ATTENTION (true multivariate cross-series learning)
# =============================================================================

class VariateAttention(nn.Module):
    """
    Operates on raw input (B, C, seq_len) BEFORE patching.
    Treats each variate as a token, runs attention over C=7 tokens.
    Attention cost: C² = 49 pairs (negligible vs 42² = 1764 temporal pairs).

    This is the key missing piece for capturing OT→load causality in ETTm1.
    Output: context-enriched (B, C, seq_len) fed into per-variate CI pipeline.
    """
    def __init__(self, seq_len, num_variates, d_attn=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_attn = d_attn
        # Project seq_len features per variate into d_attn
        self.in_proj  = nn.Linear(seq_len, d_attn)
        self.attn     = nn.MultiheadAttention(d_attn, num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(d_attn, seq_len)
        self.norm     = nn.LayerNorm(seq_len)
        self.gate     = nn.Parameter(torch.tensor(-2.0))  # init near 0, opens gradually
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

    def forward(self, x):  # x: (B, C, seq_len)
        # Project each variate's time series into d_attn embedding
        tok = self.in_proj(x)                        # (B, C, d_attn)
        ctx, _ = self.attn(tok, tok, tok)            # (B, C, d_attn) — attend across variates
        out = self.out_proj(ctx)                     # (B, C, seq_len)
        # Gated residual: starts as identity, gate opens as training progresses
        return self.norm(x + torch.sigmoid(self.gate) * out)

# =============================================================================
# SECTION 5 — UPGRADE 4: EVIDENTIAL NIG HEAD
# =============================================================================

class EvidentialHead(nn.Module):
    """Normal-Inverse-Gamma parameterisation for uncertainty quantification."""
    def __init__(self, in_dim, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(in_dim, pred_len*4)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        r = self.proj(x).view(-1, self.pred_len, 4)
        return r[...,0], F.softplus(r[...,1])+1e-4, F.softplus(r[...,2])+1.0, F.softplus(r[...,3])+1e-4


# =============================================================================
# SECTION 6 — UPGRADE 5: VARIATE GRAPH MIXER
# =============================================================================

class VariateGraphMixer(nn.Module):
    """Learned soft inter-variate mixing, initialised as identity (CI)."""
    def __init__(self, num_variates, sparsity_weight=1e-3):
        super().__init__()
        self.A  = nn.Parameter(torch.eye(num_variates))
        self.sw = sparsity_weight

    def forward(self, x):                        # (B, C, pred_len)
        return torch.einsum('ij,bjl->bil', torch.softmax(self.A, -1), x)

    def sparsity_loss(self):
        return self.sw * self.A.abs().sum()


# =============================================================================
# SECTION 7 — QUALITY: INFO-CONSTRAINED DWT
# =============================================================================

class InfoConstrainedDWT(nn.Module):
    """
    Learnable 1D DWT with three filter regularisers:
      A. Orthogonality loss: <h,g> ≈ 0
      B. Noise allocation:   penalise (1-q)*alpha*Var(LH) — if gate open but LH is noise
      C. MI proxy:           maximise corr(|LH|energy, target_variance)
    """
    def __init__(self, channels, filter_length=4):
        super().__init__()
        self.channels = channels
        self.pv = (filter_length-2)//2
        h_i = torch.tensor([[[0.7071,0.7071,0.0,0.0]]]).repeat(channels,1,1)
        g_i = torch.tensor([[[-0.7071,0.7071,0.0,0.0]]]).repeat(channels,1,1)
        self.h = nn.Parameter(h_i + torch.randn(channels,1,filter_length)*0.05)
        self.g = nn.Parameter(g_i + torch.randn(channels,1,filter_length)*0.05)
        with torch.no_grad(): self.g -= self.g.mean(-1, keepdim=True)

    def forward(self, x):
        LL = F.conv1d(x, self.h, stride=2, padding=self.pv, groups=self.channels)
        LH = F.conv1d(x, self.g, stride=2, padding=self.pv, groups=self.channels)
        return LL, LH

    def inverse(self, LL, LH):
        xL = F.conv_transpose1d(LL, self.h, stride=2, padding=self.pv, groups=self.channels)
        xH = F.conv_transpose1d(LH, self.g, stride=2, padding=self.pv, groups=self.channels)
        m  = min(xL.shape[-1], xH.shape[-1])
        return xL[...,:m] + xH[...,:m]

    def orthogonality_loss(self):
        return (self.h.view(self.channels,-1) * self.g.view(self.channels,-1)).sum(-1).abs().mean()

    def noise_allocation_loss(self, LH, q, alpha):
        return (1.0 - q.detach()) * alpha * LH.var(dim=-1).mean()

    def mi_proxy_loss(self, LH, target, min_batch=6):
        if LH.shape[0] < min_batch:
            return torch.tensor(0.0, device=LH.device)
        le = LH.abs().mean(dim=(1,2)); tv = target.squeeze(1).var(dim=-1)
        lc = le-le.mean(); tc = tv-tv.mean()
        return -(lc*tc).sum() / (lc.norm()*tc.norm()+1e-8)


# =============================================================================
# SECTION 8 — QUALITY: SIGNAL QUALITY SCORER
# =============================================================================

class SignalQualityScorer(nn.Module):
    """
    Differentiable q in (0,1). Three components:
      - ACF1: per-channel temporal autocorrelation (dead-zoned at 1.5/sqrt(T))
      - AR1 R²: closed-form AR(1) fit quality
      - Log-var-ratio: log(Var(LH)/Var(LL)) mapped via shifted sigmoid
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps      = eps
        self.combiner = nn.Linear(3, 1, bias=True)
        nn.init.constant_(self.combiner.weight, 1.0/3.0)
        nn.init.constant_(self.combiner.bias,   0.0)

    def _acf1(self, LH):
        if LH.shape[-1] < 3: return torch.zeros(1, device=LH.device)
        T  = LH.shape[-1]
        dz = min(0.35, 1.5/math.sqrt(T))       # adaptive: white noise floor
        x  = LH.reshape(-1, T) - LH.reshape(-1,T).mean(-1,keepdim=True)
        v  = (x**2).mean(-1) + self.eps
        acf = (( x[:,:-1]*x[:,1:]).mean(-1)/v).abs()
        return ((acf-dz).clamp(0)/(1.0-dz)).mean().clamp(0,1)

    def _ar1_r2(self, LH):
        if LH.shape[-1] < 4: return torch.zeros(1, device=LH.device)
        x = LH.reshape(-1, LH.shape[-1])
        y = x[:,1:]; xp = x[:,:-1]
        xpc = xp.detach()-xp.detach().mean(-1,keepdim=True)
        yc  = y.detach() -y.detach().mean(-1,keepdim=True)
        beta = (xpc*yc).sum(-1)/((xpc**2).sum(-1)+self.eps)
        yhat = beta.detach().unsqueeze(-1)*xp+(y.mean(-1)-beta.detach()*xp.mean(-1)).unsqueeze(-1)
        return (1.0-((y-yhat)**2).mean(-1)/(y.var(-1)+self.eps)).clamp(0,1).mean()

    def _log_var_ratio(self, LL, LH):
        vLH = LH.var(-1).mean() + self.eps
        vLL = LL.var(-1).mean() + self.eps
        return torch.sigmoid(torch.log(vLH/vLL)+2.0)

    def forward(self, LL, LH):
        feats = torch.stack([self._acf1(LH), self._ar1_r2(LH),
                              self._log_var_ratio(LL,LH)]).unsqueeze(0)
        return torch.sigmoid(self.combiner(feats).squeeze()*3.0)


# =============================================================================
# SECTION 9 — QUALITY: SSL LH PREDICTOR
# =============================================================================

class SSLLHPredictor(nn.Module):
    """
    ssl_loss:    GRU next-step MSE on LH — forces DWT filters to be predictable
    ssl_quality: per-channel ACF (NOT GRU fit) — directly measures structure
    """
    def __init__(self, d_model, hidden_size=32, dropout=0.1):
        super().__init__()
        self.proj_in  = nn.Linear(d_model, 8)
        self.gru      = nn.GRU(8, hidden_size, batch_first=True)
        self.proj_out = nn.Linear(hidden_size, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, LH):                       # (B, D, T)
        B, D, T = LH.shape
        ssl_loss = torch.tensor(0.0, device=LH.device)
        if T >= 3:
            xi = LH.permute(0,2,1)[:,:-1,:]     # (B, T-1, D)
            xt = LH.permute(0,2,1)[:,1:,:]
            go,_ = self.gru(self.proj_in(xi))
            ssl_loss = F.mse_loss(self.proj_out(self.drop(go)), xt)

        # ACF-based quality (not GRU fit — GRU on noise is trivially easy)
        if T < 3:
            return ssl_loss, torch.zeros(1, device=LH.device)
        dz  = min(0.35, 1.5/math.sqrt(T))
        x   = LH.reshape(-1,T) - LH.reshape(-1,T).mean(-1,keepdim=True)
        v   = (x**2).mean(-1) + 1e-6
        acf = ((x[:,:-1]*x[:,1:]).mean(-1)/v).abs()
        q   = ((acf-dz).clamp(0)/(1.0-dz)).mean().clamp(0,1)
        return ssl_loss, q


# =============================================================================
# SECTION 10 — WRAT BLOCK (SSM lanes)
# =============================================================================

class WRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, tau=0.1, dropout=0.3):
        super().__init__()
        self.env = EnvelopeExtractor(d_model)
        self.aLL = FrequencySparseAttention(d_model, num_heads)
        self.aLH = FrequencySparseAttention(d_model, num_heads, threshold=tau)
        self.tp  = nn.Linear(d_model, d_model)
        self.gp  = nn.Linear(d_model, d_model)
        self.sLL = SelectiveSSM(d_model, dropout=dropout)
        self.sLH = SelectiveSSM(d_model, dropout=dropout)
        self.n1  = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)

    def forward(self, LL, LH):
        He = self.env(LH).transpose(1,2); Ls = LL.transpose(1,2)
        Lo = self.aLL(Ls,Ls,Ls)
        Ho = self.aLH(He,He,He, ec=He)
        fused = self.tp(Lo)*torch.sigmoid(self.gp(Ho)) + Ls
        Lf = self.sLL(self.n1(fused)) + fused
        Hf = self.sLH(self.n2(Ho+He)) + Ho
        return Lf.transpose(1,2), Hf.transpose(1,2), Lo, Ho


class LearnableTauWRAT(nn.Module):
    def __init__(self, d_model, num_heads, tau_init=0.1, dropout=0.3):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(math.log(tau_init/(1-tau_init))))
        self._b  = WRATBlock(d_model, num_heads, tau_init, dropout)

    @property
    def tau(self): return torch.sigmoid(self.raw).item()

    def forward(self, LL, LH):
        self._b.aLH.tau = torch.sigmoid(self.raw)
        return self._b(LL, LH)


# =============================================================================
# SECTION 11 — COMPLETE MODEL: PWSAComplete
# =============================================================================

class PWSAComplete(nn.Module):
    """
    P-WSA++ v3 — lean architecture with VariateAttention.

    Changes from v2:
      - LH branch REMOVED (confirmed noise on ETTm1 by quality system)
      - VariateAttention ADDED before patching (true multivariate learning)
      - d_model increased from 64 to 96 (freed compute from LH branch)
      - FITS still present (captures low-frequency residual)
      - VariateGraphMixer kept for output-level mixing

    Forward flow:
      (B,C,L) →[VariateAttention]→ (B,C,L) →[per-variate CI]→
      RevIN → PatchEmb → DWT(LL only) → WRATBlock(LL) →
      TrendHead + FITSBypass → GraphMixer → output
    """
    def __init__(self, seq_len, pred_len, d_model=96, num_heads=4,
                 patch_len=16, stride=8, tau_init=0.1, dropout=0.2,
                 num_variates=7, cut_freq=None):
        super().__init__()
        self.pred_len    = pred_len
        self.num_variates = num_variates
        self._epoch      = 0   # kept for compatibility

        # ── True multivariate: attends across variates on raw input ───
        self.variate_attn = VariateAttention(seq_len, num_variates,
                                             d_attn=d_model, num_heads=num_heads,
                                             dropout=dropout)

        # ── Per-variate CI pipeline ────────────────────────────────────
        self.revin = RevIN(1)
        self.patch = PatchEmbedding(patch_len, stride, d_model)
        self.dwt   = InfoConstrainedDWT(d_model)
        self.wrat  = LearnableTauWRAT(d_model, num_heads, tau_init, dropout)
        self.fits  = FITSBypass(seq_len, pred_len, cut_freq)
        self.gm    = VariateGraphMixer(num_variates)

        # Head dim
        ps      = seq_len + stride
        num_p   = (ps - patch_len) // stride + 1
        pv      = (4-2)//2
        dwt_len = (num_p + 2*pv - 4)//2 + 1
        self.hd = dwt_len * d_model

        self.th = nn.Sequential(
            nn.Flatten(1), nn.Dropout(dropout), nn.Linear(self.hd, pred_len)
        )

    def forward(self, x):
        """
        x: (B, 1, seq_len) — single variate slice (CI dispatch from training loop)
        The training loop extracts x_all: (B, C, L), runs variate_attn on it,
        then slices per variate before calling this.
        """
        xn      = self.revin(x, 'norm')
        patches = self.patch(xn)
        LL, LH  = self.dwt(patches)

        # LL-only: LH branch removed (confirmed noise by quality system)
        LLf, _, LLl, LHl = self.wrat(LL, LH)

        # Trend head + FITS residual
        pt    = self.th(LLf)                              # (B, pred_len)
        preds = pt.unsqueeze(1) + self.fits(xn)           # trend + low-freq residual

        preds = self.revin(preds, 'denorm')
        rec   = self.dwt.inverse(LLf, LLf)  # symmetric recon from LL only

        # Return simplified tuple (loss function updated to match)
        return preds, rec, patches, LLl, LHl


# =============================================================================
# SECTION 12 — LOSS
# =============================================================================

class PWSACompleteLoss(nn.Module):
    def __init__(self, lam_recon=0.05, lam_ortho=0.005, lam_dis=0.01,
                 lam_spec=0.05, lam_nig=0.05, lam_graph=1e-3,
                 lam_ssl=0.1, lam_noise=0.05, lam_mi=0.02):
        super().__init__()
        self.L = dict(recon=lam_recon, ortho=lam_ortho, dis=lam_dis,
                      spec=lam_spec,   nig=lam_nig,     graph=lam_graph,
                      ssl=lam_ssl,     noise=lam_noise,  mi=lam_mi)

    def forward(self, preds, targets, patches, rec, dwt, LLl, LHl,
                LL_raw, LH_raw, alpha, q, ssl_loss,
                g, nu, al, be, gm=None):
        task  = F.mse_loss(preds, targets)
        fp,ft = torch.fft.rfft(preds,dim=-1), torch.fft.rfft(targets,dim=-1)
        spec  = F.l1_loss(fp.abs(), ft.abs())
        ml    = min(patches.shape[-1], rec.shape[-1])
        recon = F.mse_loss(rec[...,:ml], patches[...,:ml])
        ortho = dwt.orthogonality_loss()
        dis   = F.cosine_similarity(LLl.flatten(1), LHl.flatten(1), -1).abs().mean()

        # NIG evidential loss
        yf    = targets.squeeze(1)
        om    = 2.0*be*(1.0+nu)
        nig_l = (0.5*torch.log(torch.pi/(nu+1e-8)) - al*torch.log(om+1e-8)
                 + (al+0.5)*torch.log(nu*(yf-g)**2+om+1e-8)
                 + torch.lgamma(al+1e-8) - torch.lgamma(al+0.5+1e-8)
                 + 1e-2*(yf-g).abs()*(2*nu+al)).mean()

        noise_pen = dwt.noise_allocation_loss(LH_raw, q, alpha)
        mi_pen    = dwt.mi_proxy_loss(LH_raw, targets)
        graph_l   = gm.sparsity_loss() if gm else torch.tensor(0.0, device=preds.device)

        L = self.L
        total = (task + L['spec']*spec + L['recon']*recon + L['ortho']*ortho
                 + L['dis']*dis + L['nig']*nig_l + L['ssl']*ssl_loss
                 + L['noise']*noise_pen + L['mi']*mi_pen + L['graph']*graph_l)

        return total, dict(
            task=task.item(), spec=spec.item(), recon=recon.item(),
            ortho=ortho.item(), dis=dis.item(), nig=nig_l.item(),
            ssl=float(ssl_loss.item() if hasattr(ssl_loss,'item') else ssl_loss),
            noise=noise_pen.item(), mi=mi_pen.item() if hasattr(mi_pen,'item') else 0.0,
            graph=graph_l.item(), q=q.item(), alpha=alpha.item())


# =============================================================================
# SECTION 13 — DATASET (adaptive splits)
# =============================================================================

def _splits(n):
    te = max(1, min(int(n*0.6), n-2))
    ve = max(te+1, min(int(n*0.8), n-1))
    return te, ve

def _compute_splits(n_rows: int):
    """
    Compute train/val/test split boundaries that are ALWAYS valid
    regardless of dataset length.

    Standard ETTm1 proportions (60 / 20 / 20) with a hard minimum
    of (seq_len + pred_len + 1) rows per split enforced at call time.

    Returns (train_end, val_end) row indices into the full array.
    """
    train_end = int(n_rows * 0.60)
    val_end   = int(n_rows * 0.80)
    # Never let a split boundary sit at or beyond the end of the array
    train_end = max(1, min(train_end, n_rows - 2))
    val_end   = max(train_end + 1, min(val_end,   n_rows - 1))
    return train_end, val_end


def preflight_check(path: str, seq_len: int, pred_len: int) -> int:
    """
    Load CSV, print shape + split info, raise a clear error if any
    split is too small to form even one (seq_len + pred_len) window.
    Returns num_variates (columns - 1, dropping the date column).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ETTm1.csv not found at '{path}'.\n"
            "Download from: https://github.com/zhouhaoyi/ETDataset"
        )
    df   = pd.read_csv(path)
    data = df.iloc[:, 1:].values          # drop date/timestamp column
    n, c = data.shape
    te, ve = _compute_splits(n)
    min_rows = seq_len + pred_len + 1

    splits = {
        'train': data[:te],
        'val'  : data[te:ve],
        'test' : data[ve:],
    }
    print(f"\n[DATA] {path}  →  {n} rows × {c} variates")
    print(f"       Split boundaries: train=[0:{te}]  val=[{te}:{ve}]  test=[{ve}:{n}]")
    for name, arr in splits.items():
        status = '✓' if len(arr) >= min_rows else '✗ TOO SMALL'
        print(f"       {name:6}: {len(arr):>6} rows  (need ≥ {min_rows})  {status}")
    print()

    for name, arr in splits.items():
        if len(arr) < min_rows:
            raise ValueError(
                f"Split '{name}' has only {len(arr)} rows but seq_len+pred_len+1={min_rows}.\n"
                f"Either reduce seq_len/pred_len or use a larger dataset.\n"
                f"Dataset has {n} total rows."
            )
    return c   # number of variates


class ETTDataset(Dataset):
    """
    Adaptive-split ETT dataset.  Splits are always 60/20/20 of
    however many rows the CSV actually contains, so this works
    on any size ETT file (m1, m2, h1, h2, or a small test copy).
    """
    def __init__(self, seq_len, pred_len, split='train', path='ETTm1.csv'):
        self.seq_len = seq_len
        self.pred_len = pred_len

        df   = pd.read_csv(path)
        data = df.iloc[:, 1:].values.astype(np.float32)   # drop date col
        n    = len(data)
        te, ve = _compute_splits(n)

        raw = {'train': data[:te],
               'val'  : data[te:ve],
               'test' : data[ve:]}[split]

        # Always fit scaler on training portion only
        sc = StandardScaler()
        sc.fit(data[:te])
        self.data = torch.tensor(sc.transform(raw), dtype=torch.float32)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, i):
        x = self.data[i          : i + self.seq_len]
        y = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]
        return x.t(), y.t()   # (C, seq_len), (C, pred_len)


def get_loaders(seq_len, pred_len, bs=32, path='ETTm1.csv'):
    """Build train/val/test DataLoaders with a sanity check on lengths."""
    datasets = {s: ETTDataset(seq_len, pred_len, s, path)
                for s in ('train', 'val', 'test')}

    for name, ds in datasets.items():
        if len(ds) == 0:
            raise ValueError(
                f"Dataset split '{name}' produced 0 windows.\n"
                f"  rows in split : check preflight_check() output\n"
                f"  seq_len={seq_len}  pred_len={pred_len}\n"
                "  Either the split is empty or seq_len+pred_len > split size."
            )

    tr = DataLoader(datasets['train'], bs, shuffle=True,  drop_last=True,  num_workers=0)
    va = DataLoader(datasets['val'],   bs, shuffle=False, drop_last=True,  num_workers=0)
    te = DataLoader(datasets['test'],  bs, shuffle=False, drop_last=False, num_workers=0)
    return tr, va, te


# =============================================================================
# SECTION 14 — OPTIMIZER (separate LR for LH branch)
# =============================================================================

def make_optimizer(model, lr=1e-4, lh_lr_mult=3.0):
    """LH branch gets 3x higher LR so detail head catches up to trend head."""
    lh_ids   = {id(p) for p in model.lh_params()}
    trend_ps = [p for p in model.parameters() if id(p) not in lh_ids]
    lh_ps    = [p for p in model.parameters() if id(p) in lh_ids]
    return optim.AdamW([
        {'params': trend_ps, 'lr': lr,            'weight_decay': 1e-4},
        {'params': lh_ps,    'lr': lr*lh_lr_mult, 'weight_decay': 1e-4},
    ])


# =============================================================================
# SECTION 15 — TRAINING UTILITIES
# =============================================================================

def train_one_epoch(model, loader, opt, crit, device, num_variates):
    model.train()
    task_losses, term_log = [], defaultdict(list)
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        B, C, L = bx.shape

        # VariateAttention: all 7 variates attend to each other on raw input
        bx_va = model.variate_attn(bx)              # (B, C, L) — context-enriched
        bxf   = bx_va.reshape(B*C, 1, L)
        byf   = by.reshape(B*C, 1, -1)

        opt.zero_grad()
        preds, rec, patches, LLl, LHl = model(bxf)

        # Graph mixer over variate dimension (output-level mixing)
        pbc   = preds.squeeze(1).reshape(B, C, -1)
        preds = model.gm(pbc).reshape(B*C, 1, -1)

        loss, terms = crit(preds, byf, patches, rec,
                           model.dwt, LLl, LHl, model.gm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        task_losses.append(terms['task'])
        for k, v in terms.items(): term_log[k].append(v)
    return np.mean(task_losses), {k: np.mean(v) for k, v in term_log.items()}


@torch.no_grad()
def evaluate(model, loader, device, num_variates):
    model.eval()
    all_p, all_t = [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        B, C, L = bx.shape
        bx_va = model.variate_attn(bx)              # cross-variate enrichment
        preds = model(bx_va.reshape(B*C, 1, L))[0]
        pbc   = preds.squeeze(1).reshape(B, C, -1)
        preds = model.gm(pbc).reshape(B*C, 1, -1)
        all_p.append(preds.reshape(B, C, -1).cpu())
        all_t.append(by.cpu())
    p = torch.cat(all_p).flatten().numpy()
    t = torch.cat(all_t).flatten().numpy()
    e = p - t
    return dict(mse=float((e**2).mean()), mae=float(np.abs(e).mean()),
                dir_acc=float((np.sign(np.diff(p))==np.sign(np.diff(t))).mean()*100))


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience=patience; self.counter=0; self.best=None
        self.early_stop=False; self.state=None
    def __call__(self, val_loss, model):
        if self.best is None or val_loss < self.best:
            self.best=val_loss; self.counter=0
            self.state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            self.counter+=1
            if self.counter>=self.patience: self.early_stop=True


# =============================================================================
# SECTION 16 — MAIN TRAINING LOOP
# =============================================================================

CSV_PATH     = 'ETTm1.csv'
SEQ_LEN      = 336         # 3.5 daily cycles visible
D_MODEL      = 96          # up from 64: freed from removed LH branch
PATCH_LEN    = 16
STRIDE       = 8
NUM_VARIATES = 7
HORIZONS     = [96, 192, 336, 720]
BATCH_SIZE   = 32
EPOCHS       = 50          # longer for warmup + better winter generalisation
PATIENCE     = 15
LR           = 1e-4

# Verify CSV exists and splits are valid
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Download from github.com/zhouhaoyi/ETDataset")

df = pd.read_csv(CSV_PATH); n_rows = len(df)
te, ve = _splits(n_rows)
print(f"[DATA] {CSV_PATH}: {n_rows} rows x {len(df.columns)-1} variates")
print(f"       train=[0:{te}]  val=[{te}:{ve}]  test=[{ve}:{n_rows}]")
min_rows = SEQ_LEN + max(HORIZONS) + 1
for split, size in [('train',te), ('val',ve-te), ('test',n_rows-ve)]:
    status = 'OK' if size >= min_rows else 'TOO SMALL'
    print(f"       {split}: {size} rows [{status}]")

crit = PWSACompleteLoss()
final_results = {}

for PRED_LEN in HORIZONS:
    print(f"\n{'='*80}\n HORIZON = {PRED_LEN}\n{'='*80}")
    tr, va, te_loader = get_loaders(SEQ_LEN, PRED_LEN, BATCH_SIZE, CSV_PATH)

    torch.manual_seed(SEED)
    model = PWSAComplete(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=D_MODEL,
        num_heads=4, patch_len=PATCH_LEN, stride=STRIDE,
        tau_init=0.1, dropout=0.2, num_variates=NUM_VARIATES,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f" Parameters: {n_params:,}")

    opt = make_optimizer(model, lr=LR)
    # Linear warmup 5 epochs → cosine decay
    warmup = optim.lr_scheduler.LambdaLR(opt, lambda ep: min((ep+1)/5.0, 1.0))
    sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    es  = EarlyStopping(PATIENCE)

    val_curve, train_curve, q_curve, alpha_curve = [], [], [], []

    print(f" {'Ep':>3} | {'Train MSE':>10} | {'Val MSE':>10} | {'task':>8} | {'tau':>6}")
    print(f" {'-'*55}")

    for epoch in range(1, EPOCHS+1):
        model._epoch = epoch
        tr_loss, terms = train_one_epoch(model, tr, opt, crit, DEVICE, NUM_VARIATES)
        val_m = evaluate(model, va, DEVICE, NUM_VARIATES)['mse']
        if epoch <= 5: warmup.step()
        else:         sch.step()

        train_curve.append(tr_loss)
        val_curve.append(val_m)
        q_curve.append(0.0)
        alpha_curve.append(0.0)

        es(val_m, model)
        tau = model.wrat.tau

        tau = model.wrat.tau
        print(f" {epoch:>3} | {tr_loss:>10.5f} | {val_m:>10.5f} | {tr_loss:>8.5f} | {tau:>6.3f}")

        if es.early_stop:
            print(f" Early stop at epoch {epoch}")
            break

    model.load_state_dict(es.state)
    test_m = evaluate(model, te_loader, DEVICE, NUM_VARIATES)
    final_results[PRED_LEN] = test_m
    print(f"\n TEST H={PRED_LEN}: MSE={test_m['mse']:.4f} MAE={test_m['mae']:.4f} DirAcc={test_m['dir_acc']:.2f}%")

    # LH branch removed in v3

    # ── Plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train_curve, label='train', color='#2563eb', linestyle='--', alpha=0.7)
    axes[0].plot(val_curve,   label='val',   color='#2563eb', linewidth=2)
    axes[0].set_title(f'MSE curves H={PRED_LEN}'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(q_curve,     color='#f59e0b', linewidth=2)
    axes[1].set_title('LH quality q'); axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3)
    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    axes[2].plot(alpha_curve, color='#7c3aed', linewidth=2)
    axes[2].set_title('Gate alpha (detail contribution)')
    axes[2].set_ylim(0, 0.2); axes[2].grid(alpha=0.3)

    plt.suptitle(f'P-WSA++ Complete — H={PRED_LEN}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'plots/quality/H{PRED_LEN}_quality.png', dpi=150)
    plt.close()

# ── Final summary table ───────────────────────────────────────────
print(f"\n{'='*60}")
print(f" FINAL RESULTS SUMMARY (seq_len={SEQ_LEN})")
print(f"{'='*60}")
print(f" {'H':>5} | {'MSE':>8} | {'MAE':>8} | {'DirAcc':>8}")
print(f" {'-'*40}")
for h, r in final_results.items():
    print(f" {h:>5} | {r['mse']:>8.4f} | {r['mae']:>8.4f} | {r['dir_acc']:>7.2f}%")