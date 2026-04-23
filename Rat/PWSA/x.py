# =============================================================================
# benchmark_pwsa_vs_wrat.py  —  Head-to-Head: P-WSA v8 vs PatchWRAT
# =============================================================================
# Goal: Determine which architecture generalises better on ETTm1.
# Both models are tested under IDENTICAL conditions:
#   • Same data splits (60/20/20)
#   • Same horizons: [96, 192, 336, 720]
#   • Same seq_len: 336
#   • Same batch size: 64
#   • Same optimiser: AdamW + CosineAnnealing
#   • Same early stopping: patience=10
#   • Same evaluation metric: MSE & MAE on test set
#
# Usage:
#   python benchmark_pwsa_vs_wrat.py
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
from collections import defaultdict

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)

CSV_PATH     = 'ETTm1.csv'
SEQ_LEN      = 336
HORIZONS     = [96, 192, 336, 720]
BATCH_SIZE   = 64
EPOCHS       = 50          # capped for fair speed comparison; raise to 100 for full run
PATIENCE     = 10
LR           = 3e-4
SOTA_MSE     = {96: 0.334, 192: 0.377, 336: 0.426, 720: 0.491}  # iTransformer reference

print(f"[ENV]  Device : {DEVICE} | PyTorch {torch.__version__}")
print(f"[DATA] {CSV_PATH}  seq={SEQ_LEN}  horizons={HORIZONS}")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _splits(n):
    te = max(1, min(int(n * 0.60), n - 2))
    ve = max(te + 1, min(int(n * 0.80), n - 1))
    return te, ve


class ETTDataset(Dataset):
    """
    Loads ETTm1.csv.  mode='multi' returns all 7 variates (for P-WSA v8).
    mode='uni' returns only the OT column (for PatchWRAT, univariate).
    Splits: 60 / 20 / 20  (identical for both models).
    """
    def __init__(self, seq_len, pred_len, split='train', mode='multi'):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        df   = pd.read_csv(CSV_PATH)
        if mode == 'uni':
            raw_all = df[['OT']].values.astype(np.float32)
        else:
            raw_all = df.iloc[:, 1:].values.astype(np.float32)

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


def get_loaders(pred_len, mode='multi'):
    kw = dict(num_workers=0, pin_memory=False)
    tr = DataLoader(ETTDataset(SEQ_LEN, pred_len, 'train', mode), BATCH_SIZE, shuffle=True,  drop_last=True,  **kw)
    va = DataLoader(ETTDataset(SEQ_LEN, pred_len, 'val',   mode), BATCH_SIZE, shuffle=False, drop_last=True,  **kw)
    te = DataLoader(ETTDataset(SEQ_LEN, pred_len, 'test',  mode), BATCH_SIZE, shuffle=False, drop_last=False, **kw)
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
def evaluate(model, loader, device, model_type='pwsa'):
    model.eval()
    all_p, all_t = [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        if model_type == 'wrat':
            preds, _, _, _ = model(bx)
        else:
            preds = model(bx)
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
# MODEL A — P-WSA v8  (Multivariate, MLP-Mixer + Fixed Haar DWT + FITS)
# ─────────────────────────────────────────────────────────────────────────────

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rt = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * rt.floor_()


class MovingAvgDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.k   = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):
        p = (self.k - 1) // 2
        x_pad = torch.cat([x[..., :1].expand(*x.shape[:-1], p),
                            x,
                            x[..., -1:].expand(*x.shape[:-1], p)], dim=-1)
        trend = self.avg(x_pad)
        return x - trend, trend


class RevIN_A(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(1))
        self.b   = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean  = x.mean(-1, keepdim=True).detach()
            self.stdev = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            return ((x - self.mean) / self.stdev) * self.w.unsqueeze(-1) + self.b.unsqueeze(-1)
        x = (x - self.b.unsqueeze(-1)) / (self.w.unsqueeze(-1) + self.eps**2)
        return x * self.stdev + self.mean


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj      = nn.Linear(patch_len, d_model)
        self.norm      = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x):
        pad = x[..., -1:].expand(*x.shape[:-1], self.stride)
        x   = torch.cat([x, pad], dim=-1)
        x   = x.unfold(-1, self.patch_len, self.stride).squeeze(1)
        return self.drop(self.norm(self.proj(x))).transpose(1, 2)


def haar_dwt(x):
    s = 0.7071067811865476
    return (x[..., 0::2] + x[..., 1::2]) * s, (x[..., 0::2] - x[..., 1::2]) * s


class LearnableWaveletThreshold(nn.Module):
    def __init__(self, d_model, init_val=0.05):
        super().__init__()
        self.thresh = nn.Parameter(torch.full((1, d_model, 1), init_val))

    def forward(self, LH):
        t = F.softplus(self.thresh)
        return torch.sign(LH) * F.relu(torch.abs(LH) - t)


class PatchMixer(nn.Module):
    def __init__(self, num_patches, d_model, expansion=2, dropout=0.2, dp_prob=0.1):
        super().__init__()
        self.dp_prob = dp_prob
        self.norm1   = nn.LayerNorm(num_patches)
        self.fc1     = nn.Linear(num_patches, num_patches * expansion)
        self.fc2     = nn.Linear(num_patches * expansion, num_patches)
        self.norm2   = nn.LayerNorm(d_model)
        self.fc3     = nn.Linear(d_model, d_model * expansion)
        self.fc4     = nn.Linear(d_model * expansion, d_model)
        self.drop    = nn.Dropout(dropout)
        self.ls1     = nn.Parameter(torch.full((d_model,),     1e-4))
        self.ls2     = nn.Parameter(torch.full((num_patches,), 1e-4))

    def forward(self, x):
        r  = x
        tx = self.drop(F.gelu(self.fc1(self.norm1(x))))
        tx = self.drop(self.fc2(tx))
        x  = r + drop_path(tx * self.ls1.view(1, -1, 1), self.dp_prob, self.training)
        r  = x
        cx = self.drop(F.gelu(self.fc3(self.norm2(x.transpose(1, 2)))))
        cx = self.drop(self.fc4(cx))
        x  = r + drop_path(cx.transpose(1, 2) * self.ls2.view(1, 1, -1), self.dp_prob, self.training)
        return x


class FITSBypass(nn.Module):
    def __init__(self, seq_len, pred_len, cut_freq=16):
        super().__init__()
        self.cut = cut_freq
        self.dec = nn.Linear(cut_freq * 2, pred_len)
        nn.init.xavier_uniform_(self.dec.weight, gain=0.05)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        f = torch.fft.rfft(x, dim=-1)[..., :self.cut]
        return self.dec(torch.cat([f.real, f.imag], dim=-1).squeeze(1)).unsqueeze(1)


class PWSAv8(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=64, n_layers=3,
                 patch_len=16, stride=8, dropout=0.2, dp_prob=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.decomp   = MovingAvgDecomp(25)
        self.revin    = RevIN_A()
        self.patch    = PatchEmbedding(patch_len, stride, d_model, dropout * 0.5)
        self.fits     = FITSBypass(seq_len, pred_len)
        self.wt       = LearnableWaveletThreshold(d_model)
        ps            = seq_len + stride
        num_p         = (ps - patch_len) // stride + 1
        self.mixers   = nn.ModuleList([
            PatchMixer(num_p, d_model, dropout=dropout, dp_prob=dp_prob)
            for _ in range(n_layers)
        ])
        self.head_norm  = nn.LayerNorm(d_model * num_p)
        self.head       = nn.Linear(d_model * num_p, pred_len)
        self.head_drop  = nn.Dropout(dropout)
        self.trend_proj = nn.Linear(seq_len, pred_len)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1); nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.trend_proj.weight, gain=0.1); nn.init.zeros_(self.trend_proj.bias)

    def _per_variate(self, x):
        res, trend = self.decomp(x)
        rn  = self.revin(res, 'norm')
        p   = self.patch(rn)
        LL, LH = haar_dwt(p)
        out = torch.cat([LL, self.wt(LH)], dim=-1)
        for m in self.mixers: out = m(out)
        pr  = self.head(self.head_drop(self.head_norm(out.flatten(1)))).unsqueeze(1)
        pr  = pr + self.fits(rn)
        pr  = self.revin(pr, 'denorm')
        return pr + self.trend_proj(trend)

    def forward(self, x):
        B, C, _ = x.shape
        return torch.cat([self._per_variate(x[:, c:c+1, :]) for c in range(C)], dim=1)


def pwsa_loss(preds, targets, lam_spec=0.1):
    mse  = F.mse_loss(preds, targets)
    spec = F.l1_loss(torch.fft.rfft(preds, dim=-1).abs(),
                     torch.fft.rfft(targets, dim=-1).abs())
    return mse + lam_spec * spec, mse.item()


def train_pwsa(model, loader, opt):
    model.train()
    losses = []
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        loss, mse = pwsa_loss(model(bx), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        losses.append(mse)
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL B — PatchWRAT  (Univariate, Learned Conv DWT + Sparse Attention)
# ─────────────────────────────────────────────────────────────────────────────

class RevIN_B(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(1))
        self.b   = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean  = x.mean(-1, keepdim=True).detach()
            self.stdev = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            return ((x - self.mean) / self.stdev) * self.w.unsqueeze(-1) + self.b.unsqueeze(-1)
        x = (x - self.b.unsqueeze(-1)) / (self.w.unsqueeze(-1) + self.eps**2)
        return x * self.stdev + self.mean


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

    def forward(self, q_x, k_x, v_x, energy=None):
        B, L, D = q_x.shape; Dh = D // self.H
        Q = self.q(q_x).view(B, L, self.H, Dh).transpose(1, 2)
        K = self.k(k_x).view(B, -1, self.H, Dh).transpose(1, 2)
        V = self.v(v_x).view(B, -1, self.H, Dh).transpose(1, 2)
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        if energy is not None:
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

    @property
    def tau(self): return torch.sigmoid(self.raw_tau).item()

    def forward(self, LL, LH):
        tau = torch.sigmoid(self.raw_tau).item()
        self.lh_attn.threshold = tau
        ll_s = LL.transpose(1, 2); lh_s = LH.transpose(1, 2)
        ll_o = self.ll_attn(ll_s, ll_s, ll_s)
        lh_o = self.lh_attn(lh_s, lh_s, lh_s, energy=lh_s)
        cr_o = self.cross_attn(ll_o, lh_o, lh_o)
        ll_f = self.n1(ll_s + ll_o + cr_o);  ll_f = self.mlp_ll(ll_f) + ll_f
        lh_f = self.n2(lh_s + lh_o);         lh_f = self.mlp_lh(lh_f) + lh_f
        return ll_f.transpose(1, 2), lh_f.transpose(1, 2)


class PatchWRAT(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=32, num_heads=4, dropout=0.3):
        super().__init__()
        self.revin = RevIN_B()
        self.dwt   = LearnableDWT1D(1, d_model)
        self.wrat  = WRATBlock(d_model, num_heads, dropout=dropout)
        l_half     = seq_len // 2
        flat_dim   = l_half * d_model * 2
        self.head  = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, pred_len)
        )
        self.sc    = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        xn           = self.revin(x, 'norm')
        LL, LH       = self.dwt(xn)
        LL_o, LH_o  = self.wrat(LL, LH)
        fused        = torch.cat([LL_o, LH_o], dim=1)
        preds        = self.revin(self.head(fused).unsqueeze(1), 'denorm')
        xr           = self.sc(self.revin(self.dwt.inverse(LL_o, LH_o), 'denorm'))
        return preds, xr, LL, LH


def wrat_loss(preds, targets, x_orig, x_recon, dwt, lam_r=1.0, lam_o=0.1):
    task  = F.mse_loss(preds, targets)
    n     = min(x_orig.shape[-1], x_recon.shape[-1])
    recon = F.mse_loss(x_recon[..., :n], x_orig[..., :n])
    h_f   = dwt.h.view(dwt.h.shape[0], -1)
    g_f   = dwt.g.view(dwt.g.shape[0], -1)
    ortho = (h_f * g_f).sum().abs()
    return task + lam_r * recon + lam_o * ortho, task.item()


def train_wrat(model, loader, opt):
    model.train()
    losses = []
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        preds, xr, _, _ = model(bx)
        loss, mse = wrat_loss(preds, by, bx, xr, model.dwt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(mse)
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  (shared, parametric)
# ─────────────────────────────────────────────────────────────────────────────

def run_model(name, model, train_fn, val_loader, test_loader, mtype):
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warmup, cos = make_scheduler(opt, EPOCHS)
    es  = EarlyStopping(PATIENCE)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_fn(model, tr_loader, opt)
        val_m   = evaluate(model, val_loader, DEVICE, mtype)['mse']
        if epoch <= 5: warmup.step()
        else:          cos.step()
        improved = es(val_m, model)
        if epoch % 10 == 0 or improved:
            print(f"    [{name}] ep{epoch:>3}  train={tr_loss:.5f}  val={val_m:.5f}"
                  f"  {'✓' if improved else ''}")
        if es.early_stop:
            print(f"    [{name}] Early stop @ ep{epoch}  best_val={es.best:.5f}")
            break

    model.load_state_dict(es.state)
    return evaluate(model, test_loader, DEVICE, mtype)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found.")

results = {h: {} for h in HORIZONS}

for PRED_LEN in HORIZONS:
    print(f"\n{'='*70}")
    print(f"  HORIZON = {PRED_LEN}  |  seq={SEQ_LEN}")
    print(f"{'='*70}")

    # ── P-WSA v8  (multivariate) ──
    tr_loader, va_loader, te_loader = get_loaders(PRED_LEN, mode='multi')
    torch.manual_seed(SEED)
    pwsa = PWSAv8(SEQ_LEN, PRED_LEN, d_model=64, n_layers=3,
                  patch_len=16, stride=8, dropout=0.2, dp_prob=0.1).to(DEVICE)
    n_pwsa = sum(p.numel() for p in pwsa.parameters() if p.requires_grad)
    print(f"\n  [P-WSA v8]  params={n_pwsa:,}  mode=multivariate(7)")
    res_pwsa = run_model('P-WSA v8', pwsa, train_pwsa, va_loader, te_loader, 'pwsa')
    results[PRED_LEN]['pwsa'] = res_pwsa

    # ── PatchWRAT  (univariate — OT only) ──
    tr_loader, va_loader, te_loader = get_loaders(PRED_LEN, mode='uni')
    torch.manual_seed(SEED)
    wrat = PatchWRAT(SEQ_LEN, PRED_LEN, d_model=32, num_heads=4, dropout=0.3).to(DEVICE)
    n_wrat = sum(p.numel() for p in wrat.parameters() if p.requires_grad)
    print(f"\n  [PatchWRAT] params={n_wrat:,}  mode=univariate(OT)")
    res_wrat = run_model('PatchWRAT', wrat, train_wrat, va_loader, te_loader, 'wrat')
    results[PRED_LEN]['wrat'] = res_wrat

    # ── Per-horizon summary ──
    print(f"\n  ┌─ Horizon {PRED_LEN} Results {'─'*38}┐")
    print(f"  │  {'Model':<14} {'MSE':>8}  {'MAE':>8}  {'DirAcc':>8}  {'vs iTransf':>12}  │")
    print(f"  │  {'─'*58}  │")
    for tag, label in [('pwsa','P-WSA v8'), ('wrat','PatchWRAT')]:
        r   = results[PRED_LEN][tag]
        gap = r['mse'] - SOTA_MSE.get(PRED_LEN, 0)
        sgn = '+' if gap > 0 else '-'
        print(f"  │  {label:<14} {r['mse']:>8.4f}  {r['mae']:>8.4f}  "
              f"{r['dir_acc']:>7.1f}%  {sgn}{abs(gap):.4f}          │")
    print(f"  └{'─'*62}┘")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("  FINAL BENCHMARK SUMMARY")
print(f"  (iTransformer SOTA reference for ETTm1)")
print(f"{'='*70}")
print(f"  {'H':>5} │ {'P-WSA v8 MSE':>13} │ {'PatchWRAT MSE':>13} │ {'Winner':>12} │ SOTA")
print(f"  {'─'*5}─┼─{'─'*13}─┼─{'─'*13}─┼─{'─'*12}─┼─{'─'*6}")

pwsa_wins = wrat_wins = 0
for h in HORIZONS:
    pm  = results[h]['pwsa']['mse']
    wm  = results[h]['wrat']['mse']
    win = 'P-WSA v8' if pm < wm else 'PatchWRAT'
    if pm < wm: pwsa_wins += 1
    else:        wrat_wins += 1
    sota = SOTA_MSE.get(h, '—')
    print(f"  {h:>5} │ {pm:>13.4f} │ {wm:>13.4f} │ {win:>12} │ {sota}")

print(f"\n  Overall wins  →  P-WSA v8: {pwsa_wins}/4   PatchWRAT: {wrat_wins}/4")

rec = 'P-WSA v8' if pwsa_wins > wrat_wins else ('PatchWRAT' if wrat_wins > pwsa_wins else 'Tie')
print(f"\n  ✦ RECOMMENDATION: {rec}")

if rec == 'P-WSA v8':
    print("    P-WSA v8 wins — the MLP-Mixer + fixed Haar + FITS architecture")
    print("    generalises better across horizons. Proceed with v8 and tune")
    print("    d_model / n_layers / lam_spec for further gains.")
elif rec == 'PatchWRAT':
    print("    PatchWRAT wins — learned conv filters + sparse attention provide")
    print("    better signal decomposition. Consider scaling d_model and adding")
    print("    multi-variate support as the next step.")
else:
    print("    Both models tied. Consider an ensemble or deeper ablation study.")

print(f"\n{'='*70}\n")