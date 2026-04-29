# =============================================================================
# run_pwsa_v8.py  —  P-WSA v8.0 (The Ideal Generalization Baseline)
# =============================================================================
# FINAL ARCHITECTURE FOR TEST-SET ALIGNMENT:
# [1] Trend Dropout Removed: Trend projection is pure, preventing baseline collapse.
# [2] Zero-Jitter Default: Allows perfect calibration to the Pure MSE loss metric.
# [3] Wavelet Calibration: Learnable threshold initializes at 0.05 for better
#     early-epoch high-frequency retention.
# [4] Strict CI & Decomposition: Maintained for absolute distribution shift robustness.
# =============================================================================

from __future__ import annotations
import os, math, warnings, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    from pwsa_plots import PlotManager
    PM = PlotManager(base_dir='plots')
    print("[PLOTS] PlotManager loaded.")
except ImportError:
    PM = None
    print("[PLOTS] pwsa_plots.py not found - plots disabled.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"[ENV] Device: {DEVICE} | PyTorch {torch.__version__}")


# ── Stochastic Depth (DropPath) ──────────────────────────────────────────────

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  
    return x.div(keep_prob) * random_tensor


# ── Moving Average Decomposition ─────────────────────────────────────────────

class MovingAvgDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end   = x[:, 0:1, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=-1)
        trend = self.avg(x_pad)
        res = x - trend
        return res, trend


# ── Learnable Wavelet Multiresolution ────────────────────────────────────────

def haar_dwt(x):
    s = 0.7071067811865476
    return (x[..., 0::2] + x[..., 1::2]) * s, \
           (x[..., 0::2] - x[..., 1::2]) * s

class LearnableWaveletThreshold(nn.Module):
    def __init__(self, d_model, init_val=0.05): # Lowered to 0.05 for fine-grained detail
        super().__init__()
        self.thresh = nn.Parameter(torch.full((1, d_model, 1), init_val))

    def forward(self, LH):
        t = F.softplus(self.thresh)
        return torch.sign(LH) * F.relu(torch.abs(LH) - t)


# ── RevIN ────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(num_features))
        self.b   = nn.Parameter(torch.zeros(num_features))

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


# ── Patch Embedding ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj      = nn.Linear(patch_len, d_model)
        self.norm      = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, x):                              
        pad = x[..., -1:].expand(*x.shape[:-1], self.stride)
        x   = torch.cat([x, pad], dim=-1)
        x   = x.unfold(-1, self.patch_len, self.stride).squeeze(1)
        return self.drop(self.norm(self.proj(x))).transpose(1, 2)  


# ── Patch Mixer ──────────────────────────────────────────────────────────────

class PatchMixer(nn.Module):
    def __init__(self, num_patches, d_model, expansion=2, dropout=0.2, dp_prob=0.1):
        super().__init__()
        self.dp_prob = dp_prob
        h_tok = num_patches * expansion
        h_ch  = d_model * expansion
        self.norm1 = nn.LayerNorm(num_patches)
        self.fc1   = nn.Linear(num_patches, h_tok)
        self.fc2   = nn.Linear(h_tok, num_patches)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc3   = nn.Linear(d_model, h_ch)
        self.fc4   = nn.Linear(h_ch, d_model)
        self.drop  = nn.Dropout(dropout)
        self.ls1   = nn.Parameter(torch.full((d_model,),     1e-4))
        self.ls2   = nn.Parameter(torch.full((num_patches,), 1e-4))

    def forward(self, x):                              
        r  = x
        tx = self.norm1(x)
        tx = self.drop(F.gelu(self.fc1(tx)))
        tx = self.drop(self.fc2(tx))
        x  = r + drop_path(tx * self.ls1.view(1, -1, 1), self.dp_prob, self.training)

        r  = x
        cx = self.norm2(x.transpose(1, 2))            
        cx = self.drop(F.gelu(self.fc3(cx)))
        cx = self.drop(self.fc4(cx))
        x  = r + drop_path(cx.transpose(1, 2) * self.ls2.view(1, 1, -1), self.dp_prob, self.training)
        return x


# ── FITS Bypass ──────────────────────────────────────────────────────────────

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


# ── Main Model (P-WSA v8) ────────────────────────────────────────────────────

class PWSAv8(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=64, n_layers=3,
                 patch_len=16, stride=8, dropout=0.2, dp_prob=0.1, num_variates=7,
                 mixer_expansion=2, use_fits=True, use_wavelet=True):
        super().__init__()
        self.pred_len     = pred_len
        self.num_variates = num_variates
        self.seq_len      = seq_len
        self.use_fits    = use_fits
        self.use_wavelet = use_wavelet

        self.decomp = MovingAvgDecomp(kernel_size=25)
        self.revin  = RevIN(1)
        self.patch  = PatchEmbedding(patch_len, stride, d_model, dropout=dropout * 0.5)
        
        if self.use_fits:
            self.fits = FITSBypass(seq_len, pred_len)
            
        if self.use_wavelet:
            self.wavelet_thresh = LearnableWaveletThreshold(d_model)

        ps        = seq_len + stride
        num_p_raw = (ps - patch_len) // stride + 1
        num_p = num_p_raw

        self.mixers = nn.ModuleList([
            PatchMixer(num_p, d_model, expansion=mixer_expansion, dropout=dropout, dp_prob=dp_prob)
            for _ in range(n_layers)
        ])

        self.head_norm  = nn.LayerNorm(d_model * num_p)
        self.head       = nn.Linear(d_model * num_p, pred_len)
        self.head_drop  = nn.Dropout(dropout)
        
        # Trend projection: Removed dropout to preserve pure baseline structures
        self.trend_proj = nn.Linear(seq_len, pred_len)
        
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.trend_proj.weight, gain=0.1)
        nn.init.zeros_(self.trend_proj.bias)

        self.d_model = d_model

    def _per_variate(self, x):                        
        res, trend = self.decomp(x)
        
        res_norm = self.revin(res, 'norm')
        patches  = self.patch(res_norm)
        
        if self.use_wavelet:
            LL, LH = haar_dwt(patches)
            LH_den = self.wavelet_thresh(LH)
            out    = torch.cat([LL, LH_den], dim=-1) 
        else:
            out = patches
            
        for mixer in self.mixers:
            out = mixer(out)
            
        flat     = self.head_drop(self.head_norm(out.flatten(1)))
        pred_res = self.head(flat).unsqueeze(1)
        
        if self.use_fits:
            pred_res = pred_res + self.fits(res_norm)
            
        pred_res   = self.revin(pred_res, 'denorm')
        pred_trend = self.trend_proj(trend) # Un-choked
        
        return pred_res + pred_trend

    def forward(self, x_all):                         
        B, C, _ = x_all.shape
        return torch.cat(
            [self._per_variate(x_all[:, c:c+1, :]) for c in range(C)],
            dim=1
        )

    def compat_forward(self, x_all):
        preds = self.forward(x_all)
        dummy = torch.zeros(1, device=x_all.device)
        return preds, dummy, dummy, dummy, dummy


# ── Loss ─────────────────────────────────────────────────────────────────────

class PWSAv8Loss(nn.Module):
    def __init__(self, lam_spec=0.1):
        super().__init__()
        self.lam_spec = lam_spec

    def forward(self, preds, targets):
        mse   = F.mse_loss(preds, targets)
        fp    = torch.fft.rfft(preds,   dim=-1)
        ft    = torch.fft.rfft(targets, dim=-1)
        spec  = F.l1_loss(fp.abs(), ft.abs())
        total = mse + self.lam_spec * spec
        return total, dict(task=mse.item(), mse=mse.item(), huber=0.0, spec=spec.item())


# ── Dataset ──────────────────────────────────────────────────────────────────

def _compute_splits(n):
    te = max(1, min(int(n * 0.60), n - 2))
    ve = max(te + 1, min(int(n * 0.80), n - 1))
    return te, ve


class ETTDataset(Dataset):
    def __init__(self, seq_len, pred_len, split='train', path='ETTm1.csv'):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        df   = pd.read_csv(path)
        data = df.iloc[:, 1:].values.astype(np.float32)
        n    = len(data)
        te, ve = _compute_splits(n)
        raw    = {'train': data[:te], 'val': data[te:ve], 'test': data[ve:]}[split]
        sc     = StandardScaler().fit(data[:te])
        self.data = torch.tensor(sc.transform(raw), dtype=torch.float32)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, i):
        x = self.data[i          : i + self.seq_len]
        y = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]
        return x.t(), y.t()


def get_loaders(seq_len, pred_len, bs=64, path='ETTm1.csv'):
    kw = dict(num_workers=0, pin_memory=True)
    tr = DataLoader(ETTDataset(seq_len, pred_len, 'train', path), bs, shuffle=True,  drop_last=True,  **kw)
    va = DataLoader(ETTDataset(seq_len, pred_len, 'val',   path), bs, shuffle=False, drop_last=True,  **kw)
    te = DataLoader(ETTDataset(seq_len, pred_len, 'test',  path), bs, shuffle=False, drop_last=False, **kw)
    return tr, va, te


# ── Optimiser ────────────────────────────────────────────────────────────────

def make_optimizer(model, lr=3e-4):
    hd_ids = {id(p) for p in list(model.head.parameters()) + list(model.head_norm.parameters()) + list(model.trend_proj.parameters())}
    rest   = [p for p in model.parameters() if id(p) not in hd_ids]
    
    param_groups = [
        {'params': list(model.head.parameters()) + list(model.head_norm.parameters()) + list(model.trend_proj.parameters()), 'lr': lr},
        {'params': rest, 'lr': lr},
    ]

    return optim.AdamW(param_groups, weight_decay=1e-3)


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, opt, crit, device, jitter_std=0.0):
    model.train()
    losses, terms_log = [], defaultdict(list)
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        
        if jitter_std > 0:
            bx = bx + torch.randn_like(bx) * jitter_std
            
        opt.zero_grad()
        preds       = model(bx)
        loss, terms = crit(preds, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        losses.append(terms['task'])
        for k, v in terms.items():
            terms_log[k].append(v)
    return np.mean(losses), {k: np.mean(v) for k, v in terms_log.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        all_p.append(model(bx).cpu())
        all_t.append(by.cpu())
    p = torch.cat(all_p).flatten().numpy()
    t = torch.cat(all_t).flatten().numpy()
    e = p - t
    return dict(mse=float((e**2).mean()), mae=float(np.abs(e).mean()),
                dir_acc=float((np.sign(np.diff(p))==np.sign(np.diff(t))).mean()*100))


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience=patience; self.min_delta=min_delta
        self.counter=0; self.best=None; self.early_stop=False; self.state=None

    def __call__(self, val_loss, model):
        improved = self.best is None or val_loss < self.best - self.min_delta
        if improved:
            self.best=val_loss; self.counter=0
            self.state={k: v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P-WSA v8 Forecasting Model")
    parser.add_argument('--no_fits', action='store_true', help='Disable FITS Bypass')
    parser.add_argument('--no_wavelet', action='store_true', help='Disable Haar DWT')
    parser.add_argument('--jitter', type=float, default=0.0, help='Input jitter std dev (Default: 0.0)')
    parser.add_argument('--dp_prob', type=float, default=0.1, help='DropPath probability')
    args = parser.parse_args()

    CSV_PATH=     'ETTm1.csv'
    SEQ_LEN=      336
    D_MODEL=      64
    N_LAYERS=     3
    PATCH_LEN=    16
    STRIDE=       8
    NUM_VARIATES= 7
    HORIZONS=     [96, 192, 336, 720]
    BATCH_SIZE=   64
    EPOCHS=       100
    PATIENCE=     10
    LR=           3e-4
    SOTA=         {96: 0.334, 192: 0.377, 336: 0.426, 720: 0.491}

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Download: github.com/zhouhaoyi/ETDataset")

    df=pd.read_csv(CSV_PATH); n_rows=len(df); te,ve=_compute_splits(n_rows)
    print(f"\n[DATA] {CSV_PATH}: {n_rows} rows x {len(df.columns)-1} variates")
    
    print("\n[CONFIG v8.0 - Ideal Baseline]")
    print(" LVS:        OFF (Decomposition Active)")
    print(" Attn:       OFF (Strict CI Active)")
    print(f" FITS:       {'OFF' if args.no_fits else 'ON'}")
    print(f" Wavelet:    {'OFF' if args.no_wavelet else 'ON (Learnable Thresholding @ 0.05)'}")
    print(f" Loss:       PURE MSE + Spectral Penalty")
    print(f" Jitter:     {args.jitter} | DropPath: {args.dp_prob}")

    crit=PWSAv8Loss(); final_results={}; all_train_curves={}; all_val_curves={}

    for PRED_LEN in HORIZONS:
        print(f"\n{'='*80}")
        print(f" HORIZON = {PRED_LEN}  |  seq={SEQ_LEN}  d={D_MODEL}  layers={N_LAYERS}")
        print(f"{'='*80}")

        tr, va, te_loader = get_loaders(SEQ_LEN, PRED_LEN, BATCH_SIZE, CSV_PATH)
        torch.manual_seed(SEED)
        
        model = PWSAv8(seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=D_MODEL,
                       n_layers=N_LAYERS, patch_len=PATCH_LEN, stride=STRIDE,
                       dropout=0.2, dp_prob=args.dp_prob, num_variates=NUM_VARIATES,
                       use_fits=not args.no_fits, 
                       use_wavelet=not args.no_wavelet).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" Trainable parameters: {n_params:,}")

        opt    = make_optimizer(model, lr=LR)
        warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5)
        sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS-5), eta_min=LR*0.02)
        es     = EarlyStopping(patience=PATIENCE)
        train_curve, val_curve = [], []

        print(f"\n  {'Ep':>4} | {'Train MSE':>10} | {'Val MSE':>10} | {'Gap':>6} | *")
        print(f"  {'-'*52}")

        for epoch in range(1, EPOCHS+1):
            tr_loss, _ = train_one_epoch(model, tr, opt, crit, DEVICE, jitter_std=args.jitter)
            val_m      = evaluate(model, va, DEVICE)['mse']
            if epoch <= 5: warmup.step()
            else:          sch.step()
            train_curve.append(tr_loss); val_curve.append(val_m)
            improved  = es(val_m, model)
            gap_ratio = val_m / (tr_loss + 1e-8)
            print(f"  {epoch:>4} | {tr_loss:>10.5f} | {val_m:>10.5f} | "
                  f"{gap_ratio:>5.2f}x | {'*' if improved else ''}")
            if es.early_stop:
                print(f"\n  Early stop @ ep {epoch}  (best val={es.best:.5f})")
                break

        model.load_state_dict(es.state)
        test_m = evaluate(model, te_loader, DEVICE)
        final_results[PRED_LEN] = test_m
        gap = test_m['mse'] - SOTA.get(PRED_LEN, 0)
        print(f"\n  TEST H={PRED_LEN}: MSE={test_m['mse']:.4f}  MAE={test_m['mae']:.4f}  "
              f"DirAcc={test_m['dir_acc']:.1f}%  "
              f"({'above' if gap>0 else 'below'} iTransformer by {abs(gap):.4f})")
        all_train_curves[PRED_LEN]=train_curve; all_val_curves[PRED_LEN]=val_curve

        if PM is not None:
            try:
                PM.plot_learning_curves(train_curve, val_curve, PRED_LEN)
            except Exception as e:
                print(f"  [PLOTS] learning curves skipped: {e}")
            try:
                class _CW(nn.Module):
                    def __init__(self, m):
                        super().__init__(); self._m=m
                        self.dwt=None; self.wrat=None; self.gm=None
                        self.variate_attn = lambda x: x 
                    def forward(self, x): return self._m.compat_forward(x)
                PM.plot_predictions(_CW(model), te_loader, PRED_LEN, DEVICE,
                                    n_samples=3, n_variates=NUM_VARIATES)
            except Exception as e:
                print(f"  [PLOTS] predictions skipped: {e}")

    print(f"\n{'='*70}")
    print(f" FINAL RESULTS  (v8.0 | seq={SEQ_LEN} | d={D_MODEL} | layers={N_LAYERS})")
    print(f"{'='*70}")
    print(f"  {'H':>5} | {'MSE':>8} | {'MAE':>8} | {'DirAcc':>8} | {'vs iTransf':>12}")
    print(f"  {'-'*60}")
    for h, r in final_results.items():
        gap  = r['mse'] - SOTA.get(h, 0)
        sign = '+' if gap > 0 else '-'
        print(f"  {h:>5} | {r['mse']:>8.4f} | {r['mae']:>8.4f} | "
              f"{r['dir_acc']:>7.1f}% | {sign}{abs(gap):.4f}")

    if PM is not None:
        try:
            PM.plot_all_horizons_summary(all_train_curves, all_val_curves, HORIZONS)
            PM.plot_final_benchmarks(final_results)
            print("[PLOTS] Done.")
        except Exception as e:
            print(f"[PLOTS] Final plots skipped: {e}")

if __name__ == "__main__":
    main()