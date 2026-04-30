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

from model import PatchWRAT

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)

CSV_PATH     = r'C:\Users\Asus\Desktop\TTS\WRAT\ETTm1.csv'
SEQ_LEN      = 336
HORIZONS     = [96] # Reduced to 96 for quick 1 epoch run as requested
BATCH_SIZE   = 64
EPOCHS       = 1    # SET TO 1 EPOCH AS REQUESTED
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
        if epoch % 10 == 0 or improved or True:
            flag = '*' if improved else ''
            print(f"    ep{epoch:>3} | train={tr_loss:.4f} | val={val_m:.4f} {flag}")
            
        if es.early_stop:
            print(f"    Early stop @ ep{epoch} | best_val={es.best:.4f}")
            break

    model.load_state_dict(es.state)
    return evaluate(model, te_loader, DEVICE), model

# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
        print(f"\n  +- Horizon {PRED_LEN} Ablation Summary {'-'*30}+")
        print(f"  |  {'Variant':<15} {'MSE':>8}  {'MAE':>8}  {'DirAcc':>8}          |")
        print(f"  |  {'-'*53}  |")
        for name in configs.keys():
            r = ablation_results[PRED_LEN][name]
            print(f"  |  {name:<15} {r['mse']:>8.4f}  {r['mae']:>8.4f}  {r['dir_acc']:>7.1f}%          |")
        print(f"  + {'-'*56}+")

    print(f"\n{'='*70}\n[WMRT Framework] Benchmark complete. Plots saved in local directory.\n{'='*70}")
