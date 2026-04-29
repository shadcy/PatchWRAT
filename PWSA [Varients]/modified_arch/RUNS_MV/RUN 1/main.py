# ============================================================
# run_patchwrat_multivariate.py  —  Multivariate PatchWRAT Benchmark
# ============================================================
# Dataset : ETTm1.csv (must be in the same folder)
# Upgrades: Channel Independence, 7-Channel Simultaneous Forecasting
# ============================================================

import math, warnings
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
# SECTION 1 — WAVELET, ATTENTION & NORMALIZATION
# ══════════════════════════════════════════════════════════════

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=-1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.unsqueeze(-1) + self.affine_bias.unsqueeze(-1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias.unsqueeze(-1)) / (self.affine_weight.unsqueeze(-1) + self.eps**2)
        x = x * self.stdev
        x = x + self.mean
        return x


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
        
        if isinstance(threshold, float):
            self.threshold = torch.tensor(threshold)
        else:
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
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        if energy_coeffs is not None:
            energy = torch.abs(energy_coeffs).mean(dim=-1)
            current_threshold = self.threshold.to(energy.device)
            gate = torch.sigmoid((energy - current_threshold) * 10.0)
            gate = gate.view(B, 1, 1, -1)
            attn_weights = attn_weights * gate
            
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class WRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_tau=0.1, dropout=0.2):
        super().__init__()
        self.intra_LL_attn  = FrequencySparseAttention(d_model, num_heads)
        self.intra_LH_attn  = FrequencySparseAttention(d_model, num_heads, threshold=sparsity_tau)
        self.cross_attn     = FrequencySparseAttention(d_model, num_heads)
        
        self.mlp_LL = nn.Sequential(
            nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model), nn.Dropout(dropout)
        )
        self.mlp_LH = nn.Sequential(
            nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model), nn.Dropout(dropout)
        )
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


class LearnableTauWRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, tau_init=0.1, dropout=0.2):
        super().__init__()
        self.raw_tau = nn.Parameter(torch.tensor(math.log(tau_init / (1.0 - tau_init))))
        self._block  = WRATBlock(d_model, num_heads, sparsity_tau=tau_init, dropout=dropout)

    @property
    def tau(self): 
        return torch.sigmoid(self.raw_tau).item()

    def forward(self, LL, LH):
        self._block.intra_LH_attn.threshold = torch.sigmoid(self.raw_tau)
        return self._block(LL, LH)


# ══════════════════════════════════════════════════════════════
# SECTION 2 — THE NEW PATCH-WRAT ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class PatchWRAT(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=16, num_heads=4, tau_init=0.1, tau_type='learnable', dropout=0.4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.revin = RevIN(num_features=1)
        self.dwt = LearnableDWT1D(1, d_model)
        
        if tau_type == 'learnable':
            self.wrat_block = LearnableTauWRATBlock(d_model, num_heads, tau_init, dropout=0.2)
        else:
            self.wrat_block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init, dropout=0.2)
            
        self.l_half = seq_len // 2
        self.flatten_dim = self.l_half * d_model * 2  
        
        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.flatten_dim, pred_len)
        )
        
        self.sc = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x, zero_lh=False):
        x_norm = self.revin(x, mode='norm')
        LL, LH = self.dwt(x_norm)
        if zero_lh: LH = torch.zeros_like(LH)
        LL_out, LH_out = self.wrat_block(LL, LH)
        
        fused = torch.cat([LL_out, LH_out], dim=1) 
        preds = self.forecast_head(fused).unsqueeze(1) 
        preds = self.revin(preds, mode='denorm')
        
        x_recon = self.dwt.inverse(LL_out, LH_out)
        x_recon = self.sc(x_recon)
        x_recon = self.revin(x_recon, mode='denorm')
        
        return preds, x_recon, LL, LH


class DualHeadWRATLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_ortho=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho

    def forward(self, preds, targets, x_orig, x_recon, dwt_layer):
        task_loss = F.mse_loss(preds, targets)
        min_len = min(x_orig.shape[-1], x_recon.shape[-1])
        recon_loss = F.mse_loss(x_recon[..., :min_len], x_orig[..., :min_len])
        
        h_flat = dwt_layer.h.view(dwt_layer.h.shape[0], -1)
        g_flat = dwt_layer.g.view(dwt_layer.g.shape[0], -1)
        ortho_loss = (h_flat * g_flat).sum().abs()
        
        total = task_loss + self.lambda_recon * recon_loss + self.lambda_ortho * ortho_loss
        return total, task_loss


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

# ══════════════════════════════════════════════════════════════
# SECTION 3 — DATASET & EVALUATION (MULTIVARIATE UPDATED)
# ══════════════════════════════════════════════════════════════

class ETTDataset(Dataset):
    def __init__(self, seq_len, pred_len, split='train', file_path=r'C:\Users\Asus\Desktop\TTS\WRAT\ETTm1.csv'):
        self.seq_len, self.pred_len = seq_len, pred_len
        df = pd.read_csv(file_path)
        
        # CHANGED: Grabs ALL 7 features instead of just 'OT'
        data = df.iloc[:, 1:].values 
        
        train_end = 12 * 30 * 24 * 4
        val_end   = train_end + 4 * 30 * 24 * 4
        raw = {'train': data[:train_end], 'val': data[train_end:val_end], 'test': data[val_end:]}[split]
        
        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)

    def inverse(self, x_norm): 
        # Handle multivariate scale and mean shapes
        return x_norm * self.scaler.scale_ + self.scaler.mean_

    def __len__(self): 
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x.t(), y.t()


def evaluate(model, loader, device, inv_fn=None):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            B, C, L = bx.shape
            
            # Channel Independence: Flatten channels into the batch dimension
            bx_flat = bx.reshape(B * C, 1, L)
            
            preds, _, _, _ = model(bx_flat)
            
            # Reshape back to multivariate [Batch, Channels, Pred_Len]
            preds = preds.reshape(B, C, -1)
            
            all_p.append(preds.cpu())
            all_t.append(by.cpu())
            
    p, t = torch.cat(all_p, 0).flatten().numpy(), torch.cat(all_t, 0).flatten().numpy()
    err = p - t
    mae, mse = float(np.abs(err).mean()), float((err**2).mean())
    dir_acc = float((np.sign(np.diff(p)) == np.sign(np.diff(t))).mean() * 100)
    model.train()
    return dict(mae=mae, mse=mse, dir_acc=dir_acc)


def train_patchwrat(model, opt, crit, loader, tau_override=None, zero_lh=False):
    model.train()
    if tau_override is not None and hasattr(model.wrat_block, 'sparsity_tau'):
        model.wrat_block.sparsity_tau = tau_override
        
    total_loss, n = 0, 0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        B, C, L = bx.shape
        
        # Channel Independence: Flatten channels into the batch dimension
        bx_flat = bx.reshape(B * C, 1, L)
        by_flat = by.reshape(B * C, 1, by.shape[-1])
        
        opt.zero_grad()
        
        preds, x_recon, _, _ = model(bx_flat, zero_lh=zero_lh)
        loss, _ = crit(preds, by_flat, bx_flat, x_recon, model.dwt)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        n += 1
        
    return total_loss / n

# ══════════════════════════════════════════════════════════════
# SECTION 4 — FILTER INSIGHT EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_and_plot_filters(model, horizon, model_type="Learnable"):
    h_weights = model.dwt.h.detach().cpu().numpy()
    g_weights = model.dwt.g.detach().cpu().numpy()
    
    h_avg = np.mean(h_weights, axis=(0, 1))
    g_avg = np.mean(g_weights, axis=(0, 1))
    
    print(f"\n --- Learned Signal Processing Insights (H={horizon} | {model_type}) ---")
    print(f" Average Low-Pass (Trend) Filter 'h' : {np.round(h_avg, 4)}")
    print(f" Average High-Pass (Noise) Filter 'g': {np.round(g_avg, 4)}")
    
    if hasattr(model.wrat_block, 'tau'):
        print(f" Final Learned Sparsity Threshold (tau): {model.wrat_block.tau:.4f}")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(h_avg, label='h (Trend)', marker='o', color='#2563eb', linewidth=2)
    plt.plot(g_avg, label='g (Noise)', marker='x', color='#dc2626', linewidth=2, linestyle='--')
    plt.title(f'Learned Filter Impulse Response (H={horizon})')
    plt.xlabel('Filter Tap Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    h_fft = np.abs(np.fft.fft(h_avg, n=64))[:32] 
    g_fft = np.abs(np.fft.fft(g_avg, n=64))[:32]
    freqs = np.linspace(0, np.pi, 32)
    
    plt.plot(freqs, h_fft, label='h (Low-Pass)', color='#2563eb', linewidth=2)
    plt.plot(freqs, g_fft, label='g (High-Pass)', color='#dc2626', linewidth=2, linestyle='--')
    plt.title('Magnitude Spectrum (Frequency Response)')
    plt.xlabel('Normalized Frequency (radians/sample)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = f'learned_filters_H{horizon}_{model_type}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f" Saved filter visualization to '{filename}'\n")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — CONFIG & EXECUTION LOOP
# ══════════════════════════════════════════════════════════════

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 336
HORIZONS   = [96, 192, 336, 720]
BATCH_SIZE = 64
D_MODEL    = 16   
EPOCHS     = 30
PATIENCE   = 10
LR         = 5e-4 

crit_wrat = DualHeadWRATLoss(lambda_recon=1.0, lambda_ortho=0.1)

print(f"Device: {DEVICE} | Horizons: {HORIZONS}")
print(f"Epochs: {EPOCHS} | Patience: {PATIENCE} | Seq_Len: {SEQ_LEN} | D_MODEL: {D_MODEL}\n")
print(">>> RUNNING MULTIVARIATE BENCHMARK (7 CHANNELS) <<< \n")

for PRED_LEN in HORIZONS:
    print(f"\n{'='*65}\n HORIZON = {PRED_LEN} \n{'='*65}")
    
    train_ds = ETTDataset(SEQ_LEN, PRED_LEN, 'train')
    val_ds   = ETTDataset(SEQ_LEN, PRED_LEN, 'val')
    test_ds  = ETTDataset(SEQ_LEN, PRED_LEN, 'test')
    
    trl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    vll = DataLoader(val_ds, BATCH_SIZE, shuffle=False, drop_last=True)
    tel = DataLoader(test_ds, BATCH_SIZE, shuffle=False, drop_last=True)
    
    wrat_lrn = PatchWRAT(SEQ_LEN, PRED_LEN, D_MODEL, tau_type='learnable', dropout=0.4).to(DEVICE)
    wrat_fix = PatchWRAT(SEQ_LEN, PRED_LEN, D_MODEL, tau_type='fixed', tau_init=0.1, dropout=0.4).to(DEVICE)
    
    opt_lrn = optim.AdamW(wrat_lrn.parameters(), lr=LR, weight_decay=1e-4)
    opt_fix = optim.AdamW(wrat_fix.parameters(), lr=LR, weight_decay=1e-4)
    
    sched_lrn = optim.lr_scheduler.CosineAnnealingLR(opt_lrn, T_max=EPOCHS, eta_min=1e-6)
    sched_fix = optim.lr_scheduler.CosineAnnealingLR(opt_fix, T_max=EPOCHS, eta_min=1e-6)
    
    early_stop_lrn = EarlyStopping(patience=PATIENCE)
    early_stop_fix = EarlyStopping(patience=PATIENCE)
    
    if PRED_LEN == HORIZONS[0]:
        params = sum(p.numel() for p in wrat_lrn.parameters())
        print(f" Parameters: {params:,} (Maintains extreme efficiency via Channel Independence)\n")
    
    print(f" {'Ep':>3} | {'LR':>8} | {'Val MSE (Learnable)':>19} | {'Val MSE (Fixed)':>15}")
    print(f" {'-'*58}")
    
    for epoch in range(1, EPOCHS+1):
        current_lr = opt_lrn.param_groups[0]['lr']
        
        if not early_stop_lrn.early_stop: train_patchwrat(wrat_lrn, opt_lrn, crit_wrat, trl)
        if not early_stop_fix.early_stop: train_patchwrat(wrat_fix, opt_fix, crit_wrat, trl)
            
        v_lrn = evaluate(wrat_lrn, vll, DEVICE)['mse'] if not early_stop_lrn.early_stop else early_stop_lrn.best_loss
        v_fix = evaluate(wrat_fix, vll, DEVICE)['mse'] if not early_stop_fix.early_stop else early_stop_fix.best_loss
        
        if not early_stop_lrn.early_stop: early_stop_lrn(v_lrn, wrat_lrn)
        if not early_stop_fix.early_stop: early_stop_fix(v_fix, wrat_fix)
        
        lrn_tag = " (Stopped)" if early_stop_lrn.early_stop and epoch == early_stop_lrn.counter + early_stop_lrn.patience else ""
        fix_tag = " (Stopped)" if early_stop_fix.early_stop and epoch == early_stop_fix.counter + early_stop_fix.patience else ""
        print(f" {epoch:>3} | {current_lr:>8.6f} | {v_lrn:>10.5f}{lrn_tag:<9} | {v_fix:>10.5f}{fix_tag:<9}")
        
        if early_stop_lrn.early_stop and early_stop_fix.early_stop:
            print(" Both models reached early stopping.")
            break
            
        if not early_stop_lrn.early_stop: sched_lrn.step()
        if not early_stop_fix.early_stop: sched_fix.step()

    wrat_lrn.load_state_dict(early_stop_lrn.best_state)
    wrat_fix.load_state_dict(early_stop_fix.best_state)
    
    res_lrn = evaluate(wrat_lrn, tel, DEVICE, test_ds.inverse)
    res_fix = evaluate(wrat_fix, tel, DEVICE, test_ds.inverse)
    
    print(f"\n MULTIVARIATE TEST RESULTS (Horizon {PRED_LEN}):")
    print(f" PatchWRAT (Learnable) -> MAE: {res_lrn['mae']:.4f} | MSE: {res_lrn['mse']:.4f} | DirAcc: {res_lrn['dir_acc']:.2f}%")
    print(f" PatchWRAT (Fixed)     -> MAE: {res_fix['mae']:.4f} | MSE: {res_fix['mse']:.4f} | DirAcc: {res_fix['dir_acc']:.2f}%")

    extract_and_plot_filters(wrat_lrn, PRED_LEN, model_type="Learnable")

print("\nMultivariate PatchWRAT Benchmark complete.")