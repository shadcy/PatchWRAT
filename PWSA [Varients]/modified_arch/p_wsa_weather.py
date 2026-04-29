# ============================================================
# run_pwsa_ablation_suite_mpi.py  —  P-WSA Publication Benchmark
# ============================================================
# Dataset : mpi_roof_2017b.csv (MPI Weather Dataset)
# Upgrades: 3-Way Ablation, Auto-Plotting, L=512, D_MODEL=64
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
# SECTION 1 — NORMALIZATION, PATCHING, & WAVELETS
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


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) 
        x = x.squeeze(1) 
        x = self.proj(x) 
        return x.transpose(1, 2)


class LearnableDWT1D(nn.Module):
    def __init__(self, channels, filter_length=4):
        super().__init__()
        self.channels = channels
        self.filter_length = filter_length
        self.padding_val = (filter_length - 2) // 2
        
        self.h = nn.Parameter(torch.randn(channels, 1, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(channels, 1, filter_length) * 0.1)
        with torch.no_grad():
            self.g -= self.g.mean(dim=-1, keepdim=True)

    def forward(self, x):
        LL = F.conv1d(x, self.h, stride=2, padding=self.padding_val, groups=self.channels)
        LH = F.conv1d(x, self.g, stride=2, padding=self.padding_val, groups=self.channels)
        return LL, LH

    def inverse(self, LL, LH):
        x_recon_L = F.conv_transpose1d(LL, self.h, stride=2, padding=self.padding_val, groups=self.channels)
        x_recon_H = F.conv_transpose1d(LH, self.g, stride=2, padding=self.padding_val, groups=self.channels)
        min_len = min(x_recon_L.shape[-1], x_recon_H.shape[-1])
        return x_recon_L[..., :min_len] + x_recon_H[..., :min_len]

# ══════════════════════════════════════════════════════════════
# SECTION 2 — DIFFERENTIABLE SPARSITY & ATTENTION
# ══════════════════════════════════════════════════════════════

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
        
        self.mlp_LL = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
        self.mlp_LH = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)

    def forward(self, LL, LH):
        LL_seq, LH_seq = LL.transpose(1, 2), LH.transpose(1, 2)
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
# SECTION 3 — PATCHED WSA ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class PatchedWSA(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=64, num_heads=4, patch_len=16, stride=8, tau_init=0.1, tau_type='learnable', dropout=0.2):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_len, self.stride = patch_len, stride
        
        self.revin = RevIN(num_features=1)
        self.patch_emb = PatchEmbedding(patch_len, stride, d_model)
        self.dwt = LearnableDWT1D(channels=d_model)
        
        if tau_type == 'learnable':
            self.wrat_block = LearnableTauWRATBlock(d_model, num_heads, tau_init, dropout)
        else:
            self.wrat_block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init, dropout=dropout)
            
        num_patches = (seq_len - patch_len) // stride + 1
        pad = (4 - 2) // 2 
        dwt_len = (num_patches + 2 * pad - 4) // 2 + 1
        self.flatten_dim = dwt_len * d_model * 2  
        
        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Dropout(dropout), nn.Linear(self.flatten_dim, pred_len)
        )

    def forward(self, x, zero_lh=False):
        x_norm = self.revin(x, mode='norm')
        patches = self.patch_emb(x_norm) 
        
        LL, LH = self.dwt(patches)
        if zero_lh: LH = torch.zeros_like(LH) # Ablation Trigger
        
        LL_out, LH_out = self.wrat_block(LL, LH)
        
        fused = torch.cat([LL_out, LH_out], dim=1) 
        preds = self.forecast_head(fused).unsqueeze(1) 
        preds = self.revin(preds, mode='denorm')
        
        patch_recon = self.dwt.inverse(LL_out, LH_out)
        return preds, patch_recon, patches, LL, LH


class DualHeadPWSA_Loss(nn.Module):
    def __init__(self, lambda_recon=0.1, lambda_ortho=0.01):
        super().__init__()
        self.lambda_recon, self.lambda_ortho = lambda_recon, lambda_ortho

    def forward(self, preds, targets, patches_orig, patches_recon, dwt_layer):
        task_loss = F.mse_loss(preds, targets)
        min_len = min(patches_orig.shape[-1], patches_recon.shape[-1])
        recon_loss = F.mse_loss(patches_recon[..., :min_len], patches_orig[..., :min_len])
        
        h_flat, g_flat = dwt_layer.h.view(dwt_layer.channels, -1), dwt_layer.g.view(dwt_layer.channels, -1)
        ortho_loss = (h_flat * g_flat).sum(dim=-1).abs().mean()
        
        total = task_loss + self.lambda_recon * recon_loss + self.lambda_ortho * ortho_loss
        return total, task_loss.item() 

# ══════════════════════════════════════════════════════════════
# SECTION 4 — DATASET, TRAINING, & VISUALIZATION (WEATHER UPDATE)
# ══════════════════════════════════════════════════════════════

class WeatherDataset(Dataset):
    def __init__(self, seq_len, pred_len, split='train', file_path=r'C:\Users\Asus\Desktop\TTS\Rat\mpi_roof_2017b\mpi_roof_2017b.csv'):
        self.seq_len, self.pred_len = seq_len, pred_len
        
        # 1. Load with specific encoding and drop NaN sensor readings
        df = pd.read_csv(file_path, encoding='latin1')
        
        # 2. Automatically drop 'Date Time' string column, keep only numerical sensors
        df = df.select_dtypes(include=[np.number]).dropna()
        data = df.values 
        
        # 3. Dynamic 70% / 10% / 20% Split (Adapts to dataset length)
        n = len(data)
        train_end = int(n * 0.7)
        val_end   = int(n * 0.8)
        raw = {'train': data[:train_end], 'val': data[train_end:val_end], 'test': data[val_end:]}[split]
        
        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)

    def inverse(self, x_norm): return x_norm * self.scaler.scale_ + self.scaler.mean_
    def __len__(self): return max(0, len(self.data) - self.seq_len - self.pred_len)
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x.t(), y.t()

def evaluate(model, loader, device, zero_lh=False):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            B, C, L = bx.shape
            preds, _, _, _, _ = model(bx.reshape(B * C, 1, L), zero_lh=zero_lh)
            all_p.append(preds.reshape(B, C, -1).cpu())
            all_t.append(by.cpu())
            
    p, t = torch.cat(all_p, 0).flatten().numpy(), torch.cat(all_t, 0).flatten().numpy()
    err = p - t
    mae, mse = float(np.abs(err).mean()), float((err**2).mean())
    dir_acc = float((np.sign(np.diff(p)) == np.sign(np.diff(t))).mean() * 100)
    model.train()
    return dict(mae=mae, mse=mse, dir_acc=dir_acc)

def plot_learning_curves(train_dict, val_dict, horizon):
    plt.figure(figsize=(10, 5))
    colors = {'Learnable': '#2563eb', 'Fixed': '#16a34a', 'No_HF': '#dc2626'}
    
    for key in train_dict.keys():
        plt.plot(range(1, len(val_dict[key])+1), val_dict[key], label=f'Val MSE ({key})', color=colors[key], linewidth=2)
        plt.plot(range(1, len(train_dict[key])+1), train_dict[key], label=f'Train MSE ({key})', color=colors[key], linestyle='--', alpha=0.5)

    plt.title(f'P-WSA Ablation Learning Curves (H={horizon})')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'learning_curves_H{horizon}.png', dpi=150)
    plt.close()

def plot_final_bar_charts(final_results, horizons):
    labels = ['Learnable \u03C4', 'Fixed \u03C4=0.1', 'No HF Branch']
    colors = ['#2563eb', '#16a34a', '#dc2626']
    x = np.arange(len(horizons))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, lbl in enumerate(['Learnable', 'Fixed', 'No_HF']):
        mses = [final_results[h][lbl]['mse'] for h in horizons]
        plt.bar(x + (i-1)*width, mses, width, label=labels[i], color=colors[i], edgecolor='black')
    
    plt.ylabel('Test MSE (Lower is Better)')
    plt.title('P-WSA MPI Weather Benchmarks: MSE by Horizon')
    plt.xticks(x, [f'H={h}' for h in horizons])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('final_ablation_mse.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for i, lbl in enumerate(['Learnable', 'Fixed', 'No_HF']):
        accs = [final_results[h][lbl]['dir_acc'] for h in horizons]
        plt.bar(x + (i-1)*width, accs, width, label=labels[i], color=colors[i], edgecolor='black')
    
    plt.ylabel('Directional Accuracy % (Higher is Better)')
    plt.title('P-WSA MPI Weather Benchmarks: Directional Accuracy by Horizon')
    plt.xticks(x, [f'H={h}' for h in horizons])
    plt.ylim(45, 60) 
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('final_ablation_diracc.png', dpi=150)
    plt.close()

def extract_and_plot_filters(model, horizon):
    h_weights = np.mean(model.dwt.h.detach().cpu().numpy(), axis=(0, 1))
    g_weights = np.mean(model.dwt.g.detach().cpu().numpy(), axis=(0, 1))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(h_weights, label='h (Trend)', marker='o', color='#2563eb', linewidth=2)
    plt.plot(g_weights, label='g (Noise)', marker='x', color='#dc2626', linewidth=2, linestyle='--')
    plt.title(f'Learned Filter Impulse Response (H={horizon})')
    plt.xlabel('Filter Tap Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, np.pi, 32), np.abs(np.fft.fft(h_weights, n=64))[:32], label='h (Low-Pass)', color='#2563eb')
    plt.plot(np.linspace(0, np.pi, 32), np.abs(np.fft.fft(g_weights, n=64))[:32], label='g (High-Pass)', color='#dc2626', linestyle='--')
    plt.title('Magnitude Spectrum (Frequency Response)')
    plt.xlabel('Normalized Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'learned_filters_H{horizon}.png', dpi=150)
    plt.close()

# ══════════════════════════════════════════════════════════════
# SECTION 5 — CONFIG & EXECUTION LOOP
# ══════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience, self.counter, self.best_loss, self.early_stop = patience, 0, None, False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN, D_MODEL, PATCH_LEN, STRIDE = 512, 64, 16, 8
HORIZONS   = [12, 24, 48, 96, 192, 336, 720]
BATCH_SIZE = 32
EPOCHS     = 30
PATIENCE   = 10
LR         = 5e-4 

crit_pwsa = DualHeadPWSA_Loss(lambda_recon=0.1, lambda_ortho=0.01)

print(f"Device: {DEVICE} | P-WSA Settings -> L: {SEQ_LEN} | D: {D_MODEL} | Patch: {PATCH_LEN}")
print(">>> RUNNING FULL P-WSA ABLATION SUITE (MPI WEATHER) <<< \n")

final_results = {h: {} for h in HORIZONS}

for PRED_LEN in HORIZONS:
    print(f"\n{'='*75}\n HORIZON = {PRED_LEN} \n{'='*75}")
    
    # Swapped to WeatherDataset
    train_ds, val_ds, test_ds = [WeatherDataset(SEQ_LEN, PRED_LEN, split) for split in ['train', 'val', 'test']]
    trl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    vll = DataLoader(val_ds, BATCH_SIZE, shuffle=False, drop_last=True)
    tel = DataLoader(test_ds, BATCH_SIZE, shuffle=False, drop_last=True)
    
    m_lrn = PatchedWSA(SEQ_LEN, PRED_LEN, D_MODEL, tau_type='learnable').to(DEVICE)
    m_fix = PatchedWSA(SEQ_LEN, PRED_LEN, D_MODEL, tau_type='fixed', tau_init=0.1).to(DEVICE)
    m_abl = PatchedWSA(SEQ_LEN, PRED_LEN, D_MODEL, tau_type='learnable').to(DEVICE) 
    
    opt_lrn = optim.AdamW(m_lrn.parameters(), lr=LR, weight_decay=1e-4)
    opt_fix = optim.AdamW(m_fix.parameters(), lr=LR, weight_decay=1e-4)
    opt_abl = optim.AdamW(m_abl.parameters(), lr=LR, weight_decay=1e-4)
    
    sch_lrn = optim.lr_scheduler.CosineAnnealingLR(opt_lrn, T_max=EPOCHS, eta_min=1e-6)
    sch_fix = optim.lr_scheduler.CosineAnnealingLR(opt_fix, T_max=EPOCHS, eta_min=1e-6)
    sch_abl = optim.lr_scheduler.CosineAnnealingLR(opt_abl, T_max=EPOCHS, eta_min=1e-6)
    
    es_lrn, es_fix, es_abl = EarlyStopping(PATIENCE), EarlyStopping(PATIENCE), EarlyStopping(PATIENCE)
    
    trk_train = {'Learnable': [], 'Fixed': [], 'No_HF': []}
    trk_val   = {'Learnable': [], 'Fixed': [], 'No_HF': []}
    
    print(f" {'Ep':>3} | {'Val MSE (Lrn)':>15} | {'Val MSE (Fix)':>15} | {'Val MSE (No_HF)':>15}")
    print(f" {'-'*65}")
    
    for epoch in range(1, EPOCHS+1):
        m_lrn.train(); m_fix.train(); m_abl.train()
        l_lrn, l_fix, l_abl = [], [], []
        
        for bx, by in trl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            B, C, L = bx.shape
            bx_flat, by_flat = bx.reshape(B*C, 1, L), by.reshape(B*C, 1, -1)
            
            if not es_lrn.early_stop:
                opt_lrn.zero_grad()
                p, rec, orig, _, _ = m_lrn(bx_flat)
                loss, t_loss = crit_pwsa(p, by_flat, orig, rec, m_lrn.dwt)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m_lrn.parameters(), 1.0); opt_lrn.step()
                l_lrn.append(t_loss)
                
            if not es_fix.early_stop:
                opt_fix.zero_grad()
                p, rec, orig, _, _ = m_fix(bx_flat)
                loss, t_loss = crit_pwsa(p, by_flat, orig, rec, m_fix.dwt)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m_fix.parameters(), 1.0); opt_fix.step()
                l_fix.append(t_loss)
                
            if not es_abl.early_stop:
                opt_abl.zero_grad()
                p, rec, orig, _, _ = m_abl(bx_flat, zero_lh=True)
                loss, t_loss = crit_pwsa(p, by_flat, orig, rec, m_abl.dwt)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m_abl.parameters(), 1.0); opt_abl.step()
                l_abl.append(t_loss)
        
        if l_lrn: trk_train['Learnable'].append(np.mean(l_lrn))
        if l_fix: trk_train['Fixed'].append(np.mean(l_fix))
        if l_abl: trk_train['No_HF'].append(np.mean(l_abl))
        
        v_lrn = evaluate(m_lrn, vll, DEVICE)['mse'] if not es_lrn.early_stop else es_lrn.best_loss
        v_fix = evaluate(m_fix, vll, DEVICE)['mse'] if not es_fix.early_stop else es_fix.best_loss
        v_abl = evaluate(m_abl, vll, DEVICE, zero_lh=True)['mse'] if not es_abl.early_stop else es_abl.best_loss
        
        if not es_lrn.early_stop: es_lrn(v_lrn, m_lrn); trk_val['Learnable'].append(v_lrn); sch_lrn.step()
        if not es_fix.early_stop: es_fix(v_fix, m_fix); trk_val['Fixed'].append(v_fix); sch_fix.step()
        if not es_abl.early_stop: es_abl(v_abl, m_abl); trk_val['No_HF'].append(v_abl); sch_abl.step()
        
        s_lrn = "*" if es_lrn.early_stop and epoch == es_lrn.counter + es_lrn.patience else ""
        s_fix = "*" if es_fix.early_stop and epoch == es_fix.counter + es_fix.patience else ""
        s_abl = "*" if es_abl.early_stop and epoch == es_abl.counter + es_abl.patience else ""
        
        print(f" {epoch:>3} | {v_lrn:>14.5f}{s_lrn:1} | {v_fix:>14.5f}{s_fix:1} | {v_abl:>14.5f}{s_abl:1}")
        
        if es_lrn.early_stop and es_fix.early_stop and es_abl.early_stop:
            print(" All variants reached early stopping.")
            break

    m_lrn.load_state_dict(es_lrn.best_state)
    m_fix.load_state_dict(es_fix.best_state)
    m_abl.load_state_dict(es_abl.best_state)
    
    res_lrn = evaluate(m_lrn, tel, DEVICE)
    res_fix = evaluate(m_fix, tel, DEVICE)
    res_abl = evaluate(m_abl, tel, DEVICE, zero_lh=True)
    
    final_results[PRED_LEN] = {'Learnable': res_lrn, 'Fixed': res_fix, 'No_HF': res_abl}
    
    print(f"\n --- TEST RESULTS (H={PRED_LEN}) ---")
    print(f" Learnable Tau : MSE={res_lrn['mse']:.4f} | MAE={res_lrn['mae']:.4f} | DirAcc={res_lrn['dir_acc']:.2f}% | Final Tau={m_lrn.wrat_block.tau:.4f}")
    print(f" Fixed Tau=0.1 : MSE={res_fix['mse']:.4f} | MAE={res_fix['mae']:.4f} | DirAcc={res_fix['dir_acc']:.2f}%")
    print(f" No HF Branch  : MSE={res_abl['mse']:.4f} | MAE={res_abl['mae']:.4f} | DirAcc={res_abl['dir_acc']:.2f}%")

    plot_learning_curves(trk_train, trk_val, PRED_LEN)
    extract_and_plot_filters(m_lrn, PRED_LEN)

plot_final_bar_charts(final_results, HORIZONS)
print("\n>>> Full P-WSA Ablation Suite (MPI WEATHER) Complete! <<<")