# ============================================================
# run_pwsa_multivariate.py  —  Patched WSA (P-WSA) Benchmark
# ============================================================
# Dataset : ETTm1.csv (must be in the same folder)
# Upgrades: Pre-DWT Patching, L=512, D_MODEL=64, Channel Independence
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
        # x shape: [B*C, 1, L]
        # Unfold extracts sliding local blocks from the sequence
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B*C, 1, N_patches, P_len]
        x = x.squeeze(1) # [B*C, N_patches, P_len]
        
        # Project patches to D_MODEL
        x = self.proj(x) # [B*C, N_patches, D_MODEL]
        
        # Transpose for Conv1d (DWT) -> [B*C, D_MODEL, N_patches]
        return x.transpose(1, 2)


class LearnableDWT1D(nn.Module):
    def __init__(self, channels, filter_length=4):
        super().__init__()
        self.channels = channels
        self.filter_length = filter_length
        self.padding_val = (filter_length - 2) // 2
        
        # Depthwise 1D Convolution filters (one pair per embedding dimension)
        self.h = nn.Parameter(torch.randn(channels, 1, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(channels, 1, filter_length) * 0.1)
        with torch.no_grad():
            self.g -= self.g.mean(dim=-1, keepdim=True)

    def forward(self, x):
        # x: [B*C, D_MODEL, N_patches]
        # Applying grouped convolution to process each channel's frequencies independently
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
            # The Differentiable Sigmoid Gate (k=10.0)
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
# SECTION 3 — PATCHED WSA ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class PatchedWSA(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=64, num_heads=4, patch_len=16, stride=8, tau_init=0.1, tau_type='learnable', dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        
        self.revin = RevIN(num_features=1)
        self.patch_emb = PatchEmbedding(patch_len, stride, d_model)
        self.dwt = LearnableDWT1D(channels=d_model)
        
        if tau_type == 'learnable':
            self.wrat_block = LearnableTauWRATBlock(d_model, num_heads, tau_init, dropout)
        else:
            self.wrat_block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init, dropout=dropout)
            
        # Calculate resulting dimensions dynamically
        num_patches = (seq_len - patch_len) // stride + 1
        pad = (4 - 2) // 2 
        dwt_len = (num_patches + 2 * pad - 4) // 2 + 1
        self.flatten_dim = dwt_len * d_model * 2  
        
        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.flatten_dim, pred_len)
        )

    def forward(self, x, zero_lh=False):
        x_norm = self.revin(x, mode='norm')
        
        # 1. Patching
        patches = self.patch_emb(x_norm) # [B*C, D_MODEL, N_patches]
        
        # 2. Wavelet Decomposition on Patches
        LL, LH = self.dwt(patches)
        if zero_lh: LH = torch.zeros_like(LH)
        
        # 3. Hierarchical Attention
        LL_out, LH_out = self.wrat_block(LL, LH)
        
        # 4. Forecasting
        fused = torch.cat([LL_out, LH_out], dim=1) 
        preds = self.forecast_head(fused).unsqueeze(1) 
        preds = self.revin(preds, mode='denorm')
        
        # 5. Patch Reconstruction (for regularization/future pre-training)
        patch_recon = self.dwt.inverse(LL_out, LH_out)
        
        return preds, patch_recon, patches, LL, LH


class DualHeadPWSA_Loss(nn.Module):
    def __init__(self, lambda_recon=0.1, lambda_ortho=0.01):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho

    def forward(self, preds, targets, patches_orig, patches_recon, dwt_layer):
        task_loss = F.mse_loss(preds, targets)
        
        # Reconstruct the patches in latent space
        min_len = min(patches_orig.shape[-1], patches_recon.shape[-1])
        recon_loss = F.mse_loss(patches_recon[..., :min_len], patches_orig[..., :min_len])
        
        h_flat = dwt_layer.h.view(dwt_layer.channels, -1)
        g_flat = dwt_layer.g.view(dwt_layer.channels, -1)
        ortho_loss = (h_flat * g_flat).sum(dim=-1).abs().mean()
        
        total = task_loss + self.lambda_recon * recon_loss + self.lambda_ortho * ortho_loss
        return total, task_loss

# ══════════════════════════════════════════════════════════════
# SECTION 4 — DATASET & EVALUATION
# ══════════════════════════════════════════════════════════════

class ETTDataset(Dataset):
    def __init__(self, seq_len, pred_len, split='train',  file_path=r'C:\Users\Asus\Desktop\TTS\WRAT\ETTm1.csv'):
        self.seq_len, self.pred_len = seq_len, pred_len
        df = pd.read_csv(file_path)
        data = df.iloc[:, 1:].values 
        
        train_end = 12 * 30 * 24 * 4
        val_end   = train_end + 4 * 30 * 24 * 4
        raw = {'train': data[:train_end], 'val': data[train_end:val_end], 'test': data[val_end:]}[split]
        
        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)

    def inverse(self, x_norm): 
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
            bx_flat = bx.reshape(B * C, 1, L)
            preds, _, _, _, _ = model(bx_flat)
            preds = preds.reshape(B, C, -1)
            all_p.append(preds.cpu())
            all_t.append(by.cpu())
            
    p, t = torch.cat(all_p, 0).flatten().numpy(), torch.cat(all_t, 0).flatten().numpy()
    err = p - t
    mae, mse = float(np.abs(err).mean()), float((err**2).mean())
    dir_acc = float((np.sign(np.diff(p)) == np.sign(np.diff(t))).mean() * 100)
    model.train()
    return dict(mae=mae, mse=mse, dir_acc=dir_acc)


def train_pwsa(model, opt, crit, loader, tau_override=None, zero_lh=False):
    model.train()
    if tau_override is not None and hasattr(model.wrat_block, 'sparsity_tau'):
        model.wrat_block.sparsity_tau = tau_override
        
    total_loss, n = 0, 0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        B, C, L = bx.shape
        bx_flat = bx.reshape(B * C, 1, L)
        by_flat = by.reshape(B * C, 1, by.shape[-1])
        
        opt.zero_grad()
        
        preds, patches_recon, patches_orig, _, _ = model(bx_flat, zero_lh=zero_lh)
        loss, _ = crit(preds, by_flat, patches_orig, patches_recon, model.dwt)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        n += 1
        
    return total_loss / n

def extract_and_plot_filters(model, horizon, model_type="Learnable"):
    h_weights = model.dwt.h.detach().cpu().numpy()
    g_weights = model.dwt.g.detach().cpu().numpy()
    # Average across all 64 D_MODEL embedding channels to see the global wavelet shape
    h_avg = np.mean(h_weights, axis=(0, 1))
    g_avg = np.mean(g_weights, axis=(0, 1))
    
    print(f"\n --- Learned Signal Processing Insights (H={horizon} | {model_type}) ---")
    print(f" Average Low-Pass (Trend) Filter 'h' : {np.round(h_avg, 4)}")
    print(f" Average High-Pass (Noise) Filter 'g': {np.round(g_avg, 4)}")
    if hasattr(model.wrat_block, 'tau'):
        print(f" Final Learned Sparsity Threshold (tau): {model.wrat_block.tau:.4f}")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — CONFIG & EXECUTION LOOP
# ══════════════════════════════════════════════════════════════

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

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 512   # UPGRADED: 512 Look-back window (Matches PatchTST/64)
D_MODEL    = 64    # UPGRADED: 64 Embedding Dimension
PATCH_LEN  = 16
STRIDE     = 8
HORIZONS   = [96, 192, 336, 720]
BATCH_SIZE = 32    # Lowered slightly for memory safety on L=512
EPOCHS     = 30    # FULL 30 EPOCH RUN
PATIENCE   = 10
LR         = 5e-4 

crit_pwsa = DualHeadPWSA_Loss(lambda_recon=0.1, lambda_ortho=0.01)

print(f"Device: {DEVICE} | Horizons: {HORIZONS}")
print(f"P-WSA Settings -> L: {SEQ_LEN} | D_MODEL: {D_MODEL} | Patch: {PATCH_LEN} | Epochs: {EPOCHS}\n")
print(">>> RUNNING FULL P-WSA MULTIVARIATE BENCHMARK <<< \n")

for PRED_LEN in HORIZONS:
    print(f"\n{'='*65}\n HORIZON = {PRED_LEN} \n{'='*65}")
    
    train_ds = ETTDataset(SEQ_LEN, PRED_LEN, 'train')
    val_ds   = ETTDataset(SEQ_LEN, PRED_LEN, 'val')
    test_ds  = ETTDataset(SEQ_LEN, PRED_LEN, 'test')
    
    trl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    vll = DataLoader(val_ds, BATCH_SIZE, shuffle=False, drop_last=True)
    tel = DataLoader(test_ds, BATCH_SIZE, shuffle=False, drop_last=True)
    
    model_lrn = PatchedWSA(SEQ_LEN, PRED_LEN, D_MODEL, tau_type='learnable').to(DEVICE)
    opt_lrn = optim.AdamW(model_lrn.parameters(), lr=LR, weight_decay=1e-4)
    sched_lrn = optim.lr_scheduler.CosineAnnealingLR(opt_lrn, T_max=EPOCHS, eta_min=1e-6)
    early_stop_lrn = EarlyStopping(patience=PATIENCE)
    
    if PRED_LEN == HORIZONS[0]:
        params = sum(p.numel() for p in model_lrn.parameters())
        print(f" New Parameters: {params:,} (Scaled for P-WSA)\n")
    
    print(f" {'Ep':>3} | {'LR':>8} | {'Val MSE (Learnable)':>19}")
    print(f" {'-'*40}")
    
    for epoch in range(1, EPOCHS+1):
        current_lr = opt_lrn.param_groups[0]['lr']
        
        if not early_stop_lrn.early_stop: 
            train_pwsa(model_lrn, opt_lrn, crit_pwsa, trl)
            
        v_lrn = evaluate(model_lrn, vll, DEVICE)['mse'] if not early_stop_lrn.early_stop else early_stop_lrn.best_loss
        
        if not early_stop_lrn.early_stop: 
            early_stop_lrn(v_lrn, model_lrn)
        
        lrn_tag = " (Stopped)" if early_stop_lrn.early_stop and epoch == early_stop_lrn.counter + early_stop_lrn.patience else ""
        print(f" {epoch:>3} | {current_lr:>8.6f} | {v_lrn:>10.5f}{lrn_tag:<9}")
        
        if early_stop_lrn.early_stop:
            print(" Early stopping reached.")
            break
            
        sched_lrn.step()

    # Load best state for Test evaluation
    model_lrn.load_state_dict(early_stop_lrn.best_state)
    
    res_lrn = evaluate(model_lrn, tel, DEVICE)
    
    print(f"\n FULL P-WSA TEST RESULTS (Horizon {PRED_LEN}):")
    print(f" P-WSA (Learnable) -> MAE: {res_lrn['mae']:.4f} | MSE: {res_lrn['mse']:.4f} | DirAcc: {res_lrn['dir_acc']:.2f}%")

    extract_and_plot_filters(model_lrn, PRED_LEN, model_type="Learnable")

print("\nP-WSA Full Benchmark complete.")