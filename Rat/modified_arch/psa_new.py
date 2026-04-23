# ============================================================
# run_pwsa_ablation_suite.py  —  P-WSA Publication Benchmark
# ============================================================
import os
import urllib.request
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
# SECTION 0 — AUTOMATED DATASET RETRIEVAL
# ══════════════════════════════════════════════════════════════
def download_ett_datasets(data_dir='./dataset', target='ETTh1.csv'):
    """Downloads only the specified target dataset to save time."""
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
    
    file_path = os.path.join(data_dir, target)
    if not os.path.exists(file_path):
        print(f"Downloading {target} from official repository...")
        urllib.request.urlretrieve(base_url + target, file_path)
    return {target.split('.')[0]: file_path}


# ══════════════════════════════════════════════════════════════
# SECTION 1 — NORMALIZATION, PATCHING, & WAVELETS
# ══════════════════════════════════════════════════════════════
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features, self.eps, self.affine = num_features, eps, affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=-1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight.unsqueeze(-1) + self.affine_bias.unsqueeze(-1)
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias.unsqueeze(-1)) / (self.affine_weight.unsqueeze(-1) + self.eps**2)
            x = x * self.stdev + self.mean
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.proj = nn.Linear(patch_len, d_model)
        num_patches = (seq_len - patch_len) // stride + 1
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)
        x = self.proj(x)  # [B, num_patches, d_model]
        n = x.shape[1]    # actual number of patches
        pos_emb = self.position_embedding[:, :n, :]  # trim along patch dim
        if pos_emb.shape[1] < n:                     # pad if needed (rare)
            pad = torch.zeros(1, n - pos_emb.shape[1], x.shape[2], device=x.device)
            pos_emb = torch.cat([pos_emb, pad], dim=1)
        x = x + pos_emb   # add positional encoding in [B, num_patches, d_model] space
        return x.transpose(1, 2)  # → [B, d_model, num_patches] for DWT

class LearnableDWT1D(nn.Module):
    def __init__(self, channels, filter_length=4):
        super().__init__()
        self.channels, self.padding_val = channels, (filter_length - 2) // 2
        self.h = nn.Parameter(torch.randn(channels, 1, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(channels, 1, filter_length) * 0.1)
        with torch.no_grad(): self.g -= self.g.mean(dim=-1, keepdim=True)

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
        self.num_heads, self.d_model = num_heads, d_model
        self.threshold = torch.tensor(threshold) if isinstance(threshold, float) else threshold
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = [nn.Linear(d_model, d_model) for _ in range(4)]

    def forward(self, q_x, k_x, v_x, energy_coeffs=None):
        B, L, D = q_x.shape
        H, D_h = self.num_heads, D // self.num_heads
        
        Q = self.q_proj(q_x).view(B, L, H, D_h).transpose(1, 2)
        K = self.k_proj(k_x).view(B, -1, H, D_h).transpose(1, 2)
        V = self.v_proj(v_x).view(B, -1, H, D_h).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D_h ** 0.5)
        attn_weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        
        if energy_coeffs is not None:
            energy = torch.abs(energy_coeffs).mean(dim=-1)
            gate = torch.sigmoid((energy - self.threshold.to(energy.device)) * 10.0).view(B, 1, 1, -1)
            attn_weights = attn_weights * gate
            
        out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class WRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_tau=0.1, dropout=0.2):
        super().__init__()
        self.intra_LL_attn = FrequencySparseAttention(d_model, num_heads)
        self.intra_LH_attn = FrequencySparseAttention(d_model, num_heads, threshold=sparsity_tau)
        self.cross_attn    = FrequencySparseAttention(d_model, num_heads)
        self.mlp_LL = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
        self.mlp_LH = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, LL, LH):
        LL_seq, LH_seq = LL.transpose(1, 2), LH.transpose(1, 2)
        LL_out    = self.intra_LL_attn(LL_seq, LL_seq, LL_seq)
        LH_out    = self.intra_LH_attn(LH_seq, LH_seq, LH_seq, energy_coeffs=LH_seq)
        cross_out = self.cross_attn(LL_out, LH_out, LH_out)
        
        LL_fused  = self.norm1(LL_seq + LL_out + cross_out)
        LH_fused  = self.norm2(LH_seq + LH_out)
        return (self.mlp_LL(LL_fused) + LL_fused).transpose(1, 2), (self.mlp_LH(LH_fused) + LH_fused).transpose(1, 2)

class LearnableTauWRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, tau_init=0.1, dropout=0.2):
        super().__init__()
        self.raw_tau = nn.Parameter(torch.tensor(math.log(tau_init / (1.0 - tau_init))))
        self._block  = WRATBlock(d_model, num_heads, sparsity_tau=tau_init, dropout=dropout)

    @property
    def tau(self): return torch.sigmoid(self.raw_tau).item()

    def forward(self, LL, LH):
        self._block.intra_LH_attn.threshold = torch.sigmoid(self.raw_tau)
        return self._block(LL, LH)

class AdaptiveMultiresolutionFusion(nn.Module):
    def __init__(self, dwt_len, d_model):
        super().__init__()
        self.ll_gate = nn.Parameter(torch.ones(1, dwt_len, d_model))
        self.lh_gate = nn.Parameter(torch.ones(1, dwt_len, d_model) * 0.5)

    def forward(self, LL, LH):
        LL_scaled = LL.transpose(1, 2) * self.ll_gate
        LH_scaled = LH.transpose(1, 2) * self.lh_gate
        return torch.cat([LL_scaled, LH_scaled], dim=1) 

# ══════════════════════════════════════════════════════════════
# SECTION 3 — PATCHED WSA ARCHITECTURE
# ══════════════════════════════════════════════════════════════
class PatchedWSA(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=64, num_heads=4, patch_len=16, stride=8, tau_init=0.1, tau_type='learnable', dropout=0.2):
        super().__init__()
        self.revin = RevIN(num_features=1)
        self.patch_emb = PatchEmbedding(seq_len, patch_len, stride, d_model)
        self.dwt = LearnableDWT1D(channels=d_model)
        self.wrat_block = LearnableTauWRATBlock(d_model, num_heads, tau_init, dropout) if tau_type == 'learnable' else WRATBlock(d_model, num_heads, sparsity_tau=tau_init, dropout=dropout)
            
        num_patches = (seq_len - patch_len) // stride + 1
        dwt_len = (num_patches + 2 * ((4 - 2) // 2) - 4) // 2 + 1
        
        self.fusion = AdaptiveMultiresolutionFusion(dwt_len, d_model)
        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Dropout(dropout), nn.Linear(dwt_len * d_model * 2, pred_len)
        )

    def forward(self, x, zero_lh=False):
        x_norm = self.revin(x, mode='norm')
        patches = self.patch_emb(x_norm) 
        LL, LH = self.dwt(patches)
        if zero_lh: LH = torch.zeros_like(LH) 
        LL_out, LH_out = self.wrat_block(LL, LH)
        
        fused = self.fusion(LL_out, LH_out) 
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
        return task_loss + self.lambda_recon * recon_loss + self.lambda_ortho * ortho_loss, task_loss.item()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — DATASET, TRAINING, & VISUALIZATION
# ══════════════════════════════════════════════════════════════
class ETTDataset(Dataset):
    def __init__(self, seq_len, pred_len, file_path, split='train'):
        self.seq_len, self.pred_len = seq_len, pred_len
        df = pd.read_csv(file_path)
        data = df.iloc[:, 1:].values 
        
        is_minute = 'm' in os.path.basename(file_path)
        points_per_day = 24 * 4 if is_minute else 24
        
        train_end = 12 * 30 * points_per_day
        val_end   = train_end + 4 * 30 * points_per_day
        raw = {'train': data[:train_end], 'val': data[train_end:val_end], 'test': data[val_end:]}[split]
        
        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end]) # Strict rule: Fit ONLY on training data to prevent leakage
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)

    def __len__(self): return max(0, len(self.data) - self.seq_len - self.pred_len)
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x.t(), y.t()

def evaluate_and_plot(model, loader, device, ds_name, horizon, save_dir='./results/plots/', zero_lh=False):
    """Calculates strict multivariate MSE/MAE and saves sample plots."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_p, all_t = [], []
    
    saved_plot = False # Track if we've saved a plot for this evaluation pass
    
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            B, C, L = bx.shape
            preds, _, _, _, _ = model(bx.reshape(B * C, 1, L), zero_lh=zero_lh)
            
            # Reshape back to [Batch, Channel, Pred_Len]
            preds = preds.reshape(B, C, -1).cpu()
            targets = by.cpu()
            
            all_p.append(preds)
            all_t.append(targets)
            
            # Save visual samples for the first batch only
            if not saved_plot:
                plt.figure(figsize=(10, 4))
                # Plotting the last channel (often the target variable in ETT datasets like 'OT')
                plt.plot(targets[0, -1, :].numpy(), label='Ground Truth', color='blue', linewidth=2)
                plt.plot(preds[0, -1, :].numpy(), label='PWSA Prediction', color='orange', linestyle='--', linewidth=2)
                plt.title(f"{ds_name} - Horizon {horizon} (Target Feature)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_path = os.path.join(save_dir, f"{ds_name}_H{horizon}_sample.png")
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                plt.close()
                saved_plot = True
            
    # Standard Multivariate Calculation: Flattening across all dimensions yields average error per point
    p = torch.cat(all_p, 0).flatten().numpy()
    t = torch.cat(all_t, 0).flatten().numpy()
    
    err = p - t
    mae = float(np.abs(err).mean())
    mse = float((err**2).mean())
    
    return dict(mae=mae, mse=mse)

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience, self.counter, self.best_loss, self.early_stop, self.best_state = patience, 0, None, False, None
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss, self.best_state, self.counter = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

# ══════════════════════════════════════════════════════════════
# SECTION 5 — CONFIG & MASTER EXECUTION LOOP
# ══════════════════════════════════════════════════════════════
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ENFORCED SOTA PARAMETERS
SEQ_LEN = 512       # Look-back window L=512
HORIZONS = [96, 192, 336, 720] # Target Horizons T

D_MODEL, PATCH_LEN, STRIDE = 64, 16, 8
BATCH_SIZE = 32
EPOCHS     = 10 # Adjust as needed
LR         = 1e-3 

print(">>> FETCHING DATASET <<<")
# Restricted strictly to 1 dataset for current testing phase
dataset_paths = download_ett_datasets(target='ETTh1.csv')
crit_pwsa = DualHeadPWSA_Loss()

for ds_name, file_path in dataset_paths.items():
    print(f"\n\n{'='*80}")
    print(f"🚀 INITIATING TRAINING FOR: {ds_name.upper()}")
    print(f"   Look-back Window (L): {SEQ_LEN}")
    print(f"{'='*80}")
    
    final_results = {h: {} for h in HORIZONS}
    
    for PRED_LEN in HORIZONS:
        print(f"\n--- {ds_name} | HORIZON (T) = {PRED_LEN} ---")
        
        train_ds = ETTDataset(SEQ_LEN, PRED_LEN, file_path, split='train')
        val_ds   = ETTDataset(SEQ_LEN, PRED_LEN, file_path, split='val')
        test_ds  = ETTDataset(SEQ_LEN, PRED_LEN, file_path, split='test')
        
        trl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
        vll = DataLoader(val_ds, BATCH_SIZE, shuffle=False, drop_last=True)
        tel = DataLoader(test_ds, BATCH_SIZE, shuffle=False, drop_last=True)
        
        m_lrn = PatchedWSA(SEQ_LEN, PRED_LEN, D_MODEL, tau_type='learnable').to(DEVICE)
        opt_lrn = optim.AdamW(m_lrn.parameters(), lr=LR)
        es_lrn = EarlyStopping()
        
        for epoch in range(1, EPOCHS+1):
            m_lrn.train()
            
            for bx, by in trl:
                bx_flat, by_flat = bx.to(DEVICE).reshape(-1, 1, SEQ_LEN), by.to(DEVICE).reshape(-1, 1, PRED_LEN)
                
                if not es_lrn.early_stop:
                    opt_lrn.zero_grad()
                    p, rec, orig, _, _ = m_lrn(bx_flat)
                    loss, _ = crit_pwsa(p, by_flat, orig, rec, m_lrn.dwt)
                    loss.backward(); opt_lrn.step()
                    
            # Use MSE as the standard validation metric
            v_metrics = evaluate_and_plot(m_lrn, vll, DEVICE, ds_name, PRED_LEN)
            v_lrn = v_metrics['mse']
            
            if not es_lrn.early_stop: es_lrn(v_lrn, m_lrn)
            
            print(f" Ep {epoch:>2} | Val MSE: {v_lrn:.4f}")
            if es_lrn.early_stop: 
                print("   Early stopping triggered.")
                break

        # Load best weights and evaluate on Test set
        m_lrn.load_state_dict(es_lrn.best_state)
        
        # This will also save your plots to ./results/plots/
        res_lrn = evaluate_and_plot(m_lrn, tel, DEVICE, ds_name, PRED_LEN, save_dir='./results/plots/')
        print(f" Test Results: MSE={res_lrn['mse']:.4f} | MAE={res_lrn['mae']:.4f}")
        print(f" Plots saved to: ./results/plots/")

print("\n>>> EXECUTION COMPLETE <<<")