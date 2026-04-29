import os, sys, math, zipfile, warnings, copy, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SECTION 1 — MODEL ARCHITECTURE (Optimized for Classification)
# ══════════════════════════════════════════════════════════════

class RevIN(nn.Module):
    """Reversible Instance Normalization to handle non-stationary LOB data."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None
        self.std  = None

    def normalize(self, x):
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std  = x.std(dim=-1, keepdim=True) + self.eps
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

class LearnableDWT1D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_length=4):
        super().__init__()
        self.h = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.padding_val = (filter_length - 2) // 2
        with torch.no_grad(): self.g -= self.g.mean(dim=-1, keepdim=True)

    def forward(self, x):
        LL = F.conv1d(x, self.h, stride=2, padding=self.padding_val)
        LH = F.conv1d(x, self.g, stride=2, padding=self.padding_val)
        return LL, LH

class FrequencySparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, threshold=0.1):
        super().__init__()
        self.num_heads, self.d_model, self.threshold = num_heads, d_model, threshold
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q_x, k_x, v_x, energy_coeffs=None):
        B, L, D = q_x.shape
        H, D_h = self.num_heads, D // self.num_heads
        Q = self.q_proj(q_x).view(B, L, H, D_h).transpose(1, 2)
        K = self.k_proj(k_x).view(B, -1, H, D_h).transpose(1, 2)
        V = self.v_proj(v_x).view(B, -1, H, D_h).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D_h ** 0.5)
        
        if energy_coeffs is not None:
            energy = torch.abs(energy_coeffs).mean(dim=-1)
            # Straight-Through Estimator (STE) for differentiability
            soft_mask = torch.sigmoid((energy - self.threshold) * 10.0)
            hard_mask = (energy > self.threshold).float()
            mask_val = hard_mask.detach() - soft_mask.detach() + soft_mask
            scores = scores + (1.0 - mask_val.view(B, 1, 1, -1)) * -1e9
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

class WRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_tau=0.1):
        super().__init__()
        self.sparsity_tau = sparsity_tau
        self.intra_LL_attn = FrequencySparseAttention(d_model, num_heads)
        self.intra_LH_attn = FrequencySparseAttention(d_model, num_heads, threshold=sparsity_tau)
        self.cross_attn = FrequencySparseAttention(d_model, num_heads)
        self.mlp_LL = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.mlp_LH = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, LL, LH):
        LL_s, LH_s = LL.transpose(1, 2), LH.transpose(1, 2)
        LL_out = self.intra_LL_attn(LL_s, LL_s, LL_s)
        LH_out = self.intra_LH_attn(LH_s, LH_s, LH_s, energy_coeffs=LH_s)
        cross_out = self.cross_attn(LL_out, LH_out, LH_out)
        LL_f = self.norm1(LL_s + LL_out + cross_out)
        LH_f = self.norm2(LH_s + LH_out)
        return (self.mlp_LL(LL_f) + LL_f).transpose(1, 2), (self.mlp_LH(LH_f) + LH_f).transpose(1, 2)

class WRATClassifier(nn.Module):
    def __init__(self, in_channels, d_model, num_heads, seq_len, tau_init=0.1):
        super().__init__()
        self.revin = RevIN(in_channels)
        self.dwt   = LearnableDWT1D(in_channels, d_model)
        self.block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * (seq_len // 2), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3) # Up, Down, Neutral
        )

    def forward(self, x):
        x = self.revin.normalize(x)
        LL, LH = self.dwt(x)
        LL_o, LH_o = self.block(LL, LH)
        return self.classifier(LL_o)

# ══════════════════════════════════════════════════════════════
# SECTION 2 — DATA AUTOMATION (FI-2010)
# ══════════════════════════════════════════════════════════════

def download_fi2010():
    url = "https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/FI2010/Train_Dst_NoAuction_Dec_CF_7.txt"
    if not os.path.exists("FI2010_train.txt"):
        print("Downloading FI-2010 Benchmark...")
        urllib.request.urlretrieve(url, "FI2010_train.txt")
    return "FI2010_train.txt"

class FI2010Dataset(Dataset):
    def __init__(self, file_path, seq_len=10, k_horizon=10, split='train'):
        self.seq_len, self.k_idx = seq_len, {10:0, 50:2, 100:3}[k_horizon]
        data = np.loadtxt(file_path)
        
        # Features: top 40 rows | Labels: bottom 5 rows
        features = data[:40, :].T 
        labels = data[-5:, :].T.astype(int) - 1 # Map 1,2,3 -> 0,1,2
        
        split_idx = int(0.8 * len(features))
        if split == 'train':
            self.x, self.y = features[:split_idx], labels[:split_idx]
        else:
            self.x, self.y = features[split_idx:], labels[split_idx:]

    def __len__(self): return len(self.x) - self.seq_len
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx:idx+self.seq_len].T, dtype=torch.float32), \
               torch.tensor(self.y[idx+self.seq_len-1, self.k_idx], dtype=torch.long)

# ══════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def run_experiment():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = download_fi2010()
    K_HORIZONS, TAU_VALS = [10, 50, 100], [0.01, 0.1, 0.5]
    final_summary = {}

    for K in K_HORIZONS:
        print(f"\n▶ Testing Horizon k={K}")
        train_ds = FI2010Dataset(DATA_PATH, k_horizon=K, split='train')
        test_ds = FI2010Dataset(DATA_PATH, k_horizon=K, split='test')
        trl = DataLoader(train_ds, batch_size=128, shuffle=True)
        tel = DataLoader(test_ds, batch_size=128)
        
        horizon_results = {}
        for tau in TAU_VALS:
            model = WRATClassifier(40, 64, 4, 10, tau_init=tau).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            # Weighted loss to handle imbalance
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.6, 1.0]).to(DEVICE))
            
            # Training
            model.train()
            for ep in range(3): # Fast ablation epochs
                for bx, by in trl:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for bx, by in tel:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    all_p.append(model(bx).argmax(-1).cpu()); all_t.append(by.cpu())
            
            p, t = torch.cat(all_p).numpy(), torch.cat(all_t).numpy()
            f1 = f1_score(t, p, average='macro')
            horizon_results[f"Tau_{tau}"] = f1
            print(f"   Tau={tau:<5} | F1-Macro: {f1:.4f}")
        
        final_summary[K] = horizon_results

    # Final Table
    print("\n" + "="*40 + "\nFINAL ABLATION SUMMARY (F1-Macro)\n" + "="*40)
    print(f"{'Tau Variant':<15} | k=10   | k=50   | k=100")
    for tau in TAU_VALS:
        name = f"Tau_{tau}"
        row = f"{name:<15} | " + " | ".join(f"{final_summary[k][name]:.4f}" for k in K_HORIZONS)
        print(row)

if __name__ == "__main__":
    run_experiment()