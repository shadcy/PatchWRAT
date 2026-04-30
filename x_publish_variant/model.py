import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(1, num_features, 1))
        self.b   = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, mode):
        # x shape: (B, C, L)
        if mode == 'norm':
            self.mean  = x.mean(-1, keepdim=True).detach()
            self.stdev = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            return ((x - self.mean) / self.stdev) * self.w + self.b
        return ((x - self.b) / self.w) * self.stdev + self.mean

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

    def forward(self, q_x, k_x, v_x, energy=None, use_sparse=True):
        B, L, D = q_x.shape; Dh = D // self.H
        Q = self.q(q_x).view(B, L, self.H, Dh).transpose(1, 2)
        K = self.k(k_x).view(B, -1, self.H, Dh).transpose(1, 2)
        V = self.v(v_x).view(B, -1, self.H, Dh).transpose(1, 2)
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        
        if use_sparse and energy is not None:
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

    def forward(self, LL, LH, use_sparse=True):
        tau = torch.sigmoid(self.raw_tau).item()
        self.lh_attn.threshold = tau
        ll_s = LL.transpose(1, 2); lh_s = LH.transpose(1, 2)
        
        ll_o = self.ll_attn(ll_s, ll_s, ll_s, use_sparse=False) # LL is always dense
        lh_o = self.lh_attn(lh_s, lh_s, lh_s, energy=lh_s, use_sparse=use_sparse)
        cr_o = self.cross_attn(ll_o, lh_o, lh_o, use_sparse=False)
        
        ll_f = self.n1(ll_s + ll_o + cr_o);  ll_f = self.mlp_ll(ll_f) + ll_f
        lh_f = self.n2(lh_s + lh_o);         lh_f = self.mlp_lh(lh_f) + lh_f
        return ll_f.transpose(1, 2), lh_f.transpose(1, 2)

class PatchWRAT(nn.Module):
    def __init__(self, seq_len, pred_len, num_channels=7, d_model=32, num_heads=4, 
                 dropout=0.3, use_revin=True, use_sparse=False):
        super().__init__()
        self.use_revin  = use_revin
        self.use_sparse = use_sparse
        self.pred_len   = pred_len
        self.num_ch     = num_channels

        if self.use_revin:
            self.revin = RevIN(num_channels)
            
        self.dwt  = LearnableDWT1D(1, d_model)
        self.wrat = WRATBlock(d_model, num_heads, dropout=dropout)
        
        l_half    = seq_len // 2
        flat_dim  = l_half * d_model * 2
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, pred_len)
        )
        self.sc   = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        B, C, L = x.shape
        
        if self.use_revin:
            x = self.revin(x, 'norm')

        # Channel Independence: merge Batch and Channel dims
        x_ci = x.reshape(B * C, 1, L)
        
        LL, LH     = self.dwt(x_ci)
        LL_o, LH_o = self.wrat(LL, LH, use_sparse=self.use_sparse)
        
        fused      = torch.cat([LL_o, LH_o], dim=1)
        preds      = self.head(fused).unsqueeze(1) # (B*C, 1, pred_len)
        
        # Reshape back to multivariate dims
        preds = preds.reshape(B, C, self.pred_len)
        xr    = self.sc(self.dwt.inverse(LL_o, LH_o)).reshape(B, C, -1)
        LL    = LL.reshape(B, C, -1)
        LH    = LH.reshape(B, C, -1)

        if self.use_revin:
            preds = self.revin(preds, 'denorm')
            xr    = self.revin(xr, 'denorm')

        return preds, xr, LL, LH
