"""
============================================================
Wavelet Residual Attention Transformer (WRAT)
============================================================
Standalone module for time-series forecasting.
Includes fixed-tau, adaptive-tau, and learnable-tau variants.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableDWT1D(nn.Module):
    """1D Discrete Wavelet Transform with learnable filters."""
    def __init__(self, in_channels, out_channels, filter_length=4):
        super().__init__()
        self.h = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.filter_length = filter_length
        self.padding_val = (filter_length - 2) // 2
        with torch.no_grad():
            self.g -= self.g.mean(dim=-1, keepdim=True)

    def forward(self, x):
        # This specific padding calculation is designed to ensure that the output 
        # sequence length is exactly half of the input sequence length
        LL = F.conv1d(x, self.h, stride=2, padding=self.padding_val)
        LH = F.conv1d(x, self.g, stride=2, padding=self.padding_val)
        return LL, LH

    def inverse(self, LL, LH):
        x_recon_L = F.conv_transpose1d(LL, self.h, stride=2, padding=self.padding_val)
        x_recon_H = F.conv_transpose1d(LH, self.g, stride=2, padding=self.padding_val)
        min_len = min(x_recon_L.shape[-1], x_recon_H.shape[-1])
        return x_recon_L[..., :min_len] + x_recon_H[..., :min_len]


class FrequencySparseAttention(nn.Module):
    """Energy-thresholded sparse attention mechanism."""
    def __init__(self, d_model, num_heads, threshold=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model   = d_model
        self.threshold = threshold
        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q_x, k_x, v_x, energy_coeffs=None):
        B, L, D = q_x.shape
        H  = self.num_heads
        D_h = D // H
        Q = self.q_proj(q_x).view(B, L, H, D_h).transpose(1, 2)
        K = self.k_proj(k_x).view(B, -1, H, D_h).transpose(1, 2)
        V = self.v_proj(v_x).view(B, -1, H, D_h).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D_h ** 0.5)
        
        if energy_coeffs is not None:
            energy = torch.abs(energy_coeffs).mean(dim=-1)
            if isinstance(self.threshold, torch.Tensor):
                temperature = 10.0 
                soft_mask = torch.sigmoid((energy - self.threshold) * temperature)
                hard_mask = (energy > self.threshold).float()
                mask_val = hard_mask.detach() - soft_mask.detach() + soft_mask
                mask_val = mask_val.view(B, 1, 1, -1)
                scores = scores + (1.0 - mask_val) * -1e9 
            else:
                mask   = energy > self.threshold
                mask   = mask.view(B, 1, 1, -1)
                scores = scores.masked_fill(~mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class WRATBlock(nn.Module):
    """Core Wavelet Residual Attention Transformer block."""
    def __init__(self, d_model, num_heads, sparsity_tau=0.1):
        super().__init__()
        self.sparsity_tau   = sparsity_tau
        self.intra_LL_attn  = FrequencySparseAttention(d_model, num_heads)
        self.intra_LH_attn  = FrequencySparseAttention(d_model, num_heads, threshold=sparsity_tau)
        self.cross_attn     = FrequencySparseAttention(d_model, num_heads)
        self.mlp_LL = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.mlp_LH = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
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
    """WRATBlock with tau as a trainable sigmoid-bounded parameter."""
    def __init__(self, d_model, num_heads, tau_init=0.1):
        super().__init__()
        self.raw_tau = nn.Parameter(torch.tensor(
            math.log(tau_init / (1.0 - tau_init))))   # inverse sigmoid
        self._block  = WRATBlock(d_model, num_heads, sparsity_tau=tau_init)

    @property
    def tau(self):
        return torch.sigmoid(self.raw_tau).item() # Fine to keep .item() here for logging

    def forward(self, LL, LH):
        # PASS THE TENSOR directly, do not use .item()
        self._block.intra_LH_attn.threshold = torch.sigmoid(self.raw_tau)
        return self._block(LL, LH)

class WaveletTransformerLoss(nn.Module):
    """Custom loss function balancing task MSE, reconstruction, and filter orthogonality."""
    def __init__(self, lambda_recon=1.0, lambda_ortho=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho

    def forward(self, preds, targets, x_orig, x_recon, dwt_layer):
        task_loss  = F.mse_loss(preds, targets)
        recon_loss = F.mse_loss(x_recon, x_orig) if x_recon is not None else 0.0
        
        # Orthogonality: penalise dot product between h and g filters
        h_flat = dwt_layer.h.view(dwt_layer.h.shape[0], -1)
        g_flat = dwt_layer.g.view(dwt_layer.g.shape[0], -1)
        ortho_loss = (h_flat * g_flat).sum().abs()
        
        total = task_loss + self.lambda_recon * recon_loss + self.lambda_ortho * ortho_loss
        return total, task_loss, recon_loss, ortho_loss


class WRATModel(nn.Module):
    """
    Complete End-to-End WRAT architecture.
    Ties together the DWT, WRAT blocks, and final projection.
    """
    def __init__(self, in_channels=1, d_model=64, num_heads=4, tau_type='fixed', tau_init=0.1):
        super().__init__()
        self.dwt = LearnableDWT1D(in_channels, d_model)
        
        if tau_type == 'learnable':
            self.block = LearnableTauWRATBlock(d_model, num_heads, tau_init)
        else:
            self.block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init)
            
        self.projection = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, zero_lh=False):
        # 1. Decomposition
        LL, LH = self.dwt(x)
        
        if zero_lh: 
            LH = torch.zeros_like(LH)
            
        # 2. Sparse Attention Processing
        LL_out, LH_out = self.block(LL, LH)
        
        # 3. Inverse DWT & Final Projection
        recon = self.dwt.inverse(LL_out, LH_out)
        preds = self.projection(recon)
        
        return preds