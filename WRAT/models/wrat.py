import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableDWT1D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_length=4):
        super().__init__()
        # Low-pass filter (h) and High-pass filter (g)
        self.h = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(out_channels, in_channels, filter_length) * 0.1)
        self.filter_length = filter_length
        # Calculate padding to ensure output length is exactly seq_len // 2 for stride 2
        self.padding_val = (filter_length - 2) // 2 

        # Ensure zero-mean initialization for high-pass
        with torch.no_grad():
            self.g -= self.g.mean(dim=-1, keepdim=True)

    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        # Low-frequency band (LL) - Approximation coefficients
        LL = F.conv1d(x, self.h, stride=2, padding=self.padding_val)
        # High-frequency band (LH) - Detail coefficients
        LH = F.conv1d(x, self.g, stride=2, padding=self.padding_val)
        return LL, LH

    def inverse(self, LL, LH):
        # Learnable Inverse DWT using transposed convolution
        x_recon_L = F.conv_transpose1d(LL, self.h, stride=2, padding=self.padding_val)
        x_recon_H = F.conv_transpose1d(LH, self.g, stride=2, padding=self.padding_val)

        # Match lengths if padding caused mismatch 
        min_len = min(x_recon_L.shape[-1], x_recon_H.shape[-1])
        return x_recon_L[..., :min_len] + x_recon_H[..., :min_len]

class FrequencySparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, threshold=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.threshold = threshold

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q_x, k_x, v_x, energy_coeffs=None):
        B, L, D = q_x.shape
        H = self.num_heads
        D_h = D // H

        Q = self.q_proj(q_x).view(B, L, H, D_h).transpose(1, 2)
        K = self.k_proj(k_x).view(B, -1, H, D_h).transpose(1, 2)
        V = self.v_proj(v_x).view(B, -1, H, D_h).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D_h ** 0.5)

        # Apply Wavelet Energy Sparsity Mask
        if energy_coeffs is not None:
            # Calculate energy (magnitude) of detail coefficients
            energy = torch.abs(energy_coeffs).mean(dim=-1) # (B, L_k)
            # Create boolean mask where energy > tau
            mask = energy > self.threshold # (B, L_k)
            mask = mask.view(B, 1, 1, -1) # Broadcast for heads and queries

            # Apply mask: Set low-energy connections to -inf before softmax
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # Handle potential NaNs if a whole row is masked out
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class WRATBlock(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_tau=0.1):
        super().__init__()
        # Attention modules
        self.intra_LL_attn = FrequencySparseAttention(d_model, num_heads)
        self.intra_LH_attn = FrequencySparseAttention(d_model, num_heads, threshold=sparsity_tau)
        self.cross_attn = FrequencySparseAttention(d_model, num_heads) # LL queries LH

        # MLPs for processing post-attention
        self.mlp_LL = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.mlp_LH = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, LL, LH):
        B, C, L = LL.shape
        # Transpose to (B, L, C) for attention
        LL_seq = LL.transpose(1, 2)
        LH_seq = LH.transpose(1, 2)

        # 1. Intra-scale Attention
        LL_out = self.intra_LL_attn(LL_seq, LL_seq, LL_seq)
        LH_out = self.intra_LH_attn(LH_seq, LH_seq, LH_seq, energy_coeffs=LH_seq)

        # 2. Cross-scale Attention (Low freq trend queries High freq details)
        cross_out = self.cross_attn(LL_out, LH_out, LH_out)

        # Combine and apply MLP
        LL_fused = self.norm1(LL_seq + LL_out + cross_out)
        LH_fused = self.norm2(LH_seq + LH_out)

        LL_final = self.mlp_LL(LL_fused) + LL_fused
        LH_final = self.mlp_LH(LH_fused) + LH_fused

        # Transpose back to (B, C, L)
        return LL_final.transpose(1, 2), LH_final.transpose(1, 2)
