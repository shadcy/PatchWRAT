"""
=============================================================================
PatchWRAT — Patch-based Wavelet Routing Attention Transformer
=============================================================================
Architecture Overview
---------------------
  Input (B, C, L)
      │
  RevIN          — per-channel instance normalisation (removes distribution shift)
      │
  PatchEmbedding — unfold time axis into overlapping patches → project to d_model
      │
  LearnableDWT1D — depthwise conv filters learn low-pass (h) / high-pass (g)
      │         decompose embedded patches into:
      ├── LL (trend / approximation coefficients)
      └── LH (detail / noise coefficients)
              │
          WRATBlock / LearnableTauWRATBlock
              ├─ Intra-LL  Self-Attention   (captures long-range trend)
              ├─ Intra-LH  Sparse Attention (energy-gated, threshold τ)
              └─ Cross      Attention       (LL queries LH → trend-guided detail)
              │
  Concat [LL_out, LH_out] → Flatten → Dropout → Linear → Forecast
      │
  RevIN.denorm   — restore original scale

Auxiliary outputs (for reconstruction loss & analysis):
  patch_recon   — inverse-DWT of processed LL/LH bands
  patches       — raw patch embeddings (pre-DWT)
  LL, LH        — raw wavelet decompositions (pre-attention)

Loss = MSE_forecast + λ_recon · MSE_reconstruction + λ_ortho · |⟨h,g⟩|

Reference: PatchWRAT (CS728 Project), IIT Bombay, 2025


Note:
LL and LH is the naming convention actually represent Lowpass and Highpass only
=============================================================================
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

__all__ = [
    "RevIN",
    "PatchEmbedding",
    "LearnableDWT1D",
    "FrequencySparseAttention",
    "WRATBlock",
    "LearnableTauWRATBlock",
    "PatchedWSA",
    "DualHeadPWSA_Loss",
]


# =============================================================================
# 1.  Reversible Instance Normalisation (RevIN)
# =============================================================================

class RevIN(nn.Module):
    """
    Per-channel, per-sample normalisation that stores statistics during the
    forward pass and reverses them on the output — preventing look-ahead and
    keeping the model's internal computation distribution-agnostic.

    Parameters
    ----------
    num_features : int
        Number of input channels / variates.
    eps : float
        Numerical stability epsilon (default 1e-5).
    affine : bool
        Whether to learn per-channel scale and bias (default True).

    Forward Signature
    -----------------
    forward(x, mode) where mode ∈ {'norm', 'denorm'}
        x shape expected: (B, C, L)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got '{mode}'")

    # ------------------------------------------------------------------
    def _get_statistics(self, x: torch.Tensor) -> None:
        """Cache mean and std along the time axis (dim=-1)."""
        self.mean  = x.mean(dim=-1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight.unsqueeze(-1) + self.affine_bias.unsqueeze(-1)
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias.unsqueeze(-1)) / (
                self.affine_weight.unsqueeze(-1) + self.eps ** 2
            )
        return x * self.stdev + self.mean


# =============================================================================
# 2.  Patch Embedding
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Segment a univariate time series into overlapping patches and linearly
    project each patch into a d_model-dimensional token.

    Parameters
    ----------
    patch_len : int   — length of each patch window.
    stride    : int   — step between consecutive patches.
    d_model   : int   — embedding dimension.

    Input  : (B, 1, L)         — single-channel series
    Output : (B, d_model, N)   — N patch tokens, channels-first for conv ops
    """

    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj      = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) → unfold → (B, 1, N, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.squeeze(1)          # (B, N, patch_len)
        x = self.proj(x)          # (B, N, d_model)
        return x.transpose(1, 2)  # (B, d_model, N)  — channels-first


# =============================================================================
# 3.  Learnable 1-D DWT
# =============================================================================

class LearnableDWT1D(nn.Module):
    """
    Learnable 1-D Discrete Wavelet Transform using depthwise 1-D convolutions.

    Instead of fixed Haar / Daubechies wavelets, the low-pass (h) and
    high-pass (g) filters are trained jointly with the rest of the network.
    An orthogonality regularisation term (‖h·g^T‖) keeps the filters
    physically interpretable during training.

    Parameters
    ----------
    channels      : int — number of input feature channels (= d_model).
    filter_length : int — filter tap count (default 4, like Daubechies-2).

    Forward
    -------
    Input  : (B, C, N)   — patch embeddings
    Output : (LL, LH) each of shape (B, C, N//2)
        LL  — low-frequency  / trend  approximation coefficients
        LH  — high-frequency / detail noise coefficients

    Inverse
    -------
    inverse(LL, LH) → reconstructed (B, C, N') — used for reconstruction loss
    """

    def __init__(self, channels: int, filter_length: int = 4):
        super().__init__()
        self.channels      = channels
        self.filter_length = filter_length
        # Same-padding formula for stride-2 conv with this filter length
        self.padding_val   = (filter_length - 2) // 2

        # Learnable filters — depthwise (groups = channels)
        self.h = nn.Parameter(torch.randn(channels, 1, filter_length) * 0.1)
        self.g = nn.Parameter(torch.randn(channels, 1, filter_length) * 0.1)

        # Initialise g with zero-mean (high-pass prior)
        with torch.no_grad():
            self.g -= self.g.mean(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Decompose: x → (LL, LH)."""
        LL = F.conv1d(x, self.h, stride=2, padding=self.padding_val, groups=self.channels)
        LH = F.conv1d(x, self.g, stride=2, padding=self.padding_val, groups=self.channels)
        return LL, LH

    def inverse(self, LL: torch.Tensor, LH: torch.Tensor) -> torch.Tensor:
        """Reconstruct from wavelet sub-bands using transposed convolutions."""
        x_L = F.conv_transpose1d(LL, self.h, stride=2, padding=self.padding_val, groups=self.channels)
        x_H = F.conv_transpose1d(LH, self.g, stride=2, padding=self.padding_val, groups=self.channels)
        min_len = min(x_L.shape[-1], x_H.shape[-1])
        return x_L[..., :min_len] + x_H[..., :min_len]


# =============================================================================
# 4.  Frequency-Sparse Attention
# =============================================================================

class FrequencySparseAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with an optional differentiable
    wavelet-energy sparsity gate for the high-frequency branch.

    When `energy_coeffs` is provided (for the LH branch), the attention
    weights are soft-masked by a sigmoid gate:

        gate  = σ( (energy − τ) × 10 )
        weights = softmax(QK^T / √d_h) ⊙ gate

    This suppresses attention on low-energy (noise) positions while keeping
    the operation fully differentiable — so τ (or raw_τ) can be trained.

    Parameters
    ----------
    d_model   : int   — model dimension.
    num_heads : int   — number of attention heads (d_model must be divisible).
    threshold : float — initial sparsity threshold τ (ignored if overridden externally).
    """

    def __init__(self, d_model: int, num_heads: int, threshold: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model   = d_model
        # Store threshold as tensor so it can be replaced by a Parameter gradient path
        self.threshold = torch.tensor(threshold)

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------------
    def forward(
        self,
        q_x: torch.Tensor,
        k_x: torch.Tensor,
        v_x: torch.Tensor,
        energy_coeffs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q_x, k_x, v_x : (B, L, D)
        energy_coeffs  : (B, L_k, D) — LH coefficients for energy gating.

        Returns
        -------
        out : (B, L, D)
        """
        B, L, D = q_x.shape
        H, D_h  = self.num_heads, D // self.num_heads

        Q = self.q_proj(q_x).view(B, L,    H, D_h).transpose(1, 2)  # (B,H,L,D_h)
        K = self.k_proj(k_x).view(B, -1,   H, D_h).transpose(1, 2)
        V = self.v_proj(v_x).view(B, -1,   H, D_h).transpose(1, 2)

        scores       = torch.matmul(Q, K.transpose(-2, -1)) / (D_h ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # -- Energy-based sparsity gate (LH branch only) ------------------
        if energy_coeffs is not None:
            energy = torch.abs(energy_coeffs).mean(dim=-1)          # (B, L_k)
            tau    = self.threshold.to(energy.device)
            gate   = torch.sigmoid((energy - tau) * 10.0)           # soft gate
            gate   = gate.view(B, 1, 1, -1)                         # broadcast
            attn_weights = attn_weights * gate

        out = torch.matmul(attn_weights, V)                          # (B,H,L,D_h)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# =============================================================================
# 5.  WRAT Block (Fixed τ)
# =============================================================================

class WRATBlock(nn.Module):
    """
    Wavelet Routing Attention Transformer Block.

    Processes LL and LH wavelet sub-bands independently and then fuses them
    via cross-scale attention (trend-guided detail routing):

        1. Intra-LL attention  : models long-range trend dependencies
        2. Intra-LH attention  : energy-sparse — suppresses noise
        3. Cross attention     : LL queries attend to LH values
                                 (trend draws on relevant detail)
        4. MLP residual blocks : per-band feed-forward processing

    Parameters
    ----------
    d_model      : int   — embedding dimension.
    num_heads    : int   — attention heads.
    sparsity_tau : float — fixed energy threshold τ for LH attention.
    dropout      : float — dropout probability in MLP layers.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sparsity_tau: float = 0.1,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Attention sub-modules
        self.intra_LL_attn = FrequencySparseAttention(d_model, num_heads)
        self.intra_LH_attn = FrequencySparseAttention(d_model, num_heads, threshold=sparsity_tau)
        self.cross_attn    = FrequencySparseAttention(d_model, num_heads)

        # Feed-forward networks (4× expansion, GELU)
        def _mlp(d):
            return nn.Sequential(
                nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d * 4, d), nn.Dropout(dropout),
            )

        self.mlp_LL = _mlp(d_model)
        self.mlp_LH = _mlp(d_model)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    def forward(self, LL: torch.Tensor, LH: torch.Tensor):
        """
        Parameters
        ----------
        LL, LH : (B, d_model, N)

        Returns
        -------
        LL_final, LH_final : (B, d_model, N)
        """
        # Transpose to (B, N, d_model) for attention
        LL_seq = LL.transpose(1, 2)
        LH_seq = LH.transpose(1, 2)

        # 1. Intra-scale self-attention
        LL_out    = self.intra_LL_attn(LL_seq, LL_seq, LL_seq)
        LH_out    = self.intra_LH_attn(LH_seq, LH_seq, LH_seq, energy_coeffs=LH_seq)

        # 2. Cross-scale attention: trend queries detail
        cross_out = self.cross_attn(LL_out, LH_out, LH_out)

        # 3. Residual fusion + LayerNorm
        LL_fused  = self.norm1(LL_seq + LL_out + cross_out)
        LH_fused  = self.norm2(LH_seq + LH_out)

        # 4. MLP with residual
        LL_final  = self.mlp_LL(LL_fused) + LL_fused
        LH_final  = self.mlp_LH(LH_fused) + LH_fused

        return LL_final.transpose(1, 2), LH_final.transpose(1, 2)


# =============================================================================
# 6.  Learnable-τ WRAT Block
# =============================================================================

class LearnableTauWRATBlock(nn.Module):
    """
    Wraps WRATBlock with a learnable sparsity threshold τ.

    τ is parameterised in logit-space to keep it bounded in (0, 1):
        raw_τ  ~ unconstrained scalar parameter
        τ      = sigmoid(raw_τ)

    The threshold is injected into the LH attention module before each
    forward pass, making it fully differentiable end-to-end.

    Parameters
    ----------
    d_model   : int   — embedding dimension.
    num_heads : int   — attention heads.
    tau_init  : float — initial value of τ (default 0.1).
    dropout   : float — dropout for MLP layers.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        tau_init: float = 0.1,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Logit initialisation: sigmoid(raw_tau) ≈ tau_init
        self.raw_tau = nn.Parameter(
            torch.tensor(math.log(tau_init / (1.0 - tau_init)))
        )
        self._block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init, dropout=dropout)

    @property
    def tau(self) -> float:
        """Current effective threshold value (read-only)."""
        return torch.sigmoid(self.raw_tau).item()

    def forward(self, LL: torch.Tensor, LH: torch.Tensor):
        # Inject current τ into attention before forward
        self._block.intra_LH_attn.threshold = torch.sigmoid(self.raw_tau)
        return self._block(LL, LH)


# =============================================================================
# 7.  PatchedWSA — Full Architecture
# =============================================================================

class PatchedWSA(nn.Module):
    """
    Patch-based Wavelet Sparse Attention (PatchWRAT) for multivariate
    time-series forecasting.

    Data flow (channel-independent — each variate processed separately):
        (B·C, 1, L)  →  RevIN  →  PatchEmbed  →  DWT  →  WRATBlock
                      →  Concat [LL_out, LH_out]  →  Head  →  RevIN⁻¹
                      →  (B·C, 1, H)

    Parameters
    ----------
    seq_len   : int   — input sequence length L.
    pred_len  : int   — forecast horizon H.
    d_model   : int   — patch embedding / attention dimension (default 64).
    num_heads : int   — attention heads (default 4).
    patch_len : int   — patch window size (default 16).
    stride    : int   — patch stride (default 8, 50% overlap).
    tau_init  : float — initial τ for LH sparsity gate (default 0.1).
    tau_type  : str   — 'learnable' (gradient-trained) or 'fixed'.
    dropout   : float — dropout probability (default 0.2).

    Returns (from forward)
    -----------------------
    preds       : (B·C, 1, H)   — denormalised forecast
    patch_recon : (B·C, d_model, N') — reconstructed patch embeddings (for loss)
    patches     : (B·C, d_model, N)  — raw patch embeddings (pre-DWT)
    LL          : (B·C, d_model, N//2) — low-freq wavelet coeffs (pre-attention)
    LH          : (B·C, d_model, N//2) — high-freq wavelet coeffs (pre-attention)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        num_heads: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        tau_init: float = 0.1,
        tau_type: str = "learnable",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.patch_len = patch_len
        self.stride    = stride

        # -- Sub-modules ---------------------------------------------------
        self.revin     = RevIN(num_features=1)
        self.patch_emb = PatchEmbedding(patch_len, stride, d_model)
        self.dwt       = LearnableDWT1D(channels=d_model)

        if tau_type == "learnable":
            self.wrat_block = LearnableTauWRATBlock(d_model, num_heads, tau_init, dropout)
        else:
            self.wrat_block = WRATBlock(d_model, num_heads, sparsity_tau=tau_init, dropout=dropout)

        # -- Forecast head dimension (computed analytically) ---------------
        num_patches   = (seq_len - patch_len) // stride + 1
        pad           = (4 - 2) // 2           # DWT filter_length=4
        dwt_out_len   = (num_patches + 2 * pad - 4) // 2 + 1
        flatten_dim   = dwt_out_len * d_model * 2   # LL + LH concatenated

        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(flatten_dim, pred_len),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, zero_lh: bool = False):
        """
        Parameters
        ----------
        x       : (B, 1, L) — single-variate input (channel-independent)
        zero_lh : bool      — ablation flag: zeros the LH branch if True
        """
        # 1. Normalise
        x_norm  = self.revin(x, mode="norm")

        # 2. Patch embedding
        patches = self.patch_emb(x_norm)           # (B, d_model, N)

        # 3. Wavelet decomposition
        LL, LH  = self.dwt(patches)                # each (B, d_model, N//2)
        if zero_lh:
            LH = torch.zeros_like(LH)              # ablation: no HF branch

        # 4. Dual-band attention
        LL_out, LH_out = self.wrat_block(LL, LH)

        # 5. Forecast head
        fused = torch.cat([LL_out, LH_out], dim=1) # (B, 2·d_model, N//2)
        preds = self.forecast_head(fused).unsqueeze(1)  # (B, 1, H)
        preds = self.revin(preds, mode="denorm")

        # 6. Auxiliary reconstruction (for loss)
        patch_recon = self.dwt.inverse(LL_out, LH_out)

        return preds, patch_recon, patches, LL, LH


# =============================================================================
# 8.  Dual-Head PatchWRAT Loss
# =============================================================================

class DualHeadPWSA_Loss(nn.Module):
    """
    Composite training objective for PatchWRAT:

        L_total = L_forecast
                + λ_recon  · L_reconstruction
                + λ_ortho  · L_orthogonality

    Components
    ----------
    L_forecast      : MSE between predicted and target sequences.
    L_reconstruction: MSE between raw patch embeddings and their inverse-DWT
                      reconstruction — encourages energy-preserving wavelet filters.
    L_orthogonality : |⟨h_flat, g_flat⟩| — penalises overlap between low-pass
                      and high-pass filters (promotes filter separation).

    Parameters
    ----------
    lambda_recon : float — weight for reconstruction loss (default 0.1).
    lambda_ortho : float — weight for orthogonality loss  (default 0.01).
    """

    def __init__(self, lambda_recon: float = 0.1, lambda_ortho: float = 0.01):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        patches_orig: torch.Tensor,
        patches_recon: torch.Tensor,
        dwt_layer: LearnableDWT1D,
    ):
        """
        Parameters
        ----------
        preds, targets   : (B, 1, H)
        patches_orig     : (B, d_model, N)   — raw patch embeddings
        patches_recon    : (B, d_model, N')  — inverse-DWT output
        dwt_layer        : reference to the LearnableDWT1D module

        Returns
        -------
        total_loss : scalar tensor (for backward)
        task_loss  : float         (for logging only)
        """
        # Forecast loss
        task_loss  = F.mse_loss(preds, targets)

        # Reconstruction loss (trim to matching length)
        min_len    = min(patches_orig.shape[-1], patches_recon.shape[-1])
        recon_loss = F.mse_loss(
            patches_recon[..., :min_len], patches_orig[..., :min_len]
        )

        # Filter orthogonality loss: penalise ⟨h, g⟩
        h_flat     = dwt_layer.h.view(dwt_layer.channels, -1)
        g_flat     = dwt_layer.g.view(dwt_layer.channels, -1)
        ortho_loss = (h_flat * g_flat).sum(dim=-1).abs().mean()

        total = (
            task_loss
            + self.lambda_recon * recon_loss
            + self.lambda_ortho * ortho_loss
        )
        return total, task_loss.item()
