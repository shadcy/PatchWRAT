import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1), :]

class VanillaTransformerBaseline(nn.Module):
    def __init__(self, in_channels=1, d_model=16, num_heads=4, seq_len=128):
        super().__init__()
        # Standard Tokenization (Strided Conv to match DWT's sequence halving)
        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=2, stride=2)

        self.pos_encoder = PositionalEncoding(d_model)

        # Standard Dense Self-Attention Block
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            batch_first=True, activation='gelu'
        )
        # Using 2 layers to roughly match the parameter count of the WRAT block
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        # Standard Linear Projection Head mapped back to original sequence length
        self.output_head = nn.Sequential(
            nn.ConvTranspose1d(d_model, in_channels, kernel_size=2, stride=2),
            nn.Conv1d(in_channels, in_channels, kernel_size=1) # Amplitude matching
        )

    def forward(self, x):
        # 1. Embed and halve sequence: (B, C, L) -> (B, D, L/2)
        x_emb = self.patch_embed(x)

        # 2. Transformer expects (B, L, D) for batch_first=True
        x_seq = x_emb.transpose(1, 2)
        x_seq = self.pos_encoder(x_seq)

        # 3. Dense Attention (NO sparsity)
        attn_out = self.transformer_encoder(x_seq)

        # 4. Project back to (B, C, L)
        attn_out = attn_out.transpose(1, 2)
        preds = self.output_head(attn_out)

        return preds
