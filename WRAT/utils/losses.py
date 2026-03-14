import torch
import torch.nn as nn

class WaveletTransformerLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_ortho=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho
        self.mse = nn.MSELoss()

    def forward(self, preds, targets, x_original, x_recon, dwt_layer):
        # 1. Task Loss (e.g., forecasting)
        loss_task = self.mse(preds, targets)

        # 2. Reconstruction Loss
        # Ensure x_recon matches x_original length
        min_len = min(x_original.shape[-1], x_recon.shape[-1])
        loss_recon = self.mse(x_original[..., :min_len], x_recon[..., :min_len])

        # 3. Orthogonality Loss for filters (h * h^T = I)
        h = dwt_layer.h.view(dwt_layer.h.shape[0], -1)
        identity = torch.eye(h.shape[0], device=h.device)
        loss_ortho = self.mse(torch.mm(h, h.t()), identity)

        # High-pass zero mean penalty
        g_mean = dwt_layer.g.mean(dim=-1).abs().sum()

        total_loss = loss_task + (self.lambda_recon * loss_recon) + (self.lambda_ortho * loss_ortho) + (0.01 * g_mean)
        return total_loss, loss_task, loss_recon, loss_ortho
