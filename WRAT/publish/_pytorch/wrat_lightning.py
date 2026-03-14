import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your core modules here
from wrat import WRATModel, WaveletTransformerLoss

class WRATLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the Wavelet Residual Attention Transformer.
    Handles training loops, multi-part loss computation, and metric logging.
    """
    def __init__(
        self,
        in_channels=1, 
        d_model=64, 
        num_heads=4, 
        tau_type='learnable', 
        tau_init=0.1,
        learning_rate=1e-3,
        lambda_recon=1.0,
        lambda_ortho=0.1,
        weight_decay=1e-4
    ):
        super().__init__()
        # Saves arguments to self.hparams for automatic checkpoint logging
        self.save_hyperparameters() 
        
        # Initialize Core Architecture
        self.model = WRATModel(
            in_channels=self.hparams.in_channels,
            d_model=self.hparams.d_model,
            num_heads=self.hparams.num_heads,
            tau_type=self.hparams.tau_type,
            tau_init=self.hparams.tau_init
        )
        
        # Initialize Custom Loss
        self.loss_fn = WaveletTransformerLoss(
            lambda_recon=self.hparams.lambda_recon,
            lambda_ortho=self.hparams.lambda_ortho
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, stage="train"):
        """Computes forward pass and multi-part loss for both train and val."""
        # Assuming batch yields (inputs, targets)
        # Inputs shape: (B, C, L)
        x, y = batch
        
        # Forward pass
        preds = self(x)
        
        # We need the intermediate DWT outputs for the custom loss
        LL, LH = self.model.dwt(x)
        recon = self.model.dwt.inverse(LL, LH)
        
        # Calculate complex loss
        total_loss, task_loss, recon_loss, ortho_loss = self.loss_fn(
            preds=preds, 
            targets=y, 
            x_orig=x, 
            x_recon=recon, 
            dwt_layer=self.model.dwt
        )
        
        # Log all loss components
        self.log(f"{stage}_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_task_mse", task_loss, sync_dist=True)
        self.log(f"{stage}_recon_mse", recon_loss, sync_dist=True)
        self.log(f"{stage}_ortho_penalty", ortho_loss, sync_dist=True)
        
        # If using learnable tau, log its value to monitor sparsity over time
        if self.hparams.tau_type == 'learnable':
            self.log(f"{stage}_tau_value", self.model.block.tau, sync_dist=True)
            
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")
        
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        """AdamW with Cosine Annealing is generally best for Transformers."""
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        # Decays learning rate down to 10% of the initial LR
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100, 
            eta_min=self.hparams.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }