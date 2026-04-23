import sys
import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_pytorch')))

# Import your WRAT Lightning module
from wrat_lightning import WRATLightningModule

# ==========================================
# 1. PyTorch Dataset for Time Series
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=96, pred_len=96):
        """
        Args:
            data (np.array): Scaled multivariate time series data (L, C).
            seq_len (int): Lookback window.
            pred_len (int): Prediction horizon.
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # Ensure we have enough data for a full input sequence + target sequence
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # Input: [index : index + seq_len]
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # Target: [index + seq_len : index + seq_len + pred_len]
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # Extract and transpose from (Length, Channels) to (Channels, Length) for 1D Convs
        seq_x = self.data[s_begin:s_end].T
        seq_y = self.data[r_begin:r_end].T

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)


# ==========================================
# 2. PyTorch Lightning DataModule
# ==========================================
class ETTh1DataModule(pl.LightningDataModule):
    def __init__(self, seq_len=96, pred_len=96, batch_size=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.data_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        self.file_path = "ETTh1.csv"
        self.scaler = StandardScaler()

    def prepare_data(self):
        """Downloads the ETTh1 dataset if it doesn't exist."""
        if not os.path.exists(self.file_path):
            print("Downloading ETTh1 dataset...")
            urllib.request.urlretrieve(self.data_url, self.file_path)
            print("Download complete.")

    def setup(self, stage=None):
        """Loads CSV, splits into Train/Val/Test, and scales the data."""
        df_raw = pd.read_csv(self.file_path)
        
        # Drop the 'date' column, keep only the 7 feature columns
        df_data = df_raw.drop(columns=['date'])
        
        # Standard Informer split for ETTh1 (8640, 2880, 2880)
        num_train = 12 * 30 * 24
        num_val   = 4 * 30 * 24
        num_test  = 4 * 30 * 24

        # Notice the overlap subtraction for val and test
        train_data = df_data.iloc[:num_train].values
        val_data   = df_data.iloc[num_train - self.seq_len : num_train + num_val].values
        test_data  = df_data.iloc[num_train + num_val - self.seq_len : num_train + num_val + num_test].values

        # Fit scaler ONLY on train data to prevent data leakage
        self.scaler.fit(train_data)
        
        train_scaled = self.scaler.transform(train_data)
        val_scaled   = self.scaler.transform(val_data)
        test_scaled  = self.scaler.transform(test_data)

        if stage == 'fit' or stage is None:
            self.train_dataset = TimeSeriesDataset(train_scaled, self.seq_len, self.pred_len)
            self.val_dataset   = TimeSeriesDataset(val_scaled, self.seq_len, self.pred_len)
            
        if stage == 'test' or stage is None:
            self.test_dataset  = TimeSeriesDataset(test_scaled, self.seq_len, self.pred_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


# ==========================================
# 3. Execution Script
# ==========================================
if __name__ == "__main__":
    # Hyperparameters based on SOTA standards
    SEQ_LEN = 96
    PRED_LEN = 96
    BATCH_SIZE = 32
    IN_CHANNELS = 7 # ETTh1 has 7 variables

    # 1. Initialize Data
    datamodule = ETTh1DataModule(seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)

    # 2. Initialize Model (Using your WRAT wrapper)
    pl_model = WRATLightningModule(
        in_channels=IN_CHANNELS, 
        d_model=64, 
        num_heads=4, 
        tau_type='adaptive',
        learning_rate=1e-4,
        lambda_recon=5.0,
        lambda_ortho=1.0
    )

    print(f"Dataset ready. Set up to predict horizon {PRED_LEN} from lookback {SEQ_LEN}.")
    # 2. Save the absolute best weights
    checkpoint = ModelCheckpoint(
        monitor="val_loss", 
        dirpath="checkpoints", 
        filename="wrat-etth1-best", 
        save_top_k=1, 
        mode="min"
    )

    # 3. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=100, 
        accelerator="auto", 
        devices="auto",
        callbacks=[checkpoint],
        enable_progress_bar=True
    )

    # 4. Train and Test!
    print("Starting Training...")
    trainer.fit(pl_model, datamodule=datamodule)
    
    print("Starting Testing on Best Checkpoint...")
    trainer.test(ckpt_path="best", datamodule=datamodule)
