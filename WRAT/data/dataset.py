import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import urllib.request # Import urllib.request for downloading

class ETTm1Dataset(Dataset):
    def __init__(self, seq_len=128, split='train', file_path='ETTm1.csv', target_col='OT'):
        self.seq_len = seq_len

        # Download Dataset if not exists
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
        if not os.path.exists(file_path):
            print("Downloading ETTm1 dataset...")
            urllib.request.urlretrieve(url, file_path)
            print("Download complete!")

        df = pd.read_csv(file_path)
        data = df[target_col].values.reshape(-1, 1)

        # Standard ETTm1 splits (15-min intervals)
        train_end  = 12 * 30 * 24 * 4        # 17,280
        val_end    = train_end + 4 * 30 * 24 * 4   # 23,040
        # test = val_end : end

        if split == 'train':
            raw = data[:train_end]
        elif split == 'val':
            raw = data[train_end:val_end]
        else:
            raw = data[val_end:]

        # Fit scaler ONLY on train to avoid leakage
        self.scaler = StandardScaler()
        train_raw = data[:train_end]
        self.scaler.fit(train_raw)
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx          : idx + self.seq_len]
        y = self.data[idx + 1      : idx + self.seq_len + 1]
        return x.t(), y.t()   # → (1, seq_len)
