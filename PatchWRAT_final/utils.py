"""
=============================================================================
PatchWRAT — Utilities
=============================================================================
Modules
-------
  Datasets
    ETTDataset      — ETTm1 / ETTh1 benchmark (standard splits)
    WeatherDataset  — MPI Roof 2017b weather sensor dataset

  Training Helpers
    EarlyStopping   — patience-based checkpoint with best-state restore
    evaluate        — full metric suite (MSE, MAE, RMSE, MAPE, R², DirAcc)
    count_parameters — human-readable parameter count

  Visualisation
    plot_learning_curves  — train/val MSE per ablation variant
    plot_final_bar_charts — grouped bar charts for MSE and DirAcc
    plot_learned_filters  — impulse response + magnitude spectrum of h / g
    plot_predictions      — overlay of ground-truth vs. forecast sample
=============================================================================
"""

import os
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

__all__ = [
    # datasets
    "ETTDataset",
    "WeatherDataset",
    # training helpers
    "EarlyStopping",
    "evaluate",
    "count_parameters",
    # visualisation
    "plot_learning_curves",
    "plot_final_bar_charts",
    "plot_learned_filters",
    "plot_predictions",
]

# ---------------------------------------------------------------------------
# Colour palette shared across all plots
# ---------------------------------------------------------------------------
_PALETTE = {
    "Learnable": "#2563eb",   # blue
    "Fixed":     "#16a34a",   # green
    "No_HF":     "#dc2626",   # red
}
_LABEL_MAP = {
    "Learnable": "Learnable τ",
    "Fixed":     "Fixed τ=0.1",
    "No_HF":     "No HF Branch",
}


# =============================================================================
# 1.  Datasets
# =============================================================================

class ETTDataset(Dataset):
    """
    Electricity Transformer Temperature dataset (ETTm1 / ETTh1).

    Uses the standard chronological split:
        Train : first 12 months  (12·30·24·4 time-steps for ETTm1)
        Val   : next  4 months
        Test  : remainder

    Parameters
    ----------
    seq_len   : int  — look-back window length.
    pred_len  : int  — forecast horizon.
    split     : str  — one of 'train', 'val', 'test'.
    file_path : str  — path to the CSV file (must have a date column as col 0).

    Item shape
    ----------
    x : (C, seq_len)   — normalised input
    y : (C, pred_len)  — normalised target
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        split: str = "train",
        file_path: str = "ETTm1.csv",
    ):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        df   = pd.read_csv(file_path)
        data = df.iloc[:, 1:].values.astype(np.float32)   # drop date column

        # Standard ETT chronological splits
        train_end = 12 * 30 * 24 * 4
        val_end   = train_end + 4 * 30 * 24 * 4

        split_map = {
            "train": data[:train_end],
            "val":   data[train_end:val_end],
            "test":  data[val_end:],
        }
        raw = split_map[split]

        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx: int):
        x = self.data[idx            : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x.t(), y.t()   # (C, L), (C, H)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.scaler.scale_ + self.scaler.mean_


class WeatherDataset(Dataset):
    """
    MPI Roof 2017b multi-sensor weather dataset.

    Dynamic 70 / 10 / 20 % split (adapts to any dataset length).
    Non-numeric columns (e.g. 'Date Time') are automatically dropped.

    Parameters
    ----------
    seq_len   : int  — look-back window length.
    pred_len  : int  — forecast horizon.
    split     : str  — one of 'train', 'val', 'test'.
    file_path : str  — path to the CSV file (latin-1 encoded).

    Item shape
    ----------
    x : (C, seq_len)
    y : (C, pred_len)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        split: str = "train",
        file_path: str = "mpi_roof_2017b.csv",
    ):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        df   = pd.read_csv(file_path, encoding="latin1")
        data = df.select_dtypes(include=[np.number]).dropna().values.astype(np.float32)

        n         = len(data)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.80)

        split_map = {
            "train": data[:train_end],
            "val":   data[train_end:val_end],
            "test":  data[val_end:],
        }
        raw = split_map[split]

        self.scaler = StandardScaler()
        self.scaler.fit(data[:train_end])
        self.data = torch.tensor(self.scaler.transform(raw), dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx: int):
        x = self.data[idx            : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x.t(), y.t()

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.scaler.scale_ + self.scaler.mean_


# =============================================================================
# 2.  Training Helpers
# =============================================================================

class EarlyStopping:
    """
    Monitors validation loss and stops training when no improvement is seen
    for `patience` consecutive epochs. Saves the best model state in CPU
    memory so it can be restored after the run.

    Usage
    -----
    >>> es = EarlyStopping(patience=10)
    >>> for epoch in ...:
    ...     es(val_loss, model)
    ...     if es.early_stop:
    ...         break
    >>> model.load_state_dict(es.best_state)

    Parameters
    ----------
    patience : int   — epochs without improvement before stopping.
    delta    : float — minimum improvement to reset the counter (default 0).
    """

    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_loss  = None
        self.best_state: Optional[dict] = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss  = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model: torch.nn.Module) -> None:
        """Load the best saved state back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def evaluate(model, loader, device, zero_lh: bool = False) -> Dict[str, float]:
    """
    Run a full evaluation pass and return a comprehensive metric dictionary.

    Metrics (all computed on normalised predictions for fair cross-model
    comparison; directional accuracy is scale-invariant by definition):

        MSE    — Mean Squared Error
        MAE    — Mean Absolute Error
        RMSE   — Root Mean Squared Error
        MAPE   — Mean Absolute Percentage Error (%)
        sMAPE  — Symmetric MAPE (%)
        R²     — Coefficient of determination
        Corr   — Pearson correlation coefficient
        DirAcc — Directional accuracy (%) — sign agreement on consecutive diffs

    Parameters
    ----------
    model   : PatchedWSA instance
    loader  : DataLoader yielding (x, y) batches  x:(B,C,L)  y:(B,C,H)
    device  : torch.device
    zero_lh : bool — ablation flag passed to model.forward

    Returns
    -------
    dict[str, float]
    """
    model.eval()
    all_p, all_t = [], []

    with torch.no_grad():
        for bx, by in loader:
            bx, by  = bx.to(device), by.to(device)
            B, C, L = bx.shape
            preds, *_ = model(bx.reshape(B * C, 1, L), zero_lh=zero_lh)
            all_p.append(preds.reshape(B, C, -1).cpu())
            all_t.append(by.cpu())

    p = torch.cat(all_p, 0).flatten().numpy()
    t = torch.cat(all_t, 0).flatten().numpy()

    err   = p - t
    eps   = 1e-8
    mae   = float(np.abs(err).mean())
    mse   = float((err ** 2).mean())
    rmse  = float(mse ** 0.5)
    mape  = float((np.abs(err) / (np.abs(t) + eps)).mean() * 100)
    smape = float((2 * np.abs(err) / (np.abs(p) + np.abs(t) + eps)).mean() * 100)

    ss_res = (err ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2     = float(1 - ss_res / (ss_tot + eps))
    corr   = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 else 0.0

    dp, dt  = np.diff(p), np.diff(t)
    dir_acc = float((np.sign(dp) == np.sign(dt)).mean() * 100)

    model.train()
    return dict(mae=mae, mse=mse, rmse=rmse, mape=mape, smape=smape,
                r2=r2, corr=corr, dir_acc=dir_acc)


def count_parameters(model: torch.nn.Module) -> str:
    """
    Return a human-readable string with total and trainable parameter counts.

    Example: "Total: 1.23 M  |  Trainable: 1.23 M"
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _fmt(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f} M"
        if n >= 1_000:
            return f"{n / 1_000:.1f} K"
        return str(n)

    return f"Total: {_fmt(total)}  |  Trainable: {_fmt(trainable)}"


# =============================================================================
# 3.  Visualisation
# =============================================================================

def plot_learning_curves(
    train_dict: Dict[str, List[float]],
    val_dict:   Dict[str, List[float]],
    horizon: int,
    save_dir: str = ".",
) -> None:
    """
    Plot training and validation MSE curves for all ablation variants.

    Parameters
    ----------
    train_dict : {'Learnable': [...], 'Fixed': [...], 'No_HF': [...]}
    val_dict   : same structure
    horizon    : prediction horizon (used in title and filename)
    save_dir   : directory to save the PNG
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, color in _PALETTE.items():
        if key not in train_dict or not train_dict[key]:
            continue
        ax.plot(range(1, len(val_dict[key]) + 1),   val_dict[key],
                label=f"Val MSE ({_LABEL_MAP[key]})",   color=color, linewidth=2)
        ax.plot(range(1, len(train_dict[key]) + 1), train_dict[key],
                label=f"Train MSE ({_LABEL_MAP[key]})", color=color,
                linestyle="--", alpha=0.5)

    ax.set_title(f"PatchWRAT Ablation — Learning Curves (H={horizon})", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, f"learning_curves_H{horizon}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {path}")


def plot_final_bar_charts(
    final_results: Dict[int, Dict[str, Dict]],
    horizons: List[int],
    save_dir: str = ".",
) -> None:
    """
    Grouped bar charts for MSE and Directional Accuracy across horizons.

    Parameters
    ----------
    final_results : {horizon: {'Learnable': metrics_dict, ...}}
    horizons      : list of forecast horizons
    save_dir      : output directory
    """
    variants = ["Learnable", "Fixed", "No_HF"]
    labels   = [_LABEL_MAP[v] for v in variants]
    colors   = [_PALETTE[v]   for v in variants]
    x, w     = np.arange(len(horizons)), 0.25

    # --- MSE chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (v, c) in enumerate(zip(variants, colors)):
        mses = [final_results[h][v]["mse"] for h in horizons]
        ax.bar(x + (i - 1) * w, mses, w, label=labels[i], color=c, edgecolor="black")
    ax.set_ylabel("Test MSE  (lower is better)")
    ax.set_title("PatchWRAT — MSE by Forecast Horizon", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in horizons])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(save_dir, "final_ablation_mse.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {path}")

    # --- Directional Accuracy chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (v, c) in enumerate(zip(variants, colors)):
        accs = [final_results[h][v]["dir_acc"] for h in horizons]
        ax.bar(x + (i - 1) * w, accs, w, label=labels[i], color=c, edgecolor="black")
    ax.set_ylabel("Directional Accuracy %  (higher is better)")
    ax.set_title("PatchWRAT — Directional Accuracy by Horizon", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in horizons])
    ax.set_ylim(45, 65)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(save_dir, "final_ablation_diracc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {path}")


def plot_learned_filters(model, horizon: int, save_dir: str = ".") -> None:
    """
    Visualise the learnable DWT filter weights (h, g) as:
        Left  — Impulse response (filter tap amplitudes)
        Right — Magnitude spectrum (FFT of each filter)

    Parameters
    ----------
    model    : PatchedWSA instance (must have .dwt attribute)
    horizon  : used in title and filename
    save_dir : output directory
    """
    h_w = model.dwt.h.detach().cpu().numpy().mean(axis=(0, 1))  # (filter_len,)
    g_w = model.dwt.g.detach().cpu().numpy().mean(axis=(0, 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Impulse response
    ax = axes[0]
    ax.plot(h_w, marker="o", color=_PALETTE["Learnable"], linewidth=2, label="h  (low-pass / trend)")
    ax.plot(g_w, marker="x", color=_PALETTE["No_HF"],     linewidth=2, linestyle="--", label="g  (high-pass / detail)")
    ax.set_title(f"Learned Filter — Impulse Response (H={horizon})")
    ax.set_xlabel("Filter Tap Index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(alpha=0.3)

    # Magnitude spectrum
    ax = axes[1]
    freqs = np.linspace(0, np.pi, 32)
    ax.plot(freqs, np.abs(np.fft.fft(h_w, n=64))[:32],
            color=_PALETTE["Learnable"], label="h  (low-pass)")
    ax.plot(freqs, np.abs(np.fft.fft(g_w, n=64))[:32],
            color=_PALETTE["No_HF"],     linestyle="--", label="g  (high-pass)")
    ax.set_title("Filter — Magnitude Spectrum")
    ax.set_xlabel("Normalised Frequency (rad / sample)")
    ax.set_ylabel("|H(ω)|")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"PatchWRAT Learned DWT Filters  (H={horizon})", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, f"learned_filters_H{horizon}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path}")


def plot_predictions(
    model,
    loader,
    device,
    horizon: int,
    n_samples: int = 3,
    save_dir: str = ".",
) -> None:
    """
    Plot ground-truth vs. predicted sequences for a few random samples.

    Parameters
    ----------
    model     : PatchedWSA instance
    loader    : test DataLoader
    device    : torch.device
    horizon   : forecast horizon (for filename)
    n_samples : number of samples to plot
    save_dir  : output directory
    """
    model.eval()
    preds_list, truth_list = [], []

    with torch.no_grad():
        for bx, by in loader:
            bx, by  = bx.to(device), by.to(device)
            B, C, L = bx.shape
            preds, *_ = model(bx.reshape(B * C, 1, L))
            preds_list.append(preds.reshape(B, C, -1).cpu())
            truth_list.append(by.cpu())
            if len(preds_list) * bx.shape[0] >= n_samples:
                break

    preds_all = torch.cat(preds_list, 0)   # (N, C, H)
    truth_all = torch.cat(truth_list, 0)

    n_show = min(n_samples, preds_all.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 3 * n_show), squeeze=False)

    for i in range(n_show):
        ax = axes[i][0]
        # Use first variate (channel 0)
        ax.plot(truth_all[i, 0].numpy(), label="Ground Truth",
                color="#1e293b", linewidth=1.5)
        ax.plot(preds_all[i, 0].numpy(), label="Prediction",
                color=_PALETTE["Learnable"], linewidth=1.5, linestyle="--")
        ax.set_title(f"Sample {i+1}  —  H={horizon}")
        ax.set_xlabel("Forecast Step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"PatchWRAT Forecasts  (H={horizon})", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, f"predictions_H{horizon}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    model.train()
    print(f"  [plot] saved → {path}")
