import torch
import numpy as np

def evaluate(model_fn, loader, device, scaler=None):
    """
    model_fn : callable bx → preds
    scaler   : if provided, inverse-transforms before computing MAPE/R²
               (keeps MAE/MSE on normalised scale for fair comparison)
    Returns  : dict of all metrics
    """
    all_preds, all_targets = [], []

    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            preds = model_fn(bx)
            L = min(preds.shape[-1], by.shape[-1])
            all_preds.append(preds[..., :L].cpu())
            all_targets.append(by[..., :L].cpu())

    p = torch.cat(all_preds,   dim=0).flatten().numpy()   # (N,)
    t = torch.cat(all_targets, dim=0).flatten().numpy()   # (N,)

    # ── Core regression metrics (normalised scale) ──────────────
    err   = p - t
    mae   = np.abs(err).mean()
    mse   = (err ** 2).mean()
    rmse  = mse ** 0.5
    maxe  = np.abs(err).max()

    # ── MAPE / sMAPE (avoid div-by-zero with small epsilon) ──────
    eps    = 1e-8
    mape   = (np.abs(err) / (np.abs(t) + eps)).mean() * 100
    smape  = (2 * np.abs(err) / (np.abs(p) + np.abs(t) + eps)).mean() * 100

    # ── R² ───────────────────────────────────────────────────────
    ss_res = (err ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2     = 1 - ss_res / (ss_tot + eps)

    # ── Pearson correlation ───────────────────────────────────────
    corr = np.corrcoef(p, t)[0, 1] if len(p) > 1 else 0.0

    # ── Directional accuracy ──────────────────────────────────────
    # Compare sign of consecutive differences
    dp = np.diff(p)
    dt = np.diff(t)
    dir_acc = (np.sign(dp) == np.sign(dt)).mean() * 100

    return dict(
        mae=mae, mse=mse, rmse=rmse, maxe=maxe,
        mape=mape, smape=smape,
        r2=r2, corr=corr,
        dir_acc=dir_acc
    )
