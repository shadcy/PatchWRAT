"""
=============================================================================
PatchWRAT — Training Entry Point
=============================================================================
Usage
-----
    # ETTm1 dataset (default)
    python train.py

    # MPI Weather dataset
    python train.py --dataset weather --data_path path/to/mpi_roof_2017b.csv

    # Quick smoke-test (1 horizon, 3 epochs)
    python train.py --horizons 96 --epochs 3 --batch_size 16

Key Flags
---------
  --dataset     : 'ett' or 'weather'
  --data_path   : path to CSV file
  --seq_len     : look-back window  (default 512)
  --horizons    : space-separated list, e.g. 96 192 336 720
  --d_model     : embedding dim     (default 64)
  --num_heads   : attention heads   (default 4)
  --patch_len   : patch size        (default 16)
  --stride      : patch stride      (default 8)
  --tau_type    : 'learnable' or 'fixed'
  --tau_init    : initial τ value   (default 0.1)
  --dropout     : dropout rate      (default 0.2)
  --epochs      : max training epochs (default 30)
  --patience    : early-stopping patience (default 10)
  --batch_size  : mini-batch size   (default 32)
  --lr          : learning rate     (default 5e-4)
  --save_dir    : output directory for plots and checkpoints
=============================================================================
"""

import argparse
import os
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import PatchedWSA, DualHeadPWSA_Loss
from utils import (
    ETTDataset, WeatherDataset,
    EarlyStopping, evaluate, count_parameters,
    plot_learning_curves, plot_final_bar_charts,
    plot_learned_filters, plot_predictions,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="PatchWRAT Training Script")

    # Data
    p.add_argument("--dataset",   type=str, default="ett",
                   choices=["ett", "weather"])
    p.add_argument("--data_path", type=str, default=None,
                   help="Override default CSV path.")

    # Sequence
    p.add_argument("--seq_len",  type=int, default=512)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[96, 192, 336, 720])

    # Model
    p.add_argument("--d_model",   type=int,   default=64)
    p.add_argument("--num_heads", type=int,   default=4)
    p.add_argument("--patch_len", type=int,   default=16)
    p.add_argument("--stride",    type=int,   default=8)
    p.add_argument("--tau_type",  type=str,   default="learnable",
                   choices=["learnable", "fixed"])
    p.add_argument("--tau_init",  type=float, default=0.1)
    p.add_argument("--dropout",   type=float, default=0.2)

    # Training
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=5e-4)

    # Output
    p.add_argument("--save_dir",   type=str, default="outputs")

    return p.parse_args()


# =============================================================================
# Dataset factory
# =============================================================================

_DEFAULT_PATHS = {
    "ett":     "ETTm1.csv",
    "weather": "mpi_roof_2017b.csv",
}

def build_loaders(args, pred_len: int):
    """Return (train_loader, val_loader, test_loader)."""
    path    = args.data_path or _DEFAULT_PATHS[args.dataset]
    DS_cls  = ETTDataset if args.dataset == "ett" else WeatherDataset
    kwargs  = dict(seq_len=args.seq_len, pred_len=pred_len, file_path=path)

    train_ds = DS_cls(split="train", **kwargs)
    val_ds   = DS_cls(split="val",   **kwargs)
    test_ds  = DS_cls(split="test",  **kwargs)

    trl = DataLoader(train_ds, args.batch_size, shuffle=True,  drop_last=True)
    vll = DataLoader(val_ds,   args.batch_size, shuffle=False, drop_last=True)
    tel = DataLoader(test_ds,  args.batch_size, shuffle=False, drop_last=True)
    return trl, vll, tel


# =============================================================================
# Per-horizon training routine
# =============================================================================

def train_one_horizon(args, pred_len: int, device, criterion, save_dir: str):
    """
    Train all three ablation variants for a single forecast horizon and
    return their test metrics.

    Variants
    --------
    Learnable — full model with learnable τ
    Fixed     — full model with fixed τ = 0.1
    No_HF     — full model with LH branch zeroed (ablation)
    """
    trl, vll, tel = build_loaders(args, pred_len)

    # -- Build models -------------------------------------------------------
    model_cfg = dict(
        seq_len=args.seq_len, pred_len=pred_len,
        d_model=args.d_model, num_heads=args.num_heads,
        patch_len=args.patch_len, stride=args.stride,
        tau_init=args.tau_init, dropout=args.dropout,
    )
    m_lrn = PatchedWSA(**model_cfg, tau_type="learnable").to(device)
    m_fix = PatchedWSA(**model_cfg, tau_type="fixed").to(device)
    m_abl = PatchedWSA(**model_cfg, tau_type="learnable").to(device)

    if pred_len == args.horizons[0]:
        print(f"\nModel size: {count_parameters(m_lrn)}\n")

    # -- Optimisers & schedulers -------------------------------------------
    def _opt(m):
        return optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    def _sch(o):
        return optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs, eta_min=1e-6)

    opt_lrn, opt_fix, opt_abl = _opt(m_lrn), _opt(m_fix), _opt(m_abl)
    sch_lrn, sch_fix, sch_abl = _sch(opt_lrn), _sch(opt_fix), _sch(opt_abl)
    es_lrn  = EarlyStopping(args.patience)
    es_fix  = EarlyStopping(args.patience)
    es_abl  = EarlyStopping(args.patience)

    trk_train = {"Learnable": [], "Fixed": [], "No_HF": []}
    trk_val   = {"Learnable": [], "Fixed": [], "No_HF": []}

    # -- Training loop ------------------------------------------------------
    print(f" {'Ep':>3} | {'Val MSE (Lrn)':>15} | {'Val MSE (Fix)':>15} | {'Val MSE (No_HF)':>15}")
    print(f" {'-' * 65}")

    for epoch in range(1, args.epochs + 1):
        m_lrn.train(); m_fix.train(); m_abl.train()
        l_lrn, l_fix, l_abl = [], [], []

        for bx, by in trl:
            bx, by = bx.to(device), by.to(device)
            B, C, L = bx.shape
            bx_f, by_f = bx.reshape(B * C, 1, L), by.reshape(B * C, 1, -1)

            if not es_lrn.early_stop:
                opt_lrn.zero_grad()
                p, rec, orig, _, _ = m_lrn(bx_f)
                loss, tl = criterion(p, by_f, orig, rec, m_lrn.dwt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m_lrn.parameters(), 1.0)
                opt_lrn.step()
                l_lrn.append(tl)

            if not es_fix.early_stop:
                opt_fix.zero_grad()
                p, rec, orig, _, _ = m_fix(bx_f)
                loss, tl = criterion(p, by_f, orig, rec, m_fix.dwt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m_fix.parameters(), 1.0)
                opt_fix.step()
                l_fix.append(tl)

            if not es_abl.early_stop:
                opt_abl.zero_grad()
                p, rec, orig, _, _ = m_abl(bx_f, zero_lh=True)
                loss, tl = criterion(p, by_f, orig, rec, m_abl.dwt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m_abl.parameters(), 1.0)
                opt_abl.step()
                l_abl.append(tl)

        if l_lrn: trk_train["Learnable"].append(np.mean(l_lrn))
        if l_fix: trk_train["Fixed"].append(np.mean(l_fix))
        if l_abl: trk_train["No_HF"].append(np.mean(l_abl))

        v_lrn = evaluate(m_lrn, vll, device)["mse"]           if not es_lrn.early_stop else es_lrn.best_loss
        v_fix = evaluate(m_fix, vll, device)["mse"]           if not es_fix.early_stop else es_fix.best_loss
        v_abl = evaluate(m_abl, vll, device, zero_lh=True)["mse"] if not es_abl.early_stop else es_abl.best_loss

        if not es_lrn.early_stop: es_lrn(v_lrn, m_lrn); trk_val["Learnable"].append(v_lrn); sch_lrn.step()
        if not es_fix.early_stop: es_fix(v_fix, m_fix); trk_val["Fixed"].append(v_fix);     sch_fix.step()
        if not es_abl.early_stop: es_abl(v_abl, m_abl); trk_val["No_HF"].append(v_abl);    sch_abl.step()

        print(f" {epoch:>3} | {v_lrn:>15.5f} | {v_fix:>15.5f} | {v_abl:>15.5f}")

        if es_lrn.early_stop and es_fix.early_stop and es_abl.early_stop:
            print("  All variants converged (early stopping).")
            break

    # -- Restore best states & test ----------------------------------------
    es_lrn.restore(m_lrn); es_fix.restore(m_fix); es_abl.restore(m_abl)

    res_lrn = evaluate(m_lrn, tel, device)
    res_fix = evaluate(m_fix, tel, device)
    res_abl = evaluate(m_abl, tel, device, zero_lh=True)

    print(f"\n --- TEST RESULTS (H={pred_len}) ---")
    tau_str = f"  |  τ={m_lrn.wrat_block.tau:.4f}" if hasattr(m_lrn.wrat_block, "tau") else ""
    print(f"  Learnable : MSE={res_lrn['mse']:.4f}  MAE={res_lrn['mae']:.4f}"
          f"  DirAcc={res_lrn['dir_acc']:.2f}%{tau_str}")
    print(f"  Fixed τ   : MSE={res_fix['mse']:.4f}  MAE={res_fix['mae']:.4f}"
          f"  DirAcc={res_fix['dir_acc']:.2f}%")
    print(f"  No HF     : MSE={res_abl['mse']:.4f}  MAE={res_abl['mae']:.4f}"
          f"  DirAcc={res_abl['dir_acc']:.2f}%")

    # -- Per-horizon plots --------------------------------------------------
    plot_learning_curves(trk_train, trk_val, pred_len, save_dir)
    plot_learned_filters(m_lrn, pred_len, save_dir)
    plot_predictions(m_lrn, tel, device, pred_len, save_dir=save_dir)

    # -- Save best checkpoint ----------------------------------------------
    ckpt_path = os.path.join(save_dir, f"patchwrat_H{pred_len}.pt")
    torch.save({"model_state": es_lrn.best_state, "args": vars(args)}, ckpt_path)
    print(f"  [ckpt] saved → {ckpt_path}")

    return {
        "Learnable": res_lrn,
        "Fixed":     res_fix,
        "No_HF":     res_abl,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    criterion = DualHeadPWSA_Loss(lambda_recon=0.1, lambda_ortho=0.01)

    print("=" * 70)
    print(" PatchWRAT — Patch-based Wavelet Routing Attention Transformer")
    print("=" * 70)
    print(f"  Device   : {device}")
    print(f"  Dataset  : {args.dataset.upper()}")
    print(f"  Horizons : {args.horizons}")
    print(f"  d_model  : {args.d_model}  |  heads: {args.num_heads}"
          f"  |  patch: {args.patch_len}  |  stride: {args.stride}")
    print(f"  τ type   : {args.tau_type}  |  LR: {args.lr}")
    print()

    final_results = {}

    for pred_len in args.horizons:
        print(f"\n{'=' * 70}")
        print(f"  HORIZON = {pred_len}")
        print(f"{'=' * 70}")
        final_results[pred_len] = train_one_horizon(
            args, pred_len, device, criterion, args.save_dir
        )

    # -- Summary charts -------------------------------------------------------
    plot_final_bar_charts(final_results, args.horizons, args.save_dir)

    # -- Summary table --------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'H':>5} | {'Variant':>12} | {'MSE':>8} | {'MAE':>8} | {'DirAcc':>9}")
    print(f"  {'-' * 55}")
    for h in args.horizons:
        for v in ["Learnable", "Fixed", "No_HF"]:
            m = final_results[h][v]
            print(f"  {h:>5} | {v:>12} | {m['mse']:>8.4f} | {m['mae']:>8.4f} | {m['dir_acc']:>8.2f}%")

    print(f"\n  All outputs saved to: {os.path.abspath(args.save_dir)}")
    print("  PatchWRAT run complete.")


if __name__ == "__main__":
    main()
