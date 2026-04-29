# =============================================================================
# pwsa_plots.py  —  P-WSA++ Comprehensive Diagnostic Plots
# =============================================================================
#
# FOLDER STRUCTURE PRODUCED
# -------------------------
# plots/
#   learning_curves/
#     H{h}_curves.png              train/val MSE + overfit gap + tau per epoch
#     all_horizons_summary.png     all 4 horizons on one chart
#
#   predictions/H{h}/
#     variate_{i}_{name}_sample{j}.png   pred vs GT per variate per sample
#     all_variates_grid.png              7-panel grid, first test sample
#     residuals.png                      histogram + Q-Q + ACF of residuals
#     horizon_error_profile.png          MAE at each forecast step
#
#   filters/
#     H{h}_dwt_filters.png         learned h/g taps vs Haar init
#     H{h}_filter_spectrum.png     frequency magnitude response
#     H{h}_filter_evolution.png    deviation from Haar init per tap
#     H{h}_filter_stats.png        L2 norm per channel (heatmap)
#     all_horizons_filter_norms.png h/g norm comparison across horizons
#
#   variate_graph/
#     H{h}_adjacency.png           learned 7x7 GraphMixer matrix + deviation
#     H{h}_variate_attn.png        VariateAttention gate value
#     correlation_vs_learned.png   data correlation vs learned mixing
#
#   attention/
#     H{h}_wrat_attn.png           LL attention map + entropy profile
#     H{h}_tau_bar.png             learned sparsity threshold tau
#
#   benchmarks/
#     final_mse_bar.png            test MSE all horizons
#     final_mae_bar.png            test MAE all horizons
#     sota_comparison.png          your model vs published baselines at H=96
#     all_horizons_summary.png     MSE + MAE + DirAcc line plots
#     H{h}_spectral.png            PSD of input / LL / prediction / target
#
# USAGE
# -----
#   from pwsa_plots import PlotManager
#   pm = PlotManager('plots')
#   pm.plot_learning_curves(train_curve, val_curve, pred_len, tau_curve)
#   pm.plot_predictions(model, test_loader, pred_len, device)
#   pm.plot_filters(model, pred_len)
#   pm.plot_variate_graph(model, pred_len, test_loader, device)
#   pm.plot_attention(model, test_loader, pred_len, device)
#   pm.plot_final_benchmarks(final_results)
#   pm.plot_spectral_analysis(model, test_loader, pred_len, device)
# =============================================================================

import os, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Style constants ──────────────────────────────────────────────────────────
C_BLUE   = '#2563eb'
C_GREEN  = '#16a34a'
C_PURPLE = '#7c3aed'
C_AMBER  = '#f59e0b'
C_RED    = '#dc2626'
C_GRAY   = '#6b7280'
C_TEAL   = '#0f6e56'

VARIATE_COLORS = [C_BLUE, C_GREEN, C_PURPLE, C_AMBER, C_RED, C_TEAL, '#d85a30']
SOTA_MSE_H96   = {'iTransformer': 0.454, 'PatchTST': 0.387,
                  'TimesNet': 0.338, 'DLinear': 0.299}

plt.rcParams.update({
    'font.family':          'DejaVu Sans',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'grid.alpha':           0.25,
    'grid.linewidth':       0.5,
    'savefig.dpi':          150,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {path}")


class PlotManager:

    def __init__(self, base_dir='plots',
                 variate_names=None):
        self.base   = base_dir
        self.vnames = (variate_names or
                       ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])
        for sub in ['learning_curves', 'filters',
                    'variate_graph', 'attention', 'benchmarks']:
            os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    def _p(self, *parts):
        return os.path.join(self.base, *parts)

    # =========================================================================
    # 1. LEARNING CURVES
    # =========================================================================

    def plot_learning_curves(self, train_curve, val_curve, pred_len,
                              tau_curve=None):
        epochs    = list(range(1, len(train_curve) + 1))
        best_ep   = int(np.argmin(val_curve)) + 1
        best_val  = float(min(val_curve))
        ncols     = 3 if (tau_curve and len(tau_curve) > 0) else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

        # Panel A: MSE curves
        ax = axes[0]
        ax.plot(epochs, train_curve, color=C_BLUE, linestyle='--',
                alpha=0.6, linewidth=1.5, label='train')
        ax.plot(epochs, val_curve,   color=C_BLUE, linewidth=2,   label='val')
        ax.axvline(best_ep, color=C_AMBER, linestyle=':', linewidth=1.2,
                   label=f'best ep{best_ep}')
        ax.scatter([best_ep], [best_val], color=C_AMBER, s=55, zorder=5)
        ax.annotate(f'{best_val:.4f}', xy=(best_ep, best_val),
                    xytext=(best_ep + 0.5, best_val * 1.03),
                    fontsize=8, color=C_AMBER)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
        ax.set_title(f'Train / Val MSE  H={pred_len}')
        ax.legend(fontsize=8)

        # Panel B: Overfit gap
        ax = axes[1]
        gap = [v - t for v, t in zip(val_curve, train_curve)]
        ax.fill_between(epochs, 0, gap, alpha=0.25, color=C_RED)
        ax.plot(epochs, gap, color=C_RED, linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Val − Train')
        ax.set_title('Overfit gap')

        # Panel C: Tau evolution
        if ncols == 3 and tau_curve:
            ax = axes[2]
            ax.plot(epochs[:len(tau_curve)], tau_curve[:len(epochs)],
                    color=C_PURPLE, linewidth=2)
            ax.axhline(0.1, color=C_GRAY, linestyle='--', linewidth=0.8,
                       label='init τ=0.1')
            ax.set_xlabel('Epoch'); ax.set_ylabel('τ')
            ax.set_title('Learnable sparsity τ')
            ax.set_ylim(0, 0.25); ax.legend(fontsize=8)

        fig.suptitle(f'P-WSA++ Learning Diagnostics  H={pred_len}',
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        _save(fig, self._p('learning_curves', f'H{pred_len}_curves.png'))

    def plot_all_horizons_summary(self, all_train, all_val, horizons):
        palette = [C_BLUE, C_GREEN, C_PURPLE, C_AMBER]
        fig, ax  = plt.subplots(figsize=(12, 5))
        for h, col in zip(horizons, palette):
            vc = all_val.get(h, [])
            tc = all_train.get(h, [])
            if vc:
                ax.plot(range(1, len(vc)+1), vc,  color=col,
                        linewidth=2, label=f'H={h} val')
            if tc:
                ax.plot(range(1, len(tc)+1), tc,  color=col,
                        linewidth=1, linestyle='--', alpha=0.35)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Val MSE')
        ax.set_title('All horizons — validation MSE')
        ax.legend(fontsize=9)
        _save(fig, self._p('learning_curves', 'all_horizons_summary.png'))

    # =========================================================================
    # 2. PREDICTIONS
    # =========================================================================

    def plot_predictions(self, model, loader, pred_len, device,
                         n_samples=3, n_variates=None):
        model.eval()
        folder = self._p('predictions', f'H{pred_len}')
        os.makedirs(folder, exist_ok=True)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in loader:
                bx, by  = bx.to(device), by.to(device)
                B, C, L = bx.shape
                bx_va   = model.variate_attn(bx)
                preds   = model(bx_va.reshape(B*C, 1, L))[0]
                pbc     = preds.squeeze(1).reshape(B, C, -1)
                preds   = model.gm(pbc)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(by.cpu().numpy())

        preds_arr   = np.concatenate(all_preds,   axis=0)   # (N, C, P)
        targets_arr = np.concatenate(all_targets, axis=0)   # (N, C, P)
        C  = preds_arr.shape[1]
        nv = n_variates or C
        t  = np.arange(pred_len)

        # ── 2a. Per-variate per-sample ────────────────────────────────
        for vi in range(min(nv, C)):
            vname = self.vnames[vi] if vi < len(self.vnames) else f'V{vi}'
            col   = VARIATE_COLORS[vi % len(VARIATE_COLORS)]
            for si in range(min(n_samples, preds_arr.shape[0])):
                fig, ax = plt.subplots(figsize=(11, 3.5))
                ax.plot(t, targets_arr[si, vi, :],
                        color='black', linewidth=2, label='Ground truth')
                ax.plot(t, preds_arr[si, vi, :],
                        color=col, linewidth=1.8, linestyle='--', label='Prediction')
                ax.fill_between(t, targets_arr[si,vi,:], preds_arr[si,vi,:],
                                alpha=0.12, color=col)
                mse_s = float(np.mean((preds_arr[si,vi,:] - targets_arr[si,vi,:])**2))
                ax.set_title(f'{vname}  |  sample {si}  |  MSE={mse_s:.5f}  |  H={pred_len}')
                ax.set_xlabel('Forecast step'); ax.set_ylabel('Normalised value')
                ax.legend(fontsize=9)
                _save(fig, os.path.join(folder, f'variate_{vi}_{vname}_sample{si}.png'))

        # ── 2b. All-variate grid (first sample) ───────────────────────
        rows  = math.ceil(C / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(14, 3.2 * rows))
        axes_flat = axes.flatten() if C > 2 else [axes[0], axes[1]]
        for vi in range(C):
            ax    = axes_flat[vi]
            vname = self.vnames[vi] if vi < len(self.vnames) else f'V{vi}'
            col   = VARIATE_COLORS[vi % len(VARIATE_COLORS)]
            ax.plot(t, targets_arr[0, vi, :], color='black', linewidth=1.8, label='GT')
            ax.plot(t, preds_arr[0, vi, :],   color=col,    linewidth=1.5,
                    linestyle='--', label='Pred')
            mse_v = float(np.mean((preds_arr[:, vi, :] - targets_arr[:, vi, :])**2))
            ax.set_title(f'{vname}   test MSE={mse_v:.4f}', fontsize=10)
            ax.legend(fontsize=7, loc='upper right')
            ax.xaxis.set_major_locator(MaxNLocator(5))
        for vi in range(C, len(axes_flat)):
            axes_flat[vi].set_visible(False)
        fig.suptitle(f'All Variates — First Test Sample  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, os.path.join(folder, 'all_variates_grid.png'))

        # ── 2c. Residual diagnostics ──────────────────────────────────
        residuals = (preds_arr - targets_arr).reshape(-1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(residuals, bins=80, color=C_BLUE,
                     edgecolor='white', linewidth=0.3, alpha=0.8)
        axes[0].axvline(0, color='black', linewidth=1)
        axes[0].axvline(residuals.mean(), color=C_RED, linewidth=1.5,
                        linestyle='--', label=f'mean={residuals.mean():.4f}')
        axes[0].set_xlabel('Residual'); axes[0].set_ylabel('Count')
        axes[0].set_title('Residual distribution'); axes[0].legend(fontsize=8)

        # Q-Q
        n_pts = min(len(residuals), 2000)
        idx   = np.random.choice(len(residuals), n_pts, replace=False)
        samp  = np.sort(residuals[idx])
        theo  = np.sort(np.random.randn(10000))
        theo  = np.percentile(theo, np.linspace(0.5, 99.5, n_pts))
        lim   = max(abs(theo).max(), abs(samp).max()) * 1.05
        axes[1].scatter(theo, samp, s=5, alpha=0.4, color=C_BLUE)
        axes[1].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
        axes[1].set_xlabel('Theoretical quantile')
        axes[1].set_ylabel('Sample quantile')
        axes[1].set_title('Q-Q plot (normal)')
        axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)

        # ACF
        max_lag  = 30
        res_c    = residuals - residuals.mean()
        res_var  = (res_c**2).mean() + 1e-8
        acf_vals = [(res_c[lag:] * res_c[:-lag]).mean() / res_var
                    for lag in range(1, max_lag + 1)]
        conf = 1.96 / math.sqrt(len(residuals))
        axes[2].bar(range(1, max_lag+1), acf_vals, color=C_BLUE,
                    alpha=0.7, width=0.8)
        axes[2].axhline( conf, color=C_RED, linestyle='--', linewidth=1,
                         label=f'95% CI ±{conf:.4f}')
        axes[2].axhline(-conf, color=C_RED, linestyle='--', linewidth=1)
        axes[2].axhline(0, color='black', linewidth=0.5)
        axes[2].set_xlabel('Lag'); axes[2].set_ylabel('ACF')
        axes[2].set_title('Residual autocorrelation'); axes[2].legend(fontsize=8)

        fig.suptitle(f'Residual Diagnostics  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, os.path.join(folder, 'residuals.png'))

        # ── 2d. Horizon error profile ─────────────────────────────────
        step_mae = np.abs(preds_arr - targets_arr).mean(axis=(0, 1))  # (P,)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].plot(range(1, pred_len+1), step_mae, color=C_BLUE, linewidth=2)
        axes[0].fill_between(range(1, pred_len+1), 0, step_mae,
                             alpha=0.15, color=C_BLUE)
        axes[0].set_xlabel('Forecast step'); axes[0].set_ylabel('MAE')
        axes[0].set_title(f'Mean MAE at each step  H={pred_len}')

        for vi in range(C):
            vname     = self.vnames[vi] if vi < len(self.vnames) else f'V{vi}'
            step_mae_v = np.abs(preds_arr[:,vi,:] - targets_arr[:,vi,:]).mean(0)
            axes[1].plot(range(1, pred_len+1), step_mae_v,
                         color=VARIATE_COLORS[vi % len(VARIATE_COLORS)],
                         linewidth=1.2, label=vname, alpha=0.85)
        axes[1].set_xlabel('Forecast step'); axes[1].set_ylabel('MAE')
        axes[1].set_title('Per-variate MAE profile')
        axes[1].legend(fontsize=7, ncol=2)

        fig.suptitle(f'Error Profile Along Horizon  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, os.path.join(folder, 'horizon_error_profile.png'))

    # =========================================================================
    # 3. LEARNED FILTERS
    # =========================================================================

    def plot_filters(self, model, pred_len):
        dwt = model.dwt
        h_w = dwt.h.detach().cpu().numpy()   # (C, 1, filter_len)
        g_w = dwt.g.detach().cpu().numpy()
        C, _, flen = h_w.shape
        h_mean = h_w[:, 0, :].mean(0)
        g_mean = g_w[:, 0, :].mean(0)
        h_std  = h_w[:, 0, :].std(0)
        g_std  = g_w[:, 0, :].std(0)
        haar_h = np.array([0.7071,  0.7071, 0.0, 0.0])
        haar_g = np.array([-0.7071, 0.7071, 0.0, 0.0])
        x      = np.arange(flen)

        # ── 3a. Time-domain taps ─────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, mean, std, haar, title, col in [
            (axes[0], h_mean, h_std, haar_h, 'Low-pass h (trend)',  C_BLUE),
            (axes[1], g_mean, g_std, haar_g, 'High-pass g (detail)', C_PURPLE),
        ]:
            ax.bar(x - 0.2, haar, 0.35, label='Haar init',
                   color=C_GRAY, alpha=0.5, edgecolor='black', linewidth=0.5)
            ax.bar(x + 0.2, mean, 0.35, label='Learned (mean)',
                   color=col, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.errorbar(x + 0.2, mean, yerr=std, fmt='none',
                        color='black', capsize=3, linewidth=0.8)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Tap index'); ax.set_ylabel('Coefficient')
            ax.set_title(title); ax.legend(fontsize=8)
        fig.suptitle(f'Learned DWT Filter Taps  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, self._p('filters', f'H{pred_len}_dwt_filters.png'))

        # ── 3b. Frequency response ────────────────────────────────────
        nfft = 512
        freqs = np.fft.rfftfreq(nfft)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, mean, haar, title, col in [
            (axes[0], h_mean, haar_h, 'Low-pass h',  C_BLUE),
            (axes[1], g_mean, haar_g, 'High-pass g',  C_PURPLE),
        ]:
            h_resp    = np.abs(np.fft.rfft(haar, n=nfft))
            l_resp    = np.abs(np.fft.rfft(mean, n=nfft))
            # Per-channel spread
            ch_resps  = [np.abs(np.fft.rfft(h_w[ci,0,:] if col==C_BLUE
                                             else g_w[ci,0,:], n=nfft))
                         for ci in range(min(C, 16))]
            ch_arr = np.array(ch_resps)
            ax.fill_between(freqs, ch_arr.min(0), ch_arr.max(0),
                            alpha=0.15, color=col, label='per-channel range')
            ax.plot(freqs, h_resp, color=C_GRAY, linewidth=1.5,
                    linestyle='--', label='Haar init', alpha=0.7)
            ax.plot(freqs, l_resp, color=col,   linewidth=2.0, label='Learned mean')
            ax.axvline(0.25, color=C_AMBER, linestyle=':', linewidth=1,
                       label='Nyquist/2')
            ax.set_xlabel('Normalised frequency')
            ax.set_ylabel('|H(f)|')
            ax.set_title(title + ' frequency response')
            ax.set_xlim(0, 0.5); ax.legend(fontsize=8)
        fig.suptitle(f'DWT Filter Frequency Response  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, self._p('filters', f'H{pred_len}_filter_spectrum.png'))

        # ── 3c. Deviation from Haar ───────────────────────────────────
        dh = h_mean - haar_h
        dg = g_mean - haar_g
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, delta, title, col in [
            (axes[0], dh, 'Δh  (learned − Haar)', C_BLUE),
            (axes[1], dg, 'Δg  (learned − Haar)', C_PURPLE),
        ]:
            bar_colors = [col if d >= 0 else C_RED for d in delta]
            bars = ax.bar(x, delta, color=bar_colors, edgecolor='black', linewidth=0.5)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_xlabel('Tap index'); ax.set_ylabel('Δ coefficient')
            ax.set_title(title)
            ax.bar_label(bars, fmt='%.4f', fontsize=8, padding=2)
        fig.suptitle(f'Filter Deviation from Haar Init  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, self._p('filters', f'H{pred_len}_filter_evolution.png'))

        # ── 3d. Per-channel norm heatmap ──────────────────────────────
        n_show  = min(C, 32)
        h_norms = np.linalg.norm(h_w[:n_show, 0, :], axis=1)
        g_norms = np.linalg.norm(g_w[:n_show, 0, :], axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        for ax, norms, title, col in [
            (axes[0], h_norms, 'Low-pass h  L2 norm per channel', C_BLUE),
            (axes[1], g_norms, 'High-pass g  L2 norm per channel', C_PURPLE),
        ]:
            ax.bar(range(n_show), norms, color=col, alpha=0.8,
                   edgecolor='white', linewidth=0.3)
            ax.axhline(norms.mean(), color='black', linestyle='--',
                       linewidth=1, label=f'mean={norms.mean():.3f}')
            ax.set_xlabel('Channel index'); ax.set_ylabel('L2 norm')
            ax.set_title(title); ax.legend(fontsize=8)
        fig.suptitle(f'Filter Norms per Channel  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, self._p('filters', f'H{pred_len}_filter_stats.png'))

    def plot_filter_comparison(self, filter_data_by_horizon):
        """filter_data_by_horizon = {pred_len: (h_mean_vec, g_mean_vec)}"""
        horizons = sorted(filter_data_by_horizon.keys())
        h_norms  = [np.linalg.norm(filter_data_by_horizon[h][0]) for h in horizons]
        g_norms  = [np.linalg.norm(filter_data_by_horizon[h][1]) for h in horizons]
        fig, ax  = plt.subplots(figsize=(8, 4))
        xp = np.arange(len(horizons))
        ax.bar(xp-0.2, h_norms, 0.35, label='h (low-pass)',
               color=C_BLUE,   edgecolor='black', linewidth=0.5)
        ax.bar(xp+0.2, g_norms, 0.35, label='g (high-pass)',
               color=C_PURPLE, edgecolor='black', linewidth=0.5)
        ax.set_xticks(xp)
        ax.set_xticklabels([f'H={h}' for h in horizons])
        ax.set_ylabel('Filter L2 norm')
        ax.set_title('Learned filter norms across horizons')
        ax.legend()
        _save(fig, self._p('filters', 'all_horizons_filter_norms.png'))

    # =========================================================================
    # 4. VARIATE GRAPH
    # =========================================================================

    def plot_variate_graph(self, model, pred_len,
                           loader=None, device=None):
        C      = model.gm.A.shape[0]
        vnames = self.vnames[:C]

        # ── 4a. GraphMixer adjacency ──────────────────────────────────
        A       = torch.softmax(model.gm.A.detach().cpu(), dim=-1).numpy()
        delta_A = A - np.eye(C) / C
        vmax_d  = max(abs(delta_A).max(), 1e-4)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, mat, title, cmap, vmin_v, vmax_v in [
            (axes[0], A,       'Learned mixing weight A',       'Blues',   0,       A.max()),
            (axes[1], delta_A, 'Δ from uniform (red=more mix)', 'RdBu_r', -vmax_d, vmax_d),
        ]:
            im = ax.imshow(mat, cmap=cmap, vmin=vmin_v, vmax=vmax_v, aspect='auto')
            ax.set_xticks(range(C)); ax.set_xticklabels(vnames, rotation=45, ha='right')
            ax.set_yticks(range(C)); ax.set_yticklabels(vnames)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            for i in range(C):
                for j in range(C):
                    txt_col = 'white' if mat[i,j] > vmax_v*0.6 else 'black'
                    ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                            fontsize=8, color=txt_col)
        fig.suptitle(f'Variate Dependency Graph  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, self._p('variate_graph', f'H{pred_len}_adjacency.png'))

        # ── 4b. VariateAttention gate ─────────────────────────────────
        gate_v = torch.sigmoid(model.variate_attn.gate).item()
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.barh(['VA gate'], [gate_v],       color=C_TEAL,
                edgecolor='black', linewidth=0.5, label='gate open')
        ax.barh(['VA gate'], [1.0 - gate_v], left=[gate_v],
                color=C_GRAY, alpha=0.3, edgecolor='black', linewidth=0.5,
                label='gate closed')
        ax.axvline(0.5, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Gate value  (0=identity, 1=full cross-variate enrichment)')
        ax.set_title(f'VariateAttention gate = {gate_v:.4f}  H={pred_len}\n'
                     f'({"open — cross-variate active" if gate_v > 0.3 else "near-identity — variates mostly independent"})')
        ax.legend(fontsize=8)
        _save(fig, self._p('variate_graph', f'H{pred_len}_variate_attn.png'))

        # ── 4c. Data correlation vs learned graph ─────────────────────
        if loader is not None and device is not None:
            bx, _ = next(iter(loader))
            raw   = bx[0].cpu().numpy()   # (C, seq_len)
            data_corr = np.corrcoef(raw)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, mat, title, cmap, vmin_v, vmax_v in [
                (axes[0], data_corr, 'Input data correlation', 'RdYlGn', -1, 1),
                (axes[1], A,         'Learned mixing (GraphMixer)', 'Blues', 0, A.max()),
            ]:
                im = ax.imshow(mat, cmap=cmap, vmin=vmin_v, vmax=vmax_v, aspect='auto')
                ax.set_xticks(range(C)); ax.set_xticklabels(vnames, rotation=45, ha='right')
                ax.set_yticks(range(C)); ax.set_yticklabels(vnames)
                ax.set_title(title); plt.colorbar(im, ax=ax)
                for i in range(C):
                    for j in range(C):
                        ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                                fontsize=7)
            fig.suptitle(f'Data Correlation vs Learned Graph  H={pred_len}',
                         fontweight='bold')
            plt.tight_layout()
            _save(fig, self._p('variate_graph', 'correlation_vs_learned.png'))

    # =========================================================================
    # 5. ATTENTION WEIGHTS
    # =========================================================================

    def plot_attention(self, model, loader, pred_len, device):
        model.eval()
        bx, _ = next(iter(loader))
        bx     = bx.to(device)
        B, C, L = bx.shape

        with torch.no_grad():
            bx_va   = model.variate_attn(bx)
            bxf     = bx_va.reshape(B*C, 1, L)
            xn      = model.revin(bxf, 'norm')
            patches = model.patch(xn)
            LL, LH  = model.dwt(patches)
            LL_t    = LL.transpose(1, 2)            # (B*C, T, D)
            aLL     = model.wrat._b.aLL
            D       = LL_t.shape[-1]
            H_n, Dh = aLL.H, D // aLL.H
            Q = aLL.q(LL_t).view(-1, LL_t.shape[1], H_n, Dh).transpose(1, 2)
            K = aLL.k(LL_t).view(-1, LL_t.shape[1], H_n, Dh).transpose(1, 2)
            attn = F.softmax(Q @ K.transpose(-2,-1) / Dh**0.5, dim=-1)  # (B*C, H, T, T)

        # Average over heads; show first sample
        attn_mean = attn[0].mean(0).cpu().numpy()  # (T, T)
        T = attn_mean.shape[0]
        entropy = -(attn_mean * np.log(attn_mean + 1e-8)).sum(axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im = axes[0].imshow(attn_mean, cmap='viridis', aspect='auto')
        axes[0].set_xlabel('Key patch'); axes[0].set_ylabel('Query patch')
        axes[0].set_title(f'LL attention weights\n(heads averaged, sample 0)  H={pred_len}')
        plt.colorbar(im, ax=axes[0], label='Attention weight')

        axes[1].plot(range(T), entropy, color=C_BLUE, linewidth=2)
        axes[1].fill_between(range(T), 0, entropy, alpha=0.18, color=C_BLUE)
        axes[1].axhline(math.log(T), color=C_GRAY, linestyle='--', linewidth=1,
                        label=f'max entropy log({T})={math.log(T):.2f}')
        axes[1].set_xlabel('Query patch'); axes[1].set_ylabel('Entropy (nats)')
        axes[1].set_title('Attention entropy  (high=diffuse, low=focused)')
        axes[1].legend(fontsize=8)

        fig.suptitle(f'WRATBlock Attention Analysis  H={pred_len}', fontweight='bold')
        plt.tight_layout()
        _save(fig, self._p('attention', f'H{pred_len}_wrat_attn.png'))

        # Tau bar
        tau_val = model.wrat.tau
        fig, ax  = plt.subplots(figsize=(7, 2.5))
        ax.barh(['τ  (sparsity)'], [tau_val], color=C_BLUE,
                edgecolor='black', linewidth=0.5)
        ax.barh(['τ  (sparsity)'], [0.5 - tau_val], left=[tau_val],
                color=C_GRAY, alpha=0.25)
        ax.axvline(0.1, color=C_AMBER, linestyle='--', linewidth=1,
                   label='init τ=0.1')
        ax.set_xlim(0, 0.5)
        ax.set_xlabel('τ value')
        ax.set_title(f'Learned sparsity threshold τ = {tau_val:.4f}  H={pred_len}')
        ax.legend(fontsize=8)
        _save(fig, self._p('attention', f'H{pred_len}_tau_bar.png'))

    # =========================================================================
    # 6. BENCHMARKS
    # =========================================================================

    def plot_final_benchmarks(self, final_results, sota=None):
        horizons = sorted(final_results.keys())
        mses  = [final_results[h]['mse']     for h in horizons]
        maes  = [final_results[h]['mae']      for h in horizons]
        daccs = [final_results[h]['dir_acc']  for h in horizons]
        xlabels = [f'H={h}' for h in horizons]

        # ── MSE bar ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(xlabels, mses, color=C_BLUE,
                      edgecolor='black', linewidth=0.5)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        ax.set_ylabel('Test MSE (lower is better)')
        ax.set_title('Test MSE by Forecast Horizon')
        _save(fig, self._p('benchmarks', 'final_mse_bar.png'))

        # ── MAE bar ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(xlabels, maes, color=C_GREEN,
                      edgecolor='black', linewidth=0.5)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        ax.set_ylabel('Test MAE (lower is better)')
        ax.set_title('Test MAE by Forecast Horizon')
        _save(fig, self._p('benchmarks', 'final_mae_bar.png'))

        # ── SOTA comparison (H=96) ────────────────────────────────────
        sota_data = dict(sota or SOTA_MSE_H96)
        if 96 in final_results:
            sota_data['P-WSA++ (ours)'] = final_results[96]['mse']
        sorted_sota = sorted(sota_data.items(), key=lambda x: x[1], reverse=True)
        names, vals  = zip(*sorted_sota)
        bar_cols = [C_TEAL if 'ours' in n else C_GRAY for n in names]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, vals, color=bar_cols,
                       edgecolor='black', linewidth=0.5)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        ax.set_xlabel('Test MSE at H=96  (lower is better)')
        ax.set_title('SOTA Comparison — ETTm1 Multivariate H=96\n'
                     '(P-WSA++ CI mode; published SOTA uses full multivariate)')
        _save(fig, self._p('benchmarks', 'sota_comparison.png'))

        # ── All-horizons summary ──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, vals_list, ylabel, title, col, marker in [
            (axes[0], mses,  'Test MSE', 'Test MSE',          C_BLUE,   'o'),
            (axes[1], maes,  'Test MAE', 'Test MAE',          C_GREEN,  's'),
            (axes[2], daccs, 'Dir Acc %','Direction accuracy', C_PURPLE, '^'),
        ]:
            ax.plot(xlabels, vals_list, f'{marker}-', color=col,
                    linewidth=2, markersize=9)
            for xl, v in zip(xlabels, vals_list):
                label = f'{v:.3f}' if col != C_PURPLE else f'{v:.1f}%'
                ax.annotate(label, xy=(xl, v),
                            xytext=(0, 9), textcoords='offset points',
                            ha='center', fontsize=9)
            ax.set_ylabel(ylabel); ax.set_title(title)
            if col == C_PURPLE:
                ax.axhline(50, color=C_GRAY, linestyle='--',
                           linewidth=1, label='random 50%')
                ax.legend(fontsize=8)
        fig.suptitle('P-WSA++ Final Test Results — All Horizons', fontweight='bold')
        plt.tight_layout()
        _save(fig, self._p('benchmarks', 'all_horizons_summary.png'))

    # =========================================================================
    # 7. SPECTRAL ANALYSIS
    # =========================================================================

    def plot_spectral_analysis(self, model, loader, pred_len, device):
        model.eval()
        bx, by = next(iter(loader))
        bx, by  = bx.to(device), by.to(device)
        B, C, L = bx.shape

        with torch.no_grad():
            bx_va   = model.variate_attn(bx)
            bxf     = bx_va.reshape(B*C, 1, L)
            xn      = model.revin(bxf, 'norm')
            patches = model.patch(xn)
            LL, LH  = model.dwt(patches)
            preds, rec, patches_o, LLl, LHl = model(bxf)
            pbc     = preds.squeeze(1).reshape(B, C, -1)
            preds   = model.gm(pbc).reshape(B*C, 1, -1)

        vi       = 0                                     # plot first variate
        inp_sig  = xn[vi, 0, :].cpu().numpy()
        pred_sig = preds[vi, 0, :].cpu().numpy()
        tgt_sig  = by[0, vi, :].cpu().numpy()
        ll_sig   = LL[vi, 0, :].cpu().numpy()

        def psd(sig):
            f = np.fft.rfft(sig)
            return np.fft.rfftfreq(len(sig)), np.abs(f)**2

        inp_fr,  inp_p  = psd(inp_sig)
        ll_fr,   ll_p   = psd(ll_sig)
        pred_fr, pred_p = psd(pred_sig)
        tgt_fr,  tgt_p  = psd(tgt_sig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        axes[0,0].plot(inp_sig, color=C_GRAY, linewidth=1, alpha=0.8)
        axes[0,0].set_title('Input (RevIN normalised)')
        axes[0,0].set_xlabel('Time step'); axes[0,0].set_ylabel('Value')

        axes[0,1].plot(tgt_sig,  color='black',  linewidth=2, label='Target')
        axes[0,1].plot(pred_sig, color=C_BLUE,   linewidth=1.8,
                       linestyle='--', label='Prediction')
        axes[0,1].fill_between(range(pred_len), tgt_sig, pred_sig,
                               alpha=0.12, color=C_BLUE)
        axes[0,1].legend(fontsize=8)
        axes[0,1].set_title('Prediction vs target (normalised)')
        axes[0,1].set_xlabel('Forecast step')

        axes[1,0].semilogy(inp_fr, inp_p + 1e-8, color=C_GRAY,
                           linewidth=1.2, label='Input PSD')
        axes[1,0].semilogy(ll_fr,  ll_p  + 1e-8, color=C_BLUE,
                           linewidth=1.8, linestyle='--', label='LL subband PSD')
        axes[1,0].set_xlabel('Normalised frequency')
        axes[1,0].set_ylabel('PSD (log)')
        axes[1,0].set_title('Input vs LL subband power spectrum')
        axes[1,0].legend(fontsize=8)

        axes[1,1].semilogy(tgt_fr,  tgt_p  + 1e-8, color='black',
                           linewidth=2, label='Target PSD')
        axes[1,1].semilogy(pred_fr, pred_p + 1e-8, color=C_BLUE,
                           linewidth=1.8, linestyle='--', label='Prediction PSD')
        axes[1,1].set_xlabel('Normalised frequency')
        axes[1,1].set_ylabel('PSD (log)')
        axes[1,1].set_title('Target vs prediction power spectrum')
        axes[1,1].legend(fontsize=8)

        vname = self.vnames[vi] if vi < len(self.vnames) else f'V{vi}'
        fig.suptitle(f'Spectral Analysis — H={pred_len}, {vname}', fontweight='bold')
        plt.tight_layout()
        _save(fig, self._p('benchmarks', f'H{pred_len}_spectral.png'))