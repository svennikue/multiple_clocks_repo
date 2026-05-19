#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting helpers for human single-cell RSA and encoding results.

Functions
---------
p_to_stars(p)
plot_rsa_heatmap(results, models, rois, title, save_path)
    Heatmap of t-values per model × ROI, with significance stars.
    Two panels: (1) unique regression per model, (2) combined regression.

Encoding result plots (all take an explicit `models` list):
plot_neuron_fit(diag, save_dir, suptitle)
plot_perm_histogram(diag, save_dir, bins)
plot_best_neuron_per_roi_model(diagnostics_all, results_df, save_dir)
plot_r_distribution_grid(results_df, models, save_path, alpha, reg_alpha)
plot_significance_proportion(results_df, models, save_path, alpha,
                             chance_level, reg_alpha)
plot_dsr_coef_matrix(X_slice, coefs, neuron_label, model_name, save_path,
                     n_phases, n_clocks_per_phase, fold_r, p_perm)
    Per-neuron coefficient-weighted regressor matrix for DSR-family models.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats


# ── Significance helper ────────────────────────────────────────────────

def p_to_stars(p: float) -> str:
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


# # ── Main plotting function ─────────────────────────────────────────────

# def plot_rsa_heatmap(
#     results: dict,
#     models: list[str],
#     combo_models: list[str],
#     rois: list[str],
#     title: str = 'RSA results',
#     save_path: str | None = None,
#     perm_p_unique: dict | None = None,
#     perm_p_combined: dict | None = None,
# ) -> plt.Figure:
#     """
#     Plot a summary heatmap of RSA t-values for unique and combined models.

#     Parameters
#     ----------
#     results : dict
#         Nested dict: results[roi]['unique'][model] = (t, beta, p)
#                      results[roi]['combined']['t']    shape (n_models,)
#                      results[roi]['combined']['p']    shape (n_models,)
#                      results[roi]['n_neurons']        int
#     models : list of str
#         Model names in the order they appear in the regression.
#     combo_models : list of str
#         Model names in the order they appear in the combined regression.

#     rois : list of str
#         ROI names to include (must be keys in results).
#     title : str
#         Figure suptitle.
#     save_path : str or None
#         If given, save figure to this path.
#     perm_p_unique : dict or None
#         If provided, use permutation p-values for significance annotation
#         in the unique-regression panel.
#         Format: perm_p_unique[roi][model] = float p-value.
#     perm_p_combined : dict or None
#         Same as perm_p_unique but for the combined-regression panel.

#     Returns
#     -------
#     fig : plt.Figure
#     """
#     present_rois = [r for r in rois if r in results]
#     n_rois   = len(present_rois)
#     n_models = len(models)
#     n_combo_models = len(combo_models)

#     # ── collect arrays ─────────────────────────────────────────────────
#     #unique_t    = np.full((n_models, n_rois), np.nan)
#     unique_beta = np.full((n_models, n_rois), np.nan)
#     unique_p    = np.full((n_models, n_rois), np.nan)   # OLS p (fallback)
#     unique_pp   = np.full((n_models, n_rois), np.nan)   # permutation p
#     #combo_t     = np.full((n_combo_models, n_rois), np.nan)
#     combo_beta  = np.full((n_combo_models, n_rois), np.nan)
#     combo_p     = np.full((n_combo_models, n_rois), np.nan)
#     combo_pp    = np.full((n_combo_models, n_rois), np.nan)
#     n_neurons   = []

#     for r_idx, roi in enumerate(present_rois):
#         n_neurons.append(results[roi].get('n_neurons', 0))
#         for m_idx, m in enumerate(models):
#             u = results[roi].get('unique', {}).get(m)
#             if u is not None:
#                 #unique_t[m_idx, r_idx]  = float(u[0])
#                 unique_beta[m_idx, r_idx]  = float(u[1])
#                 unique_p[m_idx, r_idx]  = float(u[2])
#             if perm_p_unique is not None:
#                 pp = perm_p_unique.get(roi, {}).get(m)
#                 if pp is not None:
#                     unique_pp[m_idx, r_idx] = float(pp)
#         combo = results[roi].get('combined')
#         if combo is not None:
#             for m_idx in range(n_combo_models):
#                 #combo_t[m_idx, r_idx]  = float(combo['t'][m_idx])
#                 combo_beta[m_idx, r_idx]  = float(combo['beta'][m_idx])
#                 combo_p[m_idx, r_idx]  = float(combo['p'][m_idx])
#                 if perm_p_combined is not None:
#                     pp = perm_p_combined.get(roi, {}).get(combo_models[m_idx])
#                     if pp is not None:
#                         combo_pp[m_idx, r_idx] = float(pp)

#     # ── choose which p-values to annotate with ─────────────────────────
#     # Use permutation p if supplied, otherwise fall back to OLS p.
#     ann_unique_p = unique_pp if perm_p_unique  is not None else unique_p
#     ann_combo_p  = combo_pp  if perm_p_combined is not None else combo_p
#     p_source     = 'perm p' if (perm_p_unique is not None) else 'OLS p'

#     # ── figure layout ──────────────────────────────────────────────────
#     fig, axes = plt.subplots(
#         1, 2,
#         figsize=(max(10, n_rois * 1.3 + 3), max(4, n_models * 0.9 + 2)),
#         constrained_layout=True,
#     )
#     fig.suptitle(f'{title}  [{p_source}]', fontsize=13, weight='bold')

#     #panel_data   = [unique_t,    combo_t]
#     panel_data = [unique_beta, combo_beta]
#     panel_ann_p  = [ann_unique_p, ann_combo_p]
#     panel_titles = ['Unique regression (per model)', 'Combined regression (all models)']

#     # abs_max = np.nanmax(np.abs(np.concatenate([unique_t, combo_t])))
#     abs_max = np.nanmax(np.abs(np.concatenate([unique_beta, combo_beta])))
#     vmax = max(abs_max, 1.0)
#     import pdb; pdb.set_trace()

#     for ax, beta_mat, ann_p_mat, ptitle in zip(
#             axes, panel_data, panel_ann_p, panel_titles):
#         im = ax.imshow(
#             beta_mat,
#             #t_mat,
#             aspect='auto',
#             cmap='RdBu_r',
#             vmin=-vmax,
#             vmax=vmax,
#             interpolation='nearest',
#         )

#         for m_idx in range(n_models):
#             for r_idx in range(n_rois):
#                 beta = beta_mat[m_idx, r_idx]
#                 # t_val = t_mat[m_idx, r_idx]
#                 p_val = ann_p_mat[m_idx, r_idx]
#                 if np.isnan(beta):
#                     continue
#                 stars = p_to_stars(p_val) if not np.isnan(p_val) else ''
#                 norm_val  = (beta + vmax) / (2 * vmax)
#                 txt_color = 'white' if abs(norm_val - 0.5) > 0.25 else 'black'
#                 ax.text(
#                     r_idx, m_idx,
#                     f'{beta:.2f}{stars}',
#                     ha='center', va='center',
#                     fontsize=7.5, color=txt_color,
#                 )

#         roi_labels = [
#             f'{r}\n(n={results[r]["n_neurons"]})' for r in present_rois
#         ]
#         ax.set_xticks(range(n_rois))
#         ax.set_xticklabels(roi_labels, rotation=40, ha='right', fontsize=8)
#         ax.set_yticks(range(n_models))
#         ax.set_yticklabels(models, fontsize=9)
#         ax.set_title(ptitle, fontsize=10, pad=6)
#         plt.colorbar(im, ax=ax, shrink=0.7, label='beta')

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f'Saved → {save_path}')

#     return fig

def plot_rsa_heatmap(
    results: dict,
    models: list[str],
    combo_models: list[str],
    rois: list[str],
    title: str = 'RSA results',
    save_path: str | None = None,
    perm_p_unique: dict | None = None,
    perm_p_combined: dict | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Plot RSA beta heatmaps as two separate figures:
    1) unique regression
    2) combined regression

    This avoids shape/label mismatches when `models` and `combo_models`
    have different lengths.

    Parameters
    ----------
    results : dict
        Nested dict: results[roi]['unique'][model] = (t, beta, p)
                     results[roi]['combined']['beta'] shape (n_models,)
                     results[roi]['combined']['p']    shape (n_models,)
                     results[roi]['n_neurons']        int
    models : list of str
        Model names in the order they appear in the unique regression.
    combo_models : list of str
        Model names in the order they appear in the combined regression.
    rois : list of str
        ROI names to include (must be keys in results).
    title : str
        Base title for the figures.
    save_path : str or None
        If given, save figures to:
            <stem>_unique<suffix>
            <stem>_combined<suffix>
    perm_p_unique : dict or None
        If provided, use permutation p-values for significance annotation
        in the unique-regression figure.
        Format: perm_p_unique[roi][model] = float p-value.
    perm_p_combined : dict or None
        Same as perm_p_unique but for the combined-regression figure.

    Returns
    -------
    fig_unique, fig_combined : tuple[plt.Figure, plt.Figure]
    """
    from pathlib import Path

    present_rois = [r for r in rois if r in results]
    n_rois = len(present_rois)
    n_models = len(models)
    n_combo_models = len(combo_models)

    # ── collect arrays ─────────────────────────────────────────────────
    unique_beta = np.full((n_models, n_rois), np.nan)
    unique_p = np.full((n_models, n_rois), np.nan)    # OLS p (fallback)
    unique_pp = np.full((n_models, n_rois), np.nan)   # permutation p

    combo_beta = np.full((n_combo_models, n_rois), np.nan)
    combo_p = np.full((n_combo_models, n_rois), np.nan)
    combo_pp = np.full((n_combo_models, n_rois), np.nan)

    for r_idx, roi in enumerate(present_rois):
        for m_idx, m in enumerate(models):
            u = results[roi].get('unique', {}).get(m)
            if u is not None:
                unique_beta[m_idx, r_idx] = float(u[1])
                unique_p[m_idx, r_idx] = float(u[2])

            if perm_p_unique is not None:
                pp = perm_p_unique.get(roi, {}).get(m)
                if pp is not None:
                    unique_pp[m_idx, r_idx] = float(pp)

        combo = results[roi].get('combined')
        if combo is not None:
            beta_vals = combo.get('beta', [])
            p_vals = combo.get('p', [])

            for m_idx in range(min(n_combo_models, len(beta_vals), len(p_vals))):
                combo_beta[m_idx, r_idx] = float(beta_vals[m_idx])
                combo_p[m_idx, r_idx] = float(p_vals[m_idx])

                if perm_p_combined is not None:
                    pp = perm_p_combined.get(roi, {}).get(combo_models[m_idx])
                    if pp is not None:
                        combo_pp[m_idx, r_idx] = float(pp)

    # ── choose which p-values to annotate with ─────────────────────────
    ann_unique_p = unique_pp if perm_p_unique is not None else unique_p
    ann_combo_p = combo_pp if perm_p_combined is not None else combo_p

    unique_p_source = 'perm p' if perm_p_unique is not None else 'OLS p'
    combo_p_source = 'perm p' if perm_p_combined is not None else 'OLS p'

    roi_labels = [f'{r}\n(n={results[r]["n_neurons"]})' for r in present_rois]

    # shared color scale across both figures
    all_beta_vals = np.concatenate([unique_beta.ravel(), combo_beta.ravel()])
    if np.all(np.isnan(all_beta_vals)):
        vmax = 1.0
    else:
        abs_max = np.nanmax(np.abs(all_beta_vals))
        vmax = min(abs_max, 1.0)

    # import pdb; pdb.set_trace()
    def _make_single_heatmap(
        beta_mat: np.ndarray,
        ann_p_mat: np.ndarray,
        y_labels: list[str],
        fig_title: str,
    ) -> plt.Figure:
        n_rows = len(y_labels)

        fig, ax = plt.subplots(
            1, 1,
            figsize=(max(6, n_rois * 1.3 + 2), max(4, n_rows * 0.55 + 2)),
            constrained_layout=True,
        )
        fig.suptitle(fig_title, fontsize=13, weight='bold')

        im = ax.imshow(
            beta_mat,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            interpolation='nearest',
        )

        for m_idx in range(beta_mat.shape[0]):
            for r_idx in range(beta_mat.shape[1]):
                beta = beta_mat[m_idx, r_idx]
                p_val = ann_p_mat[m_idx, r_idx]
                if np.isnan(beta):
                    continue

                stars = p_to_stars(p_val) if not np.isnan(p_val) else ''
                norm_val = (beta + vmax) / (2 * vmax)
                txt_color = 'white' if abs(norm_val - 0.5) > 0.25 else 'black'

                ax.text(
                    r_idx, m_idx,
                    f'{beta:.2f}{stars}',
                    ha='center', va='center',
                    fontsize=7.5, color=txt_color,
                )

        ax.set_xticks(range(n_rois))
        ax.set_xticklabels(roi_labels, rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_labels, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7, label='beta')

        return fig

    fig_unique = _make_single_heatmap(
        unique_beta,
        ann_unique_p,
        models,
        f'{title} — Unique regression [{unique_p_source}]',
    )

    fig_combined = _make_single_heatmap(
        combo_beta,
        ann_combo_p,
        combo_models,
        f'{title} — Combined regression [{combo_p_source}]',
    )

    if save_path:
        save_path = Path(save_path)
        unique_path = save_path.with_name(f'{save_path.stem}_unique{save_path.suffix}')
        combined_path = save_path.with_name(f'{save_path.stem}_combined{save_path.suffix}')

        fig_unique.savefig(unique_path, dpi=150, bbox_inches='tight')
        fig_combined.savefig(combined_path, dpi=150, bbox_inches='tight')

        print(f'Saved unique   → {unique_path}')
        print(f'Saved combined → {combined_path}')

    return fig_unique, fig_combined


# ── Encoding result plots ──────────────────────────────────────────────


def _one_sided_t_greater(vals):
    """One-sample t-test, H1: mean > 0. Returns (t, one-sided p)."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan, np.nan
    t_stat, p_two = stats.ttest_1samp(vals, 0.0, nan_policy='omit')
    if not np.isfinite(t_stat):
        return np.nan, np.nan
    p_one = (p_two / 2) if t_stat > 0 else (1 - p_two / 2)
    return float(t_stat), float(p_one)


def _draw_perm_hist(ax, diag, bins=30):
    """Draw a permutation-null histogram into an existing axis."""
    perm_rs = np.asarray(diag['perm_rs'], dtype=float)
    perm_rs = perm_rs[np.isfinite(perm_rs)]
    emp_r = diag['mean_r']
    finite_vals = np.concatenate([
        perm_rs,
        np.array([emp_r] if np.isfinite(emp_r) else [], dtype=float),
    ])
    lim = (max(0.05, 1.05 * float(np.max(np.abs(finite_vals))))
           if finite_vals.size else 1.0)
    edges = np.linspace(-lim, lim, bins + 1)
    if perm_rs.size:
        ax.hist(perm_rs, bins=edges, color='0.75',
                edgecolor='white', linewidth=0.6, label='perm null')
    ax.axvline(0, color='black', lw=0.9)
    if np.isfinite(emp_r):
        ax.axvline(emp_r, color='tab:red', lw=1.8,
                   label=f'emp r = {emp_r:.3f}')
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('mean Pearson r')
    p_perm = diag.get('p_perm', np.nan)
    p_txt = f"p_perm = {p_perm:.3f}" if np.isfinite(p_perm) else "p_perm = n/a"
    ax.set_title(f"perm null   {p_txt}", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, loc='upper left', frameon=False)


def plot_neuron_fit(diag, save_dir=None, suptitle=None):
    """Per-fold timecourse: y_test (actual) vs y_pred (predicted).

    Parameters
    ----------
    diag : dict
        Single (neuron, model) diagnostic dict as produced by
        ``analyse_one_neuron`` — must contain keys:
        ``y_pred_per_fold``, ``y_test_per_fold``, ``r_per_fold``,
        ``neuron``, ``roi``, ``model``, ``mean_r``, ``p_perm``.
        Optionally ``configs`` (list of config strings for fold labels).
    save_dir : str or None
        Directory to save the figure in.  Filename is auto-generated.
    suptitle : str or None
        Override the auto-generated suptitle.
    """
    y_pred_per_fold = [np.asarray(yp) for yp in diag['y_pred_per_fold']]
    y_test_per_fold = [np.asarray(yt) for yt in diag['y_test_per_fold']]
    r_per_fold = list(diag['r_per_fold'])
    n_folds = len(y_pred_per_fold)

    fig, axes = plt.subplots(
        n_folds, 1,
        figsize=(10, 1.4 * n_folds + 0.5),
        sharex=False, constrained_layout=True,
    )
    if n_folds == 1:
        axes = [axes]

    sub_configs = diag.get('configs', [])
    for fold_idx, ax in enumerate(axes):
        yt = y_test_per_fold[fold_idx]
        yp = y_pred_per_fold[fold_idx]
        if yt.size == 0:
            ax.set_visible(False)
            continue
        x = np.arange(yt.size)

        # Dual y-axes: neuron on left, prediction on right (independent scales).
        ax2 = ax.twinx()
        ax.plot(x, yt, color='0.30', lw=1.2, label='neuron')
        ax2.plot(x, yp, color='tab:red', lw=1.2, alpha=0.85, label='predicted')

        r_txt = (f"r = {r_per_fold[fold_idx]:.3f}"
                 if np.isfinite(r_per_fold[fold_idx]) else "r = n/a")
        cfg_label = (sub_configs[fold_idx] if fold_idx < len(sub_configs)
                     else f'fold {fold_idx}')
        ax.set_title(f"held-out {cfg_label}   {r_txt}", fontsize=10, loc='left')
        ax.tick_params(labelsize=8)
        ax2.tick_params(labelsize=8, labelcolor='tab:red')
        ax.set_ylabel('neuron (a.u.)', fontsize=7, color='0.40')
        ax2.set_ylabel('predicted', fontsize=7, color='tab:red')
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        if fold_idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=8, loc='upper right', frameon=False)

    title = suptitle or (
        f"{diag['neuron']} ({diag['roi']}) — {diag['model']}  "
        f"mean r = {diag['mean_r']:.3f}   "
        f"p_perm = {diag['p_perm']:.3f}"
    )
    fig.suptitle(title, fontsize=11, fontweight='bold')

    if save_dir is not None:
        fname = f"fit_{diag['neuron']}_{diag['model']}.png".replace('/', '_')
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    return fig


def plot_perm_histogram(diag, save_dir=None, bins=30):
    """Permutation null vs empirical mean r, symmetric x-axis around 0.

    Parameters
    ----------
    diag : dict
        Single (neuron, model) diagnostic dict (same format as
        ``plot_neuron_fit``).
    save_dir : str or None
        Directory to save the figure in.
    bins : int
        Number of histogram bins.
    """
    perm_rs = np.asarray(diag['perm_rs'], dtype=float)
    perm_rs = perm_rs[np.isfinite(perm_rs)]
    emp_r = diag['mean_r']

    finite_vals = np.concatenate([
        perm_rs,
        np.array([emp_r] if np.isfinite(emp_r) else [], dtype=float),
    ])
    lim = (max(0.05, 1.05 * float(np.max(np.abs(finite_vals))))
           if finite_vals.size else 1.0)
    edges = np.linspace(-lim, lim, bins + 1)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    if perm_rs.size:
        ax.hist(perm_rs, bins=edges, color='0.75',
                edgecolor='white', linewidth=0.6, label='permutation null')
    ax.axvline(0, color='black', lw=0.9)
    if np.isfinite(emp_r):
        ax.axvline(emp_r, color='tab:red', lw=1.8,
                   label=f'empirical r = {emp_r:.3f}')
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('mean Pearson r (across folds)')
    ax.set_ylabel('count')
    p_perm = diag.get('p_perm', np.nan)
    p_txt = f"p_perm = {p_perm:.3f}" if np.isfinite(p_perm) else "p_perm = n/a"
    ax.set_title(
        f"{diag['neuron']} ({diag['roi']}) — {diag['model']}   {p_txt}",
        fontsize=10,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=9, frameon=False)

    if save_dir is not None:
        fname = f"perm_{diag['neuron']}_{diag['model']}.png".replace('/', '_')
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    return fig


def plot_best_neuron_per_roi_model(diagnostics_all, results_df, save_dir):
    """For each (ROI, model), plot the best neuron's permutation null and
    best-fold actual-vs-predicted timecourse.

    Parameters
    ----------
    diagnostics_all : dict
        Nested dict ``{sub_str: {neuron_label: {model: diag_dict}}}``.
    results_df : pd.DataFrame
        Per-(subject, neuron, model) results with columns
        ``roi``, ``model``, ``mean_r``, ``subject``, ``neuron``.
    save_dir : str
        Directory to save figures in (created if absent).
    """
    if results_df.empty:
        return
    os.makedirs(save_dir, exist_ok=True)
    rdf = results_df.dropna(subset=['mean_r']).copy()
    n_plots = 0
    for (roi, model), grp in rdf.groupby(['roi', 'model']):
        best_row = grp.loc[grp['mean_r'].idxmax()]
        sub_str = best_row['subject']
        neuron_label = best_row['neuron']
        diag = (diagnostics_all.get(sub_str, {})
                .get(neuron_label, {}).get(model))
        if diag is None:
            continue
        r_per_fold = np.asarray(diag['r_per_fold'], dtype=float)
        if not np.isfinite(r_per_fold).any():
            continue
        best_fold = int(np.nanargmax(r_per_fold))
        sub_configs = diag.get('configs', [])
        cfg_label = (sub_configs[best_fold] if best_fold < len(sub_configs)
                     else f'fold {best_fold}')

        fig, axes = plt.subplots(
            1, 2, figsize=(11, 3.5), constrained_layout=True,
            gridspec_kw=dict(width_ratios=[1, 2]),
        )
        _draw_perm_hist(axes[0], diag, bins=30)

        yt = np.asarray(diag['y_test_per_fold'][best_fold], dtype=float)
        yp = np.asarray(diag['y_pred_per_fold'][best_fold], dtype=float)
        x = np.arange(yt.size)

        ax_ts = axes[1]
        ax_ts2 = ax_ts.twinx()
        ax_ts.plot(x, yt, color='0.30', lw=1.2, label='neuron')
        ax_ts2.plot(x, yp, color='tab:red', lw=1.2, alpha=0.85, label='predicted')
        ax_ts.set_title(
            f"best fold: held-out {cfg_label}   r = {r_per_fold[best_fold]:.3f}",
            fontsize=10, loc='left',
        )
        ax_ts.set_xlabel('time bin')
        ax_ts.tick_params(labelsize=9)
        ax_ts2.tick_params(labelsize=9, labelcolor='tab:red')
        ax_ts.set_ylabel('neuron (a.u.)', fontsize=8, color='0.40')
        ax_ts2.set_ylabel('predicted', fontsize=8, color='tab:red')
        ax_ts.spines['top'].set_visible(False)
        ax_ts2.spines['top'].set_visible(False)
        lines1, labels1 = ax_ts.get_legend_handles_labels()
        lines2, labels2 = ax_ts2.get_legend_handles_labels()
        ax_ts.legend(lines1 + lines2, labels1 + labels2,
                     fontsize=9, loc='upper right', frameon=False)

        fig.suptitle(
            f"Best neuron in {roi} for model '{model}': "
            f"sub-{sub_str} {neuron_label}   "
            f"mean r = {diag['mean_r']:.3f}   "
            f"p_perm = {diag['p_perm']:.3f}",
            fontsize=11, fontweight='bold',
        )
        fname = (f"best_{roi}_{model}_sub-{sub_str}_{neuron_label}.png"
                 .replace('/', '_'))
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        n_plots += 1
    print(f"Saved {n_plots} best-neuron showcase plots → {save_dir}")


def plot_r_distribution_grid(results_df, models, save_path=None,
                              alpha=0.05, reg_alpha=None):
    """Per (ROI, model) histogram of cross-validated mean r values.

    Significant neurons (p_perm < alpha) are coloured red; grey = n.s.
    A one-sided t-test (H1: mean > 0) is annotated in each panel; panels
    with p_t < alpha get a bold frame.  Each panel uses its own x-axis
    limits so the distribution shape is visible regardless of neuron count.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``roi``, ``model``, ``mean_r``, ``p_perm``.
    models : list of str
        Ordered list of model names; controls column order.
    save_path : str or None
        Full path to save the figure.
    alpha : float
        Significance threshold for both perm p and t-test.
    reg_alpha : float or None
        Regularisation strength (e.g. ElasticNet alpha) — included in the
        suptitle for reference if provided.
    """
    if results_df.empty:
        return None
    rdf = results_df.dropna(subset=['mean_r']).copy()
    rois = sorted(rdf['roi'].unique())
    mods = [m for m in models if m in rdf['model'].unique()]
    if not rois or not mods:
        return None

    n_rows, n_cols = len(rois), len(mods)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.1 * n_cols + 1, 2.1 * n_rows + 0.8),
        constrained_layout=True,
    )
    axes = np.atleast_2d(np.asarray(axes)).reshape(n_rows, n_cols)

    for r, roi in enumerate(rois):
        for c, model in enumerate(mods):
            ax = axes[r, c]
            sub = rdf[(rdf['roi'] == roi) & (rdf['model'] == model)]
            vals = sub['mean_r'].to_numpy(dtype=float)
            p_perm_arr = sub['p_perm'].to_numpy(dtype=float)
            if vals.size == 0:
                ax.set_visible(False)
                continue

            panel_lim = max(0.05, 1.05 * float(np.max(np.abs(vals))))
            edges = np.linspace(-panel_lim, panel_lim, 21)

            sig_mask = np.isfinite(p_perm_arr) & (p_perm_arr < alpha)
            ax.hist(vals[~sig_mask], bins=edges, color='0.75',
                    edgecolor='white', linewidth=0.6,
                    label=f"n.s. (n={int((~sig_mask).sum())})")
            if sig_mask.any():
                ax.hist(vals[sig_mask], bins=edges, color='tab:red',
                        edgecolor='white', linewidth=0.6, alpha=0.85,
                        label=f"sig (n={int(sig_mask.sum())})")
            ax.axvline(0, color='black', lw=0.9)

            t_stat, p_t = _one_sided_t_greater(vals)
            ax.text(0.04, 0.97,
                    f"t={t_stat:.2f}\np={p_t:.3f}\nN={vals.size}",
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=8, linespacing=1.05)

            if np.isfinite(p_t) and p_t < alpha:
                for spine in ax.spines.values():
                    spine.set_linewidth(2.5)
                    spine.set_color('black')
            else:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(model, fontsize=9)
            if c == 0:
                ax.set_ylabel(roi, fontsize=9)

    reg_str = f"  |  reg alpha={reg_alpha}" if reg_alpha is not None else ""
    fig.suptitle(
        f"Cross-validated mean r per (ROI × model){reg_str}  |  "
        f"alpha={alpha}  |  "
        f"red = perm-significant (p<{alpha})  |  "
        f"bold frame = one-sided t-test p<{alpha}",
        fontsize=10, fontweight='bold',
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_significance_proportion(results_df, models, save_path=None,
                                  alpha=0.05, chance_level=0.05,
                                  reg_alpha=None):
    """Grouped bar plot: proportion of significant neurons per (ROI, model).

    One-sided binomial test vs chance_level; a star is drawn above bars
    that pass.  Uses the Dark2 colormap for ROI colours.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``roi``, ``model``, ``p_perm``.
    models : list of str
        Ordered list of model names; controls x-axis order.
    save_path : str or None
        Full path to save the figure.
    alpha : float
        Significance threshold for both binomial test and bar annotation.
    chance_level : float
        Expected proportion under the null (dashed reference line).
    reg_alpha : float or None
        Regularisation strength — included in the title if provided.
    """
    if results_df.empty:
        return None
    rdf = results_df.dropna(subset=['mean_r']).copy()
    rois = sorted(rdf['roi'].unique())
    mods = [m for m in models if m in rdf['model'].unique()]
    if not rois or not mods:
        return None

    n_models = len(mods)
    n_rois = len(rois)
    width = 0.8 / max(n_rois, 1)
    x_base = np.arange(n_models)

    fig, ax = plt.subplots(
        figsize=(max(8, n_models * 1.0 + 2), 4.5),
        constrained_layout=True,
    )
    cmap = plt.get_cmap('Set3')

    max_prop = 0.0
    for r_idx, roi in enumerate(rois):
        props, n_total_arr, sig_marks = [], [], []
        for m in mods:
            sub = rdf[(rdf['roi'] == roi) & (rdf['model'] == m)]
            n_total = int(len(sub))
            n_sig = int((sub['p_perm'] < alpha).sum())
            prop = (n_sig / n_total) if n_total > 0 else 0.0
            props.append(prop)
            n_total_arr.append(n_total)
            if n_total > 0:
                bt = stats.binomtest(n_sig, n_total, p=chance_level,
                                     alternative='greater')
                sig_marks.append(bt.pvalue < alpha)
            else:
                sig_marks.append(False)

        max_prop = max(max_prop, max(props) if props else 0.0)
        positions = x_base - 0.4 + width / 2 + r_idx * width
        n_max = max(n_total_arr) if n_total_arr else 0
        bars = ax.bar(positions, props, width=width,
                      color=cmap(r_idx),
                      label=f'{roi} (N={n_max})',
                      edgecolor='black', linewidth=0.6)
        for bar, sig in zip(bars, sig_marks):
            if sig:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, '*',
                        ha='center', va='bottom',
                        fontsize=14, fontweight='bold')

    ax.axhline(chance_level, color='black', lw=0.8, ls='--',
               label=f'chance ({chance_level:.0%})')
    ax.set_xticks(x_base)
    ax.set_xticklabels(mods, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f'proportion of neurons with p_perm < {alpha}')
    ax.set_ylim(0, max(0.3, 1.15 * max_prop))
    reg_str = f"  |  reg alpha={reg_alpha}" if reg_alpha is not None else ""
    ax.set_title(
        f"Significant-neuron proportion per (ROI × model){reg_str}  |  "
        f"alpha={alpha}  |  "
        f"* = binomial test p<{alpha} vs chance ({chance_level:.0%})",
        fontsize=10, fontweight='bold',
    )
    ax.legend(fontsize=9, loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_dsr_coef_matrix(X_slice, coefs, neuron_label, model_name, save_path,
                         n_phases=3, n_clocks_per_phase=12,
                         fold_r=None, p_perm=None,
                         phase_names=('early', 'middle', 'late')):
    """Coefficient-weighted regressor matrix for DSR-family models.

    Each row is one regressor (organised as location × phase × clock-position)
    scaled by its coefficient from the best held-out fold. White = 0, dark =
    large coefficient × regressor value, so rows with high coefficients stand
    out from the white background.

    Y-axis ticks/labels are placed every `n_clocks_per_phase` rows
    ("<loc> <phase>"). Thin dotted lines separate phases (every
    `n_clocks_per_phase` rows); thicker solid lines separate locations
    (every `n_phases * n_clocks_per_phase` rows).

    Parameters
    ----------
    X_slice : (P, T) ndarray
        Slice of the design matrix to display (typically the best fold's
        held-out config; gives one trial's worth of time bins).
    coefs : (P,) ndarray
        Coefficient vector from the best held-out fold for this model.
    neuron_label, model_name : str
    save_path : str
    n_phases, n_clocks_per_phase : int
        Layout of the regressor blocks.
    fold_r, p_perm : float or None
        Optional metadata for the title.
    phase_names : tuple of str
    """
    X_slice = np.asarray(X_slice, dtype=float)
    coefs   = np.asarray(coefs, dtype=float)
    P, T    = X_slice.shape
    rows_per_loc = n_phases * n_clocks_per_phase
    n_locs       = max(1, P // rows_per_loc)

    weighted = X_slice * coefs[:, None]
    vmax = float(np.max(weighted)) if weighted.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.022 * P + 1.5)))
    im = ax.imshow(weighted, aspect='auto', cmap='Greys',
                   vmin=0, vmax=vmax, interpolation='nearest')

    y_ticks, y_labels = [], []
    for loc_idx in range(n_locs):
        for phase_idx in range(n_phases):
            r = loc_idx * rows_per_loc + phase_idx * n_clocks_per_phase
            if r < P:
                y_ticks.append(r)
                y_labels.append(f'{loc_idx + 1} {phase_names[phase_idx]}')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)

    for r in range(n_clocks_per_phase, P, n_clocks_per_phase):
        if r % rows_per_loc == 0:
            ax.axhline(r - 0.5, color='0.15', lw=1.2)
        else:
            ax.axhline(r - 0.5, color='0.5', lw=0.5, ls=':')

    ax.set_xlabel('time bin (held-out config)', fontsize=9)
    title = f'{neuron_label} – {model_name}'
    if fold_r is not None and p_perm is not None:
        title += f'   |   best-fold r={fold_r:.3f}, p_perm={p_perm:.1e}'
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label='regressor × coef')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return None


# ── Glass-brain / 3-D mesh plots of significant cells ───────────────────

def _roi_color_map(rois):
    """Stable color per ROI using tab20 (matches cell_to_roi_MNI.py style)."""
    rois = sorted([r for r in rois if r])
    cmap = plt.get_cmap('tab20', max(len(rois), 1))
    return {r: mcolors.to_hex(cmap(i)) for i, r in enumerate(rois)}


def _effect_to_alpha(effect, alpha_min=0.25, alpha_max=1.0):
    """Map |effect size| → alpha in [alpha_min, alpha_max].

    Uses absolute value so negative effects also stand out.
    Empty / all-zero input returns alpha_max for every entry.
    """
    e = np.abs(np.asarray(effect, dtype=float))
    e = np.nan_to_num(e, nan=0.0)
    if e.size == 0 or e.max() <= 0:
        return np.full(e.shape, alpha_max)
    return alpha_min + (alpha_max - alpha_min) * (e / e.max())


def plot_significant_cells_glassbrain(
    cells_df, model_name, save_path=None,
    p_col='p_perm', effect_col='mean_r', roi_col='final_roi',
    coord_cols=('MNI_x', 'MNI_y', 'MNI_z'),
    p_threshold=0.1, marker_size=22, title_suffix=None,
):
    """Glass-brain plot of significant cells for one model.

    Parameters
    ----------
    cells_df : pd.DataFrame
        One row per neuron with at least:
        ``p_col``, ``effect_col``, ``roi_col`` and the three ``coord_cols``.
    model_name : str
        Used in the figure title and (if `save_path` is None) returned.
    save_path : str or None
        File to save to.
    p_threshold : float
        Cells with `p_col` < threshold are plotted; others ignored.
    marker_size : int
    title_suffix : str or None
        Extra string appended to the title.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        ``None`` if there are no significant cells.
    """
    from nilearn import plotting as nlplot

    df = cells_df.copy()
    df = df.dropna(subset=[p_col, *coord_cols])
    sig = df[df[p_col] < p_threshold].copy()
    if sig.empty:
        print(f"  [glassbrain] no cells with {p_col} < {p_threshold} "
              f"for model {model_name}; skipping.")
        return None

    rois = sig[roi_col].fillna('unknown').astype(str).tolist()
    roi_colors = _roi_color_map(set(rois))
    base_rgba = np.array([
        mcolors.to_rgba(roi_colors.get(r, '#666666')) for r in rois
    ])
    alphas = _effect_to_alpha(sig[effect_col].to_numpy())
    base_rgba[:, 3] = alphas
    coords = sig[list(coord_cols)].to_numpy(dtype=float)

    title = f'Significant cells — {model_name}  |  {p_col}<{p_threshold}  ' \
            f'(n={len(sig)})'
    if title_suffix:
        title += f'  |  {title_suffix}'

    # Cells not passing the threshold — drawn as a faint grey backdrop so
    # the figure shows where we actually recorded, regardless of significance.
    nonsig = df[df[p_col] >= p_threshold]

    display = nlplot.plot_glass_brain(
        None, display_mode='lyrz', title=title,
        black_bg=False, plot_abs=False,
    )
    if not nonsig.empty:
        # nilearn's add_markers expects either a single matplotlib colour
        # string or an (n, 4) RGBA array; a bare 4-tuple gets interpreted
        # as 4 colours and crashes.  Tile the grey RGBA per marker.
        n_nonsig = len(nonsig)
        grey_rgba = np.tile(
            np.array([0.55, 0.55, 0.55, 0.25], dtype=float),
            (n_nonsig, 1),
        )
        display.add_markers(
            nonsig[list(coord_cols)].to_numpy(dtype=float),
            marker_color=grey_rgba,
            marker_size=max(4, int(marker_size * 0.4)),
        )
    display.add_markers(
        coords, marker_color=base_rgba, marker_size=marker_size,
    )

    fig = plt.gcf()
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', markersize=6,
                   markerfacecolor=col, markeredgecolor=col, label=roi)
        for roi, col in roi_colors.items()
    ]
    if not nonsig.empty:
        handles.append(plt.Line2D(
            [0], [0], marker='o', linestyle='', markersize=5,
            markerfacecolor=(0.55, 0.55, 0.55, 0.6),
            markeredgecolor='none',
            label=f'all other cells (n={len(nonsig)})',
        ))
    if handles:
        fig.legend(handles=handles, loc='lower center',
                   ncol=min(len(handles), 6), frameon=False, fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [glassbrain] saved → {save_path}")
    return fig


def plot_significant_cells_mesh3d(
    cells_df, model_name, save_path=None,
    p_col='p_perm', effect_col='mean_r', roi_col='final_roi',
    coord_cols=('MNI_x', 'MNI_y', 'MNI_z'),
    p_threshold=0.1, marker_size=4, title_suffix=None,
    brain_mask_img=None, mesh_step_size=2,
):
    """Interactive 3D whole-brain mesh with significant cells overlaid.

    Builds a translucent grey isosurface from a whole-brain mask
    (nilearn MNI152 template by default) and plots one Scatter3d point
    per cell with `p_col` < `p_threshold`, coloured by ROI and with
    opacity scaled by ``|effect_col|``.

    Requires ``plotly`` and ``skimage.measure.marching_cubes``.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
    """
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes
    import nibabel as nib

    df = cells_df.copy()
    df = df.dropna(subset=[p_col, *coord_cols])
    sig = df[df[p_col] < p_threshold].copy()
    if sig.empty:
        print(f"  [mesh3d] no cells with {p_col} < {p_threshold} "
              f"for model {model_name}; skipping.")
        return None

    if brain_mask_img is None:
        from nilearn import datasets as nldatasets
        from nilearn.image import load_img
        brain_mask_img = load_img(nldatasets.load_mni152_brain_mask())

    data = brain_mask_img.get_fdata() > 0
    data = np.pad(data.astype(float), pad_width=1, mode='constant')
    affine = brain_mask_img.affine.copy()
    affine[:3, 3] -= brain_mask_img.affine[:3, :3].sum(axis=1)
    verts, faces, _, _ = marching_cubes(data, level=0.5,
                                        step_size=mesh_step_size)
    verts_mni = nib.affines.apply_affine(affine, verts)
    x, y, z = verts_mni.T
    i, j, k = faces.T

    rois = sig[roi_col].fillna('unknown').astype(str).tolist()
    roi_colors = _roi_color_map(set(rois))
    point_colors = [roi_colors.get(r, '#666666') for r in rois]
    alphas = _effect_to_alpha(sig[effect_col].to_numpy())

    hover = (
        'roi: ' + sig[roi_col].astype(str)
        + '<br>' + effect_col + '=' + sig[effect_col].round(3).astype(str)
        + '<br>' + p_col + '=' + sig[p_col].round(4).astype(str)
        + '<br>MNI: '
        + sig[coord_cols[0]].round(1).astype(str) + ', '
        + sig[coord_cols[1]].round(1).astype(str) + ', '
        + sig[coord_cols[2]].round(1).astype(str)
    )
    if 'neuron' in sig.columns:
        hover = 'neuron: ' + sig['neuron'].astype(str) + '<br>' + hover

    # Faint-grey backdrop of every recorded cell that did NOT pass the
    # threshold, so the figure also shows where we recorded overall.
    nonsig = df[df[p_col] >= p_threshold]

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        opacity=0.12, color='lightgray',
        name='brain', hoverinfo='skip',
    ))
    if not nonsig.empty:
        fig.add_trace(go.Scatter3d(
            x=nonsig[coord_cols[0]],
            y=nonsig[coord_cols[1]],
            z=nonsig[coord_cols[2]],
            mode='markers',
            marker=dict(size=max(2, marker_size * 0.55),
                        color='lightgrey', opacity=0.35,
                        line=dict(width=0)),
            name=f'other cells (n={len(nonsig)})',
            hoverinfo='skip',
        ))
    fig.add_trace(go.Scatter3d(
        x=sig[coord_cols[0]], y=sig[coord_cols[1]], z=sig[coord_cols[2]],
        mode='markers',
        marker=dict(size=marker_size, color=point_colors,
                    opacity=1.0, line=dict(width=0)),
        # Plotly Scatter3d does not support per-point opacity, so we encode
        # effect size in marker size instead, capped to a reasonable range.
        hovertext=hover, hoverinfo='text', name='significant cells',
    ))
    # Scale marker size by effect: bigger = stronger effect.
    sizes = marker_size * (0.6 + 0.8 * alphas)
    fig.data[-1].marker.size = sizes.tolist()

    title = f'Significant cells (3D) — {model_name}  |  ' \
            f'{p_col}<{p_threshold}  (n={len(sig)})'
    if title_suffix:
        title += f'  |  {title_suffix}'
    fig.update_layout(
        title=title, width=950, height=800,
        scene=dict(xaxis_title='MNI x', yaxis_title='MNI y',
                   zaxis_title='MNI z', aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if save_path is not None:
        fig.write_html(save_path)
        print(f"  [mesh3d] saved → {save_path}")
    return fig


def plot_roi_electrodes_glassbrain(
    electrodes_per_roi, save_path=None,
    title='ROI electrode locations',
    marker_size=20, per_roi_panels=True,
):
    """Schematic glass-brain of electrode locations grouped by ROI.

    Parameters
    ----------
    electrodes_per_roi : dict[str, ndarray]
        ``{roi_name: (n, 3) MNI coords}``.  ROIs with zero electrodes are
        skipped silently.
    save_path : str or None
    title : str
        Figure suptitle for the per-ROI panel plot, or main title for the
        combined view.
    marker_size : int
    per_roi_panels : bool
        If True (default), draw one small glass-brain per ROI in a grid.
        If False, draw all electrodes on a single glass-brain coloured by
        ROI.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    from nilearn import plotting as nlplot

    coords_per_roi = {}
    for roi, arr in electrodes_per_roi.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] == 3:
            coords_per_roi[roi] = a
    if not coords_per_roi:
        print("  [roi-electrodes] no electrodes to plot.")
        return None

    roi_colors = _roi_color_map(coords_per_roi.keys())

    if not per_roi_panels:
        # Single combined glass-brain.
        display = nlplot.plot_glass_brain(
            None, display_mode='lyrz', title=title,
            black_bg=False, plot_abs=False,
        )
        for roi, coords in coords_per_roi.items():
            display.add_markers(coords,
                                marker_color=roi_colors[roi],
                                marker_size=marker_size)
        fig = plt.gcf()
        handles = [
            plt.Line2D([0], [0], marker='o', linestyle='', markersize=6,
                       markerfacecolor=col, markeredgecolor=col,
                       label=f'{r} (n={len(coords_per_roi[r])})')
            for r, col in roi_colors.items()
        ]
        fig.legend(handles=handles, loc='lower center',
                   ncol=min(len(handles), 5), frameon=False, fontsize=8)
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  [roi-electrodes] saved → {save_path}")
        return fig

    # Multi-panel: one mini glass-brain per ROI.
    rois = list(coords_per_roi)
    n = len(rois)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.2, nrows * 2.4),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=11, weight='bold')

    for ax_idx, roi in enumerate(rois):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        coords = coords_per_roi[roi]
        display = nlplot.plot_glass_brain(
            None, display_mode='z', axes=ax,
            black_bg=False, plot_abs=False,
        )
        display.add_markers(coords,
                            marker_color=roi_colors[roi],
                            marker_size=marker_size * 0.6)
        ax.set_title(f'{roi} (n={len(coords)})', fontsize=9)

    # Hide unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [roi-electrodes] saved → {save_path}")
    return fig