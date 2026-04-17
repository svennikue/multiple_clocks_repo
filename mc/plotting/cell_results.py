#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting helpers for human single-cell RSA results.

Functions
---------
p_to_stars(p)
plot_rsa_heatmap(results, models, rois, title, save_path)
    Heatmap of t-values per model × ROI, with significance stars.
    Two panels: (1) unique regression per model, (2) combined regression.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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