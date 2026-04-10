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


# ── Main plotting function ─────────────────────────────────────────────

def plot_rsa_heatmap(
    results: dict,
    models: list[str],
    rois: list[str],
    title: str = 'DSR RSA results',
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a summary heatmap of RSA t-values for single and combined models.

    Parameters
    ----------
    results : dict
        Nested dict: results[roi]['unique'][model] = (t, beta, p)
                     results[roi]['combined']['t']    shape (n_models,)
                     results[roi]['combined']['p']    shape (n_models,)
                     results[roi]['n_neurons']        int
    models : list of str
        Model names in the order they appear in the combined regression.
    rois : list of str
        ROI names to include (must be keys in results).
    title : str
        Figure suptitle.
    save_path : str or None
        If given, save figure to this path.

    Returns
    -------
    fig : plt.Figure
    """
    present_rois = [r for r in rois if r in results]
    n_rois   = len(present_rois)
    n_models = len(models)

    # ── collect arrays ─────────────────────────────────────────────────
    unique_t = np.full((n_models, n_rois), np.nan)
    unique_p = np.full((n_models, n_rois), np.nan)
    combo_t  = np.full((n_models, n_rois), np.nan)
    combo_p  = np.full((n_models, n_rois), np.nan)
    n_neurons = []

    for r_idx, roi in enumerate(present_rois):
        n_neurons.append(results[roi].get('n_neurons', 0))
        for m_idx, m in enumerate(models):
            u = results[roi].get('unique', {}).get(m)
            if u is not None:
                unique_t[m_idx, r_idx] = float(u[0])
                unique_p[m_idx, r_idx] = float(u[2])
        combo = results[roi].get('combined')
        if combo is not None:
            for m_idx in range(n_models):
                combo_t[m_idx, r_idx] = float(combo['t'][m_idx])
                combo_p[m_idx, r_idx] = float(combo['p'][m_idx])

    # ── figure layout ──────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(10, n_rois * 1.3 + 3), max(4, n_models * 0.9 + 2)),
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=13, weight='bold')

    panel_data  = [unique_t,  combo_t]
    panel_p     = [unique_p,  combo_p]
    panel_titles = ['Unique regression (per model)', 'Combined regression (all models)']

    abs_max = np.nanmax(np.abs(np.concatenate([unique_t, combo_t])))
    vmax = max(abs_max, 1.0)

    for ax, t_mat, p_mat, ptitle in zip(axes, panel_data, panel_p, panel_titles):
        im = ax.imshow(
            t_mat,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            interpolation='nearest',
        )

        # annotate each cell
        for m_idx in range(n_models):
            for r_idx in range(n_rois):
                t_val = t_mat[m_idx, r_idx]
                p_val = p_mat[m_idx, r_idx]
                if np.isnan(t_val):
                    continue
                stars = p_to_stars(p_val)
                # pick text colour for readability
                norm_val = (t_val + vmax) / (2 * vmax)
                bg_lum = 0.3 + 0.4 * norm_val  # rough luminance proxy
                txt_color = 'white' if abs(norm_val - 0.5) > 0.25 else 'black'
                ax.text(
                    r_idx, m_idx,
                    f'{t_val:.2f}{stars}',
                    ha='center', va='center',
                    fontsize=7.5, color=txt_color,
                )

        roi_labels = [
            f'{r}\n(n={results[r]["n_neurons"]})' for r in present_rois
        ]
        ax.set_xticks(range(n_rois))
        ax.set_xticklabels(roi_labels, rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(ptitle, fontsize=10, pad=6)

        plt.colorbar(im, ax=ax, shrink=0.7, label='t-value')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')

    return fig
