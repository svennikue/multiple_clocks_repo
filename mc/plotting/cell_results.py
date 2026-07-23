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

Brain plots (lazy-import nilearn / plotly / nibabel inside the function):
plot_peaklag_glassbrain(coords, lags, save_path, title, n_lags, ...)
    Glass-brain marker scatter coloured by an integer 'peak lag'.
plot_peaklag_3d_mesh(coords, lags, save_path, title, n_lags, ...)
    Interactive 3D plotly scatter on a translucent MNI brain mesh (HTML).
"""
from __future__ import annotations

import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats


# Canonical orderings used by the shared heatmap/histogram/bar helpers
# so RSA and encoding analyses are visually comparable. Override with the
# ``roi_order`` / ``model_order`` kwargs if you want a non-default layout.
# ACC sits at the top-left of the heatmap, DSR is the left-most column.
CANONICAL_ROI_ORDER = [
    'ACC',
    'medial_CC',
    'HC_anterior',
    'HC_mid',
    'HC_posterior',
    'EC',
    'Parahippocampal',
    'PCC',
    'posterior_CC',
    'medialOFC',
    'OFC11',
    'OFC13',
    'ventral_ACC',
    'Visual',
]

# Encoding-analysis model names (encoding_analysis_simple.py).
CANONICAL_ENC_MODEL_ORDER = [
    'dsr',
    'dsr_only_fut',
    'dsr_now_next',
    'state',
    'state_phase',
    'phase',
    'midnight',
    'location',
    'bttn_curr',
    'bttn_prev',
    'bttn_next',
    'uncover',
]

# RSA model names (RSA_DSR_ROIs_simple.py uses 'dsr_old' rather than 'dsr').
CANONICAL_RSA_MODEL_ORDER = [
    'dsr_old',
    'dsr_fmri',
    'dsr_old_now_next',
    'state',
    'state_phase',
    'phase',
    'midnight',
    'repeat_counter',
    'location',
    'bttn_curr',
    'bttn_prev',
    'bttn_next',
    'uncover',
]


# ── Project-wide colour conventions (see CLAUDE.md → 🎨 section) ──────
# A single source of truth so every figure uses the same colour for the
# same categorical variable. Override only with explicit reason.
STATE_QUADRANT_COLORS = ('#F15A29', '#F7931E', '#C7C6E2', '#6B60AA')  # A,B,C,D

# Phases (early, middle, late) — pastel pink → bordeaux ramp.
PHASE_COLORS = ('#FCDDE3', '#D7657F', '#5C1027')

# Locations on the 3×3 grid (1…9) — dark teal top-left → light green bottom-right.
LOCATION_COLORS = (
    '#0a607a', '#7eb1c4', '#b6d4e0',
    '#175e62', '#5b9b8d', '#c8e0d0',
    '#0e3d3a', '#3d8b7d', '#a7d9b2',
)

# ROIs — fixed era_brewer "Showgirl2" assignment so the same ROI keeps
# the same hue across every figure in the project. Order taken from
# CLAUDE.md. ROIs not in this dict fall back to era_brewer overflow
# colours in _roi_color_map.
ROI_COLORS_SHOWGIRL2 = {
    # Canonical CLAUDE.md mapping (corrected 2026-06; previous indices for
    # medialOFC/ACC/HC_anterior/HC_mid/PCC did not match the project spec).
    'EC':              0,
    'ACC':             1,
    'HC_anterior':     2,
    'PCC':             3,
    'medialOFC':       4,
    'Parahippocampal': 5,
    'HC_mid':          6,
}

# Colours that sit outside the Showgirl2 palette (for ROIs added later —
# ``get_roi_colour`` returns these when the ROI isn't in the palette dict).
_EXTRA_ROI_COLORS = {
    'Precuneus': '#5C1027',    # bordeaux (added 2026-07-23)
}


def get_roi_colour(roi):
    """Look up an ROI colour, falling back to the palette index if defined."""
    idx = ROI_COLORS_SHOWGIRL2.get(roi)
    if idx is not None:
        return SHOWGIRL2_DISCRETE[idx]
    if roi in _EXTRA_ROI_COLORS:
        return _EXTRA_ROI_COLORS[roi]
    return '#888888'

# Canonical Showgirl2 palette as returned by ``era_brewer.era_brew(
# 'Showgirl2', n=7)`` — single source of truth for the project palette.
# Hardcoded to (a) avoid re-importing era_brewer everywhere and (b) sit
# next to the CLAUDE.md ROI index map so the colours can be inspected
# at a glance.  Verified against era_brewer 2026-06-29.
#
# IMPORTANT: era_brewer's n=7 interpolation gives indices 3 and 6 the
# same value (#C1DCBF), so PCC (idx 3) and HC_mid (idx 6) would visually
# collide if rendered side by side. ``HC_mid`` is therefore promoted
# to a darker green sampled from the higher-n gradient so the two are
# distinguishable in figures.
SHOWGIRL2_DISCRETE = [
    '#B74C2D',   # 0  EC               — dark red
    '#448363',   # 1  ACC / mPFC       — dark teal-green
    '#CCB178',   # 2  HC_anterior      — tan
    '#C1DCBF',   # 3  PCC              — pale green
    '#DC673E',   # 4  medialOFC        — red (orange-red)
    '#7BB594',   # 5  Parahippocampal  — sage
    '#629E7E',   # 6  HC_mid           — mid-dark green
    #                                    (override of era_brewer's idx 6
    #                                     duplicate of idx 3, sampled
    #                                     from era_brew('Showgirl2', n=12)
    #                                     to stay on the same gradient)
]


# Display-name overrides for ROI strings used in plotting helpers.
# The ROI *key* in the data files stays unchanged (e.g. 'ACC') — this
# only renames the label that appears on axes, legends, titles, etc.
# Edit here to propagate the rename everywhere that goes through the
# shared helpers.
ROI_DISPLAY_NAMES = {
    'ACC': 'mPFC',
}


def roi_display(name):
    """Return the display label for an ROI key, falling back to itself."""
    return ROI_DISPLAY_NAMES.get(name, name)

# Dark green used for the empirical / observed-value vertical line on
# permutation-null histograms (matches LOCATION_COLORS[6]).
OBSERVED_VALUE_COLOR = '#0e3d3a'


def _order_keep_present(canonical, present):
    """Reorder `present` using `canonical` as a priority list.

    Items missing from `canonical` are appended in their original order so
    nothing is silently dropped.
    """
    seen = set()
    out = [x for x in canonical if x in present and not (x in seen or seen.add(x))]
    # out.extend(x for x in present if x not in seen)
    return out


def _wrap_label(label, width=10):
    """Wrap a label to multi-line if longer than `width` characters.

    Underscores are treated as soft break points (e.g. ``HC_anterior``
    becomes ``HC\nanterior`` rather than being squashed).
    """
    if label is None:
        return ''
    s = str(label)
    if len(s) <= width:
        return s
    if '_' in s:
        parts = s.split('_')
        # Greedy newline insertion to keep lines roughly under `width`.
        lines, cur = [], parts[0]
        for p in parts[1:]:
            if len(cur) + 1 + len(p) <= width:
                cur = cur + '_' + p
            else:
                lines.append(cur)
                cur = p
        lines.append(cur)
        return '\n'.join(lines)
    return '\n'.join(textwrap.wrap(s, width=width, break_long_words=False))


def _pval_to_stars(p, alphas=(0.05, 0.01, 0.001)):
    """Return '*'/'**'/'***'/'' for the given p-value thresholds (ascending)."""
    if not np.isfinite(p):
        return ''
    star = ''
    for a in alphas:
        if p < a:
            star += '*'
    return star


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
                                  reg_alpha=None, use_fdr=True,
                                  legend_outside=True,
                                  rois=None, model_order=None,
                                  base_fontsize=12):
    """Grouped bar plot: proportion of significant neurons per (ROI, model).

    One-sided binomial test vs `chance_level`. If `use_fdr=True` the
    binomial p-values are BH-adjusted across all (ROI × model) cells and
    stars reflect the adjusted q: 0.05=*, 0.01=**, 0.001=***. Legend is
    placed outside the axes by default.
    """
    if results_df.empty:
        return None
    rdf = results_df.dropna(subset=['mean_r']).copy()
    present_rois = set(rdf['roi'].unique())
    present_models = set(rdf['model'].unique())
    rois_order = _order_keep_present(
        rois if rois is not None else CANONICAL_ROI_ORDER, present_rois)
    mods = _order_keep_present(
        model_order if model_order is not None else models, present_models)
    if not rois_order or not mods:
        return None

    # First pass: collect props, n_total, raw binomial p (per (roi, model)).
    props = np.zeros((len(rois_order), len(mods)))
    n_total_mat = np.zeros_like(props, dtype=int)
    n_sig_mat = np.zeros_like(props, dtype=int)
    pbin_mat = np.full_like(props, np.nan)
    for i, roi in enumerate(rois_order):
        for j, m in enumerate(mods):
            sub = rdf[(rdf['roi'] == roi) & (rdf['model'] == m)]
            n_total = int(len(sub))
            n_sig = int((sub['p_perm'] < alpha).sum())
            n_total_mat[i, j] = n_total
            n_sig_mat[i, j] = n_sig
            props[i, j] = (n_sig / n_total) if n_total else 0.0
            if n_total > 0:
                bt = stats.binomtest(n_sig, n_total, p=chance_level,
                                     alternative='greater')
                pbin_mat[i, j] = bt.pvalue

    # Stars use either raw or FDR-adjusted binomial p.
    if use_fdr:
        flat = pbin_mat.flatten()
        flat_q = bh_fdr(flat)
        p_for_stars = flat_q.reshape(pbin_mat.shape)
        sig_label = 'q (BH-FDR)'
    else:
        p_for_stars = pbin_mat
        sig_label = 'p (binomial vs chance)'

    n_models = len(mods)
    n_rois = len(rois_order)
    width = 0.8 / max(n_rois, 1)
    x_base = np.arange(n_models)

    fig, ax = plt.subplots(
        figsize=(max(9, n_models * 1.1 + 3.5), 5.0),
        constrained_layout=True,
    )
    cmap = plt.get_cmap('Set3')

    max_h = 0.0
    for i, roi in enumerate(rois_order):
        positions = x_base - 0.4 + width / 2 + i * width
        n_max = int(n_total_mat[i].max()) if n_total_mat[i].size else 0
        bars = ax.bar(positions, props[i], width=width,
                      color=cmap(i),
                      label=f'{roi} (N={n_max})',
                      edgecolor='black', linewidth=0.6)
        max_h = max(max_h, float(props[i].max()) if props[i].size else 0.0)
        for j, bar in enumerate(bars):
            stars = _pval_to_stars(p_for_stars[i, j])
            if stars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005, stars,
                        ha='center', va='bottom',
                        fontsize=base_fontsize + 2, fontweight='bold')

    ax.axhline(chance_level, color='black', lw=0.8, ls='--',
               label=f'chance ({chance_level:.0%})')
    ax.set_xticks(x_base)
    ax.set_xticklabels(mods, rotation=30, ha='right',
                       fontsize=base_fontsize)
    ax.set_ylabel(f'proportion of neurons with p_perm < {alpha}',
                  fontsize=base_fontsize)
    ax.set_ylim(0, max(0.3, 1.15 * max_h))
    reg_str = f"  |  reg alpha={reg_alpha}" if reg_alpha is not None else ""
    ax.set_title(
        f"Significant-neuron proportion per (ROI × model){reg_str}\n"
        f"stars = {sig_label}:  * <0.05   ** <0.01   *** <0.001",
        fontsize=base_fontsize + 1, fontweight='bold',
    )

    if legend_outside:
        ax.legend(fontsize=base_fontsize - 1,
                  loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  frameon=False, borderaxespad=0.0)
    else:
        ax.legend(fontsize=base_fontsize - 1, loc='upper right',
                  frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        base, _ = os.path.splitext(save_path)
        fig.savefig(base + '.svg', bbox_inches='tight')
    return fig


def plot_significance_counts_bars(results_df, models, save_path=None,
                                   alpha=0.05, chance_level=0.05,
                                   rois=None, model_order=None,
                                   use_fdr=True, base_fontsize=13,
                                   panel_w=4.2, panel_h=4.4):
    """Per-ROI grey total + coloured significant counts, one panel per model.

    Background grey bar = total number of cells in that ROI (for this
    model). Foreground coloured bar = number with `p_perm < alpha`.
    Stars use BH-corrected binomial p (or raw p if `use_fdr=False`):
    0.05=*, 0.01=**, 0.001=***. Saves to PNG + SVG + PDF.
    """
    if results_df is None or results_df.empty:
        return None
    rdf = results_df.dropna(subset=['mean_r']).copy()
    present_rois = set(rdf['roi'].unique())
    present_models = set(rdf['model'].unique())
    rois_order = _order_keep_present(
        rois if rois is not None else CANONICAL_ROI_ORDER, present_rois)
    mods = _order_keep_present(
        model_order if model_order is not None else models, present_models)
    if not rois_order or not mods:
        return None
    # import pdb; pdb.set_trace()
    # Gather counts + raw binomial p per (roi, model).
    n_total = np.zeros((len(rois_order), len(mods)), dtype=int)
    n_sig = np.zeros_like(n_total)
    pbin = np.full(n_total.shape, np.nan, dtype=float)
    for i, roi in enumerate(rois_order):
        for j, m in enumerate(mods):
            sub = rdf[(rdf['roi'] == roi) & (rdf['model'] == m)]
            n_total[i, j] = int(len(sub))
            n_sig[i, j] = int((sub['p_perm'] < alpha).sum())
            if n_total[i, j] > 0:
                bt = stats.binomtest(int(n_sig[i, j]), int(n_total[i, j]),
                                     p=chance_level, alternative='greater')
                pbin[i, j] = bt.pvalue
    if use_fdr:
        p_for_stars = bh_fdr(pbin.flatten()).reshape(pbin.shape)
        sig_label = 'q (BH-FDR)'
    else:
        p_for_stars = pbin
        sig_label = 'p (binomial)'

    n_models = len(mods)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(panel_w * n_models + 1.0, panel_h),
        constrained_layout=True, sharey=False,
    )
    if n_models == 1:
        axes = [axes]

    cmap = plt.get_cmap('Set2')
    x = np.arange(len(rois_order))

    for j, m in enumerate(mods):
        ax = axes[j]
        # Background grey bar = total per ROI.
        ax.bar(x, n_total[:, j], width=0.78,
               color='0.82', edgecolor='0.5', linewidth=0.8,
               label='total cells')
        # Foreground coloured bar = significant per ROI.
        ax.bar(x, n_sig[:, j], width=0.78,
               color=cmap(j), edgecolor='black', linewidth=0.7,
               label=f'p_perm<{alpha:g}')
        # Star annotations.
        for i in range(len(rois_order)):
            stars = _pval_to_stars(p_for_stars[i, j])
            if stars:
                h = max(n_total[i, j], 1)
                ax.text(x[i], h + max(n_total[:, j].max() * 0.02, 0.5),
                        stars,
                        ha='center', va='bottom',
                        fontsize=base_fontsize + 2,
                        fontweight='bold')

        ax.set_title(m, fontsize=base_fontsize + 2, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([_wrap_label(r, width=11) for r in rois_order],
                           rotation=35, ha='right',
                           fontsize=base_fontsize)
        ax.set_ylabel('number of cells', fontsize=base_fontsize + 1)
        ax.tick_params(axis='y', labelsize=base_fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(n_total[:, j].max() * 1.20, 1))

    # Shared legend (right-of-figure).
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper left', bbox_to_anchor=(1.0, 0.95),
               fontsize=base_fontsize, frameon=False)

    fig.suptitle(
        f'Significant-cell counts per ROI (one panel per model)\n'
        f'stars = {sig_label}:  * <0.05   ** <0.01   *** <0.001',
        fontsize=base_fontsize + 2, fontweight='bold',
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        base, _ = os.path.splitext(save_path)
        fig.savefig(base + '.svg', bbox_inches='tight')
        fig.savefig(base + '.pdf', bbox_inches='tight')
        print(f'  [sig-counts-bars] saved → {save_path} (+ .svg, .pdf)')
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


def make_brain_anatomy_figure(mesh_step=3):
    """Plotly Figure with the standard anatomical background — a transparent
    whole-brain shell, with the hippocampus and ACC shaded grey and the
    entorhinal cortex shaded dark grey inside it (matches the plotly brains
    in scripts/cell_to_roi_MNI.py).

    Build it once and reuse via the `bg_fig` argument of plot_peaklag_3d_mesh.
    """
    import plotly.graph_objects as go
    import nibabel as nib
    from nilearn import datasets
    from skimage.measure import marching_cubes

    def _mesh(binary, affine, step):
        binary = np.asarray(binary)
        if binary.sum() == 0:
            return None
        v, f, _, _ = marching_cubes(binary.astype(float), level=0.5,
                                    step_size=step)
        x, y, z = nib.affines.apply_affine(affine, v).T
        i, j, k = f.T
        return x, y, z, i, j, k

    def _atlas_mask(atlas, patterns):
        img = atlas.maps
        img = nib.load(img) if isinstance(img, str) else img
        idxs = [n for n, lab in enumerate(atlas.labels)
                if any(p.lower() in str(lab).lower() for p in patterns)]
        return np.isin(img.get_fdata(), idxs), img.affine

    fig = go.Figure()
    try:
        bm = datasets.load_mni152_brain_mask()
        m = _mesh(bm.get_fdata() > 0, bm.affine, mesh_step)
        if m is not None:
            x, y, z, i, j, k = m
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                          color='lightgray', opacity=0.06, name='brain',
                          hoverinfo='skip', showlegend=True))
    except Exception as exc:
        print(f"  brain shell unavailable ({exc}).")

    try:
        ho_sub  = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
        ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        juelich = datasets.fetch_atlas_juelich('maxprob-thr25-2mm')
        for atlas, patterns, color, opacity, name in [
            (ho_sub,  ['hippocampus'],                        'gray',    0.30, 'hippocampus'),
            (juelich, ['entorhinal'],                         'dimgray', 0.50, 'EC'),
            (ho_cort, ['cingulate gyrus, anterior division'], 'gray',    0.30, 'ACC'),
        ]:
            mask, affine = _atlas_mask(atlas, patterns)
            m = _mesh(mask, affine, 1)
            if m is None:
                continue
            x, y, z, i, j, k = m
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                          color=color, opacity=opacity, name=name,
                          hoverinfo='skip', showlegend=True))
    except Exception as exc:
        print(f"  anatomy meshes unavailable ({exc}).")
    return fig


def _strength_to_alpha(strengths, n, lo=0.15, hi=1.0):
    """Map a per-marker effect-strength array to opacity in [lo, hi].

    Stronger effects -> more opaque. Returns an (n,) array; all `hi` when
    `strengths` is None or unusable. Negative strengths clip to 0.
    """
    if strengths is None:
        return np.full(n, hi)
    s = np.asarray(strengths, dtype=float)
    if s.size != n or not np.isfinite(s).any():
        return np.full(n, hi)
    s = np.clip(np.nan_to_num(s), 0.0, None)
    smax = float(np.nanmax(s))
    if smax <= 0:
        return np.full(n, hi)
    return lo + (hi - lo) * np.clip(s / smax, 0.0, 1.0)


def plot_peaklag_glassbrain(coords, lags, save_path, title,
                            strengths=None, n_lags=12, marker_size=4,
                            display_mode='ortho', cmap_name='YlOrRd'):
    """Glass-brain marker scatter coloured by an integer 'peak lag'.

    Parameters
    ----------
    coords : (N, 3) array of MNI x/y/z coordinates. Jitter duplicate
        coordinates beforehand if every cell should be individually visible.
    lags : (N,) array of integer lag values in 0 .. n_lags-1.
    save_path : str
    title : str
    strengths : (N,) array or None
        Per-neuron effect strength -> per-marker opacity (stronger = more
        opaque). None -> all markers fully opaque.
    n_lags : int
        Number of lag bins; sets the YlOrRd colour mapping (0 .. n_lags-1).
    marker_size : int
        Glass-brain marker size (small so individual cells are visible).
    display_mode : str
        nilearn glass-brain display mode ('ortho' = 3 orthogonal views).
    """
    from nilearn import plotting
    import matplotlib.colors as mcolors

    coords = np.asarray(coords, dtype=float)
    lags   = np.asarray(lags, dtype=float)
    if coords.shape[0] == 0:
        print(f"  skip {save_path}: no neurons.")
        return None

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=max(n_lags - 1, 1))
    alphas = _strength_to_alpha(strengths, coords.shape[0])
    # RGBA per marker: hue = peak lag, opacity = effect strength.
    marker_colors = [(*cmap(norm(v))[:3], float(a))
                     for v, a in zip(lags, alphas)]

    display = plotting.plot_glass_brain(
        None, display_mode=display_mode, title=title,
        black_bg=False, plot_abs=False)
    display.add_markers(coords, marker_color=marker_colors,
                        marker_size=marker_size)

    # Discrete legend: one colour per lag.
    fig = plt.gcf()
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=6,
                          markerfacecolor=cmap(norm(l)),
                          markeredgecolor='none', label=str(l))
               for l in range(n_lags)]
    fig.legend(handles=handles, loc='lower center', ncol=n_lags,
               frameon=False, fontsize=7, title='DSR peak lag',
               handletextpad=0.1, columnspacing=0.8)

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {save_path}")
    return None


def plot_peaklag_3d_mesh(coords, lags, save_path, title,
                         strengths=None, n_lags=12, marker_size=3,
                         cmap_name='YlOrRd', bg_fig=None):
    """Interactive 3D plotly scatter coloured by peak lag, on the standard
    anatomical brain background (transparent shell + grey hippocampus +
    dark-grey EC + grey ACC). Written to an HTML file.

    Parameters as in plot_peaklag_glassbrain.  `strengths` (one per neuron)
    maps to per-marker opacity.  `bg_fig` is a Figure from
    make_brain_anatomy_figure() — built on demand if not supplied (slow, so
    pass a pre-built one when calling repeatedly).
    """
    import plotly.graph_objects as go
    import matplotlib.colors as mcolors

    coords = np.asarray(coords, dtype=float)
    lags   = np.asarray(lags, dtype=float)
    if coords.shape[0] == 0:
        print(f"  skip {save_path}: no neurons.")
        return None

    base = bg_fig if bg_fig is not None else make_brain_anatomy_figure()
    fig = go.Figure(base)

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=max(n_lags - 1, 1))
    alphas = _strength_to_alpha(strengths, coords.shape[0])
    # Per-marker rgba: hue = peak lag, opacity = effect strength. (Plotly
    # Scatter3d has only a scalar marker.opacity, so opacity is baked in.)
    rgba = []
    for v, a in zip(lags, alphas):
        r, g, b, _ = cmap(norm(v))
        rgba.append(f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})')

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=marker_size, color=rgba),
        name='DSR neurons'))

    # Dummy trace carrying the lag colour bar (the scatter above uses baked-in
    # rgba so it cannot also render a numeric scale).
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode='markers',
        marker=dict(color=[0], colorscale=cmap_name, cmin=0,
                    cmax=max(n_lags - 1, 1), showscale=True,
                    colorbar=dict(title='DSR<br>peak lag', len=0.6)),
        showlegend=False, hoverinfo='skip', name=''))

    fig.update_layout(
        title=title, width=950, height=800,
        scene=dict(xaxis_title='MNI x', yaxis_title='MNI y',
                   zaxis_title='MNI z', aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(save_path)
    print(f"Wrote {save_path}")
    return None


# ── Glass-brain / 3-D mesh plots of significant cells ───────────────────

def _roi_color_map(rois):
    """Stable per-ROI colour from the canonical Showgirl2 palette
    (see CLAUDE.md / ``SHOWGIRL2_DISCRETE``).

    ROIs listed in ``ROI_COLORS_SHOWGIRL2`` use their fixed index in the
    7-colour palette. ROIs outside that mapping cycle through a small
    extra-shades list (alphabetical order for stability across runs).
    """
    rois = [r for r in rois if r]
    known = [r for r in rois if r in ROI_COLORS_SHOWGIRL2]
    extra = sorted(r for r in rois if r not in ROI_COLORS_SHOWGIRL2)
    out = {r: SHOWGIRL2_DISCRETE[ROI_COLORS_SHOWGIRL2[r]] for r in known}
    # A small overflow palette for ROIs not in the canonical 7. Distinct
    # neutral tones so they don't clash with the 7 mapped hues.
    extra_palette = ['#888888', '#bdbdbd', '#5C1027', '#0e3d3a',
                     '#7eb1c4', '#3d8b7d', '#a7d9b2']
    for i, r in enumerate(extra):
        out[r] = extra_palette[i % len(extra_palette)]
    return out


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
    roi_colors=None,
):
    """Schematic glass-brain of electrode locations grouped by ROI.

    Parameters
    ----------
    electrodes_per_roi : dict[str, ndarray]
        ``{roi_name: (n, 3) MNI coords}``.  ROIs with zero electrodes are
        skipped silently.
    save_path : str or None
        If the path ends in ``.pdf`` or ``.svg`` it is saved as a vector
        figure (preferred for publication panels). PNG/JPG also supported.
    title : str
        Figure suptitle for the per-ROI panel plot, or main title for the
        combined view.
    marker_size : int
    per_roi_panels : bool
        If True (default), draw one small glass-brain per ROI in a grid.
        If False, draw all electrodes on a single glass-brain coloured by
        ROI.
    roi_colors : dict[str, str] or None
        Optional override for the ROI -> colour mapping (e.g. to enforce
        the canonical CLAUDE.md palette from a caller-side dict). When
        None (default), falls back to :func:`_roi_color_map` which uses
        ``ROI_COLORS_SHOWGIRL2``. Any ROI missing from the provided dict
        is filled in from the default map.

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

    default_colors = _roi_color_map(coords_per_roi.keys())
    if roi_colors is None:
        roi_colors = default_colors
    else:
        # Caller-supplied colours win; fall back to the default for any ROI
        # the caller didn't list.
        roi_colors = {r: roi_colors.get(r, default_colors[r])
                       for r in coords_per_roi}

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
                       label=f'{roi_display(r)} (n={len(coords_per_roi[r])})')
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
        ax.set_title(f'{roi_display(roi)} (n={len(coords)})', fontsize=9)

    # Hide unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [roi-electrodes] saved → {save_path}")
    return fig


def plot_roi_beta_glassbrain(
    roi_betas,
    roi_pvals=None,
    only_rois=None,
    roi_label_column='alt_final_roi',
    brainnetome_nii=None,
    brainnetome_lut=None,
    vmax=None,
    cmap_name='RdBu_r',
    alpha_threshold=0.05,
    display_mode='lyrz',
    overlay_alpha=0.85,
    roi_cell_coords=None,
    cell_sphere_radius_mm=8.0,
    sig_outline_color='black',
    sig_outline_linewidth=2.0,
    title=None,
    save_path=None,
):
    """Glass-brain with each ROI shaded by a heatmap value.

    Mirrors the cell colours in
    :func:`plot_roi_model_heatmap` (RSA_DSR_ROIs_simple.py): symmetric
    RdBu_r centred on 0, with ``vmax = max(|beta|)`` across the plotted
    ROIs.  Each ROI mask is drawn as a single-colour overlay.

    Parameters
    ----------
    roi_betas : dict[str, float]
        ROI name -> beta (heatmap value).
    roi_pvals : dict[str, float] or None
        Optional ROI -> p-value.  ROIs with ``p < alpha_threshold`` are
        listed in a small footer annotation.
    only_rois : iterable[str] or None
        Restrict plotting to these ROIs (e.g. ROIs that actually have
        cells in the recording sample).
    roi_label_column : {'alt_final_roi', 'final_roi'}
        Selects the labelling scheme used to look up anatomical masks
        in :mod:`mc.plotting.roi_atlas`.
    brainnetome_nii, brainnetome_lut : str or None
        Paths to the Brainnetome atlas + LUT.  Defaults come from
        :mod:`mc.plotting.roi_atlas`.
    vmax : float or None
        Symmetric colour limit; defaults to ``max(|beta|)`` over plotted ROIs.
    cmap_name : str
    alpha_threshold : float
    display_mode : str
        Forwarded to nilearn's ``plot_glass_brain``.
    overlay_alpha : float
        Per-ROI overlay opacity (0..1).
    roi_cell_coords : dict[str, (n, 3) ndarray] or None
        Cell MNI positions per ROI. When supplied, each anatomical mask
        is intersected with the union of spheres of radius
        ``cell_sphere_radius_mm`` around that ROI's cells, so only the
        recorded part of a large region is shaded.
    cell_sphere_radius_mm : float or None
        Sphere radius in mm. ``None`` disables the restriction.
    sig_outline_color, sig_outline_linewidth :
        Contour drawn around ROIs whose ``p_perm < alpha_threshold``.
    title : str or None
    save_path : str or None
    """
    from nilearn import plotting as nlplot
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, Normalize
    from mc.plotting.roi_atlas import (
        make_roi_mask, restrict_mask_to_cell_spheres,
    )

    only_set = set(only_rois) if only_rois is not None else None
    rois_to_plot = [r for r, v in roi_betas.items()
                    if (only_set is None or r in only_set)
                    and v is not None and np.isfinite(v)]
    if not rois_to_plot:
        print("  [roi-beta-glassbrain] no ROIs to plot.")
        return None

    if vmax is None or not np.isfinite(vmax) or vmax <= 0:
        vmax = max(abs(roi_betas[r]) for r in rois_to_plot)
        vmax = float(vmax) if vmax > 0 else 1.0

    base_cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    display = nlplot.plot_glass_brain(
        None, display_mode=display_mode,
        title=title or 'ROI beta glass-brain',
        black_bg=False, plot_abs=False,
    )

    sig_rois_drawn, plotted, skipped = [], [], []
    for roi in rois_to_plot:
        try:
            mask_img = make_roi_mask(
                roi, roi_label_column=roi_label_column,
                brainnetome_nii=brainnetome_nii,
                brainnetome_lut=brainnetome_lut,
            )
        except Exception as e:
            print(f"  [roi-beta-glassbrain] mask for {roi!r} failed: {e}")
            skipped.append(roi)
            continue
        if mask_img is None or mask_img.get_fdata().sum() == 0:
            skipped.append(roi)
            continue

        if (roi_cell_coords is not None
                and cell_sphere_radius_mm is not None
                and roi in roi_cell_coords):
            coords = np.asarray(roi_cell_coords[roi], dtype=float)
            if coords.size > 0:
                mask_img = restrict_mask_to_cell_spheres(
                    mask_img, coords,
                    radius_mm=float(cell_sphere_radius_mm),
                )
        if mask_img.get_fdata().sum() == 0:
            skipped.append(roi)
            continue

        rgba = base_cmap(norm(roi_betas[roi]))
        single_cmap = ListedColormap([rgba])
        display.add_overlay(mask_img, threshold=0.5,
                            alpha=overlay_alpha, cmap=single_cmap)

        # Significance outline drawn on top of the same (possibly
        # cell-restricted) mask so the border matches what's coloured.
        if roi_pvals is not None:
            p = roi_pvals.get(roi, np.nan)
            if np.isfinite(p) and p < alpha_threshold:
                display.add_contours(
                    mask_img, levels=[0.5],
                    colors=sig_outline_color,
                    linewidths=sig_outline_linewidth,
                )
                sig_rois_drawn.append(roi)
        plotted.append(roi)

    if skipped:
        print(f"  [roi-beta-glassbrain] no mask available for: {skipped}")

    fig = plt.gcf()

    # Horizontal colour-bar in the bottom-right corner of the figure,
    # well clear of all brain panels.
    sm = cm.ScalarMappable(cmap=base_cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.74, 0.06, 0.22, 0.022])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('beta', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Footer caption listing significant ROIs (the contour already marks
    # them on the brain; this is the legend for the outline).
    if sig_rois_drawn:
        fig.text(0.02, 0.02,
                 f"outlined: p_perm < {alpha_threshold:g}  "
                 f"({', '.join(sig_rois_drawn)})",
                 fontsize=8, color='black')

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [roi-beta-glassbrain] saved → {save_path}")
    return fig


# ── Publication-figure helpers (state overlap, polar state cells) ─────
# The plot_state_polar_clock helper mirrors the one in
# scripts/plotting_human_cells.py so the publication script can import
# from one canonical place without depending on the scripts/ folder.

def smooth_circular(arr, sigma=2):
    """Gaussian-smooth a circular (e.g. 360-bin polar) trace."""
    from scipy.ndimage import gaussian_filter1d
    arr = np.asarray(arr, dtype=float)
    extended = np.concatenate([arr, arr, arr])
    smoothed = gaussian_filter1d(extended, sigma=sigma)
    return smoothed[len(arr):2 * len(arr)]


def plot_state_polar_clock(firing_across_states, title_string='',
                           ax=None, rlim=None, fontsize_labels=28,
                           fontsize_title=14, title_pad=18):
    """Polar plot of a 360-bin trial-averaged trace with A/B/C/D quadrants.

    Conventions: 0° at 12 o'clock, clockwise, A→B→C→D at 3/6/9/12.
    Each quadrant is coloured (orange/yellow/light-purple/purple) and gets
    a transparent wedge whose radius = that quadrant's mean activity.

    Parameters
    ----------
    firing_across_states : (n_bins,) array_like
        Smoothed firing-rate trace covering a full ABCD cycle (usually 360).
    title_string : str
        Title written above the polar; pass '' for no title.
    ax : matplotlib polar Axes or None
        Reuses an existing polar axes if given; otherwise creates a new
        figure + axes.
    rlim : (rmin, rmax) or None
        Shared radial limits.  If None, the trace's own min/max are used.
    """
    vals = np.asarray(firing_across_states, dtype=float)
    n_bins = vals.size
    if n_bins < 4:
        raise ValueError("Need at least 4 bins to define quadrants.")

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
        created_fig = True

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)

    # Plot the four coloured quadrant arcs.
    edges = np.linspace(0, n_bins, 5, dtype=int)
    for i in range(4):
        s, e = edges[i], edges[i + 1]
        if e > s:
            ax.plot(theta[s:e], vals[s:e],
                    color=STATE_QUADRANT_COLORS[i], linewidth=3)

    if rlim is None:
        rmin = float(np.nanmin(vals))
        rmax = float(np.nanmax(vals))
    else:
        rmin, rmax = rlim
    ax.set_ylim(rmin, rmax)

    # Quadrant-mean wedges (transparent).
    for i in range(4):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        m = float(np.nanmean(vals[s:e]))
        center_idx = (s + e) / 2.0
        center_ang = (center_idx / n_bins) * 2 * np.pi
        width = ((e - s) / n_bins) * 2 * np.pi
        if np.isfinite(m):
            ax.bar(center_ang, max(0, m - rmin),
                   width=width, bottom=rmin,
                   color=STATE_QUADRANT_COLORS[i], alpha=0.25,
                   edgecolor='none', zorder=0, align='center')

    # A/B/C/D labels at 3/6/9/12 o'clock.
    label_angles = np.deg2rad([0, 90, 180, 270])
    letters = ['A', 'B', 'C', 'D']
    pad = 0.12 * (rmax - rmin) if (np.isfinite(rmin) and np.isfinite(rmax)) else 0.1
    label_r = rmax + pad
    for lab, ang, col in zip(letters, label_angles, STATE_QUADRANT_COLORS):
        if np.isclose(ang, 0):              ha, va = 'center', 'bottom'
        elif np.isclose(ang, np.pi / 2):    ha, va = 'left', 'center'
        elif np.isclose(ang, np.pi):        ha, va = 'center', 'top'
        elif np.isclose(ang, 3 * np.pi / 2): ha, va = 'right', 'center'
        else:                                ha, va = 'center', 'center'
        ax.text(ang, label_r, lab, ha=ha, va=va,
                fontsize=fontsize_labels, fontweight='bold', color=col,
                clip_on=False)

    ax.set_xticks([])
    ax.grid(True)
    if title_string:
        # `pad` lifts the title clear of the 'A' label that sits at the
        # top of the polar; the default value of 18 works for most layouts.
        ax.set_title(title_string, va='bottom', fontsize=fontsize_title,
                     pad=title_pad)
    if created_fig:
        plt.tight_layout()
    return ax


def plot_state_overlap_stacked_bars(overlap_df, save_path=None,
                                    title='State-cell overlap per ROI',
                                    min_n_cells=20):
    """Stacked bars per ROI: encoding-only / CV-only / both / neither.

    Parameters
    ----------
    overlap_df : DataFrame
        Must have columns:
            roi, n_total, n_enc_only, n_cv_only, n_both,
            jaccard, percent_overlap (optional)
    min_n_cells : int
        ROIs with n_total < min_n_cells are flagged with '*' in the label.
    """
    df = overlap_df.copy().sort_values('n_total', ascending=False)
    rois = df['roi'].tolist()
    x = np.arange(len(rois))
    width = 0.7

    both = df['n_both'].to_numpy(dtype=float)
    enc_only = df['n_enc_only'].to_numpy(dtype=float)
    cv_only = df['n_cv_only'].to_numpy(dtype=float)
    neither = (df['n_total'].to_numpy(dtype=float)
               - both - enc_only - cv_only)
    neither = np.clip(neither, 0, None)

    fig, ax = plt.subplots(figsize=(max(7.5, 0.95 * len(rois)), 5.0))
    b1 = ax.bar(x, both, width, color='#2c7fb8', label='significant in both')
    b2 = ax.bar(x, enc_only, width, bottom=both, color='#9ecae1',
                label='encoding only')
    b3 = ax.bar(x, cv_only, width, bottom=both + enc_only, color='#fdae6b',
                label='CV-tuning only')
    b4 = ax.bar(x, neither, width, bottom=both + enc_only + cv_only,
                color='lightgray', label='neither')

    # Jaccard annotations
    for xi, j_val, n_t in zip(x, df['jaccard'], df['n_total']):
        ax.text(xi, n_t + 0.5, f'J={j_val:.2f}',
                ha='center', va='bottom', fontsize=9)

    labels = [f'{r}\n(n={n})' + (' *' if n < min_n_cells else '')
              for r, n in zip(rois, df['n_total'])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('# cells')
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [state-overlap stacked-bars] saved → {save_path}")
    return fig, ax


def plot_state_method_scatter(merged_df, save_path=None,
                              title='State CV consistency vs encoding state r',
                              alpha_perm=0.05, color_by='roi'):
    """Scatter of CV state consistency (x) vs encoding state mean_r (y).

    Each point is one cell. Cells significant in both methods are filled;
    cells significant in only one are open. Reports per-ROI Pearson r in
    the legend.
    """
    df = merged_df.dropna(subset=['state_cv_consistency', 'enc_state_mean_r']).copy()
    rois = sorted(df['roi'].dropna().unique().tolist())
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for i, roi in enumerate(rois):
        g = df[df['roi'] == roi]
        if g.empty:
            continue
        color = cmap(i % 10)
        sig_both = g[(g['cv_p_perm'] < alpha_perm)
                     & (g['enc_p_perm'] < alpha_perm)]
        sig_one = g[((g['cv_p_perm'] < alpha_perm)
                    | (g['enc_p_perm'] < alpha_perm))
                   & ~((g['cv_p_perm'] < alpha_perm)
                       & (g['enc_p_perm'] < alpha_perm))]
        sig_none = g[(g['cv_p_perm'] >= alpha_perm)
                     & (g['enc_p_perm'] >= alpha_perm)]
        # Compute per-ROI correlation across ALL cells of that ROI.
        if len(g) >= 3:
            r = np.corrcoef(g['state_cv_consistency'], g['enc_state_mean_r'])[0, 1]
            label = f'{roi} n={len(g)} r={r:+.2f}'
        else:
            label = f'{roi} n={len(g)}'
        ax.scatter(sig_none['state_cv_consistency'],
                   sig_none['enc_state_mean_r'],
                   s=14, color=color, alpha=0.25, edgecolor='none')
        ax.scatter(sig_one['state_cv_consistency'],
                   sig_one['enc_state_mean_r'],
                   s=28, facecolor='none', edgecolor=color, lw=1.0)
        ax.scatter(sig_both['state_cv_consistency'],
                   sig_both['enc_state_mean_r'],
                   s=36, color=color, edgecolor='black', lw=0.6,
                   label=label)

    ax.axhline(0, color='0.6', lw=0.6, ls=':')
    ax.axvline(0, color='0.6', lw=0.6, ls=':')
    ax.set_xlabel('CV state consistency (Pearson r)')
    ax.set_ylabel('Encoding state model mean_r')
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [state-method scatter] saved → {save_path}")
    return fig, ax


def plot_single_state_cell_polar(mean_trace, traces_per_config, configs,
                                 cell_label, roi,
                                 cv_consistency, enc_state_mean_r,
                                 cv_p_perm=None, enc_p_perm=None,
                                 save_path=None, smooth_sigma=4,
                                 n_cols=3):
    """One cell's polar overview.

    Subplots laid out in a `n_cols`-wide grid:
      [mean across configs] [config 1] [config 2] ...

    The mean is computed by the caller (so this function doesn't have to
    know about correct-trial filtering); we just smooth and render here.
    """
    n_configs = len(configs)
    n_panels = 1 + n_configs
    n_rows = int(np.ceil(n_panels / n_cols))

    # Shared r-limits across all polar panels so wedge heights are
    # directly comparable.
    smoothed_mean = smooth_circular(mean_trace, sigma=smooth_sigma)
    smoothed_per_cfg = [smooth_circular(t, sigma=smooth_sigma)
                       if (t is not None and np.isfinite(t).any())
                       else None for t in traces_per_config]
    all_finite = [smoothed_mean] + [t for t in smoothed_per_cfg if t is not None]
    if all_finite:
        rmin = float(np.nanmin([np.nanmin(t) for t in all_finite]))
        rmax = float(np.nanmax([np.nanmax(t) for t in all_finite]))
    else:
        rmin, rmax = 0.0, 1.0
    rlim = (rmin, rmax)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 4.8 * n_rows),
                             subplot_kw=dict(projection='polar'),
                             squeeze=False)

    # Subplot 0: mean across configs.
    plot_state_polar_clock(
        smoothed_mean, title_string='mean across configs',
        ax=axes[0, 0], rlim=rlim,
        fontsize_labels=18, fontsize_title=11, title_pad=18)

    # Per-config subplots.
    for i, (cfg, trace) in enumerate(zip(configs, smoothed_per_cfg)):
        idx = i + 1
        ax = axes[idx // n_cols, idx % n_cols]
        if trace is None:
            ax.set_axis_off()
            continue
        plot_state_polar_clock(
            trace, title_string=f'cfg {cfg}',
            ax=ax, rlim=rlim,
            fontsize_labels=16, fontsize_title=9, title_pad=14)

    # Hide unused axes.
    for k in range(n_panels, n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        ax.axis('off')

    # Suptitle with the cell-level statistics.
    parts = [f'{cell_label}  [{roi}]']
    if cv_consistency is not None and np.isfinite(cv_consistency):
        cv_str = f'CV-consistency = {cv_consistency:+.3f}'
        if cv_p_perm is not None and np.isfinite(cv_p_perm):
            cv_str += f'  (p_perm = {cv_p_perm:.3g})'
        parts.append(cv_str)
    if enc_state_mean_r is not None and np.isfinite(enc_state_mean_r):
        enc_str = f'encoding state mean_r = {enc_state_mean_r:+.3f}'
        if enc_p_perm is not None and np.isfinite(enc_p_perm):
            enc_str += f'  (p_perm = {enc_p_perm:.3g})'
        parts.append(enc_str)
    suptitle = '\n'.join(parts)
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    # Polar axes don't play nicely with tight_layout; use subplots_adjust.
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.04,
                        wspace=0.35, hspace=0.45)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [single-cell polar] saved → {save_path}")
    plt.close(fig)
    return fig


# ── Fig 1 helper: DSR confound-filter grid ─────────────────────────────
def plot_dsr_confound_filter_grid(panels, save_path=None,
                                  suptitle=None, n_bins=25):
    """Grid of histograms: rows = ROIs, columns = exclusion filters.

    `panels` is a nested dict:
        panels[roi][filter_label] = {
            'r_all':  (n_total,) ndarray of mean_r for *all* DSR cells,
            'r_kept': (n_kept,)  ndarray of mean_r for *kept* DSR cells,
            'n_total':  int,
            'n_kept':   int,
            'p_kept_>0': float,   # one-sided t-test of kept-mean > 0
            'p_shift':  float,    # Mann-Whitney kept vs excluded
        }
    The first column should be the unfiltered scenario (we still expect
    `r_all == r_kept` there).
    """
    rois = list(panels.keys())
    if not rois:
        print('  [fig1] no ROIs to plot.')
        return None
    filter_labels = list(panels[rois[0]].keys())
    n_rows, n_cols = len(rois), len(filter_labels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.8 * n_rows),
                             sharex=False, sharey='row',
                             squeeze=False)

    # Per-row x-range based on the full distribution of that ROI.
    for i, roi in enumerate(rois):
        all_vals = panels[roi][filter_labels[0]]['r_all']
        if all_vals.size == 0:
            lim = 0.1
        else:
            lim = max(0.05, 1.05 * float(np.nanmax(np.abs(all_vals))))
        bins = np.linspace(-lim, lim, n_bins + 1)
        for j, fl in enumerate(filter_labels):
            ax = axes[i, j]
            d = panels[roi][fl]
            r_all = d['r_all']; r_kept = d['r_kept']
            ax.hist(r_all, bins=bins, color='0.55', alpha=0.35,
                    edgecolor='none')
            ax.hist(r_kept, bins=bins, color='tab:blue', alpha=0.80,
                    edgecolor='none')
            ax.axvline(0, color='0.4', lw=0.7, ls=':')
            ax.axvline(float(np.nanmean(r_all)),
                       color='0.35', lw=1.0)
            if r_kept.size:
                ax.axvline(float(np.nanmean(r_kept)),
                           color='tab:blue', lw=1.5)
            ax.set_title(
                (f"{fl}\nn={d['n_kept']}/{d['n_total']}   "
                 f"p>0={d['p_kept_>0']:.1e}\n"
                 f"shift p={d['p_shift']:.1e}"),
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f'{roi}\n# cells', fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel('DSR mean_r', fontsize=8)
            ax.set_xlim(-lim, lim)
            ax.spines[['top', 'right']].set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'  [fig1 DSR filter grid] saved → {save_path}')
    plt.close(fig)
    return fig


# ── Fig 1 inset helper: ACC state-encoders vs non-state-encoders ──────
def plot_acc_state_vs_dsr_inset(r_state_encoders, r_non_state_encoders,
                                save_path=None,
                                title='ACC: DSR mean_r by state-encoder status'):
    """Side-by-side box+strip plot of DSR mean_r for state-encoding vs
    non-state-encoding ACC cells, with a one-sided Mann-Whitney p."""
    a = np.asarray(r_state_encoders, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(r_non_state_encoders, dtype=float)
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        print('  [fig1 inset] not enough cells in one group; skipping.')
        return None
    try:
        mw = stats.mannwhitneyu(b, a, alternative='greater')
        p_mw = float(mw.pvalue)
    except Exception:
        p_mw = np.nan

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    parts = ax.boxplot([a, b], positions=[0, 1], widths=0.5,
                       patch_artist=True, showfliers=False)
    for patch, c in zip(parts['boxes'], ['salmon', 'steelblue']):
        patch.set_facecolor(c); patch.set_alpha(0.6); patch.set_edgecolor('0.3')
    rng = np.random.default_rng(0)
    ax.scatter(rng.uniform(-0.18, 0.18, size=a.size), a, s=10,
               color='salmon', alpha=0.5, edgecolor='none')
    ax.scatter(1 + rng.uniform(-0.18, 0.18, size=b.size), b, s=10,
               color='steelblue', alpha=0.5, edgecolor='none')
    ax.axhline(0, color='0.5', lw=0.6, ls=':')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'state-encoder\n(n={a.size})',
                        f'non state-encoder\n(n={b.size})'])
    ax.set_ylabel('DSR mean_r')
    ax.set_title(f'{title}\nMW one-sided p(non > state) = {p_mw:.1e}',
                 fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'  [fig1 ACC inset] saved → {save_path}')
    plt.close(fig)
    return fig


# ── Fig 3 helper: per-ROI DSR lag profile ─────────────────────────────
def plot_dsr_lag_overlay(lag_means_by_roi, lag_sems_by_roi=None,
                        friedman_p_by_roi=None,
                        n_cells_by_roi=None, save_path=None,
                        bold_rois=('ACC',), reference_y=1.0,
                        title='DSR coefficient lag profile per ROI'):
    """Line plot of mean per-lag DSR coefficient strength per ROI.

    Each ROI is one line over lags 0..11. `bold_rois` are highlighted with
    thicker lines and an annotated Friedman p.
    """
    rois = list(lag_means_by_roi.keys())
    n_lags = max(len(v) for v in lag_means_by_roi.values()) if rois else 12
    x = np.arange(n_lags)
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, roi in enumerate(rois):
        m = np.asarray(lag_means_by_roi[roi], dtype=float)
        lw = 2.6 if roi in bold_rois else 1.1
        alpha = 1.0 if roi in bold_rois else 0.6
        color = 'crimson' if roi in bold_rois else cmap(i % 10)
        n_cells = n_cells_by_roi.get(roi, 'n?') if n_cells_by_roi else 'n?'
        label = f'{roi} (n={n_cells})'
        if friedman_p_by_roi and roi in friedman_p_by_roi:
            p = friedman_p_by_roi[roi]
            label += f'  Friedman p={p:.2g}' if np.isfinite(p) else ''
        ax.plot(x, m, color=color, lw=lw, marker='o',
                ms=5 if roi in bold_rois else 3.5,
                alpha=alpha, label=label)
        if lag_sems_by_roi and roi in lag_sems_by_roi:
            s = np.asarray(lag_sems_by_roi[roi], dtype=float)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.12)

    ax.axhline(reference_y, color='0.5', lw=0.8, ls='--',
               label=f'uniform = {reference_y:g}')
    ax.set_xticks(x)
    ax.set_xlabel('lag (anchor → current; 0 = current, 1..11 = future/past)')
    ax.set_ylabel('mean coefficient (per-neuron mean-normalised)')
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc='best')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'  [fig3 lag overlay] saved → {save_path}')
    plt.close(fig)
    return fig


# ── Fig 4 helper: example DSR cell (best anchor + lag + actual/pred) ──
def plot_example_dsr_cell(anchor_grid_mag, best_anchor_loc, best_anchor_phase,
                          lag_profile, lag_profile_label,
                          actual_trace, predicted_trace,
                          cell_label, roi,
                          best_fold_r, best_fold_cfg,
                          save_path=None, smooth_sigma=4):
    """Per-cell 3-panel publication figure.

    Parameters
    ----------
    anchor_grid_mag : (9, 3) ndarray
        Per-(location, anchor-phase) magnitude of the DSR coefficients
        summed over lags. Heatmapped in panel 1.
    best_anchor_loc : int  (0..8)
        Location index of the dominant anchor (highlighted in panel 1).
    best_anchor_phase : int  (0..2)
        Within-state phase index of the dominant anchor.
    lag_profile : (n_lags,) ndarray
        The 12-value coefficient vector at the best anchor.
    lag_profile_label : str
        Annotation describing what's plotted (e.g. preferred lag idx).
    actual_trace, predicted_trace : (360,) ndarray
        Trial-averaged actual and model-predicted activity for the cell's
        best-fold held-out config.
    """
    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 2.0],
                         wspace=0.35)

    # ── Panel 1: anchor heatmap ──
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(anchor_grid_mag, cmap='magma', aspect='auto')
    ax1.set_xticks(range(anchor_grid_mag.shape[1]))
    ax1.set_xticklabels([f'phase {i}' for i in range(anchor_grid_mag.shape[1])])
    ax1.set_yticks(range(anchor_grid_mag.shape[0]))
    ax1.set_yticklabels([f'loc {i + 1}' for i in range(anchor_grid_mag.shape[0])])
    ax1.set_xlabel('within-state phase')
    ax1.set_ylabel('grid location')
    ax1.set_title('DSR anchor preference (Σ|coef| over lags)',
                  fontsize=10)
    # Highlight best anchor.
    ax1.add_patch(plt.Rectangle((best_anchor_phase - 0.5,
                                 best_anchor_loc - 0.5),
                                1, 1, fill=False, edgecolor='cyan',
                                lw=2.5))
    cbar = fig.colorbar(im, ax=ax1, shrink=0.85)
    cbar.set_label('|coef| sum')

    # ── Panel 2: lag profile at best anchor ──
    ax2 = fig.add_subplot(gs[0, 1])
    lags = np.arange(len(lag_profile))
    colors = ['lightgray' if i != int(np.argmax(np.abs(lag_profile)))
              else 'crimson' for i in lags]
    ax2.bar(lags, lag_profile, color=colors, edgecolor='0.3', lw=0.5)
    ax2.axhline(0, color='0.4', lw=0.6)
    ax2.set_xticks(lags)
    ax2.set_xlabel('lag (0 = now)')
    ax2.set_ylabel('coefficient')
    ax2.set_title(f'lag profile at best anchor\n'
                  f'(loc={best_anchor_loc + 1}, phase={best_anchor_phase}) '
                  f'— {lag_profile_label}', fontsize=10)
    ax2.spines[['top', 'right']].set_visible(False)

    # ── Panel 3: actual vs predicted trace ──
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.arange(len(actual_trace))
    actual_s = smooth_circular(actual_trace, sigma=smooth_sigma)
    pred_s = smooth_circular(predicted_trace, sigma=smooth_sigma)
    ax3.plot(bins, actual_s, color='steelblue', lw=1.6, label='actual')
    ax3.plot(bins, pred_s, color='crimson', lw=1.3, label='predicted',
             alpha=0.85)
    # Mark state boundaries.
    for s in [90, 180, 270]:
        ax3.axvline(s, color='0.7', lw=0.5, ls=':')
    for s, name in enumerate(['A', 'B', 'C', 'D']):
        ax3.text(s * 90 + 45, ax3.get_ylim()[1] * 0.97
                 if ax3.get_ylim()[1] > 0 else 0,
                 name, ha='center', va='top',
                 color=STATE_QUADRANT_COLORS[s], fontweight='bold')
    ax3.set_xlabel('bin (0..360, 4 states × 90 bins)')
    ax3.set_ylabel('firing (smoothed)')
    ax3.set_title(f'actual vs predicted — best fold (cfg {best_fold_cfg}, '
                  f'r={best_fold_r:+.3f})', fontsize=10)
    ax3.legend(fontsize=8, frameon=False, loc='upper right')
    ax3.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'{cell_label}  [{roi}]', fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'  [fig4 example DSR cell] saved → {save_path}')
    plt.close(fig)
    return fig


# ── Fig 5 helper: side-by-side DSR-vs-RSA per-ROI bars ────────────────
def plot_dsr_rsa_comparison(rsa_df, enc_df, save_path=None,
                            title='DSR signal: RSA β vs encoding mean_r',
                            sort_by='rsa_beta', alpha=0.05):
    """Per-ROI side-by-side bars. `rsa_df` and `enc_df` must each have:
        roi, beta_or_r, p, n_cells
    `beta_or_r` is the standardised effect size for that method's DSR
    signal (β for RSA, mean_r for encoding). Both are dimensionless and
    typically in similar [-0.1, +0.2] ranges so they share a y-axis.
    """
    merged = rsa_df.rename(columns={'beta_or_r': 'rsa_beta',
                                    'p': 'rsa_p',
                                    'n_cells': 'rsa_n'}).merge(
        enc_df.rename(columns={'beta_or_r': 'enc_r', 'p': 'enc_p',
                               'n_cells': 'enc_n'}),
        on='roi', how='outer')
    merged = merged.sort_values(sort_by, ascending=False).reset_index(drop=True)

    rois = merged['roi'].tolist()
    x = np.arange(len(rois))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(rois)), 5.5))
    b1 = ax.bar(x - w / 2, merged['rsa_beta'], w,
                color='#264653', edgecolor='black', lw=0.6, label='RSA β')
    b2 = ax.bar(x + w / 2, merged['enc_r'], w,
                color='#2A9D8F', edgecolor='black', lw=0.6,
                label='encoding mean_r')

    def _stars(p):
        return ('***' if (np.isfinite(p) and p < 0.001) else
                '**' if (np.isfinite(p) and p < 0.01) else
                '*' if (np.isfinite(p) and p < alpha) else '')

    ymax = max(np.nanmax(merged['rsa_beta'].values),
               np.nanmax(merged['enc_r'].values), 0.0)
    ymin = min(np.nanmin(merged['rsa_beta'].values),
               np.nanmin(merged['enc_r'].values), 0.0)
    pad = 0.05 * (ymax - ymin) if (ymax > ymin) else 0.02
    for xi, (_, row) in enumerate(merged.iterrows()):
        ax.text(xi - w / 2, row['rsa_beta'] + (pad if row['rsa_beta'] >= 0 else -pad),
                _stars(row['rsa_p']), ha='center',
                va='bottom' if row['rsa_beta'] >= 0 else 'top',
                fontsize=10)
        ax.text(xi + w / 2, row['enc_r'] + (pad if row['enc_r'] >= 0 else -pad),
                _stars(row['enc_p']), ha='center',
                va='bottom' if row['enc_r'] >= 0 else 'top',
                fontsize=10)

    ax.axhline(0, color='0.4', lw=0.6)
    labels = [f'{r}\n(RSA n={int(rn) if pd.notna(rn) else "?"} | '
              f'enc n={int(en) if pd.notna(en) else "?"})'
              for r, rn, en in zip(rois, merged['rsa_n'], merged['enc_n'])]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Standardised DSR signal')
    ax.set_title(title)
    ax.legend(frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'  [fig5 DSR-vs-RSA] saved → {save_path}')
    plt.close(fig)


# ── Publication-figure helpers for encoding_analysis_simple.py ────────
# 1. Per (ROI, model) t-test of mean_r against 0  (one-sided, H1: mean > 0)
# 2. ROI × model heatmap of those t-values
# 3. Publication histogram (sister of plot_r_distribution_grid)
# 4. Top-N cell fits (sister of plot_best_neuron_per_roi_model)
# 5. Benjamini-Hochberg FDR adjustment


def compute_roi_model_tstats(results_df, models=None, alpha=0.05):
    """One-sample t-test of `mean_r > 0` per (ROI, model).

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns `roi`, `model`, `mean_r`; `p_perm` is optional
        and used to compute the per-(ROI,model) significant-cell count.
    models : iterable of str or None
        If given, restrict to these models.
    alpha : float
        Threshold for counting how many cells have `p_perm < alpha`.

    Returns
    -------
    pd.DataFrame with rows = (roi, model) and columns:
        n_cells, mean_r, sem_r, t, p_t, n_sig_perm, prop_sig_perm
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    rdf = results_df.dropna(subset=['mean_r']).copy()
    if models is not None:
        rdf = rdf[rdf['model'].isin(list(models))]
    rows = []
    for (roi, model), g in rdf.groupby(['roi', 'model'], sort=False):
        vals = g['mean_r'].to_numpy(dtype=float)
        t, p_t = _one_sided_t_greater(vals)
        if 'p_perm' in g.columns:
            p_perms = g['p_perm'].to_numpy(dtype=float)
            n_sig = int(np.sum(p_perms < alpha))
            prop_sig = float(n_sig / len(p_perms)) if len(p_perms) else np.nan
        else:
            n_sig, prop_sig = 0, np.nan
        rows.append({
            'roi':            roi,
            'model':          model,
            'n_cells':        int(vals.size),
            'mean_r':         float(np.nanmean(vals)) if vals.size else np.nan,
            'sem_r':          float(stats.sem(vals)) if vals.size > 1 else np.nan,
            't':              t,
            'p_t':            p_t,
            'n_sig_perm':     n_sig,
            'prop_sig_perm':  prop_sig,
        })
    return pd.DataFrame(rows).sort_values(['model', 'roi']).reset_index(drop=True)


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Returns an array the same length as `pvals`; NaN inputs stay NaN.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_ok = np.empty(n)
    q_ok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = q_ok
    return q


def plot_roi_model_heatmap(stats_df, models=None, rois=None,
                            value_col='t', annot_col='p_t', sig_col='p_t',
                            n_col='n_cells', alpha=0.05, vmax=None,
                            cmap_name='RdBu_r', save_path=None,
                            title=None, value_label='t-statistic',
                            base_fontsize=14):
    """Shared ROI × model heatmap used by both RSA and encoding scripts.

    Each cell shows `value_col` as its colour, `annot_col` written inside,
    and a thick black box drawn around cells where `sig_col < alpha`.
    ROIs and models are ordered using `rois` / `models` if given, otherwise
    via :data:`CANONICAL_ROI_ORDER` / :data:`CANONICAL_ENC_MODEL_ORDER`.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Long table with columns ``roi``, ``model``, `value_col`, `annot_col`,
        `sig_col`, and (optionally) `n_col`.
    models, rois : list[str] or None
        Explicit display order. ``None`` -> canonical default, ROIs/models
        actually present, with absent entries appended at the end.
    value_col, annot_col, sig_col : str
        Column names used for colour, in-cell text, and the significance
        outline respectively.
    n_col : str
        Per-(ROI, model) cell count, displayed in the row label.
    alpha : float
        Outline threshold applied to `sig_col`.
    vmax : float or None
        Symmetric colour limit; defaults to ``max(|value|)``.
    cmap_name : str
    save_path : str or None
        Saved as PNG and also as SVG (same basename).
    title : str or None
    value_label : str
        Colour-bar label.
    base_fontsize : int
        All other text scales relative to this (default 14, larger than
        the previous default).
    """
    if stats_df is None or stats_df.empty:
        return None, None

    # Refuse to silently hide configuration bugs: duplicated model names in
    # the explicit display list (e.g. a combo accidentally containing 'state'
    # twice) collapse to a single column and the loop below would pick one
    # row at random via iloc[0]. Same for repeated (roi, model) pairs in the
    # long-format input. Raise so the caller fixes the upstream list / df.
    if models is not None:
        dup_in_arg = [m for m in set(models) if list(models).count(m) > 1]
        if dup_in_arg:
            raise ValueError(
                f"plot_roi_model_heatmap: `models` contains duplicate "
                f"entries {sorted(set(dup_in_arg))}. Fix the upstream combo "
                f"definition; silent deduplication would hide collinear "
                f"regressors.")
    dup_rows = (stats_df.groupby(['roi', 'model']).size()
                .loc[lambda s: s > 1])
    if not dup_rows.empty:
        raise ValueError(
            f"plot_roi_model_heatmap: stats_df has duplicate (roi, model) "
            f"rows — first few:\n{dup_rows.head().to_string()}\n"
            f"Each (roi, model) must appear exactly once; otherwise the "
            f"plotted cell value depends on row order.")

    present_rois = set(stats_df['roi'].dropna().unique())
    present_models = set(stats_df['model'].dropna().unique())
    rois_order = _order_keep_present(
        rois if rois is not None else CANONICAL_ROI_ORDER, present_rois)
    cols_order = _order_keep_present(
        models if models is not None else CANONICAL_ENC_MODEL_ORDER,
        present_models)
    if not rois_order or not cols_order:
        return None, None

    val_mat = np.full((len(rois_order), len(cols_order)), np.nan)
    annot_mat = np.full_like(val_mat, np.nan)
    sig_mat = np.full_like(val_mat, np.nan)

    for i, roi in enumerate(rois_order):
        for j, m in enumerate(cols_order):
            r = stats_df[(stats_df['roi'] == roi) & (stats_df['model'] == m)]
            if r.empty:
                continue
            val_mat[i, j] = float(r[value_col].iloc[0])
            annot_mat[i, j] = float(r[annot_col].iloc[0])
            sig_mat[i, j] = float(r[sig_col].iloc[0])

    # Per-ROI cell count (max across this ROI's models).
    roi_n = {}
    if n_col in stats_df.columns:
        for roi in rois_order:
            g = stats_df[stats_df['roi'] == roi]
            roi_n[roi] = int(g[n_col].max()) if len(g) else 0
    roi_labels = [
        (f'{_wrap_label(r, width=12)}\n(n={roi_n[r]})'
         if roi_n.get(r) is not None
         else _wrap_label(r, width=12))
        for r in rois_order
    ]

    if vmax is None or not np.isfinite(vmax) or vmax <= 0:
        finite = val_mat[np.isfinite(val_mat)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmax = vmax if vmax > 0 else 1.0

    figsize = (1.85 * len(cols_order) + 3.5,
               0.95 * len(rois_order) + 2.4)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(val_mat, cmap=cmap_name, vmin=-vmax, vmax=vmax,
                   aspect='auto')

    ax.set_xticks(np.arange(len(cols_order)))
    ax.set_xticklabels(cols_order, rotation=35, ha='right',
                       fontsize=base_fontsize + 1)
    ax.set_yticks(np.arange(len(rois_order)))
    ax.set_yticklabels(roi_labels, fontsize=base_fontsize)

    for i in range(len(rois_order)):
        for j in range(len(cols_order)):
            p = annot_mat[i, j]
            if np.isfinite(p):
                ax.text(j, i, f'p={p:.3f}',
                        ha='center', va='center',
                        fontsize=base_fontsize - 1, color='black')
            s = sig_mat[i, j]
            if np.isfinite(s) and s < alpha:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor='black', linewidth=3.0))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(value_label, fontsize=base_fontsize + 1)
    cbar.ax.tick_params(labelsize=base_fontsize)

    ax.set_title(title or f'ROI × model  (outline: p < {alpha})',
                 fontsize=base_fontsize + 3, fontweight='bold')

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        base, _ = os.path.splitext(save_path)
        fig.savefig(base + '.svg', bbox_inches='tight')
        print(f'  [roi-model-heatmap] saved → {save_path} (+ .svg)')
    return fig, ax


# Backwards-compatible alias for the previous name.
def plot_roi_tstat_heatmap(stats_df, models, save_path=None,
                            alpha=0.05, value_col='t', annot_col='p_t',
                            sig_col='p_t', vmax=None,
                            cmap_name='RdBu_r', title=None,
                            base_fontsize=14):
    return plot_roi_model_heatmap(
        stats_df, models=models, value_col=value_col,
        annot_col=annot_col, sig_col=sig_col, vmax=vmax,
        cmap_name=cmap_name, save_path=save_path, alpha=alpha,
        title=title or (f'ROI × model — one-sided t-test of mean_r > 0  '
                        f'(outline: p_t < {alpha})'),
        value_label='t-statistic (mean_r > 0)',
        base_fontsize=base_fontsize,
    )


def plot_publication_r_histogram(results_df, models, save_path=None,
                                  alpha=0.05, bins=21, base_fontsize=14,
                                  rois=None, model_order=None,
                                  panel_w=3.3, panel_h=2.7,
                                  p_t_per_roi_model=None):
    """Publication-ready ROI × model histogram of held-out mean r values.

    * No text inside the histograms — `n = n_sig/n_total` is shown above
      each panel, all other meta lives at the figure edges.
    * Larger default panels (so frequency = 1 bars are visible) and bigger
      fonts (default 14, was 13).
    * ROI labels on the leftmost column are wrapped to multiple lines so
      long names (HC_anterior, medialOFC) stay readable.
    * Significance is communicated by a thick black frame only — no
      circle markers in the corner.
    * If `p_t_per_roi_model` is supplied (dict-of-dicts keyed by
      ``[model][roi]`` -> p-value), the frame uses that p (e.g. FDR-
      adjusted); otherwise an in-panel one-sided t-test is computed.
    * Saves to PNG, SVG and PDF for publication.
    """
    if results_df is None or results_df.empty:
        return None
    rdf = results_df.dropna(subset=['mean_r']).copy()
    present_rois = set(rdf['roi'].unique())
    present_models = set(rdf['model'].unique())
    rois_order = _order_keep_present(
        rois if rois is not None else CANONICAL_ROI_ORDER, present_rois)
    rois_order = [r for r in rois_order
                  if (model_order or models) is None
                  or any(((rdf['roi'] == r) & (rdf['model'] == m)).any()
                         for m in (model_order or models))]
    mods = _order_keep_present(
        model_order if model_order is not None else models, present_models)
    if not rois_order or not mods:
        return None

    n_rows, n_cols = len(rois_order), len(mods)
    # Generous panel size so freq=1 bars are visible.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols + 2.2, panel_h * n_rows + 1.8),
        constrained_layout=True,
    )
    axes = np.atleast_2d(np.asarray(axes)).reshape(n_rows, n_cols)

    for r, roi in enumerate(rois_order):
        for c, model in enumerate(mods):
            ax = axes[r, c]
            sub = rdf[(rdf['roi'] == roi) & (rdf['model'] == model)]
            vals = sub['mean_r'].to_numpy(dtype=float)
            p_perm_arr = sub['p_perm'].to_numpy(dtype=float) \
                if 'p_perm' in sub.columns else np.full(vals.size, np.nan)
            if vals.size == 0:
                ax.axis('off')
                continue

            panel_lim = max(0.05, 1.05 * float(np.max(np.abs(vals))))
            edges = np.linspace(-panel_lim, panel_lim, bins)
            sig_mask = np.isfinite(p_perm_arr) & (p_perm_arr < alpha)

            ax.hist(vals[~sig_mask], bins=edges, color='0.75',
                    edgecolor='white', linewidth=0.6)
            if sig_mask.any():
                ax.hist(vals[sig_mask], bins=edges, color='tab:red',
                        edgecolor='white', linewidth=0.6, alpha=0.9)
            ax.axvline(0, color='black', lw=1.0)

            n_sig = int(sig_mask.sum())
            n_tot = int(vals.size)

            # Panel significance from either supplied p_t (e.g. FDR) or
            # a one-sided t-test computed here.
            if (p_t_per_roi_model is not None
                    and model in p_t_per_roi_model
                    and roi in p_t_per_roi_model[model]):
                p_panel = float(p_t_per_roi_model[model][roi])
            else:
                _, p_panel = _one_sided_t_greater(vals)

            is_sig = np.isfinite(p_panel) and p_panel < alpha
            if is_sig:
                for spine in ax.spines.values():
                    spine.set_linewidth(3.0)
                    spine.set_color('black')
            else:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            ax.tick_params(labelsize=base_fontsize - 1)

            # Panel title = "n = sig/total" above the data area. Adds the
            # model name on top of the first row only.
            if r == 0:
                ax.set_title(f'{model}\nn = {n_sig}/{n_tot}',
                             fontsize=base_fontsize + 1,
                             fontweight='bold', pad=8)
            else:
                ax.set_title(f'n = {n_sig}/{n_tot}',
                             fontsize=base_fontsize, pad=4)

            if c == 0:
                ax.set_ylabel(_wrap_label(roi, width=11),
                              fontsize=base_fontsize + 1,
                              fontweight='bold')
            if r == n_rows - 1:
                ax.set_xlabel('held-out mean r',
                              fontsize=base_fontsize)

    suptitle = (
        f'Held-out r per (ROI × model)  |  alpha = {alpha}  |  '
        f'red = perm-significant cells  |  bold frame = panel p < {alpha}'
    )
    fig.suptitle(suptitle, fontsize=base_fontsize + 2, fontweight='bold')

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        base, _ = os.path.splitext(save_path)
        fig.savefig(base + '.svg', bbox_inches='tight')
        fig.savefig(base + '.pdf', bbox_inches='tight')
        print(f'  [pub-hist] saved → {save_path} (+ .svg, .pdf)')
    return fig


def plot_top_n_cells_per_model(diagnostics_all, results_df, models, n=5,
                                save_dir=None, base_fontsize=12):
    """Save publication-ready fit plots for the top-`n` cells per model.

    For each model in `models`, the n cells with the highest `mean_r`
    across all ROIs are plotted: permutation null on the left, best-fold
    actual-vs-predicted trace on the right. Larger fonts than the diagnostic
    plots, and saved as png + svg + pdf so they survive being shrunk for
    figures.
    """
    if results_df is None or results_df.empty or not diagnostics_all:
        return
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    rdf = results_df.dropna(subset=['mean_r']).copy()
    saved = 0
    for model in models:
        sub = rdf[rdf['model'] == model]
        if sub.empty:
            continue
        top = sub.nlargest(n, 'mean_r')
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            sub_str = row['subject']
            neuron_label = row['neuron']
            roi = row['roi']
            diag = (diagnostics_all.get(sub_str, {})
                    .get(neuron_label, {}).get(model))
            if diag is None:
                continue
            r_per_fold = np.asarray(diag.get('r_per_fold', []), dtype=float)
            if not r_per_fold.size or not np.isfinite(r_per_fold).any():
                continue
            best_fold = int(np.nanargmax(r_per_fold))
            sub_configs = diag.get('configs', [])
            cfg_label = (sub_configs[best_fold]
                         if best_fold < len(sub_configs)
                         else f'fold {best_fold}')

            fig, axes = plt.subplots(
                1, 2, figsize=(12.5, 4.2), constrained_layout=True,
                gridspec_kw=dict(width_ratios=[1, 2]),
            )

            # Perm null (left).
            _draw_perm_hist(axes[0], diag, bins=30)
            axes[0].set_title('permutation null',
                              fontsize=base_fontsize + 1)
            axes[0].set_xlabel('mean Pearson r',
                               fontsize=base_fontsize)
            axes[0].tick_params(labelsize=base_fontsize - 1)
            for txt in axes[0].get_legend().get_texts():
                txt.set_fontsize(base_fontsize - 1)

            # Actual vs predicted (right).
            yt = np.asarray(diag['y_test_per_fold'][best_fold], dtype=float)
            yp = np.asarray(diag['y_pred_per_fold'][best_fold], dtype=float)
            x = np.arange(yt.size)

            ax_ts = axes[1]
            ax_ts2 = ax_ts.twinx()
            ax_ts.plot(x, yt, color='0.30', lw=1.6, label='neuron')
            ax_ts2.plot(x, yp, color='tab:red', lw=1.6, alpha=0.9,
                        label='predicted')
            ax_ts.set_title(
                f'best fold: held-out {cfg_label}   '
                f'r = {r_per_fold[best_fold]:.3f}',
                fontsize=base_fontsize + 1, loc='left',
            )
            ax_ts.set_xlabel('time bin', fontsize=base_fontsize)
            ax_ts.tick_params(labelsize=base_fontsize - 1)
            ax_ts2.tick_params(labelsize=base_fontsize - 1,
                               labelcolor='tab:red')
            ax_ts.set_ylabel('neuron (a.u.)',
                             fontsize=base_fontsize - 1,
                             color='0.40')
            ax_ts2.set_ylabel('predicted',
                              fontsize=base_fontsize - 1,
                              color='tab:red')
            ax_ts.spines['top'].set_visible(False)
            ax_ts2.spines['top'].set_visible(False)
            lines1, labels1 = ax_ts.get_legend_handles_labels()
            lines2, labels2 = ax_ts2.get_legend_handles_labels()
            ax_ts.legend(lines1 + lines2, labels1 + labels2,
                         fontsize=base_fontsize - 1,
                         loc='upper right', frameon=False)

            fig.suptitle(
                f'#{rank} cell for model "{model}"   '
                f'sub-{sub_str} {neuron_label}   roi: {roi}   '
                f'mean r = {diag.get("mean_r", float("nan")):.3f}   '
                f'p_perm = {diag.get("p_perm", float("nan")):.3f}',
                fontsize=base_fontsize + 1, fontweight='bold',
            )
            fname = (f'top{rank}_{model}_{roi}_sub-{sub_str}_{neuron_label}'
                     .replace('/', '_'))
            base = os.path.join(save_dir, fname)
            fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
            fig.savefig(base + '.svg', bbox_inches='tight')
            fig.savefig(base + '.pdf', bbox_inches='tight')
            plt.close(fig)
            saved += 1
    print(f'  [top-cell fits] saved {saved} figures → {save_dir}')


def plot_top_n_cells_per_roi_model(diagnostics_all, results_df,
                                    targets=None, models=None, rois=None,
                                    n=5, save_dir=None, base_fontsize=13):
    """Publication-ready fit plots for the top-`n` cells within (ROI, model).

    `targets` is a list of either ``(model, roi)`` or ``(model, roi, n_i)``
    tuples specifying exactly which subsets to render. If `targets` is
    ``None``, iterate over the Cartesian product of `models × rois`
    (each with `n` cells).

    Saves one PNG + SVG + PDF per cell into `save_dir`.
    """
    if results_df is None or results_df.empty or not diagnostics_all:
        return
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    rdf = results_df.dropna(subset=['mean_r']).copy()

    pairs = []
    if targets is None:
        if not models or not rois:
            return
        for m in models:
            for roi in rois:
                pairs.append((m, roi, int(n)))
    else:
        for t in targets:
            if len(t) == 2:
                pairs.append((t[0], t[1], int(n)))
            else:
                pairs.append((t[0], t[1], int(t[2])))

    saved = 0
    for model, roi, n_i in pairs:
        sub = rdf[(rdf['model'] == model) & (rdf['roi'] == roi)]
        if sub.empty:
            print(f'  [top-cells per (roi, model)] no cells for '
                  f'({roi}, {model}); skipping.')
            continue
        top = sub.nlargest(n_i, 'mean_r')
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            sub_str = row['subject']
            neuron_label = row['neuron']
            diag = (diagnostics_all.get(sub_str, {})
                    .get(neuron_label, {}).get(model))
            if diag is None:
                continue
            r_per_fold = np.asarray(diag.get('r_per_fold', []), dtype=float)
            if not r_per_fold.size or not np.isfinite(r_per_fold).any():
                continue
            best_fold = int(np.nanargmax(r_per_fold))
            sub_configs = diag.get('configs', [])
            cfg_label = (sub_configs[best_fold]
                         if best_fold < len(sub_configs)
                         else f'fold {best_fold}')

            fig, axes = plt.subplots(
                1, 2, figsize=(13, 4.3), constrained_layout=True,
                gridspec_kw=dict(width_ratios=[1, 2]),
            )
            _draw_perm_hist(axes[0], diag, bins=30)
            axes[0].set_title('permutation null',
                              fontsize=base_fontsize + 1)
            axes[0].set_xlabel('mean Pearson r', fontsize=base_fontsize)
            axes[0].tick_params(labelsize=base_fontsize - 1)
            leg = axes[0].get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_fontsize(base_fontsize - 1)

            yt = np.asarray(diag['y_test_per_fold'][best_fold], dtype=float)
            yp = np.asarray(diag['y_pred_per_fold'][best_fold], dtype=float)
            x = np.arange(yt.size)
            ax_ts = axes[1]
            ax_ts2 = ax_ts.twinx()
            ax_ts.plot(x, yt, color='0.30', lw=1.6, label='neuron')
            ax_ts2.plot(x, yp, color='tab:red', lw=1.6, alpha=0.9,
                        label='predicted')
            ax_ts.set_title(
                f'best fold: held-out {cfg_label}   '
                f'r = {r_per_fold[best_fold]:.3f}',
                fontsize=base_fontsize + 1, loc='left',
            )
            ax_ts.set_xlabel('time bin', fontsize=base_fontsize)
            ax_ts.tick_params(labelsize=base_fontsize - 1)
            ax_ts2.tick_params(labelsize=base_fontsize - 1,
                               labelcolor='tab:red')
            ax_ts.set_ylabel('neuron (a.u.)',
                             fontsize=base_fontsize - 1, color='0.40')
            ax_ts2.set_ylabel('predicted',
                              fontsize=base_fontsize - 1, color='tab:red')
            ax_ts.spines['top'].set_visible(False)
            ax_ts2.spines['top'].set_visible(False)
            lines1, labels1 = ax_ts.get_legend_handles_labels()
            lines2, labels2 = ax_ts2.get_legend_handles_labels()
            ax_ts.legend(lines1 + lines2, labels1 + labels2,
                         fontsize=base_fontsize - 1,
                         loc='upper right', frameon=False)

            fig.suptitle(
                f'#{rank} {model} cell in {roi}   '
                f'sub-{sub_str} {neuron_label}   '
                f'mean r = {diag.get("mean_r", float("nan")):.3f}   '
                f'p_perm = {diag.get("p_perm", float("nan")):.3f}',
                fontsize=base_fontsize + 1, fontweight='bold',
            )
            fname = (f'top{rank}_{model}_{roi}_sub-{sub_str}_{neuron_label}'
                     .replace('/', '_'))
            base = os.path.join(save_dir, fname)
            fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
            fig.savefig(base + '.svg', bbox_inches='tight')
            fig.savefig(base + '.pdf', bbox_inches='tight')
            plt.close(fig)
            saved += 1
    print(f'  [top-cells per (roi, model)] saved {saved} figures → '
          f'{save_dir}')


# ─────────────────────────────────────────────────────────────────────
# RSA-pipeline helpers shared with RSA_DSR_ROIs_simple.py
# ─────────────────────────────────────────────────────────────────────


def plot_phase_mask_diagnostic(mask_matrices, n_configs, n_conds_per_config,
                                configs=None, save_path=None, suptitle=None):
    """Side-by-side N×N phase masks (one panel per mode).

    Parameters
    ----------
    mask_matrices : dict[str, ndarray]
        ``{mode_name: (N, N) bool array}``. Each matrix marks which RDM
        cells the corresponding masking mode keeps (True = kept).
    n_configs, n_conds_per_config : int
        Used to draw red config-boundary lines and tick labels.
    configs : list[str] or None
        Optional config strings; if given, used as tick labels.
    save_path : str or None
    suptitle : str or None
    """
    if not mask_matrices:
        return None
    modes = list(mask_matrices.keys())
    n = n_configs * n_conds_per_config
    fig, axes = plt.subplots(1, len(modes), figsize=(4.5 * len(modes), 4.5))
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        M = np.asarray(mask_matrices[mode]).astype(int)
        ax.imshow(M, cmap='Greys_r', vmin=0, vmax=1, aspect='equal')
        ax.set_title(f"mask: {mode}\n(white = kept, black = excluded)",
                     fontsize=10)
        for c in range(1, n_configs):
            ax.axvline(c * n_conds_per_config - 0.5, color='red', lw=0.7)
            ax.axhline(c * n_conds_per_config - 0.5, color='red', lw=0.7)
        ticks = (np.arange(n_configs) * n_conds_per_config
                 + n_conds_per_config / 2)
        labels = (configs if configs is not None
                  else [str(i) for i in range(n_configs)])
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8,
                                                  rotation=40, ha='right')
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f'config ({n_conds_per_config} conds each)', fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_model_design_matrices(model_concat, models, n_configs,
                                n_conds_per_config, save_path=None,
                                roi_label=None):
    """One imshow panel per model showing the (conditions × features) design.

    Object-dtype columns (e.g. button labels) are factorised to integers
    for display only. Red horizontal lines mark config boundaries every
    ``n_conds_per_config`` rows. Run-1 half only.
    """
    show = [m for m in models if m in model_concat]
    if not show:
        return None
    half1 = slice(0, n_configs * n_conds_per_config)
    fig, axes = plt.subplots(1, len(show),
                              figsize=(2.2 * len(show), 4.5),
                              constrained_layout=True)
    if len(show) == 1:
        axes = [axes]
    for ax, mname in zip(axes, show):
        raw = np.asarray(model_concat[mname][half1])
        if raw.dtype.kind in ('O', 'U', 'S'):
            uniq, inv = np.unique(raw.ravel().astype(str), return_inverse=True)
            mat = inv.reshape(raw.shape).astype(float)
            tag = f' (categorical, {len(uniq)} levels)'
        else:
            mat = np.asarray(raw, dtype=float)
            tag = ''
        im = ax.imshow(mat, aspect='auto', cmap='viridis',
                       interpolation='nearest')
        ax.set_title(f'{mname}{tag}\n{mat.shape[0]}×{mat.shape[1]}',
                     fontsize=9)
        ax.set_xlabel('features', fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('conditions (run-1 half)', fontsize=8)
        for k in range(1, n_configs):
            ax.axhline(k * n_conds_per_config - 0.5,
                       color='red', lw=0.4, alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if roi_label:
        fig.suptitle(f'Model design matrices (run-1 half) — {roi_label}',
                     fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_rdm_grid(rows_to_plot, col_specs, n_rdm, n_configs,
                   n_conds_per_config, configs=None,
                   suptitle=None, save_path=None, cmap='RdBu_r'):
    """N×N RDM display grid for diagnostic comparisons.

    Each row of the grid is one matrix family (e.g. ``data``, or one
    model RDM). Each column is one cell-subset (``full``, ``block_diag``,
    ``off_block``). Cells contain the *exact* 1-D vector that fed
    evaluate_model, embedded back into the N×N upper-tri positions
    where they live — lower-tri + diagonal stay NaN so the figure shows
    only what the regression actually saw.

    Parameters
    ----------
    rows_to_plot : list[(row_label, dict)]
        ``dict`` maps column name -> ``(vec_1d, positions_mask_bool)``.
        ``positions_mask_bool`` is length ``N*(N-1)/2`` (upper-tri order
        of ``np.triu_indices(n_rdm, k=1)``) and marks where vec_1d lives.
    col_specs : list[(col_name, title_suffix)]
    n_rdm : int
        Side length of the displayed matrix.
    """
    ii, jj = np.triu_indices(n_rdm, k=1)
    cfg_centres = (np.arange(n_configs) * n_conds_per_config
                   + n_conds_per_config / 2 - 0.5)

    def _embed(vec_1d, pos_mask):
        M = np.full((n_rdm, n_rdm), np.nan, dtype=float)
        v = np.asarray(vec_1d, dtype=float)
        m = np.asarray(pos_mask, dtype=bool)
        assert v.size == int(m.sum()), (
            f'plot_rdm_grid: vec length {v.size} '
            f'!= positions_mask.sum() {int(m.sum())}')
        M[ii[m], jj[m]] = v
        return M

    def _decorate(ax):
        for k in range(1, n_configs):
            ax.axvline(k * n_conds_per_config - 0.5,
                       color='black', lw=0.4, alpha=0.6)
            ax.axhline(k * n_conds_per_config - 0.5,
                       color='black', lw=0.4, alpha=0.6)
        # Red diagonal marker: evaluate_model never sees those cells.
        ax.plot([-0.5, n_rdm - 0.5], [-0.5, n_rdm - 0.5],
                color='red', lw=0.8, alpha=0.7)
        ax.set_xticks(cfg_centres)
        ax.set_yticks(cfg_centres)
        if configs is not None:
            ax.set_xticklabels(configs, fontsize=5, rotation=60, ha='right')
            ax.set_yticklabels(configs, fontsize=5)
        ax.tick_params(length=2, pad=1)

    nrows = len(rows_to_plot)
    ncols = len(col_specs)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2 * ncols, 3.0 * nrows),
                              constrained_layout=True)
    if nrows == 1:
        axes = np.array(axes)[None, :]
    if ncols == 1:
        axes = np.array(axes)[:, None]
    for r_i, (lbl, col_dict) in enumerate(rows_to_plot):
        for c_i, (col_name, suffix) in enumerate(col_specs):
            ax = axes[r_i, c_i]
            if col_name not in col_dict:
                ax.set_visible(False)
                continue
            vec, subset = col_dict[col_name]
            M = _embed(vec, subset)
            im = ax.imshow(M, aspect='equal', cmap=cmap,
                           interpolation='nearest')
            _decorate(ax)
            if r_i == 0:
                n_pairs = int((~np.isnan(M)).sum())
                ax.set_title(f'{col_name}{suffix}\nn_pairs = {n_pairs}',
                             fontsize=8)
            if c_i == 0:
                ax.set_ylabel(lbl, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if suptitle:
        fig.suptitle(suptitle + '\n(diagonal excluded; red line marks k=0)',
                     fontsize=10)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_permutation_hist_combo_grid(perm_results_combo,
                                      empirical_combo_results,
                                      empirical_combo_results_z,
                                      combo_key, combo_models, tests,
                                      bins=30, density=True,
                                      figsize_per_panel=(2.0, 1.8),
                                      alpha=0.05, suptitle=None):
    """Permutation-null histogram grid (rows = tests, cols = sub-models).

    For each (test, sub-model) cell draws the permutation-null histogram
    (grey) and the empirical β as a vertical line in ``OBSERVED_VALUE_COLOR``.
    Significant positive effects get a black star above the bar.
    """
    cols = combo_models[combo_key]
    nrows, ncols = len(tests), len(cols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols,
                 figsize_per_panel[1] * nrows),
        sharey=False, constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for r, test in enumerate(tests):
        emp = (empirical_combo_results_z[test[:-2]]
               if test.endswith('_z') else empirical_combo_results[test])
        x_all = np.asarray(perm_results_combo[test][combo_key]['beta'],
                            dtype=float)
        beta_emp = np.asarray(emp[combo_key][1], dtype=float).ravel()
        if x_all.ndim != 2:
            raise ValueError(
                f'{test}/{combo_key}: expected permuted beta array of shape '
                f'(n_permutations, n_combo_models), got {x_all.shape}. '
                f'Store the full beta vector per permutation.')

        row_vals = np.concatenate([x_all.ravel(), beta_emp.ravel()])
        lim = np.nanmax(np.abs(row_vals))
        lim = 1.0 if (not np.isfinite(lim) or lim == 0) else 1.05 * lim
        edges = np.linspace(-lim, lim, bins + 1)

        for c, model_name in enumerate(cols):
            ax = axes[r, c]
            x = x_all[:, c]
            beta = beta_emp[c]
            p_one_sided = (np.sum(x >= beta) + 1) / (x.size + 1)

            ax.hist(x, bins=edges, density=density,
                    color='0.75', edgecolor='white', linewidth=0.6)
            ax.axvline(0, color='black', lw=0.9)
            ax.axvline(beta, color=OBSERVED_VALUE_COLOR, lw=1.8)

            if p_one_sided < 0.1:
                ax.text(0.04, 0.96, f'p={p_one_sided:.3f}',
                        transform=ax.transAxes,
                        ha='left', va='top', fontsize=8)
            if (beta > 0) and (p_one_sided < alpha):
                y0, y1 = ax.get_ylim()
                ax.set_ylim(y0, y1 * 1.15)
                ax.text(beta, y1 * 1.08, '★',
                        ha='center', va='bottom',
                        fontsize=16, fontweight='bold', color='black')
            else:
                y0, y1 = ax.get_ylim()
                ax.set_ylim(y0, y1 * 1.08)

            ax.set_xlim(-lim, lim)
            ax.tick_params(labelsize=8, length=2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if r == 0:
                ax.set_title(model_name, fontsize=9)
            if c == 0:
                ax.set_ylabel(test, fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig, axes


# ── Generic ROI × lag plots (shared across per-lag encoding analyses) ──
def _save_pdf_png(fig, save_stem, dpi=300):
    fig.savefig(save_stem + '.pdf', dpi=dpi, bbox_inches='tight')
    fig.savefig(save_stem + '.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_roi_lag_tstat_heatmap(t_matrix, lags_deg, rois,
                                q_matrix=None,
                                predicted_lags_per_roi=None,
                                save_stem=None, title=None,
                                cm_inch=1 / 2.54,
                                font_tick=9, font_axis=10, font_big=11):
    """ROI × lag heatmap of one-sample t-stats with optional FDR stars.

    Parameters
    ----------
    t_matrix : (n_rois, n_lags) array of t-stats (one-sided > 0).
    lags_deg : list of int lags in degrees, length n_lags.
    rois : list of str, length n_rois.
    q_matrix : optional (n_rois, n_lags) array of BH-FDR p-values; cells with
        q < .05/.01/.001 are starred (* / ** / ***).
    predicted_lags_per_roi : optional dict {roi: tuple of lags}; cells at
        predicted lags get a black outline.
    save_stem : path stem (without extension); writes .pdf + .png.
    """
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       font_tick,
        'pdf.fonttype':    42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    T = np.asarray(t_matrix, dtype=float)
    Q = np.asarray(q_matrix, dtype=float) if q_matrix is not None else None
    n_rois, n_lags = T.shape
    vmax = float(np.nanmax(np.abs(T))) if np.isfinite(T).any() else 1.0
    fig, ax = plt.subplots(figsize=(14 * cm_inch, max(3.5, 0.55 * n_rois) * cm_inch),
                            constrained_layout=True)
    im = ax.imshow(T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    # Outline predicted lags ----------------------------------------
    if predicted_lags_per_roi:
        for ri, roi in enumerate(rois):
            for tl in predicted_lags_per_roi.get(roi, ()):
                if tl in lags_deg:
                    ci = lags_deg.index(tl)
                    ax.add_patch(plt.Rectangle(
                        (ci - 0.5, ri - 0.5), 1, 1,
                        fill=False, edgecolor='black', lw=1.2))
    # FDR stars -----------------------------------------------------
    if Q is not None:
        for ri in range(n_rois):
            for ci in range(n_lags):
                q = Q[ri, ci]
                if not np.isfinite(q):
                    continue
                s = p_to_stars(q)
                if not s:
                    continue
                col = 'white' if abs(T[ri, ci]) > vmax * 0.55 else 'black'
                ax.text(ci, ri, s, ha='center', va='center',
                        fontsize=font_tick, fontweight='bold', color=col)
    ax.set_xticks(range(n_lags))
    ax.set_xticklabels([str(l) for l in lags_deg], fontsize=font_tick)
    ax.set_yticks(range(n_rois))
    ax.set_yticklabels([roi_display(r) for r in rois], fontsize=font_tick)
    ax.set_xlabel('lag (°)', fontsize=font_axis)
    if title:
        ax.set_title(title, fontsize=font_big)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label('t (one-sided > 0)', fontsize=font_tick)
    cb.ax.tick_params(labelsize=font_tick)
    if save_stem is not None:
        _save_pdf_png(fig, save_stem)
        return None
    return fig, ax


def plot_roi_lag_curves(curves_per_roi, lags_deg,
                         predicted_lags_per_roi=None,
                         roi_colours=None,
                         save_stem=None, title=None,
                         cm_inch=1 / 2.54,
                         font_tick=9, font_axis=10, font_big=11):
    """Per-ROI line plot of mean CV r ± SEM across lags.

    Parameters
    ----------
    curves_per_roi : dict {roi: (n_cells, n_lags) array}
    lags_deg : list of int.
    predicted_lags_per_roi : optional dict {roi: tuple of predicted lags}
        — drawn as dotted dark-green vertical lines on the matching subplot.
    roi_colours : optional dict {roi: hex} — falls back to gray.
    """
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       font_tick,
        'pdf.fonttype':    42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    rois = list(curves_per_roi.keys())
    n = len(rois)
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.4 * cm_inch * n_cols,
                                        3.6 * cm_inch * n_rows),
                              constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis('off')
    x = np.asarray(lags_deg, dtype=float)
    for ax, roi in zip(axes_flat, rois):
        M = np.asarray(curves_per_roi[roi], dtype=float)
        if M.size == 0:
            ax.axis('off'); continue
        m_curve = np.nanmean(M, axis=0)
        s_curve = np.nanstd(M, axis=0, ddof=1) / np.sqrt(
            np.maximum(np.isfinite(M).sum(axis=0), 1)
        )
        col = (roi_colours or {}).get(roi, '#888')
        ax.fill_between(x, m_curve - s_curve, m_curve + s_curve,
                        color=col, alpha=0.25, linewidth=0)
        ax.plot(x, m_curve, color=col, lw=1.6, marker='o', ms=2.5,
                label=f'n = {M.shape[0]}')
        ax.axhline(0, color='black', lw=0.5, ls='--')
        if predicted_lags_per_roi:
            for tl in predicted_lags_per_roi.get(roi, ()):
                ax.axvline(tl, color=OBSERVED_VALUE_COLOR, lw=0.9,
                           ls=':', alpha=0.8)
        ax.set_title(roi_display(roi), fontsize=font_tick)
        ax.set_xlabel('lag (°)', fontsize=font_tick)
        ax.set_ylabel('mean CV r', fontsize=font_tick)
        ax.set_xticks(lags_deg[::2])
        ax.tick_params(axis='both', labelsize=font_tick, length=2, pad=1)
        ax.legend(fontsize=font_tick - 1, frameon=False, loc='best')
    if title:
        fig.suptitle(title, fontsize=font_big)
    if save_stem is not None:
        _save_pdf_png(fig, save_stem)
        return None
    return fig, axes
