#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure builder for the single-model "current location" RSA across human
single-unit ROIs.

WHAT THIS IS
------------
The manuscript reports two different things about location coding:

  1. the JOINT GLM (`combo_models` in RSA_DSR_ROIs_simple.py), where the
     `location` regressor competes with DSR and the other controls, and
  2. this figure: `location` fitted ON ITS OWN, i.e. "without controlling
     for the concurrent code", to show that middle and anterior
     hippocampus are the only ROIs whose population geometry fits the
     current-location model at all.

This script only PLOTS (2). Nothing is refitted: the t-values and
permutation p-values are read verbatim from a completed
`RSA_DSR_ROIs_simple.py` run's `results_summary.csv` (rows with
`model == 'location'`, `test == split_halves_z`). The only number computed
here is the BH-FDR correction across the ROIs shown, which is the
pre-specified family for this single-hypothesis figure (one model, every
ROI analysed).

OUTPUTS (in <run_dir>/location_only_figure_<date>/)
  heatmap_location_only_<test>.pdf/.png     ROI x location t-heatmap,
                                            same style as pub_figures_v2
  glassbrain_location_only_sagittal.pdf/.png  left sagittal glass brain,
                                            ROI masks shaded by t,
                                            significant ROIs outlined
  glassbrain_location_only_lyrz.pdf/.png    all four standard views
  panel_location_only.pdf/.png              heatmap + sagittal brain in
                                            one panel with a shared
                                            colour bar (figure-ready)
  location_only_values.csv                  exactly the numbers plotted
  config.json                               settings used

USAGE
    conda activate env_multiple_clocks
    python scripts/RSA_DSR_location_only_figure.py [<run_dir>]
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mc.plotting.results import _stars
from mc.plotting.cell_results import plot_roi_beta_glassbrain

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
            '/ephys_humans/derivatives')
DEFAULT_RUN = os.path.join(DATA_DIR, 'group', 'DSR_RSA_simple_ROI',
                           '2026-08-31_17-57-30')

MODEL = 'location'          # single-model RSA row to plot
TEST = 'split_halves_z'     # primary RSA variant (across-run, z-scored)
ROI_LABEL_COLUMN = 'alt_final_roi'
ALPHA = 0.05
TVAL_LIM = 5.0              # symmetric colour limit, matches the glass brain

# Row order for the heatmap; ROIs missing from the run are dropped.
ROI_ORDER = ['mPFC', 'mOFC', 'PCC', 'PHC', 'HC_anterior', 'HC_mid', 'EC']
ROI_ROW_LABELS = {'HC_anterior': 'HC ant', 'HC_mid': 'HC mid'}
# The saved tables already use canonical names; kept for older runs.
ROI_NAME_MAP = {'ACC': 'mPFC', 'medialOFC': 'mOFC', 'medial OFC': 'mOFC',
                'Parahippocampal': 'PHC', 'Parahippocampus': 'PHC'}

FONT_AXIS = 11
FONT_BIG = 11
CM = 1 / 2.54


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values; NaNs pass through."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def load_location_rows(run_dir):
    """The single-model `location` row per ROI, in ROI_ORDER."""
    df = pd.read_csv(Path(run_dir) / 'results_summary.csv')
    df['roi'] = df['roi'].map(lambda r: ROI_NAME_MAP.get(r, r))
    df = df[(df['test'] == TEST) & (df['model'] == MODEL)].copy()
    if df.empty:
        raise ValueError(f"No {MODEL!r} rows for test={TEST!r} in {run_dir}")
    df = df.set_index('roi')
    rois = [r for r in ROI_ORDER if r in df.index]
    df = df.loc[rois].reset_index()
    df['q_fdr'] = bh_fdr(df['p_perm'].values)
    return df


def draw_heatmap(ax, rows):
    """One-column ROI x t heatmap, pub_figures_v2 look, into `ax`."""
    t_col = rows['t'].values[:, None]
    im = ax.imshow(t_col, cmap='RdBu_r', vmin=-TVAL_LIM, vmax=TVAL_LIM,
                   aspect='auto')
    for i, q in enumerate(rows['q_fdr'].values):
        s = _stars(q)
        if not s:
            continue
        col = 'white' if abs(t_col[i, 0]) / TVAL_LIM > 0.55 else 'black'
        ax.text(0, i, s, ha='center', va='center', color=col,
                fontsize=FONT_BIG + 1, fontweight='bold', zorder=5)
    ax.set_xticks([])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([ROI_ROW_LABELS.get(r, r) for r in rows['roi']],
                       fontsize=FONT_AXIS)
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='both', length=1.5, pad=1)
    return im


def cell_coords_per_roi(run_dir, rois):
    """MNI coordinates of the recorded cells, per ROI, from the run."""
    path = Path(run_dir) / 'roi_electrode_coords.csv'
    if not path.exists():
        print(f"[warn] {path.name} missing — masks will not be restricted "
              f"to recorded sites.")
        return None
    coords = pd.read_csv(path)
    coords['roi'] = coords['roi'].map(lambda r: ROI_NAME_MAP.get(r, r))
    out = {}
    for roi in rois:
        sub = coords[coords['roi'] == roi]
        if not sub.empty:
            out[roi] = sub[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    return out


def main(run_dir=DEFAULT_RUN):
    run_dir = Path(run_dir)
    out_dir = run_dir / f'location_only_figure_{date.today():%Y-%m-%d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_location_rows(run_dir)
    rois = rows['roi'].tolist()
    print(f"\n=== single-model '{MODEL}' RSA — {TEST} — {run_dir.name} ===")
    print(rows[['roi', 'n_neurons', 't', 'beta', 'p_perm',
                'q_fdr']].to_string(index=False))
    rows.to_csv(out_dir / 'location_only_values.csv', index=False)

    t_vals = dict(zip(rows['roi'], rows['t']))
    q_vals = dict(zip(rows['roi'], rows['q_fdr']))
    coords = cell_coords_per_roi(run_dir, rois)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    # ── 1. Stand-alone heatmap ───────────────────────────────────────
    # Same look as the pub_figures_v2 heatmaps (RdBu_r on a shared t
    # scale, bold BH-FDR stars), drawn here rather than through
    # `plot_roi_tstat_heatmap` because that helper spaces its colour bar
    # for a multi-column panel and leaves a large gap at one column.
    fig_hm = plt.figure(figsize=(4.0 * CM, 6.0 * CM))
    gs_hm = fig_hm.add_gridspec(2, 1, height_ratios=[1, 0.09],
                                left=0.42, right=0.95, top=0.97,
                                bottom=0.16, hspace=0.14)
    ax_hm = fig_hm.add_subplot(gs_hm[0, 0])
    im_hm = draw_heatmap(ax_hm, rows)
    cb_hm = fig_hm.colorbar(im_hm, cax=fig_hm.add_subplot(gs_hm[1, 0]),
                            orientation='horizontal')
    cb_hm.set_ticks([-TVAL_LIM, 0, TVAL_LIM])
    cb_hm.set_label('t vs 0', fontsize=FONT_AXIS, labelpad=1)
    cb_hm.ax.tick_params(labelsize=FONT_AXIS, length=2, pad=1)
    for ext in ('.pdf', '.png'):
        fig_hm.savefig(out_dir / f'heatmap_location_only_{TEST}{ext}',
                       dpi=300, bbox_inches='tight')
    plt.close(fig_hm)
    print(f"[fig] heatmap_location_only_{TEST}.pdf/.png")

    # ── 2. Stand-alone glass brains ──────────────────────────────────
    for tag, mode in (('sagittal', 'l'), ('lyrz', 'lyrz')):
        fig_gb = plot_roi_beta_glassbrain(
            roi_betas=t_vals, roi_pvals=q_vals,
            only_rois=rois, roi_cell_coords=coords,
            roi_label_column=ROI_LABEL_COLUMN,
            roi_display_names=ROI_ROW_LABELS,
            significance_label='q_FDR (across ROIs)',
            alpha_threshold=ALPHA,
            display_mode=mode, is_t_val=True, title='',
        )
        stem = out_dir / f'glassbrain_location_only_{tag}'
        for ext in ('.pdf', '.png'):
            fig_gb.savefig(str(stem) + ext, dpi=300, bbox_inches='tight')
        plt.close(fig_gb)
        print(f"[fig] {stem.name}.pdf/.png")

    # ── 3. Combined figure panel: heatmap + sagittal brain ───────────
    fig = plt.figure(figsize=(9.5 * CM, 5.5 * CM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.6],
                          height_ratios=[1, 0.10],
                          left=0.20, right=0.98, top=0.97, bottom=0.16,
                          wspace=0.05, hspace=0.10)
    ax_hm = fig.add_subplot(gs[0, 0])
    im = draw_heatmap(ax_hm, rows)
    ax_gb = fig.add_subplot(gs[0, 1])
    plot_roi_beta_glassbrain(
        roi_betas=t_vals, roi_pvals=q_vals,
        only_rois=rois, roi_cell_coords=coords,
        roi_label_column=ROI_LABEL_COLUMN,
        roi_display_names=ROI_ROW_LABELS,
        alpha_threshold=ALPHA,
        display_mode='l', is_t_val=True, title='',
        figure=fig, axes=ax_gb,
        draw_colorbar=False, draw_footer=False,
    )
    cax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks([-TVAL_LIM, 0, TVAL_LIM])
    cbar.set_label('t vs 0', fontsize=FONT_AXIS, labelpad=1)
    cbar.ax.tick_params(labelsize=FONT_AXIS, length=2, pad=1)
    for ext in ('.pdf', '.png'):
        fig.savefig(out_dir / f'panel_location_only{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print("[fig] panel_location_only.pdf/.png")

    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            'source_run': str(run_dir),
            'source_table': 'results_summary.csv (single-model RSA)',
            'model': MODEL, 'test': TEST,
            'rois_plotted': rois,
            'n_neurons': dict(zip(rois, rows['n_neurons'].astype(int).tolist())),
            'multiple_comparisons': (
                f'BH-FDR across the {len(rois)} ROIs shown, on the '
                f'permutation p-values of the single-model {MODEL} fit'),
            'alpha': ALPHA,
            'colour_limits_t': [-TVAL_LIM, TVAL_LIM],
            'roi_label_column': ROI_LABEL_COLUMN,
            'note': ('No refitting — t and p_perm are taken verbatim from '
                     'the source run. This is the analysis WITHOUT the DSR '
                     'and other control regressors in the model.'),
            'created': str(date.today()),
        }, f, indent=2)
    print(f"\nAll outputs → {out_dir}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN)
