#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication panels for the human-cell encoding + RSA analyses.

Loads:
- RSA single-model results (`results_summary.csv`)        — `dsr_old` per ROI
- RSA combo-model results  (`results_summary_combos.csv`) — `location`-control
- Encoding results CSV — full subjects
- Encoding results CSV — non-RSA subjects only (subject-level subset that drops
  the DSR-RSA-grouping subjects; produced by `encoding_analysis_simple.py`
  with `SUBSET_REPLOT='none_RSA'`).

Every panel is rendered in FOUR variants of the neuron subset:
    1. all              — every encoding neuron
    2. nonRSA           — only subjects NOT in the DSR-RSA grouping
    3. all_minus_state  — all subjects, drop state-encoders
    4. nonRSA_minus_state — non-RSA subjects, drop state-encoders

Subpanels per variant (saved as both PDF + PNG into
``<ENC_FULL_DIR>/publication_panels/``):
- panel_A_{variant}  : RSA + encoding heatmap pair (DSR vs Location control)
- panel_B_{variant}  : ACC DSR mean_r histogram (mean drawn as a red line)
- panel_C_{variant}  : DSR significant-cell fraction per ROI (BH-FDR stars)

A master ``combined_publication_figure.{pdf,png}`` lays the four variants
out side by side so the design can be reviewed at a glance.

@author: Svenja Küchenhoff (panel script: Claude)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import stats

import era_brewer
import mc.plotting.figure_layout as figure_layout
from mc.plotting.cell_results import (
    compute_roi_model_tstats, plot_roi_model_heatmap,
    CANONICAL_ROI_ORDER,
)


# ─── Settings ─────────────────────────────────────────────────────────────
RSA_RUN_DIR    = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
                  'ephys_humans/derivatives/group/DSR_RSA_simple_ROI/'
                  '2026-05-26_17-54-23')
ENC_FULL_DIR   = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
                  'ephys_humans/derivatives/group/encoding_analysis_simple/'
                  '2026-05-28_16-45-09-nopenality-newROIs')
ENC_NONRSA_DIR = ENC_FULL_DIR + '-none-RSA-subset'

OUT_DIR = os.path.join(ENC_FULL_DIR, 'publication_panels')
os.makedirs(OUT_DIR, exist_ok=True)

RSA_TEST            = 'across_z'
RSA_CONTROL_COMBO   = 'MRI_combo-nofdb_midn'
DSR_MODEL_RSA       = 'dsr_old'
DSR_MODEL_ENC       = 'dsr'
CONTROL_MODEL       = 'location'
STATE_MODELS_REMOVE = ['state', 'state_phase']

ROI_ORDER = ['ACC', 'medial_CC', 'HC_anterior', 'HC_mid', 'EC',
             'Parahippocampal', 'PCC', 'medialOFC', 'Visual']

ALPHA               = 0.05
TARGET_FONT_PT      = 9
CM_PER_IN           = figure_layout.CM_PER_IN
MEAN_LINE_COLOUR    = 'red'    # user-requested: mean is drawn in red


# ─── Load data ────────────────────────────────────────────────────────────
print(f"Loading RSA  results from: {RSA_RUN_DIR}")
rsa_single = pd.read_csv(os.path.join(RSA_RUN_DIR, 'results_summary.csv'))
rsa_combos = pd.read_csv(os.path.join(RSA_RUN_DIR, 'results_summary_combos.csv'))

print(f"Loading enc  (all) from:   {ENC_FULL_DIR}")
enc_full   = pd.read_csv(os.path.join(ENC_FULL_DIR,   'encoding_results.csv'))
print(f"Loading enc  (non-RSA):    {ENC_NONRSA_DIR}")
enc_nonrsa = pd.read_csv(os.path.join(ENC_NONRSA_DIR, 'encoding_results.csv'))


# ─── RSA stats table (DSR + control) — unchanged across variants ─────────
def _rsa_stats(dsr_model, control_model, combo_name, rsa_test):
    dsr_rows = rsa_single[(rsa_single['model'] == dsr_model)
                          & (rsa_single['test'] == rsa_test)].copy()
    dsr_rows = dsr_rows.rename(columns={'n_neurons': 'n_cells'})
    dsr_rows['model'] = 'DSR'
    ctl_rows = rsa_combos[(rsa_combos['combo'] == combo_name)
                          & (rsa_combos['sub_model'] == control_model)
                          & (rsa_combos['test'] == rsa_test)].copy()
    ctl_rows = ctl_rows.rename(columns={'n_neurons': 'n_cells'})
    ctl_rows['model'] = 'Location'
    return pd.concat([
        dsr_rows[['roi', 'model', 'beta', 'p_perm', 't', 'n_cells']],
        ctl_rows[['roi', 'model', 'beta', 'p_perm', 't', 'n_cells']],
    ], ignore_index=True)


rsa_stats = _rsa_stats(DSR_MODEL_RSA, CONTROL_MODEL,
                       RSA_CONTROL_COMBO, RSA_TEST)


# ─── Encoding-variant builder ─────────────────────────────────────────────
# ``variants`` is the list of all four neuron-subset choices we apply across
# every panel. Each entry yields the filtered encoding DataFrame to plot.
def _drop_state_encoders(df, state_models=STATE_MODELS_REMOVE, alpha=ALPHA):
    excl = set(df.loc[(df['model'].isin(state_models))
                      & (df['p_perm'] < alpha), 'neuron'].unique())
    return df[~df['neuron'].isin(excl)]


VARIANTS = [
    ('all',                'all encoding neurons',
     lambda d=enc_full:   d),
    ('nonRSA',             'non-RSA subjects',
     lambda d=enc_nonrsa: d),
    ('all_minus_state',    'all encoding minus state',
     lambda d=enc_full:   _drop_state_encoders(d)),
    ('nonRSA_minus_state', 'non-RSA minus state',
     lambda d=enc_nonrsa: _drop_state_encoders(d)),
]


def _enc_stats(enc_df, dsr_model=DSR_MODEL_ENC, control_model=CONTROL_MODEL,
               alpha=ALPHA):
    sub = enc_df[enc_df['model'].isin([dsr_model, control_model])]
    s = compute_roi_model_tstats(sub, models=[dsr_model, control_model],
                                 alpha=alpha)
    return s.replace({'model': {dsr_model: 'DSR',
                                 control_model: 'Location'}})


# ─── Panel drawing primitives ─────────────────────────────────────────────
def _draw_one_heatmap(ax, stats_df, value_col, p_col, n_col, value_label,
                      rois=ROI_ORDER, models=('DSR', 'Location'),
                      vmin=None, vmax=None, font_pt=TARGET_FONT_PT,
                      title=None, show_cbar=True, cbar_label=None):
    rois = [r for r in rois if r in set(stats_df['roi'])]
    val_mat = np.full((len(rois), len(models)), np.nan)
    p_mat   = np.full_like(val_mat, np.nan)
    n_per_roi = {}
    for i, roi in enumerate(rois):
        sub = stats_df[stats_df['roi'] == roi]
        if not sub.empty and n_col in sub.columns:
            n_per_roi[roi] = int(sub[n_col].max())
        for j, m in enumerate(models):
            r = sub[sub['model'] == m]
            if r.empty:
                continue
            val_mat[i, j] = float(r[value_col].iloc[0])
            p_mat[i, j]   = float(r[p_col].iloc[0])

    finite = val_mat[np.isfinite(val_mat)]
    if vmax is None:
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if vmin is None:
        vmin = -vmax

    im = ax.imshow(val_mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(len(rois)):
        for j in range(len(models)):
            p = p_mat[i, j]
            if np.isfinite(p) and p < ALPHA:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor='black',
                                           linewidth=1.5))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=font_pt, rotation=30, ha='right')
    roi_labels = [f'{r}\n(n={n_per_roi.get(r, 0)})' for r in rois]
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(roi_labels, fontsize=font_pt - 1)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title, fontsize=font_pt + 1)
    if show_cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
        cb.ax.tick_params(labelsize=font_pt - 1)
        cb.set_label(cbar_label or value_label, fontsize=font_pt - 1)
    return im


def _draw_acc_histogram(ax, enc_df, font_pt=TARGET_FONT_PT, title=None,
                        bar_colour=None):
    """Single ACC DSR histogram with a vertical red mean line + t-test star."""
    pal = era_brewer.era_brew('Lover2')
    bar_colour = bar_colour or pal[1]    # deep blue
    rows = enc_df[(enc_df['roi'] == 'ACC')
                  & (enc_df['model'] == DSR_MODEL_ENC)]
    r = rows['mean_r'].dropna().values
    n_sig = int((rows['p_perm'] < ALPHA).sum())
    if not r.size:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=font_pt, color='0.5')
        ax.set_xticks([]); ax.set_yticks([])
        return

    lim  = max(0.05, 1.05 * float(np.max(np.abs(r))))
    bins = np.linspace(-lim, lim, 22)
    ax.hist(r, bins=bins, color=bar_colour, alpha=0.65, edgecolor='none')

    mean_r = float(np.mean(r))
    ax.axvline(mean_r, color=MEAN_LINE_COLOUR, lw=1.5,
               label=f'mean = {mean_r:+.3f}')
    ax.axvline(0, color='0.4', lw=0.5, ls=':')

    if r.size >= 2:
        p_gt0 = float(stats.ttest_1samp(r, 0.0, alternative='greater').pvalue)
        sig = ('***' if p_gt0 < 0.001 else '**' if p_gt0 < 0.01
               else '*' if p_gt0 < 0.05 else 'n.s.')
        ax.text(0.97, 0.95, sig, transform=ax.transAxes,
                ha='right', va='top', fontsize=font_pt + 2, color='black')
    else:
        p_gt0 = np.nan

    ax.set_xlim(-lim, lim)
    ax.set_xlabel('mean held-out r', fontsize=font_pt)
    ax.set_ylabel('# neurons', fontsize=font_pt)
    if title:
        ax.set_title(title, fontsize=font_pt)
    legend_text = f'n={len(r)}, sig={n_sig}'
    ax.text(0.03, 0.95, legend_text, transform=ax.transAxes,
            ha='left', va='top', fontsize=font_pt - 1, color='0.25')
    ax.legend(fontsize=font_pt - 2, frameon=False, loc='upper left',
              bbox_to_anchor=(0.03, 0.86))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]; n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qok = np.empty(n); qok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = qok
    return q


def _draw_sig_bars(ax, enc_df, font_pt=TARGET_FONT_PT, title=None,
                   bar_colour=None):
    pal = era_brewer.era_brew('Lover2')
    bar_colour = bar_colour or pal[1]
    rois_present = [r for r in ROI_ORDER if r in set(enc_df['roi'])]
    n_total, n_sig, p_binom = [], [], []
    for r in rois_present:
        sub = enc_df[(enc_df['roi'] == r)
                     & (enc_df['model'] == DSR_MODEL_ENC)]
        nt = len(sub); ns = int((sub['p_perm'] < ALPHA).sum())
        n_total.append(nt); n_sig.append(ns)
        if nt:
            res = stats.binomtest(ns, nt, p=ALPHA, alternative='greater')
            p_binom.append(float(res.pvalue))
        else:
            p_binom.append(np.nan)
    q = _bh_fdr(p_binom)

    if not n_total or max(n_total) == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=font_pt, color='0.5')
        ax.set_xticks([]); ax.set_yticks([])
        return

    x = np.arange(len(rois_present))
    ax.bar(x, n_total, color='0.85', edgecolor='none', label='total')
    ax.bar(x, n_sig,   color=bar_colour, edgecolor='none',
           label='permutation p < {:.2f}'.format(ALPHA))
    for i, (ns, nt, qq) in enumerate(zip(n_sig, n_total, q)):
        ax.text(i, nt + 0.02 * max(n_total),
                f'{ns}/{nt}', ha='center', va='bottom',
                fontsize=font_pt - 2)
        sig = ('***' if qq < 0.001 else '**' if qq < 0.01
               else '*' if qq < 0.05 else '')
        if sig:
            ax.text(i, nt + 0.12 * max(n_total), sig,
                    ha='center', va='bottom', fontsize=font_pt + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(rois_present, rotation=30, ha='right',
                       fontsize=font_pt - 1)
    ax.set_ylabel('# neurons', fontsize=font_pt)
    if title:
        ax.set_title(title, fontsize=font_pt)
    ax.legend(fontsize=font_pt - 2, frameon=False, loc='upper right')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(0, max(n_total) * 1.35)


# ─── Standalone panels per variant ────────────────────────────────────────
def save_panel_A(enc_stats, save_stem, subtitle, font_pt=TARGET_FONT_PT):
    figsize = (8.5 / CM_PER_IN, 10.0 / CM_PER_IN)
    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_pt}):
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        _draw_one_heatmap(axes[0], rsa_stats,
                          value_col='beta', p_col='p_perm', n_col='n_cells',
                          value_label='empirical β', title='RSA',
                          font_pt=font_pt, cbar_label='empirical β')
        _draw_one_heatmap(axes[1], enc_stats,
                          value_col='t', p_col='p_t', n_col='n_cells',
                          value_label='t-stat',
                          title=f'Encoding\n— {subtitle}', font_pt=font_pt,
                          cbar_label='t-stat')
        axes[1].set_yticklabels([])
        fig.savefig(save_stem + '.pdf', bbox_inches='tight')
        fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {save_stem}.pdf/.png")


def save_panel_B(enc_df, save_stem, subtitle, font_pt=TARGET_FONT_PT):
    figsize = (6.5 / CM_PER_IN, 5.0 / CM_PER_IN)
    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_pt}):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _draw_acc_histogram(ax, enc_df, font_pt=font_pt,
                            title=f'ACC — DSR\n{subtitle}')
        fig.savefig(save_stem + '.pdf', bbox_inches='tight')
        fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {save_stem}.pdf/.png")


def save_panel_C(enc_df, save_stem, subtitle, font_pt=TARGET_FONT_PT):
    figsize = (10.0 / CM_PER_IN, 4.0 / CM_PER_IN)
    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_pt}):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _draw_sig_bars(ax, enc_df, font_pt=font_pt,
                       title=f'DSR — significant cells per ROI ({subtitle})')
        fig.savefig(save_stem + '.pdf', bbox_inches='tight')
        fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {save_stem}.pdf/.png")


# Run every panel for every variant + collect per-variant stats for the JSON.
variant_stats_archive = {}
for tag, subtitle, getter in VARIANTS:
    enc_df_v = getter()
    enc_stats_v = _enc_stats(enc_df_v)
    save_panel_A(enc_stats_v,
                 save_stem=os.path.join(OUT_DIR, f'panel_A_{tag}'),
                 subtitle=subtitle)
    save_panel_B(enc_df_v,
                 save_stem=os.path.join(OUT_DIR, f'panel_B_{tag}'),
                 subtitle=subtitle)
    save_panel_C(enc_df_v,
                 save_stem=os.path.join(OUT_DIR, f'panel_C_{tag}'),
                 subtitle=subtitle)
    variant_stats_archive[tag] = {
        'subtitle':           subtitle,
        'encoding_per_roi':   enc_stats_v.to_dict(orient='records'),
        'n_neurons_total':    int(enc_df_v[enc_df_v['model'] == DSR_MODEL_ENC]
                                  ['neuron'].nunique()),
    }


# ─── Master combined figure: 4 variants × 3 columns (A | B | C) ──────────
def combined_figure(save_stem, font_pt=TARGET_FONT_PT):
    A4_W   = figure_layout.A4_WIDTH_IN
    margin = 1.5 / CM_PER_IN
    fig_w  = A4_W - 2 * margin
    # 4 rows of variants + 1 placeholder row for the modelled neuron
    fig_h  = 30.0 / CM_PER_IN

    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_pt}):
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
        gs  = fig.add_gridspec(5, 4,
                               height_ratios=[1.5, 1.5, 1.5, 1.5, 0.8],
                               width_ratios=[1.2, 1.2, 1.2, 2.4],
                               hspace=0.85, wspace=0.55,
                               left=0.06, right=0.97,
                               top=0.97, bottom=0.04)

        for row_idx, (tag, subtitle, getter) in enumerate(VARIANTS):
            enc_df_v   = getter()
            enc_stats_v = _enc_stats(enc_df_v)

            # Col 0: RSA heatmap (same across rows but kept for context)
            ax_rsa = fig.add_subplot(gs[row_idx, 0])
            _draw_one_heatmap(ax_rsa, rsa_stats,
                              value_col='beta', p_col='p_perm',
                              n_col='n_cells', value_label='β',
                              title=('RSA' if row_idx == 0 else None),
                              font_pt=font_pt, cbar_label='β')

            # Col 1: encoding heatmap for this variant
            ax_enc = fig.add_subplot(gs[row_idx, 1])
            _draw_one_heatmap(ax_enc, enc_stats_v,
                              value_col='t', p_col='p_t', n_col='n_cells',
                              value_label='t',
                              title=f'Encoding\n— {subtitle}',
                              font_pt=font_pt, cbar_label='t')
            ax_enc.set_yticklabels([])

            # Col 2: ACC DSR histogram
            ax_hist = fig.add_subplot(gs[row_idx, 2])
            _draw_acc_histogram(ax_hist, enc_df_v, font_pt=font_pt,
                                title=f'ACC — DSR\n{subtitle}')

            # Col 3: per-ROI sig bars
            ax_bars = fig.add_subplot(gs[row_idx, 3])
            _draw_sig_bars(ax_bars, enc_df_v, font_pt=font_pt,
                           title=f'DSR sig / ROI — {subtitle}')

        # Bottom row: modelled-neuron placeholder spanning full width.
        ax_ex = fig.add_subplot(gs[4, :])
        ax_ex.text(0.5, 0.5,
                   'modelled DSR neuron example  —  insert '
                   'fig12_sparse_dsr_examples_*.png here',
                   ha='center', va='center', fontsize=font_pt,
                   color='0.4', style='italic')
        ax_ex.set_xticks([]); ax_ex.set_yticks([])
        for sp in ('top', 'right', 'left', 'bottom'):
            ax_ex.spines[sp].set_linestyle('--')
            ax_ex.spines[sp].set_color('0.7')

        fig.savefig(save_stem + '.pdf', bbox_inches='tight')
        fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {save_stem}.pdf/.png")


combined_figure(save_stem=os.path.join(OUT_DIR, 'combined_publication_figure'))


# ─── Stats JSON ──────────────────────────────────────────────────────────
stats_archive = {
    'inputs': {
        'rsa_run':       RSA_RUN_DIR,
        'enc_full':      ENC_FULL_DIR,
        'enc_nonRSA':    ENC_NONRSA_DIR,
    },
    'settings': {
        'rsa_test':              RSA_TEST,
        'rsa_control_combo':     RSA_CONTROL_COMBO,
        'dsr_model_rsa':         DSR_MODEL_RSA,
        'dsr_model_enc':         DSR_MODEL_ENC,
        'control_model':         CONTROL_MODEL,
        'state_models_removed':  STATE_MODELS_REMOVE,
        'alpha':                 ALPHA,
        'roi_order':             ROI_ORDER,
        'variants':              [v[0] for v in VARIANTS],
    },
    'rsa_per_roi': rsa_stats.to_dict(orient='records'),
    'variants':    variant_stats_archive,
}
with open(os.path.join(OUT_DIR, 'publication_panels_stats.json'), 'w') as f:
    json.dump(stats_archive, f, indent=2,
              default=lambda o: float(o) if hasattr(o, 'item') else str(o))

print(f"\nWrote: {OUT_DIR}/publication_panels_stats.json")
print(f"All panels saved under: {OUT_DIR}")
