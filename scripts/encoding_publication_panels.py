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
                  '2026-06-04_16-52-55')

ENC_FULL_DIR   = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
                  'ephys_humans/derivatives/group/encoding_analysis_simple/'
                  '2026-06-05_17-58-57')


ENC_NONRSA_DIR = ENC_FULL_DIR + '-none-RSA-subset'

OUT_DIR = os.path.join(ENC_FULL_DIR, 'publication_panels')
os.makedirs(OUT_DIR, exist_ok=True)

RSA_TEST            = 'between_tasks_z'   # all-correct-repeats averaged
                                          # per config, z-scored per neuron,
                                          # between-task-config cells only
                                          # (within-config block masked).
RSA_COMBO           = 'MRI_combo-nofdb_midn-state'  # the combo we report from
DSR_MODEL_RSA       = 'dsr_old'
DSR_MODEL_ENC       = 'dsr'
CONTROL_MODEL       = 'bttn_curr'
STATE_MODELS_REMOVE = ['state']

ROI_ORDER = ['ACC', 'medial_CC', 'HC_anterior', 'HC_mid', 'EC',
             'Parahippocampal', 'PCC', 'medialOFC', 'Visual']

ALPHA               = 0.05
TARGET_FONT_PT      = 9
CM_PER_IN           = figure_layout.CM_PER_IN

# Explicit user-chosen hex codes.
COL_HIST_BG   = '#BC7E6A'   # histogram: all encoding neurons
COL_HIST_SIG  = '#9A383C'   # histogram: sig cells + mean line
COL_BAR_BG    = '#7191A9'   # bar plot: total per ROI
COL_BAR_SIG   = '#2B4159'   # bar plot: sig per ROI
COL_ZERO      = '0.25'      # 0-line on the histogram


# Display-name fixes — drop underscores from labels that go on the figure.
# Underlying CSV columns keep the raw names; this only changes display.
LABEL_MAP = {
    'HC_anterior':   'HC ant',
    'HC_mid':        'HC mid',
    'medial_CC':     'medial CC',
    'bttn_curr':     'current button',
    'DSR':           'DSR',
}


def pretty(label):
    """Display version of an internal name (ROI or model)."""
    return LABEL_MAP.get(label, str(label).replace('_', ' '))


# ─── Load data ────────────────────────────────────────────────────────────
# IMPORTANT: only the *combo* RSA table is consulted. The "unique" results
# in results_summary.csv are never used for publication panels.
print(f"Loading RSA combo results from: {RSA_RUN_DIR}")
rsa_combos = pd.read_csv(os.path.join(RSA_RUN_DIR, 'results_summary_combos.csv'))

print(f"Loading enc  (all) from:   {ENC_FULL_DIR}")
enc_full   = pd.read_csv(os.path.join(ENC_FULL_DIR,   'encoding_results.csv'))
print(f"Loading enc  (non-RSA):    {ENC_NONRSA_DIR}")
enc_nonrsa = pd.read_csv(os.path.join(ENC_NONRSA_DIR, 'encoding_results.csv'))


# ─── BH-FDR helper (used by RSA + encoding stats below) ──────────────────
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


def _add_fdr_per_model(df, p_col, out_col):
    """Add ``out_col`` = BH-FDR of ``p_col`` across ROIs, computed
    SEPARATELY within each model — that is the test family
    'across ROIs, for model X'."""
    df = df.copy()
    df[out_col] = np.nan
    for m, idx in df.groupby('model').groups.items():
        df.loc[idx, out_col] = _bh_fdr(df.loc[idx, p_col].values)
    return df


# ─── RSA stats table (DSR + control) — unchanged across variants ─────────
def _rsa_stats(dsr_model, control_model, combo_name, rsa_test):
    """Pull BOTH the DSR and control betas from the combo regression — never
    from the per-model 'unique' file. Enforces test == between_tasks_z
    (all correct repeats averaged per config, z-scored per neuron,
    between-task-config cells only — within-config block masked out) at
    the data-loading boundary.  Adds ``p_perm_fdr`` = BH-FDR of
    ``p_perm`` across ROIs, per model (this is the test family the
    publication panels report)."""
    assert rsa_test == 'between_tasks_z', \
        "Publication panels only use the between_tasks_z RSA variant."
    sub = rsa_combos[(rsa_combos['combo'] == combo_name)
                     & (rsa_combos['test']  == rsa_test)].copy()
    sub = sub.rename(columns={'n_neurons': 'n_cells'})
    dsr_rows = sub[sub['sub_model'] == dsr_model].copy()
    dsr_rows['model'] = 'DSR'
    ctl_rows = sub[sub['sub_model'] == control_model].copy()
    ctl_rows['model'] = control_model
    out = pd.concat([
        dsr_rows[['roi', 'model', 'beta', 'p_perm', 't', 'n_cells']],
        ctl_rows[['roi', 'model', 'beta', 'p_perm', 't', 'n_cells']],
    ], ignore_index=True)
    out = _add_fdr_per_model(out, p_col='p_perm', out_col='p_perm_fdr')
    return out


rsa_stats = _rsa_stats(DSR_MODEL_RSA, CONTROL_MODEL,
                       RSA_COMBO, RSA_TEST)


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
    s = s.replace({'model': {dsr_model: 'DSR',
                              control_model: CONTROL_MODEL}})
    # BH-FDR of p_t across ROIs, per model — the test family the publication
    # panels report on for the encoding heatmap.
    return _add_fdr_per_model(s, p_col='p_t', out_col='p_t_fdr')


# ─── Equal-shade colour scaling for the RSA+Encoding heatmap pair ────────
def _paired_vmax(rsa_stats_df, enc_stats_df,
                 rsa_value_col='beta', enc_value_col='t',
                 anchor_roi='ACC', anchor_model='DSR',
                 models=('DSR', CONTROL_MODEL), rois=ROI_ORDER):
    """Pick vmax for the RSA heatmap and Encoding heatmap so that the
    ACC/DSR cell sits at the SAME fractional position on the RdBu_r colour
    bar in both panels (= same shade of red).

    For each panel: compute the natural max-|value| across (ROI × DSR/control)
    cells, and the fractional position of ACC/DSR within that range. Use the
    SMALLER of the two fractions as the shared target — that way neither panel
    is ever clipped, and the anchor cell matches.
    Returns ``(vmax_rsa, vmax_enc)``.
    """
    def _gather(df, value_col):
        rois_present = [r for r in rois if r in set(df['roi'])]
        vals = []
        anchor_val = np.nan
        for roi in rois_present:
            sub = df[df['roi'] == roi]
            for m in models:
                row = sub[sub['model'] == m]
                if row.empty:
                    continue
                v = float(row[value_col].iloc[0])
                vals.append(v)
                if roi == anchor_roi and m == anchor_model:
                    anchor_val = v
        max_abs = float(np.nanmax(np.abs(vals))) if vals else 1.0
        max_abs = max(max_abs, 1e-9)
        return max_abs, anchor_val

    max_r, anch_r = _gather(rsa_stats_df, rsa_value_col)
    max_e, anch_e = _gather(enc_stats_df, enc_value_col)
    if not (np.isfinite(anch_r) and np.isfinite(anch_e)
            and abs(anch_r) > 0 and abs(anch_e) > 0):
        return max_r, max_e
    frac_r = abs(anch_r) / max_r
    frac_e = abs(anch_e) / max_e
    f = min(frac_r, frac_e)
    vmax_rsa = max(abs(anch_r) / f, max_r)
    vmax_enc = max(abs(anch_e) / f, max_e)
    return vmax_rsa, vmax_enc


# ─── Panel drawing primitives ─────────────────────────────────────────────
def _draw_one_heatmap(ax, stats_df, value_col, p_col, n_col, value_label,
                      rois=ROI_ORDER, models=('DSR', CONTROL_MODEL),
                      vmin=None, vmax=None, font_pt=TARGET_FONT_PT,
                      title=None, show_cbar=True, cbar_label=None):
    """Return (im, n_per_roi). Y-axis carries ROI names only —
    cell counts go into the figure-level caption text built by the caller."""
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
    ax.set_xticklabels([pretty(m) for m in models],
                       fontsize=font_pt, rotation=30, ha='right')
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels([pretty(r) for r in rois], fontsize=font_pt)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title, fontsize=font_pt + 1)
    if show_cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=font_pt - 1)
        cb.set_label(cbar_label or value_label, fontsize=font_pt - 1)
    return im, n_per_roi


def _draw_acc_histogram(ax, enc_df, enc_stats_df=None,
                        font_pt=TARGET_FONT_PT, title=None, n_bins=14):
    """ACC DSR mean-r histogram — a zoom-in on the ACC/DSR cell of the
    encoding heatmap. The significance star reflects the SAME test family
    as the heatmap: BH-FDR of p_t across ROIs for the DSR model. The raw
    one-sided p is also returned for the figure caption.

    - COL_HIST_BG bars: every ACC encoding neuron.
    - COL_HIST_SIG bars: cells with p_perm < ALPHA (overlay).
    - 0 line dashed, mean line solid COL_HIST_SIG, both clearly visible.
    - NO legend on the axes — only the significance star is kept.
    Returns dict(n, n_sig, mean_r, p_t_gt0, p_t_fdr).
    """
    rows = enc_df[(enc_df['roi'] == 'ACC')
                  & (enc_df['model'] == DSR_MODEL_ENC)]
    r_all = rows['mean_r'].dropna().values
    r_sig = rows.loc[rows['p_perm'] < ALPHA, 'mean_r'].dropna().values
    n_sig = int(r_sig.size)
    if not r_all.size:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=font_pt, color='0.5')
        ax.set_xticks([]); ax.set_yticks([])
        return {'n': 0, 'n_sig': 0,
                'mean_r': float('nan'),
                'p_t_gt0': float('nan'), 'p_t_fdr': float('nan')}

    lim  = max(0.05, 1.05 * float(np.max(np.abs(r_all))))
    bins = np.linspace(-lim, lim, n_bins)
    ax.hist(r_all, bins=bins, color=COL_HIST_BG, alpha=0.95,
            edgecolor='white', linewidth=0.4)
    if r_sig.size:
        ax.hist(r_sig, bins=bins, color=COL_HIST_SIG, alpha=0.95,
                edgecolor='white', linewidth=0.4)

    mean_r = float(np.mean(r_all))
    ax.axvline(0,      color=COL_ZERO,     lw=1.0, ls='--', zorder=4)
    ax.axvline(mean_r, color=COL_HIST_SIG, lw=1.8, zorder=5)

    # Raw one-sided t-test (kept for caption text only).
    if r_all.size >= 2:
        p_gt0 = float(stats.ttest_1samp(r_all, 0.0,
                                        alternative='greater').pvalue)
    else:
        p_gt0 = float('nan')

    # Significance star reflects the same FDR-corrected p as the ACC/DSR
    # cell in the encoding heatmap — i.e. BH-FDR of p_t over ROIs for DSR.
    p_t_fdr = float('nan')
    if enc_stats_df is not None and 'p_t_fdr' in enc_stats_df.columns:
        m_acc_dsr = ((enc_stats_df['roi']   == 'ACC')
                     & (enc_stats_df['model'] == 'DSR'))
        if m_acc_dsr.any():
            v = enc_stats_df.loc[m_acc_dsr, 'p_t_fdr'].iloc[0]
            p_t_fdr = float(v) if np.isfinite(v) else float('nan')

    if np.isfinite(p_t_fdr):
        sig = ('***' if p_t_fdr < 0.001 else '**' if p_t_fdr < 0.01
               else '*' if p_t_fdr < 0.05 else 'n.s.')
        ax.text(0.97, 0.95, sig, transform=ax.transAxes,
                ha='right', va='top', fontsize=font_pt + 2, color='black')

    ax.set_xlim(-lim, lim)
    ax.set_xlabel('mean held-out r', fontsize=font_pt)
    ax.set_ylabel('# neurons', fontsize=font_pt)
    if title:
        ax.set_title(title, fontsize=font_pt)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    return {'n': int(r_all.size), 'n_sig': n_sig,
            'mean_r': mean_r,
            'p_t_gt0': p_gt0, 'p_t_fdr': p_t_fdr}


def _draw_sig_bars(ax, enc_df, font_pt=TARGET_FONT_PT, title=None,
                   bar_colour=None):
    """Per-ROI total vs. sig (permutation p<ALPHA) bars.
    No legend (the colour key lives in the figure caption text).
    Returns the {roi: n_total} dict so the caller can build that caption.
    """
    bar_colour = bar_colour or COL_BAR_SIG
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
    ax.bar(x, n_total, color=COL_BAR_BG,  edgecolor='none')
    ax.bar(x, n_sig,   color=bar_colour,  edgecolor='none')
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
    ax.set_xticklabels([pretty(r) for r in rois_present],
                       rotation=30, ha='right', fontsize=font_pt - 1)
    ax.set_ylabel('# neurons', fontsize=font_pt)
    if title:
        ax.set_title(title, fontsize=font_pt)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(0, max(n_total) * 1.35)
    return {r: nt for r, nt in zip(rois_present, n_total)}


# ─── Caption-text builders ────────────────────────────────────────────────
def _n_str(n_per_roi, rois=ROI_ORDER):
    """'ACC n=68, HC ant n=179, ...' — display names, ROI-ordered."""
    return ', '.join(f'{pretty(r)} n={n_per_roi[r]}'
                     for r in rois if r in n_per_roi)


def _caption_heatmap_pair(n_rsa, n_enc, variant_subtitle):
    return (f"RSA (across-tasks, excluding within-task diagonal, z-scored, " 
            f"combo '{RSA_COMBO}'): "
            f"{_n_str(n_rsa)}.\n"
            f"Encoding ({variant_subtitle}): {_n_str(n_enc)}.\n"
            f"Black box = BH-FDR-corrected p < {ALPHA:.2f} "
            f"(family: across ROIs, per model).")


def _caption_bars(n_bars):
    return (f"Bars per ROI: {COL_BAR_BG} = total encoding neurons, "
            f"{COL_BAR_SIG} = permutation p < {ALPHA:.2f}. "
            f"Stars: BH-FDR over ROIs.\nCounts — {_n_str(n_bars)}.")


def _caption_hist(hist_stats):
    """hist_stats: dict from _draw_acc_histogram."""
    n      = hist_stats['n']
    n_sig  = hist_stats['n_sig']
    m_r    = hist_stats['mean_r']
    p_t    = hist_stats['p_t_gt0']
    p_fdr  = hist_stats.get('p_t_fdr', float('nan'))
    p_str  = f"{p_t:.1e}"   if np.isfinite(p_t)   else "n/a"
    f_str  = f"{p_fdr:.1e}" if np.isfinite(p_fdr) else "n/a"
    m_str  = f"{m_r:+.4f}"  if np.isfinite(m_r)   else "n/a"
    return (f"ACC — DSR: {COL_HIST_BG} = all encoding neurons; "
            f"{COL_HIST_SIG} = permutation p < {ALPHA:.2f}. "
            f"Dashed = 0; solid {COL_HIST_SIG} = mean. "
            f"Star: FDR-corrected p of t-test mean > 0 "
            f"(family: across ROIs, for DSR — same as the "
            f"encoding-heatmap ACC/DSR cell).\n"
            f"n = {n}, n_sig (perm p<{ALPHA:.2f}) = {n_sig}, "
            f"raw t-test p = {p_str}, FDR p = {f_str}, "
            f"mean held-out r = {m_str}.")


# ─── Standalone panels per variant ────────────────────────────────────────
def save_panel_A(enc_stats, save_stem, subtitle, font_pt=TARGET_FONT_PT):
    figsize = (9.5 / CM_PER_IN, 10.5 / CM_PER_IN)
    # Equal-shade vmax so ACC/DSR matches across the RSA and Encoding panels.
    vmax_rsa, vmax_enc = _paired_vmax(rsa_stats, enc_stats)
    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_pt}):
        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={'wspace': 0.45})
        _, n_rsa = _draw_one_heatmap(
            axes[0], rsa_stats,
            value_col='beta', p_col='p_perm_fdr', n_col='n_cells',
            value_label='empirical β', title='RSA',
            font_pt=font_pt, cbar_label='β',
            vmin=-vmax_rsa, vmax=vmax_rsa)
        _, n_enc = _draw_one_heatmap(
            axes[1], enc_stats,
            value_col='t', p_col='p_t_fdr', n_col='n_cells',
            value_label='t-stat',
            title=f'Encoding\n— {subtitle}', font_pt=font_pt,
            cbar_label='t',
            vmin=-vmax_enc, vmax=vmax_enc)
        axes[1].set_yticklabels([])
        fig.text(0.5, -0.02, _caption_heatmap_pair(n_rsa, n_enc, subtitle),
                 ha='center', va='top', fontsize=font_pt - 2, color='0.25',
                 wrap=True)
        fig.savefig(save_stem + '.pdf', bbox_inches='tight')
        fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {save_stem}.pdf/.png")


def save_panel_B(enc_df, save_stem, subtitle, font_pt=TARGET_FONT_PT):
    figsize = (8.5 / CM_PER_IN, 5.5 / CM_PER_IN)
    enc_stats_v = _enc_stats(enc_df)
    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_pt}):
        fig, ax = plt.subplots(figsize=figsize)
        hist_stats = _draw_acc_histogram(
            ax, enc_df, enc_stats_df=enc_stats_v,
            font_pt=font_pt, title=f'ACC — DSR\n{subtitle}')
        fig.text(0.5, -0.05, _caption_hist(hist_stats), ha='center', va='top',
                 fontsize=font_pt - 2, color='0.25', wrap=True)
        fig.tight_layout()
        fig.savefig(save_stem + '.pdf', bbox_inches='tight')
        fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {save_stem}.pdf/.png")


def save_panel_C(enc_df, save_stem, subtitle, font_pt=TARGET_FONT_PT):
    figsize = (10.5 / CM_PER_IN, 4.5 / CM_PER_IN)
    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_pt}):
        fig, ax = plt.subplots(figsize=figsize)
        n_bars = _draw_sig_bars(
            ax, enc_df, font_pt=font_pt,
            title=f'DSR — significant cells per ROI ({subtitle})')
        fig.text(0.5, -0.08, _caption_bars(n_bars), ha='center', va='top',
                 fontsize=font_pt - 2, color='0.25', wrap=True)
        fig.tight_layout()
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


# ─── Master combined figure: 4 variants × 4 columns (RSA | enc | hist | bars)
def combined_figure(save_stem, font_pt=TARGET_FONT_PT):
    A4_W   = figure_layout.A4_WIDTH_IN
    margin = 1.5 / CM_PER_IN
    fig_w  = A4_W - 2 * margin
    # extra room at the bottom for the caption text + modelled-neuron stub
    fig_h  = 33.0 / CM_PER_IN

    # Column widths: heatmaps thinner, histogram wider, bars widest.
    width_ratios = [0.7, 0.7, 1.8, 2.4]

    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_pt}):
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
        gs  = fig.add_gridspec(5, 4,
                               height_ratios=[1.5, 1.5, 1.5, 1.5, 0.7],
                               width_ratios=width_ratios,
                               hspace=1.25, wspace=0.85,
                               left=0.07, right=0.97,
                               top=0.96, bottom=0.10)

        # Collect counts ONCE per variant so the bottom caption can list them.
        per_variant_n_rsa  = {}
        per_variant_n_enc  = {}
        per_variant_n_bars = {}
        per_variant_hist   = {}

        for row_idx, (tag, subtitle, getter) in enumerate(VARIANTS):
            enc_df_v    = getter()
            enc_stats_v = _enc_stats(enc_df_v)
            vmax_rsa, vmax_enc = _paired_vmax(rsa_stats, enc_stats_v)

            # Col 0: RSA heatmap (data identical across rows; vmax differs
            # because it's tied to the per-row encoding panel for shade match)
            ax_rsa = fig.add_subplot(gs[row_idx, 0])
            _, n_rsa = _draw_one_heatmap(
                ax_rsa, rsa_stats,
                value_col='beta', p_col='p_perm_fdr', n_col='n_cells',
                value_label='β',
                title=('RSA' if row_idx == 0 else None),
                font_pt=font_pt, cbar_label='β',
                vmin=-vmax_rsa, vmax=vmax_rsa)
            per_variant_n_rsa[tag] = n_rsa

            # Col 1: encoding heatmap for this variant
            ax_enc = fig.add_subplot(gs[row_idx, 1])
            _, n_enc = _draw_one_heatmap(
                ax_enc, enc_stats_v,
                value_col='t', p_col='p_t_fdr', n_col='n_cells',
                value_label='t',
                title=f'Encoding\n— {subtitle}',
                font_pt=font_pt, cbar_label='t',
                vmin=-vmax_enc, vmax=vmax_enc)
            ax_enc.set_yticklabels([])
            per_variant_n_enc[tag] = n_enc

            # Col 2: ACC DSR histogram (significance star uses the SAME
            # FDR-corrected p as the ACC/DSR cell in the encoding heatmap)
            ax_hist = fig.add_subplot(gs[row_idx, 2])
            hist_stats = _draw_acc_histogram(
                ax_hist, enc_df_v, enc_stats_df=enc_stats_v,
                font_pt=font_pt, title=f'ACC — DSR\n{subtitle}')
            per_variant_hist[tag] = hist_stats

            # Col 3: per-ROI sig bars
            ax_bars = fig.add_subplot(gs[row_idx, 3])
            n_bars = _draw_sig_bars(ax_bars, enc_df_v, font_pt=font_pt,
                                    title=f'DSR sig / ROI — {subtitle}')
            per_variant_n_bars[tag] = n_bars

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

        # ─── Figure caption text at the very bottom ────────────────────
        # Lines: legend explanation + per-variant cell counts + per-variant
        # ACC DSR histogram stats (n, n_sig, p, mean).
        cap_lines = [
            f"Heatmaps — RSA (left): across-task-halves z-scored betas from "
            f"combo '{RSA_COMBO}'; columns = DSR and "
            f"{pretty(CONTROL_MODEL)} (control). Encoding (right): t-stats "
            f"per ROI. Black box = BH-FDR-corrected p < {ALPHA:.2f} "
            f"(family: across ROIs, per model). vmax is paired per row so "
            f"the ACC DSR cell has the same shade in both panels.",
            f"Histograms — {COL_HIST_BG} = all ACC encoding neurons; "
            f"{COL_HIST_SIG} = permutation p < {ALPHA:.2f}. "
            f"Dashed = 0; solid {COL_HIST_SIG} = mean. "
            f"Star: FDR-corrected p of t-test mean > 0 — the SAME family "
            f"used for the ACC/DSR cell of the encoding heatmap.",
            f"Per-ROI bars — {COL_BAR_BG} = total encoding neurons; "
            f"{COL_BAR_SIG} = permutation p < {ALPHA:.2f}. "
            f"Stars: BH-FDR (greater than chance proportion {ALPHA:.2f}).",
        ]
        # RSA counts are identical across variants — print once.
        cap_lines.append(
            f"RSA cell counts: {_n_str(per_variant_n_rsa[VARIANTS[0][0]])}.")
        for (tag, subtitle, _) in VARIANTS:
            cap_lines.append(
                f"Encoding ({subtitle}): {_n_str(per_variant_n_enc[tag])}.")
        for (tag, subtitle, _) in VARIANTS:
            h = per_variant_hist[tag]
            p_str = (f"{h['p_t_gt0']:.1e}"
                     if np.isfinite(h['p_t_gt0']) else "n/a")
            f_str = (f"{h.get('p_t_fdr', float('nan')):.1e}"
                     if np.isfinite(h.get('p_t_fdr', float('nan'))) else "n/a")
            m_str = (f"{h['mean_r']:+.4f}"
                     if np.isfinite(h['mean_r']) else "n/a")
            cap_lines.append(
                f"ACC DSR histogram ({subtitle}): "
                f"n = {h['n']}, n_sig = {h['n_sig']}, "
                f"raw t-test p = {p_str}, FDR p = {f_str}, "
                f"mean held-out r = {m_str}.")
        fig.text(0.5, 0.005, '\n'.join(cap_lines),
                 ha='center', va='bottom', fontsize=font_pt - 2,
                 color='0.25', wrap=True)

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
        'rsa_combo':             RSA_COMBO,
        'dsr_model_rsa':         DSR_MODEL_RSA,
        'dsr_model_enc':         DSR_MODEL_ENC,
        'control_model':         CONTROL_MODEL,
        'state_models_removed':  STATE_MODELS_REMOVE,
        'alpha':                 ALPHA,
        'roi_order':             ROI_ORDER,
        'variants':              [v[0] for v in VARIANTS],
        'colours': {
            'hist_bg':  COL_HIST_BG,
            'hist_sig': COL_HIST_SIG,
            'bar_bg':   COL_BAR_BG,
            'bar_sig':  COL_BAR_SIG,
        },
        'label_map': LABEL_MAP,
    },
    'rsa_per_roi': rsa_stats.to_dict(orient='records'),
    'variants':    variant_stats_archive,
}
with open(os.path.join(OUT_DIR, 'publication_panels_stats.json'), 'w') as f:
    json.dump(stats_archive, f, indent=2,
              default=lambda o: float(o) if hasattr(o, 'item') else str(o))

print(f"\nWrote: {OUT_DIR}/publication_panels_stats.json")
print(f"All panels saved under: {OUT_DIR}")
