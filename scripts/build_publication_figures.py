#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-render the two main publication figures from the latest analysis runs.

Edit the RUN_PATHS dict below to point at new run directories produced by:
  - scripts/RSA_DSR_ROIs_simple.py
  - scripts/encoding_analysis_simple.py   (full + ACC-only with dsr_ideal_phases zoo)
  - scripts/spatial_peaks_simple.py       (at future lags [30,60] AND at now lags [330,0])
  - scripts/per_lag_encoding.py

Outputs two figures to:
  data/ephys_humans/derivatives/group/_publication_figures/
        Fig1_ACC_DSR_convergence.{pdf,png}
        Fig2_lag_dissociation.{pdf,png}

Style: Arial 11, Showgirl2 era_brewer palette, vector PDF (editable text in Illustrator).

@author: Svenja Küchenhoff
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_1samp, binomtest

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
from era_brewer import ERA_PALETTES


# ── EDIT THESE WHEN NEW RUNS ARE READY ────────────────────────────────
DATA_BASE = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
RUN_PATHS = {
    'rsa':           f'{DATA_BASE}/group/DSR_RSA_simple_ROI/2026-06-18_12-17-07',
    'encoding_all':  f'{DATA_BASE}/group/encoding_analysis_simple/2026-06-05_17-58-57',
    'encoding_acc':  f'{DATA_BASE}/group/encoding_analysis_simple/2026-06-18_14-31-25',
    'sp_future':     f'{DATA_BASE}/group/spatial_peaks_simple/2026-06-18_12-17-45_full_optimal_phaseresid_norep',
    'sp_now':        f'{DATA_BASE}/group/spatial_peaks_simple/2026-06-18_15-13-34_full_optimal_lags_330_0_now',
    # Per-lag encoding outputs (one CSV per ROI). The new per_lag_encoding.py
    # script puts these in a single run-tagged directory; once you have that,
    # set 'per_lag_dir' and the loader below will pick all 7 ROIs up.
    'per_lag_dir':   None,        # e.g. f'{DATA_BASE}/group/per_lag_encoding/2026-06-19_xx-xx-xx'
    # Fallback ad-hoc CSVs (ACC + HC_mid) — used if 'per_lag_dir' is None
    'per_lag_acc':   '/tmp/acc_per_lag_encoding.csv',
    'per_lag_hcm':   '/tmp/hcmid_per_lag_encoding.csv',
}
OUT_DIR = f'{DATA_BASE}/group/_publication_figures'


# ── style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SG2 = ERA_PALETTES['Showgirl2']['colors']
COLOR_FUTURE  = SG2[0]
COLOR_NOW     = SG2[5]
COLOR_NEUTRAL = SG2[3]
COLOR_ACC     = SG2[0]
COLOR_HCMID   = SG2[5]
ROI_COLORS = {
    'ACC':              SG2[0],
    'medialOFC':        SG2[1],
    'PCC':              SG2[2],
    'Parahippocampal':  SG2[3],
    'HC_anterior':      SG2[6],
    'HC_mid':           SG2[5],
    'EC':               SG2[4],
}

ROI_ORDER = ['ACC', 'medialOFC', 'PCC', 'Parahippocampal',
             'HC_anterior', 'HC_mid', 'EC']
LAGS_BINS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


def stars(p):
    if not np.isfinite(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '·'
    return ''


# ── load data ─────────────────────────────────────────────────────────

def load_rsa(roi, test):
    """dsr_fmri in fMRI_state combo; returns (beta, p, n_neurons)."""
    c = pd.read_csv(f'{RUN_PATHS["rsa"]}/results_summary_combos.csv')
    r = c[(c['roi']==roi) & (c['combo']=='fMRI_state') &
          (c['sub_model']=='dsr_fmri') & (c['test']==test)]
    if not len(r):
        return (np.nan, np.nan, 0)
    return (float(r['beta'].iloc[0]), float(r['p_perm'].iloc[0]),
            int(r['n_neurons'].iloc[0]))


def load_encoding(roi, model='dsr', run='encoding_all'):
    """Per-cell encoding results. Returns DataFrame."""
    e = pd.read_csv(f'{RUN_PATHS[run]}/encoding_results.csv')
    return e[(e['roi']==roi) & (e['model']==model)]


def load_sp(run_tag, roi):
    """Spatial-peaks fixed_lag_r_mean and p per cell."""
    p = pd.read_csv(f'{RUN_PATHS[run_tag]}/per_cell.csv')
    p = p[(p['note']=='ok') & (p['roi']==roi)]
    return p


def load_per_lag(roi):
    """Per-lag CV r per cell (returns ndarray cells × 12 lags)."""
    if RUN_PATHS['per_lag_dir'] is not None:
        path = f'{RUN_PATHS["per_lag_dir"]}/per_lag_{roi}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            return None
    else:
        # ad-hoc fallback
        if roi == 'ACC' and os.path.exists(RUN_PATHS['per_lag_acc']):
            df = pd.read_csv(RUN_PATHS['per_lag_acc'])
        elif roi == 'HC_mid' and os.path.exists(RUN_PATHS['per_lag_hcm']):
            df = pd.read_csv(RUN_PATHS['per_lag_hcm'])
        else:
            return None
    cols = [f'lag_{l}' for l in LAGS_BINS]
    return df[cols].to_numpy()


# ── Figure 1: ACC DSR convergence ─────────────────────────────────────

def build_figure_1(out_dir):
    rsa = []
    for roi in ROI_ORDER:
        for test in ['split_halves_z', 'between_tasks_z']:
            b, p, n = load_rsa(roi, test)
            rsa.append({'roi':roi, 'test':test, 'beta':b, 'p':p, 'n':n})
    rsa_df = pd.DataFrame(rsa)

    enc = []
    for roi in ROI_ORDER:
        sub = load_encoding(roi, 'dsr', 'encoding_all')
        n = len(sub); k = int((sub['p_perm']<0.05).sum())
        p_b = binomtest(k, n, p=0.05, alternative='greater').pvalue if n else np.nan
        enc.append({'roi':roi, 'n':n, 'k':k, 'frac':k/n if n else 0, 'p_binom':p_b})
    enc_df = pd.DataFrame(enc)

    # state-NS subset for ACC from the phase-residualised encoding run
    enc_acc = load_encoding('ACC', 'dsr', 'encoding_acc')
    state_sig = set(load_encoding('ACC', 'state', 'encoding_acc')
                    .query('p_perm < 0.05')['neuron'].tolist())
    acc_ns = enc_acc[~enc_acc['neuron'].isin(state_sig)]
    if len(acc_ns):
        t_ns, p_t_ns = ttest_1samp(acc_ns['mean_r'].dropna(), 0)
    else:
        t_ns, p_t_ns = np.nan, np.nan

    fig = plt.figure(figsize=(7.5, 4.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.7, wspace=0.45,
                  width_ratios=[1.3, 1.3, 1], height_ratios=[1, 1])

    # A. RSA heatmap
    ax = fig.add_subplot(gs[0, :2])
    betas = np.zeros((len(ROI_ORDER), 2)); ps = np.zeros((len(ROI_ORDER), 2))
    for i, roi in enumerate(ROI_ORDER):
        for j, t in enumerate(['split_halves_z', 'between_tasks_z']):
            r = rsa_df[(rsa_df['roi']==roi)&(rsa_df['test']==t)]
            if len(r):
                betas[i, j] = float(r['beta'].iloc[0])
                ps[i, j]    = float(r['p'].iloc[0])
    cmap = mcolors.LinearSegmentedColormap.from_list('rust_div', [SG2[6], 'white', SG2[0]])
    im = ax.imshow(betas.T, aspect='auto', cmap=cmap, vmin=-0.06, vmax=0.06)
    for i in range(len(ROI_ORDER)):
        for j in range(2):
            s = stars(ps[i, j])
            if s:
                ax.text(i, j, s, ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER, rotation=35, ha='right', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['split_halves_z\n(across runs)',
                        'between_tasks_z\n(across tasks)'], fontsize=9)
    ax.set_title('a. RSA: DSR β per ROI (combo: DSR + buttons + loc + state)', loc='left')
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, aspect=10)
    cb.set_label('β', fontsize=9); cb.ax.tick_params(labelsize=8)

    # B. Encoding %sig
    ax = fig.add_subplot(gs[0, 2])
    fracs = enc_df['frac'].values * 100
    cols = [ROI_COLORS[r] for r in ROI_ORDER]
    ax.bar(range(len(ROI_ORDER)), fracs, color=cols, edgecolor='black', linewidth=0.6, width=0.7)
    ax.axhline(5, color='gray', ls='--', lw=1, label='chance (5%)')
    for i, (_, r) in enumerate(enc_df.iterrows()):
        s = stars(r['p_binom'])
        if s:
            ax.text(i, fracs[i] + 0.5, s, ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('% DSR-sig cells', fontsize=9)
    ax.set_title('b. Encoding: per-cell\nDSR-sig per ROI', loc='left')
    ax.set_ylim(0, max(fracs) * 1.25)
    ax.legend(frameon=False, loc='upper right', fontsize=8)

    # C. ACC DSR mean_r histogram (state-NS highlight)
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(enc_acc['mean_r'].dropna(), bins=30, color='lightgray',
            alpha=0.7, edgecolor='gray', label=f'all ACC (n={len(enc_acc)})')
    ax.hist(acc_ns['mean_r'].dropna(), bins=30, color=COLOR_ACC,
            alpha=0.85, edgecolor='black', linewidth=0.5,
            label=f'state-NS (n={len(acc_ns)})')
    ax.axvline(0, color='black', lw=0.8)
    ax.axvline(acc_ns['mean_r'].mean(), color=COLOR_ACC, lw=1.5, ls='--')
    ax.set_xlabel('per-cell CV r (encoding DSR)', fontsize=9)
    ax.set_ylabel('ACC cells', fontsize=9)
    ax.set_title(f'c. ACC DSR (state-NS)\nt = {t_ns:+.2f}, p = {p_t_ns:.3f} {stars(p_t_ns)}',
                 loc='left')
    ax.legend(frameon=False, fontsize=8)

    # D. Convergence summary
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis('off')
    acc_rsa = rsa_df[(rsa_df['roi']=='ACC') & (rsa_df['test']=='split_halves_z')]
    b_acc = float(acc_rsa['beta'].iloc[0]); p_acc = float(acc_rsa['p'].iloc[0])
    acc_enc = enc_df[enc_df['roi']=='ACC'].iloc[0]
    summary = [
        'Convergent ACC DSR-encoding evidence (phase-residualised data)',
        '',
        f'•  RSA across-runs, dsr_fmri in fMRI_state combo:',
        f'         β = {b_acc:+.3f},  p = {p_acc:.3f}  {stars(p_acc)}',
        '',
        f'•  Encoding per-cell binomial: {acc_enc["k"]} / {acc_enc["n"]}',
        f'         = {acc_enc["frac"]*100:.1f}% sig vs 5% expected',
        (f'         binomial p < 0.001  {stars(acc_enc["p_binom"])}'
         if acc_enc['p_binom'] < 0.001
         else f'         binomial p = {acc_enc["p_binom"]:.3f}  {stars(acc_enc["p_binom"])}'),
        '',
        f'•  Encoding mean-r, state-NS cells:',
        f'         t = {t_ns:+.2f}, p = {p_t_ns:.3f}  {stars(p_t_ns)}',
        '',
        'Two independent statistics converge on ACC.',
    ]
    y = 0.95
    for line in summary:
        weight = 'bold' if (line.startswith('Convergent') or
                             line.startswith('Two independent')) else 'normal'
        fs = 10 if not line.startswith('•') else 9
        ax.text(0.0, y, line, transform=ax.transAxes, fontsize=fs,
                weight=weight, family='Arial', va='top')
        y -= 0.07

    fig.suptitle('Figure 1 — Human ACC encodes the Distributed Structured Representation (DSR)',
                 fontsize=12, fontweight='bold', y=1.00, x=0.05, ha='left')
    fig.savefig(f'{out_dir}/Fig1_ACC_DSR_convergence.pdf', bbox_inches='tight')
    fig.savefig(f'{out_dir}/Fig1_ACC_DSR_convergence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: Fig1_ACC_DSR_convergence.{{pdf,png}}')


# ── Figure 2: lag dissociation ────────────────────────────────────────

def build_figure_2(out_dir):
    sp_stats = []
    for roi in ROI_ORDER:
        sf = load_sp('sp_future', roi)
        sn = load_sp('sp_now', roi)
        fl_f = sf['fixed_lag_r_mean'].dropna()
        fl_n = sn['fixed_lag_r_mean'].dropna()
        if not len(fl_f) or not len(fl_n): continue
        t_f, p_t_f = ttest_1samp(fl_f, 0)
        t_n, p_t_n = ttest_1samp(fl_n, 0)
        sp_stats.append({'roi':roi, 'n':len(fl_f),
                         'fut_mean':fl_f.mean(),
                         'fut_sem': fl_f.std()/np.sqrt(len(fl_f)),
                         'now_mean':fl_n.mean(),
                         'now_sem': fl_n.std()/np.sqrt(len(fl_n)),
                         'p_fut':p_t_f, 'p_now':p_t_n})
    sp_df = pd.DataFrame(sp_stats)

    arr_acc = load_per_lag('ACC')
    arr_hcm = load_per_lag('HC_mid')

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.85, wspace=0.5, height_ratios=[1, 1])

    # A. Spatial-peaks dissociation per ROI
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(sp_df)); w = 0.38
    ax.bar(x - w/2, sp_df['fut_mean'], w, yerr=sp_df['fut_sem'],
           color=COLOR_FUTURE, edgecolor='black', linewidth=0.6, capsize=3,
           label='future lags (+30, +60)')
    ax.bar(x + w/2, sp_df['now_mean'], w, yerr=sp_df['now_sem'],
           color=COLOR_NOW, edgecolor='black', linewidth=0.6, capsize=3,
           label='now lags (−30, 0)')
    ax.axhline(0, color='black', lw=0.8)
    for i, (_, r) in enumerate(sp_df.iterrows()):
        if r['p_fut'] < 0.05:
            ax.text(i - w/2, r['fut_mean'] + r['fut_sem'] + 0.0015,
                    '*', ha='center', fontsize=14, fontweight='bold')
        if r['p_now'] < 0.05:
            ax.text(i + w/2, r['now_mean'] + r['now_sem'] + 0.0015,
                    '*', ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(sp_df['roi'], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Spatial-peaks CV r ± SEM', fontsize=10)
    ax.set_title('a. Spatial-peaks lag dissociation per ROI', loc='left')
    ax.legend(loc='upper right', frameon=False, fontsize=8)

    # B/C. Encoding per-lag profiles
    colors_lag = [COLOR_FUTURE if l in (30, 60) else
                  (COLOR_NOW if l in (0, 330) else COLOR_NEUTRAL)
                  for l in LAGS_BINS]

    def per_lag_panel(ax, arr, title):
        if arr is None or len(arr)==0:
            ax.set_title(title + '\n(no data)', loc='left'); return
        mean = np.nanmean(arr, axis=0)
        sem  = np.nanstd(arr, axis=0)/np.sqrt(np.sum(np.isfinite(arr), axis=0))
        ax.bar(LAGS_BINS, mean, yerr=sem, width=23, color=colors_lag,
               edgecolor='black', linewidth=0.5, capsize=2)
        for i, l in enumerate(LAGS_BINS):
            rs = arr[:, i]; rs = rs[np.isfinite(rs)]
            if not len(rs): continue
            t, p = ttest_1samp(rs, 0)
            if p < 0.05:
                marker = '*' if t > 0 else '·'
                ax.text(l, mean[i] + sem[i] + 0.0005, marker,
                        ha='center', fontsize=13, fontweight='bold',
                        color='black' if t > 0 else 'darkred')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel('Lag (bins ahead)', fontsize=10)
        ax.set_xticks(LAGS_BINS[::2])

    ax = fig.add_subplot(gs[1, 0])
    per_lag_panel(ax, arr_acc, 'b. Encoding per-lag — ACC')
    ax.set_ylabel('Encoding CV r ± SEM', fontsize=10)
    ax.set_title('b. Encoding per-lag — ACC', loc='left')

    ax = fig.add_subplot(gs[1, 1])
    per_lag_panel(ax, arr_hcm, 'c. Encoding per-lag — HC_mid')
    ax.set_title('c. Encoding per-lag — HC_mid', loc='left')

    # D. Per-cell preferred lag (ACC + HC_mid overlay)
    ax = fig.add_subplot(gs[1, 2])
    if arr_acc is not None and arr_hcm is not None:
        pref_acc = np.argmax(arr_acc, axis=1)
        pref_hcm = np.argmax(arr_hcm, axis=1)
        counts_acc = np.bincount(pref_acc, minlength=12) / len(pref_acc) * 100
        counts_hcm = np.bincount(pref_hcm, minlength=12) / len(pref_hcm) * 100
        w = 12
        ax.bar(np.array(LAGS_BINS) - w/2, counts_acc, width=w,
               color=COLOR_ACC, alpha=0.85, edgecolor='black', linewidth=0.4,
               label=f'ACC (n={len(pref_acc)})')
        ax.bar(np.array(LAGS_BINS) + w/2, counts_hcm, width=w,
               color=COLOR_HCMID, alpha=0.85, edgecolor='black', linewidth=0.4,
               label=f'HC_mid (n={len(pref_hcm)})')
        ax.axhline(100/12, color='gray', ls='--', lw=1)
        ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax.set_xlabel('Preferred lag (bins ahead)', fontsize=10)
    ax.set_ylabel('% of cells', fontsize=10)
    ax.set_title('d. Per-cell preferred lag', loc='left')
    ax.set_xticks(LAGS_BINS[::2])

    fig.suptitle('Figure 2 — ACC encodes near-future location; medial-temporal regions encode now',
                 fontsize=12, fontweight='bold', y=1.00, x=0.05, ha='left')
    fig.savefig(f'{out_dir}/Fig2_lag_dissociation.pdf', bbox_inches='tight')
    fig.savefig(f'{out_dir}/Fig2_lag_dissociation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: Fig2_lag_dissociation.{{pdf,png}}')


# ── main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Building publication figures → {OUT_DIR}')
    print('Using runs:')
    for k, v in RUN_PATHS.items():
        print(f'  {k:18s}  {v}')
    print()
    build_figure_1(OUT_DIR)
    build_figure_2(OUT_DIR)
    print('\nDone.')
