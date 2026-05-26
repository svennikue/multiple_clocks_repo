#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics for el-gaby encoding results.

Reads an encoding_analysis_elgaby run (encoding_results.csv + coefs.pkl)
and produces per-ROI / per-neuron diagnostics:

  - decode each coefficient index into (anchor_location, anchor_phase, lag)
    using the `clo_og` layout: i = anchor_loc * 36 + anchor_phase * 12 + lag
  - top-K coefficients per (neuron, test config)
  - 0-lag vs non-0-lag coefficient mass per neuron and ROI
  - preferred lag distribution per ROI
  - coefficient sparsity / magnitude distributions per ROI
  - optional cross-script comparison: if an encoding_analysis_simple run
    is available, scatter the el-gaby r against the simple-script r for
    matched (neuron, test_config) rows

This is the deferred Script 3 of the pipeline, but lifted earlier to
help interpret why the el-gaby run gives a different per-ROI pattern
than encoding_analysis_simple.py.

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
ELGABY_BASE = os.path.join(DATA_DIR, 'group', 'encoding_analysis_elgaby')
SIMPLE_BASE = os.path.join(DATA_DIR, 'group', 'encoding_analysis_simple')

# Which encoding_analysis_elgaby run to diagnose.  None = most recent.
ELGABY_RUN_TAG = None
# Optional: matching encoding_analysis_simple run to cross-compare against.
# None = most recent. Set to '' to disable the comparison.
SIMPLE_RUN_TAG = None

# DSR `clo_og` layout (must match no_phase_neurons=3 in model_DSR).
N_LOCATIONS = 9
N_ANCHOR_PHASES = 3
N_LAGS = 12
N_COEFS = N_LOCATIONS * N_ANCHOR_PHASES * N_LAGS   # 324

# How many top coefficients to inspect per (neuron, fold).
TOP_K = 3

# Bins for the lag-distribution histogram.
LAG_BINS = np.arange(N_LAGS + 1) - 0.5

# ROIs with fewer neurons than this are flagged but still included.
SMALL_ROI_FLAG = 30


# ── Resolve runs + output folder ─────────────────────────────────────
def find_latest_run(base):
    candidates = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))]
    if not candidates:
        return None
    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)))
    return candidates[-1]


if ELGABY_RUN_TAG is None:
    ELGABY_RUN_TAG = find_latest_run(ELGABY_BASE)
if ELGABY_RUN_TAG is None:
    raise FileNotFoundError(f"No elgaby runs found under {ELGABY_BASE}")
ELGABY_DIR = os.path.join(ELGABY_BASE, ELGABY_RUN_TAG)
print(f"Reading el-gaby run: {ELGABY_DIR}")

RESULTS_CSV = os.path.join(ELGABY_DIR, 'encoding_results.csv')
COEFS_PKL = os.path.join(ELGABY_DIR, 'coefs.pkl')
if not os.path.isfile(RESULTS_CSV):
    raise FileNotFoundError(f"Missing {RESULTS_CSV}")
if not os.path.isfile(COEFS_PKL):
    raise FileNotFoundError(f"Missing {COEFS_PKL}")

# Output folder under the el-gaby run itself.
OUT_DIR = os.path.join(ELGABY_DIR, 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Writing diagnostics into {OUT_DIR}")


# ── Load encoding results + coefs ────────────────────────────────────
results_df = pd.read_csv(RESULTS_CSV)
results_df['subject'] = results_df['subject'].map(lambda s: f'{int(s):02d}')
print(f"Loaded results CSV: {len(results_df)} (neuron, test_config) rows.")

with open(COEFS_PKL, 'rb') as f:
    coefs_blob = pickle.load(f)
coefs_store = coefs_blob['coefs']  # {sub: {neuron: {test_cfg: coefs}}}
print(f"Loaded coefs.pkl with {len(coefs_store)} subjects.")


# ── Coefficient layout decoder ───────────────────────────────────────
def coef_index_to_components(i):
    """Decode a coef index in [0, 324) into (anchor_loc, anchor_phase, lag).

    `clo_og` is built as:
        for anchor_row in range(27):                # 9 locs * 3 phases
            for lag in range(12):
                idx = anchor_row * 12 + lag
    where anchor_row = anchor_loc * 3 + anchor_phase.
    """
    if not 0 <= i < N_COEFS:
        raise IndexError(i)
    anchor_row = i // N_LAGS
    lag = i % N_LAGS
    anchor_loc = anchor_row // N_ANCHOR_PHASES
    anchor_phase = anchor_row % N_ANCHOR_PHASES
    return anchor_loc, anchor_phase, lag


# Precompute look-up arrays for vectorized decoding.
_idx = np.arange(N_COEFS)
ANCHOR_LOC_OF = _idx // (N_ANCHOR_PHASES * N_LAGS)
ANCHOR_PHASE_OF = (_idx // N_LAGS) % N_ANCHOR_PHASES
LAG_OF = _idx % N_LAGS


# ── Per-(neuron, fold) coef diagnostics ──────────────────────────────
def summarise_coefs_for_row(coefs):
    """Per-row summary of one (neuron, fold) coefficient vector."""
    abs_coefs = np.abs(coefs)
    nonzero_mask = abs_coefs > 1e-12
    n_nonzero = int(nonzero_mask.sum())

    if n_nonzero == 0:
        return {
            'n_nonzero':         0,
            'l1_norm':           0.0,
            'max_abs':           0.0,
            'top_idx':           [],
            'top_val':           [],
            'top_lag':           [],
            'top_anchor_loc':    [],
            'top_anchor_phase':  [],
            'lag_mass_0':        0.0,
            'lag_mass_nonzero':  0.0,
            'frac_top_at_lag0':  np.nan,
            'frac_l1_at_lag0':   np.nan,
            'preferred_lag':     -1,
        }

    top_ind = np.argsort(-abs_coefs)[:TOP_K]
    top_val = coefs[top_ind]

    # Mass at lag 0 vs lags > 0 (uses absolute values).
    lag0_mask = (LAG_OF == 0)
    lag_mass_0 = float(abs_coefs[lag0_mask].sum())
    lag_mass_nonzero = float(abs_coefs[~lag0_mask].sum())
    total_l1 = lag_mass_0 + lag_mass_nonzero

    # "Preferred lag" = lag of the single largest |coef|.
    preferred_lag = int(LAG_OF[np.argmax(abs_coefs)])

    frac_top_at_lag0 = float(np.mean(LAG_OF[top_ind] == 0))
    frac_l1_at_lag0 = (lag_mass_0 / total_l1) if total_l1 > 0 else np.nan

    return {
        'n_nonzero':         n_nonzero,
        'l1_norm':           float(total_l1),
        'max_abs':           float(np.max(abs_coefs)),
        'top_idx':           top_ind.tolist(),
        'top_val':           top_val.tolist(),
        'top_lag':           LAG_OF[top_ind].tolist(),
        'top_anchor_loc':    ANCHOR_LOC_OF[top_ind].tolist(),
        'top_anchor_phase':  ANCHOR_PHASE_OF[top_ind].tolist(),
        'lag_mass_0':        lag_mass_0,
        'lag_mass_nonzero':  lag_mass_nonzero,
        'frac_top_at_lag0':  frac_top_at_lag0,
        'frac_l1_at_lag0':   frac_l1_at_lag0,
        'preferred_lag':     preferred_lag,
    }


# Build per-row coef diagnostics keyed by (subject, neuron, test_config).
diag_rows = []
for _, row in results_df.iterrows():
    sub = row['subject']
    neuron = row['neuron']
    test_cfg = row['test_config']
    coefs = None
    sub_coefs = coefs_store.get(sub, {})
    cell_coefs = sub_coefs.get(neuron, {})
    coefs = cell_coefs.get(test_cfg, None)
    if coefs is None:
        diag = summarise_coefs_for_row(np.zeros(N_COEFS))
        diag['has_coefs'] = False
    else:
        diag = summarise_coefs_for_row(np.asarray(coefs, dtype=float))
        diag['has_coefs'] = True
    diag.update({
        'subject':       sub,
        'neuron':        neuron,
        'roi':           row['roi'],
        'test_config':   test_cfg,
        'state_tuned':   bool(row['state_tuned']),
        'pref_phase':    int(row['pref_phase']) if pd.notna(row['pref_phase']) else -1,
        'r_used':        row['r_used']    if 'r_used'    in row else row.get('r_4state'),
        'r_4state':      row['r_4state']  if 'r_4state'  in row else np.nan,
        'r_bins_all':    row['r_bins_all'] if 'r_bins_all' in row else row.get('r_bins'),
        'p_perm':        row['p_perm'],
    })
    diag_rows.append(diag)

diag_df = pd.DataFrame(diag_rows)
diag_path = os.path.join(OUT_DIR, 'coef_diagnostics_per_neuron_config.csv')
# top_* columns are list-typed; flatten by string-joining for CSV legibility.
list_cols = ['top_idx', 'top_val', 'top_lag',
             'top_anchor_loc', 'top_anchor_phase']
diag_df_csv = diag_df.copy()
for c in list_cols:
    diag_df_csv[c] = diag_df_csv[c].apply(
        lambda v: ';'.join(str(x) for x in v) if isinstance(v, list) else v)
diag_df_csv.to_csv(diag_path, index=False)
print(f"Saved {diag_path}")


# ── Per-ROI summaries ────────────────────────────────────────────────
def per_roi_coef_summary(diag_df, gate_state_tuned=True):
    """One row per ROI summarising coefficient diagnostics."""
    df = diag_df[diag_df['has_coefs']].copy()
    if gate_state_tuned:
        df = df[df['state_tuned']]
    rows = []
    for roi, g in df.groupby('roi', sort=False):
        rows.append({
            'roi':                       roi,
            'n_rows':                    int(len(g)),
            'n_neurons':                 int(g['neuron'].nunique()),
            'mean_n_nonzero':            float(g['n_nonzero'].mean()),
            'median_n_nonzero':          float(g['n_nonzero'].median()),
            'mean_l1':                   float(g['l1_norm'].mean()),
            'mean_max_abs':              float(g['max_abs'].mean()),
            'mean_frac_top_at_lag0':     float(g['frac_top_at_lag0'].mean()),
            'mean_frac_l1_at_lag0':      float(g['frac_l1_at_lag0'].mean()),
            'frac_with_pref_lag_eq_0':   float((g['preferred_lag'] == 0).mean()),
            'mean_lag_mass_0':           float(g['lag_mass_0'].mean()),
            'mean_lag_mass_nonzero':     float(g['lag_mass_nonzero'].mean()),
            'small_n_flag':              bool(g['neuron'].nunique() < SMALL_ROI_FLAG),
        })
    return pd.DataFrame(rows).sort_values('roi').reset_index(drop=True)


roi_summary_gated = per_roi_coef_summary(diag_df, gate_state_tuned=True)
roi_summary_all = per_roi_coef_summary(diag_df, gate_state_tuned=False)
roi_summary_gated.to_csv(os.path.join(OUT_DIR,
                                      'roi_coef_summary_state_tuned.csv'),
                         index=False)
roi_summary_all.to_csv(os.path.join(OUT_DIR, 'roi_coef_summary_all.csv'),
                       index=False)
print("Saved per-ROI coef summaries.")


# ── Plots ────────────────────────────────────────────────────────────
def plot_preferred_lag_per_roi(diag_df, out_path, gate_state_tuned=True):
    df = diag_df[diag_df['has_coefs']].copy()
    if gate_state_tuned:
        df = df[df['state_tuned']]
    rois = sorted(df['roi'].dropna().unique().tolist())
    if not rois:
        return
    n_cols = 3
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)
    for ax, roi in zip(axes.ravel(), rois):
        g = df[df['roi'] == roi]
        n_neurons = g['neuron'].nunique()
        lags = g['preferred_lag'].to_numpy()
        lags = lags[lags >= 0]
        ax.hist(lags, bins=LAG_BINS, edgecolor='k', alpha=0.85)
        ax.set_title(f"{roi}  (n_cells={n_neurons}, rows={len(g)})")
        ax.set_xlabel('preferred lag (lag=0 is the anchor)')
        ax.set_ylabel('# (neuron, config)')
        ax.set_xticks(np.arange(N_LAGS))
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    suffix = 'state-tuned only' if gate_state_tuned else 'all (neuron, config)'
    fig.suptitle(f'Preferred lag of strongest coefficient ({suffix})',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lag_mass_per_roi(diag_df, out_path, gate_state_tuned=True):
    """Per ROI, paired bar: mean L1 mass at lag=0 vs at lag>0."""
    df = diag_df[diag_df['has_coefs']].copy()
    if gate_state_tuned:
        df = df[df['state_tuned']]
    rois = sorted(df['roi'].dropna().unique().tolist())
    if not rois:
        return
    mass0 = [df.loc[df['roi'] == roi, 'lag_mass_0'].mean() for roi in rois]
    massN = [df.loc[df['roi'] == roi, 'lag_mass_nonzero'].mean()
             for roi in rois]
    n_cells = [df.loc[df['roi'] == roi, 'neuron'].nunique() for roi in rois]
    x = np.arange(len(rois))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rois)), 5))
    ax.bar(x - width / 2, mass0, width, label='lag = 0', color='tab:blue')
    ax.bar(x + width / 2, massN, width, label='lag > 0', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}\n(n={n})' for r, n in zip(rois, n_cells)],
                       rotation=45, ha='right')
    ax.set_ylabel('mean |coef| sum per (neuron, config)')
    suffix = 'state-tuned only' if gate_state_tuned else 'all (neuron, config)'
    ax.set_title(f'Coefficient mass: lag=0 vs lag>0 ({suffix})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_frac_l1_at_lag0(diag_df, out_path, gate_state_tuned=True):
    """Per ROI, fraction of |coef| L1 norm located at lag=0."""
    df = diag_df[diag_df['has_coefs']].copy()
    if gate_state_tuned:
        df = df[df['state_tuned']]
    rois = sorted(df['roi'].dropna().unique().tolist())
    if not rois:
        return
    data = [df.loc[df['roi'] == roi, 'frac_l1_at_lag0']
            .dropna().to_numpy() for roi in rois]
    n_cells = [df.loc[df['roi'] == roi, 'neuron'].nunique() for roi in rois]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rois)), 5))
    bp = ax.boxplot(data, positions=np.arange(len(rois)),
                    widths=0.6, patch_artist=True, showfliers=False)
    for patch, n in zip(bp['boxes'], n_cells):
        if n < SMALL_ROI_FLAG:
            patch.set_facecolor((0.7, 0.7, 0.7, 0.5))
        else:
            patch.set_facecolor((0.4, 0.65, 0.9, 0.6))
    ax.axhline(1.0 / N_LAGS, ls='--', color='k', lw=0.8,
               label=f'uniform = 1/{N_LAGS} = {1/N_LAGS:.3f}')
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels([f'{r}\n(n={n})' for r, n in zip(rois, n_cells)],
                       rotation=45, ha='right')
    ax.set_ylabel('frac of |coef| L1 at lag=0')
    suffix = 'state-tuned only' if gate_state_tuned else 'all'
    ax.set_title(f'Concentration of coefficient mass at lag=0 ({suffix})\n'
                 f'(grey: n_neurons < {SMALL_ROI_FLAG})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_anchor_loc_phase_per_roi(diag_df, out_path, gate_state_tuned=True):
    """Per ROI, frequency of preferred (anchor_loc, anchor_phase) for the
    top coefficient. 27 cells per ROI panel, plotted as a 9x3 heatmap."""
    df = diag_df[diag_df['has_coefs']].copy()
    if gate_state_tuned:
        df = df[df['state_tuned']]
    df = df[df['n_nonzero'] > 0]
    rois = sorted(df['roi'].dropna().unique().tolist())
    if not rois:
        return
    n_cols = 3
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3 * n_rows),
                             squeeze=False)
    for ax, roi in zip(axes.ravel(), rois):
        g = df[df['roi'] == roi]
        # Pull top-1 anchor coordinates from the list columns.
        top_loc = g['top_anchor_loc'].apply(
            lambda v: v[0] if isinstance(v, list) and v else -1).to_numpy()
        top_phase = g['top_anchor_phase'].apply(
            lambda v: v[0] if isinstance(v, list) and v else -1).to_numpy()
        mask = (top_loc >= 0) & (top_phase >= 0)
        H, _, _ = np.histogram2d(top_loc[mask], top_phase[mask],
                                 bins=[np.arange(N_LOCATIONS + 1) - 0.5,
                                       np.arange(N_ANCHOR_PHASES + 1) - 0.5])
        im = ax.imshow(H, origin='lower', aspect='auto', cmap='magma')
        ax.set_title(f"{roi}  (n_cells={g['neuron'].nunique()})")
        ax.set_xlabel('anchor phase')
        ax.set_ylabel('anchor location (0..8)')
        ax.set_xticks(np.arange(N_ANCHOR_PHASES))
        ax.set_yticks(np.arange(N_LOCATIONS))
        plt.colorbar(im, ax=ax, shrink=0.7)
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    fig.suptitle('Preferred (anchor location, anchor phase) of '
                 'strongest coefficient', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_nonzero_count_distribution(diag_df, out_path, gate_state_tuned=True):
    df = diag_df[diag_df['has_coefs']].copy()
    if gate_state_tuned:
        df = df[df['state_tuned']]
    rois = sorted(df['roi'].dropna().unique().tolist())
    if not rois:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rois)), 5))
    data = [df.loc[df['roi'] == roi, 'n_nonzero'].to_numpy() for roi in rois]
    n_cells = [df.loc[df['roi'] == roi, 'neuron'].nunique() for roi in rois]
    bp = ax.boxplot(data, positions=np.arange(len(rois)),
                    widths=0.6, patch_artist=True, showfliers=False)
    for patch, n in zip(bp['boxes'], n_cells):
        if n < SMALL_ROI_FLAG:
            patch.set_facecolor((0.7, 0.7, 0.7, 0.5))
        else:
            patch.set_facecolor((0.45, 0.7, 0.45, 0.6))
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels([f'{r}\n(n={n})' for r, n in zip(rois, n_cells)],
                       rotation=45, ha='right')
    ax.set_ylabel('# non-zero coefficients (out of 324)')
    suffix = 'state-tuned only' if gate_state_tuned else 'all'
    ax.set_title(f'Coefficient sparsity per (neuron, config) ({suffix})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


plot_preferred_lag_per_roi(diag_df,
                           os.path.join(OUT_DIR, 'preferred_lag_per_roi.png'),
                           gate_state_tuned=True)
plot_lag_mass_per_roi(diag_df,
                      os.path.join(OUT_DIR, 'lag_mass_0_vs_other_per_roi.png'),
                      gate_state_tuned=True)
plot_frac_l1_at_lag0(diag_df,
                     os.path.join(OUT_DIR, 'frac_l1_at_lag0_per_roi.png'),
                     gate_state_tuned=True)
plot_anchor_loc_phase_per_roi(
    diag_df,
    os.path.join(OUT_DIR, 'top_anchor_loc_phase_per_roi.png'),
    gate_state_tuned=True,
)
plot_nonzero_count_distribution(
    diag_df,
    os.path.join(OUT_DIR, 'sparsity_per_roi.png'),
    gate_state_tuned=True,
)
print("Saved per-ROI coef plots.")


# ── Optional cross-script comparison ─────────────────────────────────
def load_simple_results(simple_dir):
    csv = os.path.join(simple_dir, 'encoding_results.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    if 'subject' in df.columns:
        df['subject'] = df['subject'].map(lambda s: f'{int(s):02d}')
    # encoding_analysis_simple uses 'model' column with various names;
    # keep only the DSR clo_og rows.
    if 'model' in df.columns:
        df = df[df['model'] == 'dsr']
    # Test config column name in encoding_analysis_simple is 'test_config'
    # (verified by reading the script). Mean-r is 'mean_r'.
    keep = ['subject', 'neuron', 'roi', 'test_config', 'mean_r', 'p_perm']
    have = [c for c in keep if c in df.columns]
    return df[have].rename(columns={'mean_r': 'r_simple_dsr',
                                    'p_perm': 'p_simple_dsr'})


if SIMPLE_RUN_TAG is None and os.path.isdir(SIMPLE_BASE):
    SIMPLE_RUN_TAG = find_latest_run(SIMPLE_BASE)
if SIMPLE_RUN_TAG:
    simple_dir = os.path.join(SIMPLE_BASE, SIMPLE_RUN_TAG)
    print(f"Cross-comparing against simple run: {simple_dir}")
    simple_df = load_simple_results(simple_dir)
    if simple_df is not None and not simple_df.empty:
        # encoding_analysis_simple records 'test_config' per fold in
        # encoding_results.csv? Check column existence; if not, skip.
        if 'test_config' not in simple_df.columns:
            print("  encoding_analysis_simple results lack 'test_config' "
                  "column; skipping per-fold comparison.")
        else:
            elgaby_small = results_df[['subject', 'neuron', 'roi',
                                       'test_config', 'r_used',
                                       'r_4state', 'r_bins_all',
                                       'state_tuned']].copy()
            merged = elgaby_small.merge(
                simple_df, on=['subject', 'neuron', 'test_config'],
                how='inner', suffixes=('', '_s'))
            print(f"  matched {len(merged)} (neuron, test_config) rows.")
            merged_path = os.path.join(OUT_DIR, 'elgaby_vs_simple.csv')
            merged.to_csv(merged_path, index=False)

            if not merged.empty:
                # Scatter per ROI: r_simple_dsr vs r_used (el-gaby).
                rois = sorted(merged['roi'].dropna().unique().tolist())
                n_cols = 3
                n_rows = int(np.ceil(len(rois) / n_cols))
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(4 * n_cols, 3.5 * n_rows),
                                         squeeze=False)
                for ax, roi in zip(axes.ravel(), rois):
                    g = merged[merged['roi'] == roi]
                    ax.scatter(g['r_simple_dsr'], g['r_used'],
                               s=8, alpha=0.5)
                    lo = min(g['r_simple_dsr'].min(), g['r_used'].min())
                    hi = max(g['r_simple_dsr'].max(), g['r_used'].max())
                    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
                    ax.axhline(0, color='gray', lw=0.5)
                    ax.axvline(0, color='gray', lw=0.5)
                    n_neurons = g['neuron'].nunique()
                    ax.set_title(f'{roi} (n_cells={n_neurons}, '
                                 f'rows={len(g)})')
                    ax.set_xlabel('r (encoding_analysis_simple dsr)')
                    ax.set_ylabel('r (el-gaby, r_used)')
                for ax in axes.ravel()[len(rois):]:
                    ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(OUT_DIR,
                                         'scatter_simple_vs_elgaby.png'),
                            dpi=150)
                plt.close(fig)
                print("  saved scatter_simple_vs_elgaby.png")

                # Summary: mean Δr per ROI (el-gaby - simple).
                comp_rows = []
                for roi, g in merged.groupby('roi', sort=False):
                    comp_rows.append({
                        'roi':                roi,
                        'n_rows':             int(len(g)),
                        'mean_r_simple':      float(g['r_simple_dsr'].mean()),
                        'mean_r_elgaby':      float(g['r_used'].mean()),
                        'mean_delta':         float((g['r_used']
                                                     - g['r_simple_dsr']).mean()),
                        'corr_r_to_r':        float(
                            g[['r_simple_dsr', 'r_used']].corr().iloc[0, 1]),
                    })
                comp_df = pd.DataFrame(comp_rows).sort_values('roi').reset_index(drop=True)
                comp_df.to_csv(os.path.join(OUT_DIR,
                                            'elgaby_vs_simple_per_roi.csv'),
                               index=False)
                print("Cross-script ROI comparison:")
                print(comp_df.to_string(index=False))
else:
    print("No encoding_analysis_simple run to compare against; skipping.")


# ── Console summary ──────────────────────────────────────────────────
print("\nPer-ROI coef summary (state-tuned only):")
cols_show = ['roi', 'n_neurons',
             'mean_n_nonzero', 'mean_frac_l1_at_lag0',
             'frac_with_pref_lag_eq_0', 'small_n_flag']
print(roi_summary_gated[cols_show].to_string(index=False))

print(f"\nAll diagnostic files written to {OUT_DIR}")
