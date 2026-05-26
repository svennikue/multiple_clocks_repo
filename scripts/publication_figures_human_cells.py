#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-figure driver for the human single-cell DSR / state / RSA
results.

This script does NOT do any analysis — every figure here reads from
already-saved result files (CSVs, pickles) and dispatches to plotting
helpers in ``mc.plotting.cell_results``. The intent is that the *story*
of the paper lives in this file, while individual stats / fits / RSA
pipelines stay in their own dedicated scripts.

Figures
-------
fig1  ACC DSR survives motor/location-phase/state confounds.
fig2  State-cell overlap between encoding_analysis_simple (state model)
      and the cross-validated state-tuning analysis
      (wrapper_CV_identify_state_tuning.py).
fig3  Per-ROI DSR coefficient lag profile (now-vs-future emphasis).
fig4  Example "nice" DSR cells: best anchor + lag profile + actual-vs-
      predicted polar trace.
fig5  Cell-level encoding ↔ population RSA overlap (DSR signal across
      ROIs).

Run a subset of figures via FIGURES_TO_RUN. Each figure writes into
``<OUT_BASE>/<run_tag>/figN_*.png``.

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc
from mc.plotting import cell_results as cr


# ── Inputs ────────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'

# Encoding run (fine-grained ROI labels from the alt_final_roi column).
ENCODING_RUN_DIR = os.path.join(
    DATA_DIR, 'group/encoding_analysis_simple/2026-05-22_15-57-38')
ENCODING_RESULTS_CSV = os.path.join(ENCODING_RUN_DIR, 'encoding_results.csv')
ENCODING_DIAGNOSTICS_PKL = os.path.join(ENCODING_RUN_DIR, 'diagnostics.pkl')

# Cross-validated state-tuning results (per-cell CV consistency + p_perm).
STATE_CV_CSV = os.path.join(
    DATA_DIR,
    'group/state_tuning/pval_for_perms200_state_consistency_all_correct_repeats_excl_gridwise_qc_pct_neurons.csv',
)

# ROI lookup used by the encoding pipeline (alt_final_roi -> fine-grained).
ROI_TABLE_PATH = os.path.join(DATA_DIR, 'neurons_with_final_roi_labels.csv')
ROI_LABEL_COLUMN = 'alt_final_roi'

# RSA result (used by fig5).  Set to None to skip the RSA panel for now.
RSA_BETA_TABLE = None    # e.g. .../group/rsa/MRI_combo-nofdb_midn-state_betas.csv

# ── Output ─────────────────────────────────────────────────────────────
OUT_BASE = os.path.join(DATA_DIR, 'group/publication_figures')
RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)

# Which figures to run.  Options: 'fig1', 'fig2', 'fig3', 'fig4', 'fig5',
# or 'all'.
FIGURES_TO_RUN = ['all']

# Headline ROIs (used by figs 1, 3, 4).  ACC is primary; HC + Visual are
# included for the dissociation context.
HEADLINE_ROIS = ['ACC']
CONTEXT_ROIS = ['HC_anterior', 'HC_mid', 'Visual', 'medialOFC']

# Per-figure detailed settings.
FIG2 = dict(
    alpha_perm=0.05,           # significance threshold for "state-tuned"
    min_cells_per_roi=20,      # ROIs with fewer cells are flagged
    cells_per_roi_for_polar=3, # top-N polar plots per major ROI
    polar_smooth_sigma=4,
    polar_target_rois=('ACC', 'HC_anterior', 'HC_mid',
                       'medialOFC', 'Parahippocampal'),
)

QUICK_TEST = False
if QUICK_TEST:
    FIG2['cells_per_roi_for_polar'] = 1


# ── Setup ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Publication-figure run output: {OUT_DIR}")
print(f"Figures to run: {FIGURES_TO_RUN}")

_run = set(FIGURES_TO_RUN if FIGURES_TO_RUN != ['all']
           else ['fig1', 'fig2', 'fig3', 'fig4', 'fig5'])


# ── Shared helpers ─────────────────────────────────────────────────────
def _parse_neuron_label(label):
    """'01_07-07-chan120-EC' -> (subject:int, cell_idx:int).

    Returns (None, None) if the label doesn't parse.
    """
    try:
        sub_str, rest = str(label).split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def _load_roi_table():
    """Load and index the MNI/ROI table by (subject, cell idx)."""
    df = pd.read_csv(ROI_TABLE_PATH)
    df = df[['subject', 'cell idx', ROI_LABEL_COLUMN,
             'MNI_x', 'MNI_y', 'MNI_z']].copy()
    df['subject'] = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


def _attach_finegrained_roi(df, neuron_col, roi_table):
    """Add a `roi_fine` column to `df` by parsing the neuron label and
    looking up the alt_final_roi value. Rows with un-parseable labels or
    no ROI-table entry get roi_fine = NaN."""
    rois = []
    for n in df[neuron_col]:
        sub, cidx = _parse_neuron_label(n)
        if sub is None:
            rois.append(np.nan); continue
        try:
            r = roi_table.loc[(sub, cidx), ROI_LABEL_COLUMN]
        except KeyError:
            rois.append(np.nan); continue
        if isinstance(r, pd.Series):
            r = r.dropna().iloc[0] if r.notna().any() else np.nan
        rois.append(r if pd.notna(r) else np.nan)
    out = df.copy()
    out['roi_fine'] = rois
    return out


# =====================================================================
#  Fig 2 — State-cell overlap (CV-tuning ↔ encoding state model)
# =====================================================================
def run_fig2():
    print('\n========== Fig 2: state-cell overlap ==========')
    if not os.path.isfile(STATE_CV_CSV):
        print(f'  state CV CSV not found at\n  {STATE_CV_CSV}\n  skipping fig2.')
        return
    if not os.path.isfile(ENCODING_RESULTS_CSV):
        print(f'  encoding results CSV not found; skipping fig2.')
        return

    roi_table = _load_roi_table()

    # --- Load CV state tuning -----------------------------------------
    cv_df = pd.read_csv(STATE_CV_CSV)
    # Keep only one row per (session, neuron) — the file has one row each.
    cv_df = cv_df.dropna(subset=['p_perm', 'state_cv_consistency'])
    cv_df = cv_df.rename(columns={'p_perm': 'cv_p_perm'})
    cv_df = _attach_finegrained_roi(cv_df, 'neuron_id', roi_table)
    print(f"  CV state-tuning rows (with fine ROI): "
          f"{cv_df['roi_fine'].notna().sum()} / {len(cv_df)}")

    # --- Load encoding state ------------------------------------------
    enc_df = pd.read_csv(ENCODING_RESULTS_CSV)
    enc_state = (enc_df[enc_df['model'] == 'state']
                 [['neuron', 'mean_r', 'p_perm', 'roi']]
                 .rename(columns={'mean_r': 'enc_state_mean_r',
                                  'p_perm': 'enc_p_perm'})
                 .dropna(subset=['enc_state_mean_r', 'enc_p_perm']))
    enc_state = _attach_finegrained_roi(enc_state, 'neuron', roi_table)
    # Prefer ROI from the ROI table; fall back to the encoding's own column.
    enc_state['roi_fine'] = enc_state['roi_fine'].fillna(enc_state['roi'])
    print(f"  Encoding state rows: {len(enc_state)}")

    # --- Match on neuron id -------------------------------------------
    merged = enc_state.merge(
        cv_df[['neuron_id', 'state_cv_consistency', 'cv_p_perm', 'pref_state']],
        left_on='neuron', right_on='neuron_id', how='inner',
    )
    print(f"  Merged matched cells: {len(merged)}")

    alpha = FIG2['alpha_perm']
    merged['cv_sig'] = merged['cv_p_perm'] < alpha
    merged['enc_sig'] = merged['enc_p_perm'] < alpha
    merged['both_sig'] = merged['cv_sig'] & merged['enc_sig']
    merged['cv_only'] = merged['cv_sig'] & ~merged['enc_sig']
    merged['enc_only'] = ~merged['cv_sig'] & merged['enc_sig']

    # --- Per-ROI overlap table -----------------------------------------
    overlap_rows = []
    for roi, g in merged.groupby('roi_fine', sort=False):
        if pd.isna(roi):
            continue
        n_total = len(g)
        n_both = int(g['both_sig'].sum())
        n_enc_only = int(g['enc_only'].sum())
        n_cv_only = int(g['cv_only'].sum())
        n_either = n_both + n_enc_only + n_cv_only
        jaccard = n_both / n_either if n_either > 0 else np.nan
        overlap_rows.append({
            'roi': roi,
            'n_total': n_total,
            'n_both': n_both,
            'n_enc_only': n_enc_only,
            'n_cv_only': n_cv_only,
            'n_either': n_either,
            'jaccard': jaccard,
            'percent_overlap': (n_both / n_either) if n_either else np.nan,
        })
    overlap_df = pd.DataFrame(overlap_rows).sort_values('n_total',
                                                       ascending=False)
    overlap_path = os.path.join(OUT_DIR, 'fig2_state_overlap_per_roi.csv')
    overlap_df.to_csv(overlap_path, index=False)
    print(f"  saved overlap table → {overlap_path}")
    print(overlap_df.to_string(index=False))

    # --- Panel A: stacked bars ----------------------------------------
    cr.plot_state_overlap_stacked_bars(
        overlap_df,
        save_path=os.path.join(OUT_DIR, 'fig2A_state_overlap_stackedbar.png'),
        title=f'State-cell overlap per ROI (α={alpha})',
        min_n_cells=FIG2['min_cells_per_roi'],
    )

    # --- Panel B: scatter ----------------------------------------------
    cr.plot_state_method_scatter(
        merged,
        save_path=os.path.join(OUT_DIR, 'fig2B_state_method_scatter.png'),
        title=('CV state consistency vs encoding state mean_r '
               '(filled = sig in both)'),
        alpha_perm=alpha,
    )

    # --- Panel C: top state cells per ROI — one figure per cell ------
    polar_target = [r for r in FIG2['polar_target_rois']
                    if r in merged['roi_fine'].unique()]
    if not polar_target:
        print('  no overlap ROIs available for polar grid; skipping panel C')
        return

    n_per_roi = FIG2['cells_per_roi_for_polar']
    top_rows = []
    for roi in polar_target:
        g = merged[(merged['roi_fine'] == roi)
                   & merged['both_sig']].copy()
        if g.empty:
            continue
        # Rank by encoding mean_r * sign(state_cv_consistency) — picks
        # cells positive on both methods.
        g['rank_score'] = g['enc_state_mean_r'] * np.sign(
            g['state_cv_consistency'])
        g = g.sort_values('rank_score', ascending=False).head(n_per_roi)
        top_rows.append(g)
    if not top_rows:
        print('  no cells passed both methods in the polar ROIs; '
              'skipping panel C')
        return
    top_cells_df = pd.concat(top_rows, ignore_index=True)
    print(f"  Per-cell polar figures: {len(top_cells_df)} cells across "
          f"{top_cells_df['roi_fine'].nunique()} ROIs")

    polar_dir = os.path.join(OUT_DIR, 'fig2C_top_state_cells')
    os.makedirs(polar_dir, exist_ok=True)

    # Cache per-subject data (each subject is loaded once).
    wanted_subjects = sorted({s for s, _ in (_parse_neuron_label(n)
                                             for n in top_cells_df['neuron'])
                              if s is not None})
    data_cache = {}
    for sub in wanted_subjects:
        sub_str = f'{sub:02}'
        try:
            sub_data = mc.analyse.helpers_human_cells.load_norm_data(
                DATA_DIR, [sub_str])
        except Exception as exc:
            print(f'    failed to load sub-{sub_str}: {exc}')
            continue
        if f'sub-{sub_str}' in sub_data:
            data_cache[sub] = sub_data[f'sub-{sub_str}']

    for _, row in top_cells_df.iterrows():
        sub, _ = _parse_neuron_label(row['neuron'])
        sub_dict = data_cache.get(sub)
        if sub_dict is None:
            continue
        neurons = sub_dict['normalised_neurons']
        if row['neuron'] not in neurons:
            continue

        # Compute the cell's mean-across-correct-trials trace and the per-
        # config mean traces (correct trials only).
        beh = sub_dict['beh'].copy().reset_index(drop=True)
        beh['config_str'] = (
            beh['loc_A'].astype(int).astype(str) + '-' +
            beh['loc_B'].astype(int).astype(str) + '-' +
            beh['loc_C'].astype(int).astype(str) + '-' +
            beh['loc_D'].astype(int).astype(str)
        )
        neuron_df = neurons[row['neuron']].reset_index(drop=True)
        correct_mask = (beh['correct'] == 1).to_numpy()
        if not correct_mask.any():
            continue
        mean_trace = np.nanmean(
            neuron_df.iloc[correct_mask].to_numpy(), axis=0)
        configs = sorted(beh.loc[correct_mask, 'config_str']
                         .dropna().unique().tolist())
        per_cfg = []
        for cfg in configs:
            idx = beh.index[(beh['config_str'] == cfg)
                            & (beh['correct'] == 1)].to_numpy()
            if len(idx) == 0:
                per_cfg.append(None); continue
            per_cfg.append(np.nanmean(
                neuron_df.iloc[idx].to_numpy(), axis=0))

        safe_neuron = row['neuron'].replace('/', '_')
        out_path = os.path.join(polar_dir,
                                f"{row['roi_fine']}_{safe_neuron}.png")
        cr.plot_single_state_cell_polar(
            mean_trace=mean_trace,
            traces_per_config=per_cfg,
            configs=configs,
            cell_label=row['neuron'],
            roi=row['roi_fine'],
            cv_consistency=float(row['state_cv_consistency']),
            enc_state_mean_r=float(row['enc_state_mean_r']),
            cv_p_perm=float(row['cv_p_perm']),
            enc_p_perm=float(row['enc_p_perm']),
            save_path=out_path,
            smooth_sigma=FIG2['polar_smooth_sigma'],
            n_cols=3,
        )


# ── Shared encoding-results loading + helpers for figs 1/3/4 ─────────
def _parse_folds(x):
    """Parse '[v, v, ...]' fold-string from encoding_results.csv."""
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)
    s = str(x).strip().lstrip('[').rstrip(']')
    if not s:
        return np.array([], dtype=float)
    return np.fromstring(s, sep=',', dtype=float)


def _load_encoding_df(model_filter=None):
    """Load encoding_results.csv with parsed fold lists. Optionally
    restrict to a single model string."""
    df = pd.read_csv(ENCODING_RESULTS_CSV)
    df = df.dropna(subset=['mean_r']).copy()
    df['r_per_fold'] = df['r_per_fold'].map(_parse_folds)
    if model_filter is not None:
        df = df[df['model'] == model_filter]
    return df


def _paired_wilcoxon_greater(a, b):
    """One-sided Wilcoxon, H1: a > b across folds. NaN if too few/all-zero."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 2 or np.all(d == 0):
        return np.nan
    try:
        return float(stats.wilcoxon(d, alternative='greater',
                                    zero_method='wilcox').pvalue)
    except (ValueError, TypeError):
        return np.nan


def _ttest_one_sided_greater(vals):
    """One-sided one-sample t-test, H1: mean > 0."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.all(v == v[0]):
        return np.nan
    try:
        return float(stats.ttest_1samp(v, 0.0, alternative='greater').pvalue)
    except TypeError:                      # scipy < 1.6 fallback
        t, p2 = stats.ttest_1samp(v, 0.0)
        return float(p2 / 2 if t > 0 else 1 - p2 / 2)


# =====================================================================
#  Fig 1 — ACC DSR survives motor / location-phase / state confounds
# =====================================================================
FIG1 = dict(
    rois=['ACC', 'HC_posterior', 'medialOFC'],     # primary + control rows
    motor_models=['bttn_prev', 'bttn_curr', 'bttn_next', 'uncover'],
    location_phase_models=['midnight', 'location', 'phase'],
    alpha_perm=0.05,
    alpha_paired=0.05,
)


def _neurons_passing_dsr_filter(df_roi, drops, alpha):
    """Return the set of neurons in this ROI where NONE of `drops`
    paired-beats DSR (one-sided Wilcoxon across folds, p<alpha = beats).
    """
    kept = set()
    for n, g in df_roi.groupby('neuron', sort=False):
        gi = g.set_index('model')
        if 'dsr' not in gi.index:
            continue
        r_dsr = gi.loc['dsr', 'r_per_fold']
        beaten = False
        for m in drops:
            if m not in gi.index:
                continue
            p = _paired_wilcoxon_greater(gi.loc[m, 'r_per_fold'], r_dsr)
            if np.isfinite(p) and p < alpha:
                beaten = True; break
        if not beaten:
            kept.add(n)
    return kept


def run_fig1():
    print('\n========== Fig 1: ACC DSR survives confounds ==========')
    if not os.path.isfile(ENCODING_RESULTS_CSV):
        print(f'  encoding CSV not found: {ENCODING_RESULTS_CSV}')
        return

    df_all = _load_encoding_df()
    target_rois = [r for r in FIG1['rois']
                   if r in df_all['roi'].unique().tolist()]
    if not target_rois:
        print(f"  none of {FIG1['rois']} present in encoding CSV; skipping.")
        return

    # Per-ROI per-filter results.
    panels = {}
    for roi in target_rois:
        df_roi = df_all[df_all['roi'] == roi]
        dsr_rows = df_roi[df_roi['model'] == 'dsr']
        r_all = dsr_rows['mean_r'].dropna().to_numpy(dtype=float)

        # Define the four columns.
        scenarios = [
            ('all DSR cells', set(dsr_rows['neuron'])),
            ('drop motor-beaten',
             _neurons_passing_dsr_filter(df_roi, FIG1['motor_models'],
                                         FIG1['alpha_paired'])),
            ('drop location/phase-beaten',
             _neurons_passing_dsr_filter(df_roi,
                                         FIG1['location_phase_models'],
                                         FIG1['alpha_paired'])),
            ('drop state-encoders',
             set(df_roi.loc[(df_roi['model'] != 'state'),
                            'neuron'].unique())
             - set(df_roi.loc[(df_roi['model'] == 'state')
                              & (df_roi['p_perm'] < FIG1['alpha_perm']),
                              'neuron'].unique())),
        ]
        roi_panels = {}
        for label, kept in scenarios:
            kept_rows = dsr_rows[dsr_rows['neuron'].isin(kept)]
            excl_rows = dsr_rows[~dsr_rows['neuron'].isin(kept)]
            r_kept = kept_rows['mean_r'].dropna().to_numpy(dtype=float)
            r_excl = excl_rows['mean_r'].dropna().to_numpy(dtype=float)
            p_shift = (float(stats.mannwhitneyu(
                r_kept, r_excl, alternative='two-sided').pvalue)
                if (r_kept.size >= 2 and r_excl.size >= 2) else np.nan)
            roi_panels[label] = {
                'r_all':     r_all,
                'r_kept':    r_kept,
                'n_total':   len(r_all),
                'n_kept':    len(r_kept),
                'p_kept_>0': _ttest_one_sided_greater(r_kept),
                'p_shift':   p_shift,
            }
        panels[roi] = roi_panels

    cr.plot_dsr_confound_filter_grid(
        panels,
        save_path=os.path.join(OUT_DIR, 'fig1_dsr_confound_filter_grid.png'),
        suptitle=(f'DSR encoding under exclusion filters '
                  f'(paired α={FIG1["alpha_paired"]}, '
                  f'state-encoder α={FIG1["alpha_perm"]})'),
    )

    # ACC dissociation inset: DSR mean_r for state-encoders vs not.
    if 'ACC' in target_rois:
        acc_df = df_all[df_all['roi'] == 'ACC']
        acc_dsr = acc_df[acc_df['model'] == 'dsr']
        state_sig = set(acc_df.loc[(acc_df['model'] == 'state')
                                   & (acc_df['p_perm']
                                      < FIG1['alpha_perm']),
                                   'neuron'].unique())
        r_state = acc_dsr.loc[acc_dsr['neuron'].isin(state_sig),
                              'mean_r'].dropna().to_numpy(dtype=float)
        r_no_state = acc_dsr.loc[~acc_dsr['neuron'].isin(state_sig),
                                  'mean_r'].dropna().to_numpy(dtype=float)
        cr.plot_acc_state_vs_dsr_inset(
            r_state, r_no_state,
            save_path=os.path.join(
                OUT_DIR, 'fig1_acc_state_vs_dsr_inset.png'),
            title='ACC: DSR mean_r by state-encoder status',
        )


# =====================================================================
#  Fig 3 — Per-ROI DSR coefficient lag profile
# =====================================================================
FIG3 = dict(
    rois=['ACC', 'HC_anterior', 'HC_posterior', 'Visual', 'posterior_PCC',
          'medialOFC', 'EC', 'Parahippocampal'],
    bold_rois=('ACC',),
    p_perm_threshold=0.05,
    n_locations=9, n_phases=3, n_lags=12,
)


def _load_dsr_coefs_for_lag(diagnostics):
    """For each DSR-significant cell return (per-fold-normalised, then
    fold-averaged) 324-coef vector + roi + p_perm. Mirrors the followup
    script's load_dsr_coefs with `normalize='mean'`."""
    rows = []
    for sub, per_neuron in diagnostics.items():
        for n_lab, per_model in per_neuron.items():
            d = per_model.get('dsr')
            if d is None: continue
            p = d.get('p_perm', np.nan)
            if not (np.isfinite(p) and p < FIG3['p_perm_threshold']):
                continue
            coefs_list = d.get('coefs', [])
            if not coefs_list: continue
            coefs_arr = np.array(
                [np.asarray(c, dtype=float) for c in coefs_list])
            if coefs_arr.ndim != 2:
                continue
            expected = FIG3['n_locations'] * FIG3['n_phases'] * FIG3['n_lags']
            if coefs_arr.shape[1] != expected:
                continue
            # Per-fold mean-normalisation, then average across folds.
            normed = []
            for fold in coefs_arr:
                m = float(fold.mean())
                if abs(m) > 1e-12:
                    normed.append(fold / m)
            if not normed: continue
            mean_coefs = np.mean(normed, axis=0)
            rows.append({'subject': sub, 'neuron': n_lab,
                         'roi': d.get('roi'),
                         'p_perm': p, 'coefs': mean_coefs})
    return pd.DataFrame(rows)


def run_fig3():
    print('\n========== Fig 3: per-ROI DSR lag profile ==========')
    import pickle
    if not os.path.isfile(ENCODING_DIAGNOSTICS_PKL):
        print(f'  diagnostics.pkl not found: {ENCODING_DIAGNOSTICS_PKL}')
        return
    with open(ENCODING_DIAGNOSTICS_PKL, 'rb') as f:
        diagnostics = pickle.load(f)
    coefs_df = _load_dsr_coefs_for_lag(diagnostics)
    print(f"  DSR-significant cells with valid coefs: {len(coefs_df)}")
    if coefs_df.empty:
        print('  no cells; skipping fig3.'); return

    rois = [r for r in FIG3['rois'] if r in coefs_df['roi'].unique()]
    lag_means, lag_sems, friedman_p, n_cells = {}, {}, {}, {}
    for roi in rois:
        sub = coefs_df[coefs_df['roi'] == roi]
        if len(sub) < 2: continue
        A = np.stack([c.reshape(FIG3['n_locations'],
                                FIG3['n_phases'],
                                FIG3['n_lags'])
                      for c in sub['coefs']])
        lag_vals = A.mean(axis=(1, 2))            # (N, n_lags)
        lag_means[roi] = lag_vals.mean(axis=0)
        lag_sems[roi] = lag_vals.std(axis=0) / np.sqrt(len(sub))
        try:
            f = stats.friedmanchisquare(
                *[lag_vals[:, j] for j in range(lag_vals.shape[1])])
            friedman_p[roi] = float(f.pvalue)
        except (ValueError, TypeError):
            friedman_p[roi] = np.nan
        n_cells[roi] = len(sub)

    cr.plot_dsr_lag_overlay(
        lag_means_by_roi=lag_means,
        lag_sems_by_roi=lag_sems,
        friedman_p_by_roi=friedman_p,
        n_cells_by_roi=n_cells,
        save_path=os.path.join(OUT_DIR, 'fig3_dsr_lag_overlay.png'),
        bold_rois=FIG3['bold_rois'],
        title=('DSR coefficient lag profile per ROI '
               '(per-cell mean-normalised, p_perm<'
               f"{FIG3['p_perm_threshold']})"),
    )


# =====================================================================
#  Fig 4 — Example DSR cells (best anchor + lag profile + actual/pred)
# =====================================================================
FIG4 = dict(
    rois=['ACC'],
    cells_per_roi=4,
    p_perm_threshold=0.05,
    n_locations=9, n_phases=3, n_lags=12,
)


def _best_fold_index(d):
    r = np.asarray(d.get('r_per_fold', []), dtype=float)
    if r.size == 0 or not np.isfinite(r).any():
        return None
    return int(np.nanargmax(r))


def run_fig4():
    print('\n========== Fig 4: example DSR cells ==========')
    import pickle
    if not os.path.isfile(ENCODING_DIAGNOSTICS_PKL):
        print(f'  diagnostics.pkl not found: {ENCODING_DIAGNOSTICS_PKL}')
        return
    with open(ENCODING_DIAGNOSTICS_PKL, 'rb') as f:
        diagnostics = pickle.load(f)

    out_dir = os.path.join(OUT_DIR, 'fig4_example_dsr_cells')
    os.makedirs(out_dir, exist_ok=True)

    n_loc, n_ph, n_lg = (FIG4['n_locations'], FIG4['n_phases'],
                         FIG4['n_lags'])
    n_total = n_loc * n_ph * n_lg

    n_plotted = 0
    for roi in FIG4['rois']:
        # Find DSR-perm-sig cells in this ROI, ranked by mean_r.
        candidates = []
        for sub, per_neuron in diagnostics.items():
            for n_lab, per_model in per_neuron.items():
                d = per_model.get('dsr')
                if d is None: continue
                if d.get('roi') != roi: continue
                p = d.get('p_perm', np.nan)
                if not (np.isfinite(p) and p < FIG4['p_perm_threshold']):
                    continue
                mean_r = d.get('mean_r', np.nan)
                if not np.isfinite(mean_r): continue
                coefs_list = d.get('coefs', [])
                if not coefs_list: continue
                candidates.append((mean_r, sub, n_lab, d))
        if not candidates:
            print(f'  {roi}: no DSR-perm-sig cells with coefs.')
            continue
        candidates.sort(reverse=True, key=lambda t: t[0])
        picks = candidates[:FIG4['cells_per_roi']]
        print(f'  {roi}: plotting top {len(picks)} of {len(candidates)} cells')

        for mean_r, sub, n_lab, d in picks:
            best = _best_fold_index(d)
            if best is None: continue
            coefs = np.asarray(d['coefs'][best], dtype=float)
            if coefs.size != n_total: continue
            arr = coefs.reshape(n_loc, n_ph, n_lg)

            # Best anchor: argmax over (loc, phase) of Σ|coef| over lags.
            anchor_mag = np.sum(np.abs(arr), axis=2)              # (9, 3)
            flat_idx = int(np.argmax(anchor_mag))
            best_loc, best_phase = np.unravel_index(flat_idx,
                                                    anchor_mag.shape)
            lag_profile = arr[best_loc, best_phase, :]
            pref_lag = int(np.argmax(np.abs(lag_profile)))
            lag_label = f'pref lag = {pref_lag}'

            # Actual & predicted from the best fold's saved arrays.
            y_pred = np.asarray(
                d.get('y_pred_per_fold', [[]])[best], dtype=float)
            y_test = np.asarray(
                d.get('y_test_per_fold', [[]])[best], dtype=float)
            if y_pred.size == 0 or y_test.size == 0:
                continue
            best_fold_r = float(d['r_per_fold'][best])
            configs = d.get('configs', [])
            best_cfg = (configs[best] if best < len(configs) else f'fold{best}')

            safe = n_lab.replace('/', '_')
            out_path = os.path.join(out_dir, f'{roi}_{safe}.png')
            cr.plot_example_dsr_cell(
                anchor_grid_mag=anchor_mag,
                best_anchor_loc=int(best_loc),
                best_anchor_phase=int(best_phase),
                lag_profile=lag_profile,
                lag_profile_label=lag_label,
                actual_trace=y_test,
                predicted_trace=y_pred,
                cell_label=n_lab,
                roi=roi,
                best_fold_r=best_fold_r,
                best_fold_cfg=str(best_cfg),
                save_path=out_path,
            )
            n_plotted += 1
    print(f'  total example-cell figures saved: {n_plotted}')


# =====================================================================
#  Fig 5 — DSR ↔ RSA per-ROI bar comparison
# =====================================================================
def run_fig5():
    print('\n========== Fig 5: DSR ↔ RSA overlap ==========')
    if RSA_BETA_TABLE is None or not os.path.isfile(RSA_BETA_TABLE):
        print('  RSA_BETA_TABLE not set or not found; skipping fig5.')
        print('  Expected CSV columns: roi, beta_or_r, p, n_cells.')
        return

    rsa_df = pd.read_csv(RSA_BETA_TABLE)
    enc_df = _load_encoding_df(model_filter='dsr')
    # Build per-ROI mean encoding r + one-sided p + cell count.
    enc_rows = []
    for roi, g in enc_df.groupby('roi', sort=False):
        rs = g['mean_r'].dropna().to_numpy(dtype=float)
        if rs.size < 2:
            continue
        p = _ttest_one_sided_greater(rs)
        enc_rows.append({'roi': roi, 'beta_or_r': float(np.mean(rs)),
                         'p': p, 'n_cells': len(rs)})
    enc_summary = pd.DataFrame(enc_rows)

    cr.plot_dsr_rsa_comparison(
        rsa_df=rsa_df, enc_df=enc_summary,
        save_path=os.path.join(OUT_DIR, 'fig5_dsr_vs_rsa.png'),
        title='DSR signal per ROI: RSA β vs encoding mean_r',
    )


# ── Dispatch ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    if 'fig1' in _run: run_fig1()
    if 'fig2' in _run: run_fig2()
    if 'fig3' in _run: run_fig3()
    if 'fig4' in _run: run_fig4()
    if 'fig5' in _run: run_fig5()

    # Persist the run config alongside the figures.
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump({
            'run_tag': RUN_TAG,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'figures_run': list(_run),
            'encoding_run_dir': ENCODING_RUN_DIR,
            'state_cv_csv': STATE_CV_CSV,
            'roi_table_path': ROI_TABLE_PATH,
            'roi_label_column': ROI_LABEL_COLUMN,
            'rsa_beta_table': RSA_BETA_TABLE,
            'fig2_settings': FIG2,
        }, f, indent=2)
    print(f'\nDone in {time.time() - t0:.1f}s — outputs in {OUT_DIR}')
