#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
El-Gaby-style encoding analysis for human single-cell ephys.

Reads Script 1's tuning CSV (`elgaby_pref_phase`, `elgaby_state_tuned`
per (neuron, config)) and runs the published encoding regression:

  - design matrix: 324-feature DSR model from
    mc.simulation.predictions.model_DSR with no_phase_neurons=3
    (same `clo_og` as scripts/encoding_analysis_simple.py)
  - leave-one-config-out cross-validation
  - both training and test bins are restricted to the bins whose
    within-trial phase index equals the *test config's* preferred phase
    (paper-faithful, mirrors El Gaby et al. cell 21)
  - elastic net (alpha=0.01, l1=0.5, positive=True)
  - score = Pearson r between actual and predicted 4-state means at the
    preferred phase (the published metric)
  - per-(neuron, test config) circular-shift null on the actual trace
  - ROI aggregation: gated on `elgaby_state_tuned` in the test config

Outputs go to
  derivatives/group/encoding_analysis_elgaby/<run_tag>/

This is Script 2 of the 3-script pipeline:
  1. elgaby_tuning_characterization.py
  2. encoding_analysis_elgaby.py            <-- you are here
  3. elgaby_lag_analysis.py                 (reads this script's coefs)

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc
from mc.analyse import elgaby_encoding


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE = os.path.join(DATA_DIR, 'group', 'encoding_analysis_elgaby')
TUNING_BASE = os.path.join(DATA_DIR, 'group', 'elgaby_tuning')

# Which tuning run from Script 1 to pull pref_phase + state-tuning flags
# from. None = use the most recently modified folder under TUNING_BASE.
TUNING_RUN_TAG = None

# Trial layout (must match Script 1).
N_BINS_PER_TRIAL = 360
N_STATES = 4
N_PHASES = 3

# ── Ablation preset ──────────────────────────────────────────────────
# Bundles the elastic net + scoring + phase-mask settings.  Each preset
# overwrites ALPHA, POSITIVE, SCORING_MODE, and USE_PHASE_MASK below.
# 'custom' leaves them as set.
#
#   'paper'         : alpha=0.01,  positive=True,  scoring='4state',
#                     use_phase_mask=True
#                     (El-Gaby et al. faithful reproduction)
#   'simple_fit'    : alpha=0.001, positive=False, scoring='4state',
#                     use_phase_mask=True
#                     (matches the elastic-net of encoding_analysis_simple,
#                      but keeps the el-gaby phase-masked 4-state score)
#   'all_bins'      : alpha=0.01,  positive=True,  scoring='all_bins',
#                     use_phase_mask=True
#                     (paper fit + phase-masked training, but scores on
#                      every held-out bin instead of the 4-state means)
#   'no_phase_mask' : alpha=0.001, positive=True,  scoring='all_bins',
#                     use_phase_mask=False
#                     (drops the el-gaby pref_phase dependence: trains on
#                      all 360 bins, scores on all 360 bins. Most similar
#                      to scripts/encoding_analysis_simple.py's DSR fit.)
#   'custom'        : use the four settings below as written
ABLATION_PRESET = 'custom'

# Elastic net + scoring.  Used only if ABLATION_PRESET == 'custom'.
ALPHA = 0.001
L1_RATIO = 0.5
POSITIVE = True
MAX_ITER = 2000

# Held-out score:
#   '4state'   -> Pearson r over per-state means at pref_phase (paper)
#   'all_bins' -> Pearson r over all 360 held-out bins (like simple)
SCORING_MODE = '4state'

# Phase masking:
#   True  -> restrict training + test bins to pref_phase (paper)
#   False -> use all 360 bins; drops the pref_phase dependence entirely
USE_PHASE_MASK = True

# Apply preset (if not 'custom').
_ABLATIONS = {
    'paper':         dict(ALPHA=0.01,  POSITIVE=True,  SCORING_MODE='4state',
                          USE_PHASE_MASK=True),
    'simple_fit':    dict(ALPHA=0.001, POSITIVE=False, SCORING_MODE='4state',
                          USE_PHASE_MASK=True),
    'all_bins':      dict(ALPHA=0.01,  POSITIVE=True,  SCORING_MODE='all_bins',
                          USE_PHASE_MASK=True),
    'no_phase_mask': dict(ALPHA=0.001, POSITIVE=True,  SCORING_MODE='all_bins',
                          USE_PHASE_MASK=False),
}
if ABLATION_PRESET != 'custom':
    if ABLATION_PRESET not in _ABLATIONS:
        raise ValueError(f"Unknown ABLATION_PRESET: {ABLATION_PRESET!r}. "
                         f"Choose one of {list(_ABLATIONS) + ['custom']}")
    _cfg = _ABLATIONS[ABLATION_PRESET]
    ALPHA = _cfg['ALPHA']
    POSITIVE = _cfg['POSITIVE']
    SCORING_MODE = _cfg['SCORING_MODE']
    USE_PHASE_MASK = _cfg['USE_PHASE_MASK']
print(f"ABLATION_PRESET={ABLATION_PRESET!r}  "
      f"-> ALPHA={ALPHA}  POSITIVE={POSITIVE}  "
      f"SCORING_MODE={SCORING_MODE!r}  USE_PHASE_MASK={USE_PHASE_MASK}")

# Circular-shift null on the held-out neuron trace.
N_PERMUTATIONS = 500

# Per-ROI alpha used for the binomial test on per-(neuron, test-config)
# permutation p-values.
ROI_ALPHA = 0.05

# Joblib parallelism over neurons.
N_JOBS = -1

# ROI labels.
ROI_TABLE_PATH = os.path.join(DATA_DIR, 'neurons_with_final_roi_labels.csv')
ROI_LABEL_COLUMN = 'alt_final_roi'
TARGET_ROIS = None

# Subject selection.
SUBJECTS_TO_RUN = 'all'   # 'all' | 'dsr_subs' | list

# Save the 324-feature coefficients per (neuron, test config) for
# Script 3 (lag analysis). Adds a coefs.pkl alongside the CSVs.
SAVE_COEFS = True

# ROIs with fewer neurons than this are flagged but still included.
SMALL_ROI_FLAG = 30

# Quick / diagnostic mode.
QUICK_TEST = False
if QUICK_TEST:
    MAX_SUBJECTS = 1
    MAX_NEURONS_PER_SUBJECT = 5
    N_PERMUTATIONS = 100
else:
    MAX_SUBJECTS = None
    MAX_NEURONS_PER_SUBJECT = None


# ── Resolve runs + output folder ─────────────────────────────────────
def find_latest_tuning_run(base):
    candidates = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and not d.endswith('-null')]
    if not candidates:
        raise FileNotFoundError(f"No tuning runs found under {base}")
    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)))
    return candidates[-1]


if TUNING_RUN_TAG is None:
    TUNING_RUN_TAG = find_latest_tuning_run(TUNING_BASE)
TUNING_DIR = os.path.join(TUNING_BASE, TUNING_RUN_TAG)
TUNING_CSV = os.path.join(TUNING_DIR, 'tuning_per_neuron_config.csv')
if not os.path.isfile(TUNING_CSV):
    raise FileNotFoundError(f"Tuning CSV not found: {TUNING_CSV}")
print(f"Using tuning run: {TUNING_DIR}")

if SUBJECTS_TO_RUN == 'dsr_subs':
    with open(os.path.join(DATA_DIR,
                           'all_sessions_dsrRSA_grouping_summary.json')) as f:
        SUBJECTS = list(json.load(f).keys())
elif SUBJECTS_TO_RUN == 'all':
    SUBJECTS = [f'{i:02}' for i in range(1, 64)]
elif isinstance(SUBJECTS_TO_RUN, list):
    SUBJECTS = list(SUBJECTS_TO_RUN)
else:
    raise ValueError(f"Unknown SUBJECTS_TO_RUN: {SUBJECTS_TO_RUN!r}")

RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Run output: {OUT_DIR}")

run_config = {
    'run_tag':           RUN_TAG,
    'timestamp':         datetime.now().isoformat(timespec='seconds'),
    'data_dir':          DATA_DIR,
    'out_dir':           OUT_DIR,
    'tuning_run_tag':    TUNING_RUN_TAG,
    'tuning_csv':        TUNING_CSV,
    'roi_table_path':    ROI_TABLE_PATH,
    'roi_label_column':  ROI_LABEL_COLUMN,
    'target_rois':       TARGET_ROIS,
    'subjects':          SUBJECTS,
    'n_bins_per_trial':  N_BINS_PER_TRIAL,
    'n_states':          N_STATES,
    'n_phases':          N_PHASES,
    'ablation_preset':   ABLATION_PRESET,
    'alpha':             ALPHA,
    'l1_ratio':          L1_RATIO,
    'positive':          POSITIVE,
    'max_iter':          MAX_ITER,
    'scoring_mode':      SCORING_MODE,
    'use_phase_mask':    USE_PHASE_MASK,
    'n_permutations':    N_PERMUTATIONS,
    'roi_alpha':         ROI_ALPHA,
    'small_roi_flag':    SMALL_ROI_FLAG,
    'save_coefs':        SAVE_COEFS,
    'quick_test':        QUICK_TEST,
}
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(run_config, f, indent=2)


# ── Load Script 1's tuning CSV ───────────────────────────────────────
tuning_df = pd.read_csv(TUNING_CSV)
tuning_df['subject'] = tuning_df['subject'].map(lambda s: f'{int(s):02d}')
TUNING_LOOKUP = tuning_df.set_index(['subject', 'neuron', 'config'])
print(f"Loaded tuning CSV with {len(tuning_df)} (subject, neuron, config) rows.")


# ── ROI lookup (same as Script 1) ────────────────────────────────────
def parse_neuron_label(label):
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def load_roi_table(path, roi_col):
    df = pd.read_csv(path)
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


ROI_TABLE = load_roi_table(ROI_TABLE_PATH, ROI_LABEL_COLUMN)


def get_neuron_roi(label):
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return None
    try:
        roi = ROI_TABLE.loc[(sub, cell_idx), ROI_LABEL_COLUMN]
    except KeyError:
        return None
    if isinstance(roi, pd.Series):
        roi = roi.dropna().iloc[0] if roi.notna().any() else None
    return None if (roi is None or pd.isna(roi)) else str(roi)


def get_neuron_mni(label):
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return (np.nan, np.nan, np.nan)
    try:
        row = ROI_TABLE.loc[(sub, cell_idx)]
    except KeyError:
        return (np.nan, np.nan, np.nan)
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return (float(row['MNI_x']), float(row['MNI_y']), float(row['MNI_z']))


# ── Design matrix + neuron traces per subject ────────────────────────
def mode_per_bin(arr, fallback_ffill=True):
    """Return (T,) mode along axis 0 of (n_reps, T). Optionally ffill/bfill NaN."""
    m = stats.mode(arr, axis=0, keepdims=False, nan_policy='omit').mode
    if fallback_ffill:
        s = pd.Series(m)
        m = s.ffill().bfill().to_numpy()
    return m


def build_design_per_config(beh, locs_df, neurons, configs):
    """Build trial-averaged DSR design (P, 360) per config + neuron trace per config.

    Returns
    -------
    X_per_config       : list of (P, 360) ndarrays, one per config (in `configs` order)
    Y_per_config       : dict[neuron_label] -> list of (360,) ndarrays (one per config)
    config_has_data    : list[bool]  True if config had >=1 correct trial
    """
    X_per_config = []
    Y_per_config = {n_lab: [] for n_lab in neurons}
    config_has_data = []

    for cfg in configs:
        mask = (beh['config_str'] == cfg) & (beh['correct'] == 1)
        idx = beh.index[mask].to_numpy()

        if len(idx) == 0:
            X_per_config.append(np.zeros((0, N_BINS_PER_TRIAL)))
            for n_lab in neurons:
                Y_per_config[n_lab].append(np.full(N_BINS_PER_TRIAL, np.nan))
            config_has_data.append(False)
            continue

        locs_arr = locs_df.iloc[idx].to_numpy()          # (n_reps, 360)
        mode_loc = mode_per_bin(locs_arr).astype(int)
        loc_int = np.clip(mode_loc - 1, 0, 8)             # 0-indexed for model_DSR
        _, _, _, _, clo_og, _, _ = (
            mc.simulation.predictions.model_DSR(
                locations=loc_int.tolist(),
                no_phase_neurons=N_PHASES,
            )
        )
        X_per_config.append(clo_og)
        for n_lab, df in neurons.items():
            neuron_arr = df.iloc[idx].to_numpy()
            Y_per_config[n_lab].append(np.nanmean(neuron_arr, axis=0))
        config_has_data.append(True)

    return X_per_config, Y_per_config, config_has_data


# ── Per-neuron worker ────────────────────────────────────────────────
def analyse_one_neuron(neuron_label, roi_name, sub_str,
                       X_per_config, y_per_config, configs,
                       n_permutations, seed):
    """Run el-gaby CV for one neuron and return per-fold result rows."""
    # Look up pref_phase + state-tuning + phase-tuning per config from Script 1.
    pref_phase, state_tuned, phase_tuned = [], [], []
    state_tuning_p, phase_tuning_p, n_trials_cfg = [], [], []
    for cfg in configs:
        key = (sub_str, neuron_label, cfg)
        if key in TUNING_LOOKUP.index:
            row = TUNING_LOOKUP.loc[key]
            pref_phase.append(int(row['elgaby_pref_phase']))
            state_tuned.append(bool(row['elgaby_state_tuned']))
            phase_tuned.append(bool(row['elgaby_phase_tuned']))
            state_tuning_p.append(float(row['elgaby_state_tuning_p']))
            phase_tuning_p.append(float(row['elgaby_phase_tuning_p']))
            n_trials_cfg.append(int(row['n_trials']))
        else:
            pref_phase.append(-1)
            state_tuned.append(False)
            phase_tuned.append(False)
            state_tuning_p.append(np.nan)
            phase_tuning_p.append(np.nan)
            n_trials_cfg.append(0)

    rng = np.random.default_rng(seed)
    folds = elgaby_encoding.fit_elgaby_one_neuron(
        X_per_config=X_per_config,
        y_per_config=y_per_config,
        pref_phase_per_config=pref_phase,
        state_tuned_per_config=state_tuned,
        n_bins_per_trial=N_BINS_PER_TRIAL,
        n_states=N_STATES, n_phases=N_PHASES,
        alpha=ALPHA, l1_ratio=L1_RATIO,
        positive=POSITIVE, max_iter=MAX_ITER,
        scoring_mode=SCORING_MODE,
        use_phase_mask=USE_PHASE_MASK,
        n_permutations=n_permutations, rng=rng,
    )

    mni_x, mni_y, mni_z = get_neuron_mni(neuron_label)
    rows, coefs_by_fold = [], {}
    for fold in folds:
        cfg = configs[fold['test_cfg_idx']]
        rows.append({
            'subject':              sub_str,
            'neuron':               neuron_label,
            'roi':                  roi_name,
            'MNI_x':                mni_x,
            'MNI_y':                mni_y,
            'MNI_z':                mni_z,
            'test_config':          cfg,
            'pref_phase':           fold['pref_phase'],
            'state_tuned':          fold['state_tuned'],
            'phase_tuned':          bool(phase_tuned[fold['test_cfg_idx']]),
            'state_tuning_p':       state_tuning_p[fold['test_cfg_idx']],
            'phase_tuning_p':       phase_tuning_p[fold['test_cfg_idx']],
            'n_trials_test_cfg':    n_trials_cfg[fold['test_cfg_idx']],
            'n_train_bins':         fold['n_train_bins'],
            'n_test_bins':          fold['n_test_bins'],
            'r_4state':             fold['r_4state'],
            'r_bins_all':           fold['r_bins_all'],
            'scoring_mode':         fold['scoring_mode'],
            'r_used':               fold['r_used'],
            'p_perm':               fold['p_perm'],
            'all_coefs_zero':       fold['all_coefs_zero'],
        })
        coefs_by_fold[cfg] = fold['coefs']

    return rows, coefs_by_fold


# ── ROI aggregation ──────────────────────────────────────────────────
def per_roi_summary(results_df, gate_state_tuned=False, gate_phase_tuned=False,
                    alpha=ROI_ALPHA):
    """One row per ROI: paper-style population test on `r_used` (the
    score matching the active SCORING_MODE) + binomial test on per-
    (neuron, test-config) permutation p-values.

    Both gates apply at the (neuron, test_config) level using the flags
    Script 1 wrote.  Setting both True restricts to neurons that are
    BOTH state- and phase-tuned in the test config.
    """
    rows = []
    if results_df is None or results_df.empty:
        return pd.DataFrame(rows)

    df = results_df.copy()
    df = df[np.isfinite(df['r_used'])]
    if gate_state_tuned:
        df = df[df['state_tuned']]
    if gate_phase_tuned and 'phase_tuned' in df.columns:
        df = df[df['phase_tuned']]

    for roi, g in df.groupby('roi', sort=False):
        rs = g['r_used'].to_numpy(dtype=float)
        p_perms = g['p_perm'].dropna().to_numpy(dtype=float)
        n_rows = int(rs.size)
        n_neurons = int(g['neuron'].nunique())
        if n_rows == 0:
            continue

        # One-sample t-test, one-sided greater (paper-style population test).
        try:
            t_res = stats.ttest_1samp(rs, popmean=0.0, alternative='greater')
            t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
        except Exception:
            t_stat, t_p = np.nan, np.nan

        # Wilcoxon signed-rank, one-sided greater.
        try:
            wil = stats.wilcoxon(rs, alternative='greater',
                                 zero_method='wilcox')
            wil_p = float(wil.pvalue)
        except (ValueError, TypeError):
            wil_p = np.nan

        # Binomial test on per-row permutation p-values.
        n_p = int(p_perms.size)
        n_sig = int(np.sum(p_perms < alpha))
        try:
            bin_p = float(stats.binomtest(
                n_sig, n_p, p=alpha, alternative='greater').pvalue
            ) if n_p > 0 else np.nan
        except AttributeError:
            try:
                bin_p = float(stats.binom_test(
                    n_sig, n_p, p=alpha, alternative='greater'))
            except Exception:
                bin_p = np.nan

        rows.append({
            'roi':                 roi,
            'n_neurons':           n_neurons,
            'n_rows':              n_rows,
            'mean_r':              float(np.mean(rs)),
            'median_r':            float(np.median(rs)),
            'sem_r':               float(stats.sem(rs)),
            't_stat':              t_stat,
            't_p_greater':         t_p,
            'wilcoxon_p_greater':  wil_p,
            'n_sig':               n_sig,
            'n_with_p_perm':       n_p,
            'frac_sig':            (n_sig / n_p) if n_p > 0 else np.nan,
            'binomial_p_greater':  bin_p,
            'alpha':               float(alpha),
            'small_n_flag':        bool(n_neurons < SMALL_ROI_FLAG),
        })
    return pd.DataFrame(rows).sort_values('roi').reset_index(drop=True)


# ── Plots ────────────────────────────────────────────────────────────
def plot_roi_box(results_df, gate_state_tuned, out_path, title_suffix,
                 gate_phase_tuned=False):
    df = results_df[np.isfinite(results_df['r_used'])]
    if gate_state_tuned:
        df = df[df['state_tuned']]
    if gate_phase_tuned and 'phase_tuned' in df.columns:
        df = df[df['phase_tuned']]
    if df.empty:
        return
    rois = sorted(df['roi'].dropna().unique().tolist())
    data = [df.loc[df['roi'] == roi, 'r_used'].to_numpy() for roi in rois]
    n_neurons = [df.loc[df['roi'] == roi, 'neuron'].nunique() for roi in rois]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rois)), 5))
    bp = ax.boxplot(data, positions=np.arange(len(rois)),
                    widths=0.6, patch_artist=True, showfliers=False)
    # Small-n ROIs get a translucent grey fill.
    for patch, n in zip(bp['boxes'], n_neurons):
        if n < SMALL_ROI_FLAG:
            patch.set_facecolor((0.7, 0.7, 0.7, 0.5))
        else:
            patch.set_facecolor((0.35, 0.5, 0.8, 0.6))
    ax.axhline(0.0, ls='--', color='k', lw=0.8)
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels([f'{r}\n(n={n})' for r, n in zip(rois, n_neurons)],
                       rotation=45, ha='right')
    ax.set_ylabel(f'Pearson r ({SCORING_MODE})')
    ax.set_title(f'El-Gaby encoding {title_suffix}\n'
                 f'(grey boxes: n_neurons < {SMALL_ROI_FLAG})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roi_mean(roi_df, out_path, title):
    if roi_df is None or roi_df.empty:
        return
    df = roi_df.sort_values('mean_r', ascending=False)
    x = np.arange(len(df))
    colors = [(0.7, 0.7, 0.7, 0.8) if f else (0.35, 0.5, 0.8, 0.8)
              for f in df['small_n_flag']]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(df)), 5))
    ax.bar(x, df['mean_r'], yerr=df['sem_r'], color=colors,
           edgecolor='k', capsize=3)
    ax.axhline(0.0, ls='--', color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}\n(n={n})'
                        for r, n in zip(df['roi'], df['n_neurons'])],
                       rotation=45, ha='right')
    ax.set_ylabel('mean r (4-state)')
    ax.set_title(title + f'\n(grey: n_neurons < {SMALL_ROI_FLAG})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main loop ────────────────────────────────────────────────────────
def main():
    target_set = set(TARGET_ROIS) if TARGET_ROIS else None
    print("\nLoading subject data...")
    SUBJECT_DATA = {}
    for sub_str in SUBJECTS:
        SUBJECT_DATA[sub_str] = mc.analyse.helpers_human_cells.load_norm_data(
            DATA_DIR, [sub_str],
        )
    print(f"Cached data for {len(SUBJECT_DATA)} subjects.")

    all_rows = []
    coefs_store = {}     # {sub: {neuron: {test_cfg: coefs}}}
    subjects_processed = 0
    t_overall = time.time()

    for sub_str in SUBJECTS:
        if MAX_SUBJECTS is not None and subjects_processed >= MAX_SUBJECTS:
            break

        print(f"\n========== Subject sub-{sub_str} ==========")
        t_sub = time.time()
        if f"sub-{sub_str}" not in SUBJECT_DATA[sub_str]:
            print(f"  no data; skipping.")
            continue
        sub_dict = SUBJECT_DATA[sub_str][f"sub-{sub_str}"]

        beh = sub_dict['beh'].copy().reset_index(drop=True)
        beh['config_str'] = (
            beh['loc_A'].astype(int).astype(str) + '-' +
            beh['loc_B'].astype(int).astype(str) + '-' +
            beh['loc_C'].astype(int).astype(str) + '-' +
            beh['loc_D'].astype(int).astype(str)
        )
        configs = sorted(beh['config_str'].dropna().unique().tolist())
        locs_df = sub_dict['locations'].reset_index(drop=True)
        neurons_raw = sub_dict['normalised_neurons']

        # ROI filter.
        neuron_labels = []
        for n_lab in neurons_raw:
            roi = get_neuron_roi(n_lab)
            if roi is None:
                continue
            if target_set is not None and roi not in target_set:
                continue
            neuron_labels.append(n_lab)
        if MAX_NEURONS_PER_SUBJECT is not None:
            neuron_labels = neuron_labels[:MAX_NEURONS_PER_SUBJECT]
        if not neuron_labels:
            print(f"  no neurons in target ROIs.")
            continue
        neurons = {n_lab: neurons_raw[n_lab].reset_index(drop=True)
                   for n_lab in neuron_labels}
        rois = {n_lab: get_neuron_roi(n_lab) for n_lab in neurons}
        print(f"  {len(neurons)} neurons across {len(configs)} configs.")

        t_build = time.time()
        X_per_config, Y_per_config, has_data = build_design_per_config(
            beh, locs_df, neurons, configs,
        )
        print(f"  built design in {time.time() - t_build:.1f}s")

        neuron_args = [
            (n_lab, rois[n_lab], Y_per_config[n_lab])
            for n_lab in neurons
        ]
        n_jobs_eff = 1 if len(neuron_args) == 1 else N_JOBS

        t_fit = time.time()
        results = Parallel(n_jobs=n_jobs_eff, verbose=0)(
            delayed(analyse_one_neuron)(
                n_lab, roi, sub_str,
                X_per_config, y_list, configs,
                n_permutations=N_PERMUTATIONS,
                seed=abs(hash((sub_str, n_lab))) & 0xFFFFFFFF,
            )
            for n_lab, roi, y_list in neuron_args
        )
        print(f"  fit + permuted in {time.time() - t_fit:.1f}s")

        sub_coefs = {}
        for (rows, coefs_by_fold), (n_lab, _, _) in zip(results, neuron_args):
            for row in rows:
                if row.get('all_coefs_zero'):
                    pass  # silent; recorded as a column
            all_rows.extend(rows)
            sub_coefs[n_lab] = coefs_by_fold
        coefs_store[sub_str] = sub_coefs

        subjects_processed += 1
        print(f"  subject done in {time.time() - t_sub:.1f}s "
              f"(total {time.time() - t_overall:.1f}s).")

    if not all_rows:
        print("Nothing to save.")
        return

    results_df = pd.DataFrame(all_rows)
    results_path = os.path.join(OUT_DIR, 'encoding_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved {results_path}")

    # ROI summaries: 4 gate variants (all / state-only / phase-only / both).
    GATE_VARIANTS = [
        ('all',               False, False, '(all (neuron, config))'),
        ('state_tuned',       True,  False, '(state-tuned only)'),
        ('phase_tuned',       False, True,  '(phase-tuned only)'),
        ('state_and_phase',   True,  True,  '(state- AND phase-tuned)'),
    ]
    roi_summaries = {}
    for tag, gate_s, gate_p, title_suffix in GATE_VARIANTS:
        roi_df = per_roi_summary(results_df,
                                 gate_state_tuned=gate_s,
                                 gate_phase_tuned=gate_p)
        roi_summaries[tag] = roi_df
        roi_df.to_csv(os.path.join(OUT_DIR, f'roi_summary_{tag}.csv'),
                      index=False)
        plot_roi_box(results_df, gate_state_tuned=gate_s,
                     gate_phase_tuned=gate_p,
                     out_path=os.path.join(OUT_DIR, f'roi_box_{tag}.png'),
                     title_suffix=title_suffix)
        plot_roi_mean(roi_df,
                      out_path=os.path.join(OUT_DIR, f'roi_mean_{tag}.png'),
                      title=f'Mean r per ROI {title_suffix}')

    # Keep the historical alias for the script-level summary printout.
    roi_gated = roi_summaries['state_tuned']

    if SAVE_COEFS:
        coefs_path = os.path.join(OUT_DIR, 'coefs.pkl')
        with open(coefs_path, 'wb') as f:
            pickle.dump({'coefs': coefs_store,
                         'roi_table_path': ROI_TABLE_PATH,
                         'roi_label_column': ROI_LABEL_COLUMN},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {coefs_path}")

    # Quick console summary.
    print("\nROI summary (state-tuned only):")
    cols_show = ['roi', 'n_neurons', 'mean_r', 't_p_greater',
                 'wilcoxon_p_greater', 'frac_sig', 'binomial_p_greater',
                 'small_n_flag']
    if not roi_gated.empty:
        print(roi_gated[cols_show].to_string(index=False))


if __name__ == '__main__':
    main()
