#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoding analysis for human single-cell ephys (DSR paradigm).

Per subject:
  1. Build per-config behavioural / task-structure regressors for every model.
  2. Build per-neuron 360-bin traces (mean over correct repeats, or flattened
     raw repeats — see AVERAGE_REPEATS).
  3. Leave-one-config-out cross-validation: fit elastic net on the 7 training
     configs, predict the held-out config, correlate predicted vs actual.
  4. One-sided permutation p-values via circular shifts of the neuron trace
     (independent per config; the regressors are NOT permuted).

Config order is preserved across all data structures so leave-one-out works.

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
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE = os.path.join(DATA_DIR, 'group', 'encoding_analysis_simple')

# Reload mode: set to the run tag of a previous run (e.g.
# '2026-05-18_16-33-05') to skip the heavy elastic-net + permutation loop
# and just re-render the summary plots from the saved
# encoding_results.csv (+ diagnostics.pkl, if present).  None = fresh run.
RELOAD_RUN = None # '2026-05-18_16-33-05' # None

N_PHASES = 3
N_BINS_PER_TRIAL = 360
N_SUBPATHS_PER_TRIAL = 12
states = ['A', 'B', 'C', 'D']

# True  -> mode-buttons / mode-locations + mean-of-neurons => 1×360 per config
# False -> raw per-repeat regressors and neurons, flattened across repeats
AVERAGE_REPEATS = True

# Elastic net
ALPHA = 0.001 # 0.001         # 0.01 in El-Gaby; smaller = less penalty
L1_RATIO = 0.5
POSITIVE = True
MAX_ITER = 2000

# Permutations (one-sided test: emp > perm)
N_PERMUTATIONS = 500
N_JOBS = -1            # joblib parallelism over neurons

# Models to evaluate per neuron. Each must be buildable by
# build_single_trial_regressors.
# models = [
#      'dsr', 'dsr_only_fut', 'dsr_now_next','location','midnight', 
#      'phase', 'state', 'state_phase',
#     'bttn_prev', 'bttn_next', 'bttn_curr', 'uncover',
# ]
models = [
     'dsr','location',
     'phase', 'state', 'state_phase',
    'bttn_prev', 'bttn_next', 'bttn_curr', 'uncover',
]

# ROI labels are taken from the MNI-coordinate-based table produced by
# scripts/cell_to_roi_MNI.py (column `final_roi`).  Rows are matched to
# neuron labels via (subject, cell_idx) parsed from the label
# `{sub:02d}_{cell_idx:02d}-...`.
ROI_TABLE_PATH = os.path.join(
    DATA_DIR, 'neurons_with_final_roi_labels.csv'
)

# Only analyse neurons whose final_roi is in this list. None = all ROIs
# present in the table.
TARGET_ROIS = None  # e.g. ['EC', 'HC_anterior', 'ACC']

# p-value threshold for the post-hoc glass-brain plots of significant cells.
GLASSBRAIN_P_THRESHOLD = 0.1
PLOT_GLASSBRAIN_3D      = True

# Subject selection:
#   'dsr_subs' -> load from DSR JSON summary (only subjects who ran that task)
#   'all'      -> sub-01 … sub-63
#   list       -> e.g. ['02', '31', '33']
SUBJECTS_TO_RUN = 'all' #'dsr_subs' # ['31'] # 'all'

# ── Test/diagnostic mode ─────────────────────────────────────────────
# Sanity-check mode: process a tiny subset, save per-(neuron, model)
# diagnostics, and render time-course + permutation-histogram plots.
QUICK_TEST = False

if QUICK_TEST:
    MAX_SUBJECTS            = 1
    MAX_NEURONS_PER_SUBJECT = 1
    N_PERMUTATIONS          = 100
    PLOT_DIAGNOSTICS        = True
else:
    MAX_SUBJECTS            = None
    MAX_NEURONS_PER_SUBJECT = None
    PLOT_DIAGNOSTICS        = False

# Per-neuron DSR-family coefficient maps. For each neuron with p_perm <
# DSR_COEF_MAP_ALPHA on dsr / dsr_now_next / dsr_only_fut, save an imshow of
# the model's design matrix (best held-out fold's test slice) scaled row-wise
# by the best-fold coefficients. White background = zero-coef rows.
PLOT_DSR_COEF_MAPS    = True
DSR_COEF_MAP_ALPHA    = 0.05
# DSR_COEF_MAP_FAMILY   = ('dsr', 'dsr_now_next', 'dsr_only_fut')
DSR_COEF_MAP_FAMILY   = ('dsr',)   # trailing comma is required for 1-tuple


# Per-(neuron, model) diagnostics (y_pred, y_test, perm_rs) are needed for the
# best-neuron showcase plot, so we always save them.
SAVE_DIAGNOSTICS = True


# ── Per-run output folder + config dump ──────────────────────────────
if RELOAD_RUN is not None:
    RUN_TAG = RELOAD_RUN
    OUT_DIR = os.path.join(OUT_BASE, RELOAD_RUN)
    if not os.path.isdir(OUT_DIR):
        raise FileNotFoundError(f"Reload folder not found: {OUT_DIR}")
    print(f"RELOAD MODE — reading results from: {OUT_DIR}")
else:
    RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


if SUBJECTS_TO_RUN == 'dsr_subs':
    with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json')) as f:
        config_summary = json.load(f)
    SUBJECTS = list(config_summary.keys())
elif SUBJECTS_TO_RUN == 'all':
    SUBJECTS = [f'{i:02}' for i in range(1, 64)]
elif isinstance(SUBJECTS_TO_RUN, list):
    SUBJECTS = list(SUBJECTS_TO_RUN)
else:
    raise ValueError(f"Unknown SUBJECTS_TO_RUN: {SUBJECTS_TO_RUN!r}")

run_config = {
    'run_tag':          RUN_TAG,
    'timestamp':        datetime.now().isoformat(timespec='seconds'),
    'data_dir':         DATA_DIR,
    'out_dir':          OUT_DIR,
    'configs':          'derived from beh per subject',
    'AVERAGE_REPEATS':  AVERAGE_REPEATS,
    'ALPHA':            ALPHA,
    'L1_RATIO':         L1_RATIO,
    'POSITIVE':         POSITIVE,
    'MAX_ITER':         MAX_ITER,
    'N_PERMUTATIONS':   N_PERMUTATIONS,
    'N_JOBS':           N_JOBS,
    'QUICK_TEST':       QUICK_TEST,
    'MAX_SUBJECTS':     MAX_SUBJECTS,
    'MAX_NEURONS_PER_SUBJECT': MAX_NEURONS_PER_SUBJECT,
    'SAVE_DIAGNOSTICS': SAVE_DIAGNOSTICS,
    'models':           models,
    'roi_table_path':   ROI_TABLE_PATH,
    'target_rois':      TARGET_ROIS,
    'glassbrain_p_threshold': GLASSBRAIN_P_THRESHOLD,
    'subjects':         SUBJECTS,
    'N_PHASES':         N_PHASES,
    'N_BINS_PER_TRIAL': N_BINS_PER_TRIAL,
    'states':           states,
}
if RELOAD_RUN is None:
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"Run output: {OUT_DIR}")


# ── Helpers ──────────────────────────────────────────────────────────
def parse_neuron_label(label):
    """Parse a neuron label like '01_07-07-chan120-EC' into (sub:int, cell_idx:int).

    Labels come from `load_norm_data`, which builds them as
    ``f"{sub}_{cell_name}"`` where ``cell_name`` itself starts with
    ``f"{cell_idx:02d}-..."`` taken from the filename ``cell-{cell_idx:02d}-...``.
    Returns (None, None) on malformed labels.
    """
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def _load_roi_table(path):
    """Load the MNI-based ROI table and index it by (subject, cell idx)."""
    df = pd.read_csv(path)
    needed = ['subject', 'cell idx', 'final_roi',
              'MNI_x', 'MNI_y', 'MNI_z', 'electrode label']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"ROI table {path} is missing columns: {missing}"
        )
    df = df.copy()
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


ROI_TABLE = _load_roi_table(ROI_TABLE_PATH)
print(f"Loaded ROI table with {len(ROI_TABLE)} cells "
      f"({ROI_TABLE['final_roi'].nunique()} distinct ROIs) "
      f"from {ROI_TABLE_PATH}")


def get_neuron_roi(label):
    """Look up the MNI-based ROI for a neuron label, or None if missing."""
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return None
    try:
        roi = ROI_TABLE.loc[(sub, cell_idx), 'final_roi']
    except KeyError:
        return None
    if isinstance(roi, pd.Series):
        # duplicate row — take the first non-null entry
        roi = roi.dropna().iloc[0] if roi.notna().any() else None
    return None if (roi is None or pd.isna(roi)) else str(roi)


def get_neuron_mni(label):
    """Return (MNI_x, MNI_y, MNI_z) for a neuron label, or (nan,nan,nan)."""
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


def nan_safe_pearsonr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x_m, y_m = x[mask], y[mask]
    if np.std(x_m) < 1e-12 or np.std(y_m) < 1e-12:
        return np.nan
    return float(np.corrcoef(x_m, y_m)[0, 1])


def vectorized_pearsonr(Y, y):
    """Pearson r between each row of Y (n_rows, T) and the vector y (T,).

    Returns (n_rows,) ndarray of correlations. Rows with zero variance
    return NaN. Assumes no NaNs in Y / y (true for our use case).
    """
    Y = np.asarray(Y, dtype=float)
    y = np.asarray(y, dtype=float)
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    y_c = y - y.mean()
    num   = Y_c @ y_c
    denom = np.sqrt((Y_c ** 2).sum(axis=1) * (y_c ** 2).sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(denom > 1e-12, num / denom, np.nan)
    return r


def make_circular_shifts(y_test, n_perms, rng):
    """Return (n_perms, T) of independent circular shifts of y_test."""
    T = y_test.shape[0]
    shifts = rng.integers(0, T, size=n_perms)
    idx = (np.arange(T)[None, :] - shifts[:, None]) % T
    return y_test[idx]


def one_hot_buttons(btn_seq, alphabet):
    """One-hot encode a (T,) sequence of button labels using `alphabet`.

    Returns (n_alpha, T) float array. Missing/unknown labels map to all-zero.
    """
    T = len(btn_seq)
    # import pdb; pdb.set_trace()
    oh = np.zeros((len(alphabet), T), dtype=float)
    for i, lbl in enumerate(alphabet):
        oh[i] = (btn_seq == lbl).astype(float)
    return oh


def build_single_trial_regressors(loc_path, btn_path, btn_alphabet,
                                  models_to_build):
    """Build all per-trial regressors for ONE 360-bin trial.

    Parameters
    ----------
    loc_path : (360,) array of grid positions (1-9; will be 0-indexed for model_DSR)
    btn_path : (360,) array of button labels (strings) — may contain 'Return'
    btn_alphabet : list of button labels for one-hot encoding
    models_to_build : iterable of model names

    Returns
    -------
    dict[model_name] -> (P_m, 360) ndarray
    """
    out = {}

    # Locations must be ints in [0, 8] for model_DSR
    loc_int = np.asarray(loc_path).astype(int) - 1
    loc_int = np.clip(loc_int, 0, 8)

    needs_DSR = bool(set(models_to_build) & {
        'location', 'phase', 'state', 'midnight',
        'dsr', 'state_phase', 'dsr_now_next', 'dsr_only_fut'
    })
    if needs_DSR:
        loc_og, phas_og, stat_og, midn_og, clo_og, phas_stat_og, clo_now_next_og = (
            mc.simulation.predictions.model_DSR(
                locations=loc_int.tolist(), no_phase_neurons=N_PHASES,
            )
        )
        DSR_MAP = {
            'location':       loc_og,
            'phase':          phas_og,
            'state':          stat_og,
            'midnight':       midn_og,
            'dsr':            clo_og,
            'state_phase':    phas_stat_og,
            'dsr_now_next':   clo_now_next_og,
            'dsr_only_fut': clo_og - clo_now_next_og
        }
        for m in models_to_build:
            if m in DSR_MAP:
                out[m] = DSR_MAP[m]

    # Button-based regressors
    btn_curr = np.asarray([str(b) for b in btn_path])
    needs_button = bool(set(models_to_build) & {
        'bttn_curr', 'bttn_prev', 'bttn_next', 'uncover',
    })
    if needs_button:
        # uncover: indicator of Return presses (kept from the original trace).
        if 'uncover' in models_to_build:
            out['uncover'] = (btn_curr == 'Return').astype(float)[None, :]

        # For curr / prev / next: treat 'Return' (and any missing 'nan') as
        # gaps and ffill / bfill so every bin carries the most recent
        # direction button. prev / next are then a 1-bin circular shift of
        # the cleaned trace (curr).
        btn_clean = (pd.Series(btn_curr)
                     .replace(['Return', 'nan'], np.nan)
                     .ffill().bfill()
                     .to_numpy())
        btn_prev = np.roll(btn_clean,  1, axis=0)   # button 1 bin earlier
        btn_next = np.roll(btn_clean, -1, axis=0)   # button 1 bin later

        if 'bttn_curr' in models_to_build:
            out['bttn_curr'] = one_hot_buttons(btn_clean, btn_alphabet)
        if 'bttn_prev' in models_to_build:
            out['bttn_prev'] = one_hot_buttons(btn_prev,  btn_alphabet)
        if 'bttn_next' in models_to_build:
            out['bttn_next'] = one_hot_buttons(btn_next,  btn_alphabet)

    return out


def mode_per_bin(arr, fallback_ffill=True):
    """Return (T,) mode along axis 0 of (n_reps, T). Optionally ffill/bfill NaN."""
    m = stats.mode(arr, axis=0, keepdims=False, nan_policy='omit').mode
    if fallback_ffill:
        s = pd.Series(m)
        m = s.ffill().bfill().to_numpy()
    return m


def build_design_and_neurons(beh, locs_df, buttons_df, neurons, configs,
                             average_repeats=True, models_to_build=None):
    """Build per-subject regressors and neuron traces.

    Returns
    -------
    X_models   : dict[model_name] -> (P_m, T_total) ndarray
    Y          : dict[neuron_label] -> (T_total,) ndarray
    cfg_slices : list[slice]   one per config (in `configs` order)
    btn_alphabet : list[str]   button alphabet used for one-hot
    """
    models_to_build = list(models_to_build or [])

    # 1. Correct-trial row indices per config.
    cfg_indices = {}
    for c in configs:
        mask = (beh['config_str'] == c) & (beh['correct'] == 1)
        cfg_indices[c] = beh.index[mask].to_numpy()

    # 2. Button alphabet from all correct-trial buttons of this subject.
    btn_chunks = []
    for c in configs:
        if len(cfg_indices[c]):
            btn_chunks.append(buttons_df.iloc[cfg_indices[c]].to_numpy().ravel())
    if btn_chunks:
        all_buttons = np.concatenate(btn_chunks)
        valid_mask = pd.notna(all_buttons)
        # 'Return' is encoded separately as the 'uncover' regressor; drop it
        # from the alphabet so curr / prev / next don't carry an all-zero row.
        btn_alphabet = sorted({str(b) for b in all_buttons[valid_mask]
                               if str(b) != 'Return'})
    else:
        btn_alphabet = []

    # 3. Per-config X and Y.
    per_cfg_X = {m: [] for m in models_to_build}
    per_cfg_Y = {n_lab: [] for n_lab in neurons}
    cfg_lengths = []

    for c in configs:
        idx = cfg_indices[c]
        if len(idx) == 0:
            # No correct trials for this config — fill with empties so the
            # slice exists but is zero-length.
            cfg_lengths.append(0)
            for m in models_to_build:
                per_cfg_X[m].append(np.zeros((0, 0)))
            for n_lab in neurons:
                per_cfg_Y[n_lab].append(np.zeros(0))
            continue

        locs_arr    = locs_df.iloc[idx].to_numpy()      # (n_reps, 360)
        buttons_arr = buttons_df.iloc[idx].to_numpy()   # (n_reps, 360)

        if average_repeats:
            # Single mode trial per config
            mode_loc = mode_per_bin(locs_arr).astype(int)
            mode_btn = mode_per_bin(buttons_arr, fallback_ffill=False)
            X_c = build_single_trial_regressors(
                mode_loc, mode_btn, btn_alphabet, models_to_build,
            )
            for m in models_to_build:
                per_cfg_X[m].append(X_c[m])
            for n_lab, df in neurons.items():
                neuron_arr = df.iloc[idx].to_numpy()        # (n_reps, 360)
                per_cfg_Y[n_lab].append(np.nanmean(neuron_arr, axis=0))
            cfg_lengths.append(N_BINS_PER_TRIAL)

        else:
            # Per-repeat regressors and neurons, then flatten along time.
            X_per_rep = []
            for rep_i in range(locs_arr.shape[0]):
                loc_r = locs_arr[rep_i].astype(int)
                btn_r = buttons_arr[rep_i]
                X_per_rep.append(build_single_trial_regressors(
                    loc_r, btn_r, btn_alphabet, models_to_build,
                ))
            for m in models_to_build:
                per_cfg_X[m].append(
                    np.concatenate([Xr[m] for Xr in X_per_rep], axis=1)
                )
            for n_lab, df in neurons.items():
                neuron_arr = df.iloc[idx].to_numpy()
                per_cfg_Y[n_lab].append(neuron_arr.reshape(-1))
            cfg_lengths.append(N_BINS_PER_TRIAL * locs_arr.shape[0])

    # 4. Concatenate per-config blocks along time axis.
    X_models = {m: np.concatenate(per_cfg_X[m], axis=1) for m in models_to_build}
    Y = {n_lab: np.concatenate(per_cfg_Y[n_lab], axis=0) for n_lab in neurons}

    # 5. Slices that point at each config's bins in the concatenated axis.
    cfg_slices = []
    offset = 0
    for L in cfg_lengths:
        cfg_slices.append(slice(offset, offset + L))
        offset += L

    return X_models, Y, cfg_slices, btn_alphabet


def fit_encoding_cv(X, y, cfg_slices, alpha=ALPHA, l1_ratio=L1_RATIO,
                    positive=POSITIVE, max_iter=MAX_ITER):
    """Leave-one-config-out CV on a single (X, y) pair.

    X : (P, T) regressor matrix
    y : (T,) neuron trace
    cfg_slices : list of slices into the T axis (must equal the config order)

    Returns dict:
        mean_r            : float
        r_per_fold        : (n_folds,) ndarray
        coefs             : list of (P,) ndarrays
        y_pred_per_fold   : list of (n_test,) ndarrays — held-out predictions
        y_test_per_fold   : list of (n_test,) ndarrays — held-out actuals
    """
    n_folds = len(cfg_slices)
    rs = np.full(n_folds, np.nan, dtype=float)
    coefs, y_preds, y_tests = [], [], []
    T = X.shape[1]

    for fold_idx, test_slice in enumerate(cfg_slices):
        # Skip degenerate (zero-length) folds.
        if (test_slice.stop - test_slice.start) <= 1:
            coefs.append(np.zeros(X.shape[0]))
            y_preds.append(np.zeros(0))
            y_tests.append(np.zeros(0))
            continue
        mask = np.ones(T, dtype=bool)
        mask[test_slice] = False
        X_train = X[:, mask].T            # (n_train, P)
        y_train = y[mask]
        X_test  = X[:, test_slice].T      # (n_test,  P)
        y_test  = y[test_slice]

        if np.nanstd(y_train) < 1e-12:
            coefs.append(np.zeros(X.shape[0]))
            y_preds.append(np.zeros(y_test.shape[0]))
            y_tests.append(y_test)
            continue

        reg = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            positive=positive, max_iter=max_iter,
            tol=1e-3, precompute=True,
        )
        if np.isnan(y_train).any():
            keep = ~np.isnan(y_train)
            reg.fit(X_train[keep], y_train[keep])
        else:
            reg.fit(X_train, y_train)

        coefs.append(reg.coef_.copy())
        
        y_pred = X_test @ reg.coef_
        y_preds.append(y_pred)
        y_tests.append(y_test)
        import pdb; pdb.set_trace()
        rs[fold_idx] = nan_safe_pearsonr(y_test, y_pred)

    return {
        'mean_r':           float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan,
        'r_per_fold':       rs,
        'coefs':            coefs,
        'y_pred_per_fold':  y_preds,
        'y_test_per_fold':  y_tests,
    }


def analyse_one_neuron(neuron_label, roi_name, y, X_models, cfg_slices,
                       n_permutations, seed, save_diagnostics=False):
    """Run all models for one neuron.

    Permutation null: fit ElasticNet once on un-shifted data per fold, then
    for each permutation circular-shift the held-out test trace (independent
    per fold) and recompute Pearson(y_pred_fold, y_test_shifted_fold).
    Permutation statistic = mean across folds. One-sided p-value:
      p_perm = (#(perm_mean >= empirical_mean) + 1) / (n_valid_perms + 1)

    Returns
    -------
    rows         : list of dicts (one per model)
    diagnostics  : dict[model] -> per-model diagnostic data, or None
    """
    rng = np.random.default_rng(seed)
    rows = []
    diagnostics = {} if save_diagnostics else None

    for m, X in X_models.items():
        emp = fit_encoding_cv(X, y, cfg_slices)
        emp_r = emp['mean_r']

        # Permutations: test-trace shuffle only.
        if n_permutations > 0:
            perm_rs_per_fold = np.full((n_permutations, len(cfg_slices)), np.nan)
            for fold_idx in range(len(cfg_slices)):
                y_pred = emp['y_pred_per_fold'][fold_idx]
                y_test = emp['y_test_per_fold'][fold_idx]
                if y_test.size <= 1 or y_pred.size == 0:
                    continue
                if np.std(y_pred) < 1e-12:
                    continue
                Y_shifted = make_circular_shifts(y_test, n_permutations, rng)
                perm_rs_per_fold[:, fold_idx] = vectorized_pearsonr(Y_shifted, y_pred)
            perm_rs = np.nanmean(perm_rs_per_fold, axis=1)
            valid = np.isfinite(perm_rs)
            if valid.any() and np.isfinite(emp_r):
                p_perm = (np.sum(perm_rs[valid] >= emp_r) + 1) / (valid.sum() + 1)
            else:
                p_perm = np.nan
        else:
            perm_rs = np.zeros(0)
            p_perm = np.nan

        coef_flat = (np.concatenate([np.asarray(c) for c in emp['coefs']])
                     if emp['coefs'] else np.array([]))
        all_coefs_zero = coef_flat.size > 0 and bool(np.all(coef_flat == 0))

        rows.append({
            'neuron':          neuron_label,
            'roi':             roi_name,
            'model':           m,
            'mean_r':          emp_r,
            'r_per_fold':      emp['r_per_fold'].tolist(),
            'p_perm':          float(p_perm) if np.isfinite(p_perm) else np.nan,
            'n_permutations':  int(n_permutations),
            'all_coefs_zero':  all_coefs_zero,
        })

        if save_diagnostics:
            diagnostics[m] = {
                'neuron':           neuron_label,
                'roi':              roi_name,
                'model':            m,
                'mean_r':           emp_r,
                'r_per_fold':       emp['r_per_fold'].tolist(),
                'p_perm':           p_perm,
                'y_pred_per_fold':  [yp.tolist() for yp in emp['y_pred_per_fold']],
                'y_test_per_fold':  [yt.tolist() for yt in emp['y_test_per_fold']],
                'perm_rs':          perm_rs.tolist(),
                'coefs':            [c.tolist() for c in emp['coefs']],
            }

    return rows, diagnostics


if RELOAD_RUN is None:
    # ── Cache subject data ───────────────────────────────────────────────
    print("Loading subject data...")
    SUBJECT_DATA = {}
    for sub_str in SUBJECTS:
        SUBJECT_DATA[sub_str] = mc.analyse.helpers_human_cells.load_norm_data(
            DATA_DIR, [sub_str],
        )
    print(f"Cached data for {len(SUBJECT_DATA)} subjects.")


    # ── Main loop ────────────────────────────────────────────────────────
    all_rows = []
    diagnostics_all = {}   # {sub_str: {neuron_label: {model: diag_dict}}}
    target_set = set(TARGET_ROIS) if TARGET_ROIS is not None else None
    subjects_processed = 0
    T_overall_start = time.time()

    for sub_str in SUBJECTS:
        if MAX_SUBJECTS is not None and subjects_processed >= MAX_SUBJECTS:
            break

        print(f"\n========== Subject sub-{sub_str} ==========")
        sub_t0 = time.time()
        sub_dict = SUBJECT_DATA[sub_str][f"sub-{sub_str}"]

        beh = sub_dict['beh'].copy().reset_index(drop=True)
        beh['config_str'] = (
            beh['loc_A'].astype(int).astype(str) + '-' +
            beh['loc_B'].astype(int).astype(str) + '-' +
            beh['loc_C'].astype(int).astype(str) + '-' +
            beh['loc_D'].astype(int).astype(str)
        )
        # Configs determined from this subject's beh file (sorted for consistency).
        configs_sub = sorted(beh['config_str'].dropna().unique().tolist())
        locs_df    = sub_dict['locations'].reset_index(drop=True)
        buttons_df = sub_dict['buttons'].reset_index(drop=True)
        neurons    = sub_dict['normalised_neurons']

        # Filter neurons by target ROI(s).
        def neuron_keep(lbl):
            roi = get_neuron_roi(lbl)
            if roi is None:
                return False
            if target_set is None:
                return True
            return roi in target_set

        neuron_labels = [n_lab for n_lab in neurons if neuron_keep(n_lab)]
        if MAX_NEURONS_PER_SUBJECT is not None: #for diagnostics test
            neuron_labels = neuron_labels[:MAX_NEURONS_PER_SUBJECT]
        if not neuron_labels:
            print(f"  no neurons in target ROIs — skipping subject.")
            continue
        neurons_used = {n_lab: neurons[n_lab].reset_index(drop=True)
                        for n_lab in neuron_labels}
        rois_used = {n_lab: get_neuron_roi(n_lab) for n_lab in neurons_used}
        print(f"  {len(neurons_used)} neurons in target ROIs "
              f"({set(rois_used.values())}).")

        # Build design and neuron traces.
        t_build = time.time()
        X_models, Y, cfg_slices, btn_alphabet = build_design_and_neurons(
            beh, locs_df, buttons_df, neurons_used, configs_sub,
            average_repeats=AVERAGE_REPEATS, models_to_build=models,
        )
        build_dt = time.time() - t_build

        # Sanity checks.
        T_total = next(iter(X_models.values())).shape[1]
        for m, X in X_models.items():
            assert X.shape[1] == T_total, f"{m}: T={X.shape[1]} vs {T_total}"
        for n_lab, y in Y.items():
            assert y.shape[0] == T_total, f"{n_lab}: T={y.shape[0]} vs {T_total}"
        print(f"  T_total = {T_total} bins   "
              f"cfg_lengths = {[s.stop - s.start for s in cfg_slices]}")
        print(f"  feature counts: "
              f"{ {m: X.shape[0] for m, X in X_models.items()} }")
        print(f"  button alphabet: {btn_alphabet}")
        print(f"  design build: {build_dt:.2f}s")
        print(f"  fitting {len(neurons_used)} neurons × {len(models)} models "
              f"× {N_PERMUTATIONS} perms ...")

        # Parallel over neurons. With test-shuffle perms, single-neuron subjects
        # are fast enough; joblib still helps when there are many neurons.
        t_fit = time.time()
        neuron_args = [
            (n_lab, rois_used[n_lab], Y[n_lab])
            for n_lab in neurons_used
        ]
         
        n_jobs_effective = 1 if len(neuron_args) == 1 else N_JOBS
        for n_lab, roi, y in neuron_args:
            results = analyse_one_neuron(n_lab, roi, y, X_models, cfg_slices,
                                         n_permutations=N_PERMUTATIONS,seed=abs(hash((sub_str, n_lab)))& 0xFFFFFFFF, save_diagnostics=SAVE_DIAGNOSTICS)

        # results = Parallel(n_jobs=n_jobs_effective, verbose=0)(
        #     delayed(analyse_one_neuron)(
        #         n_lab, roi, y, X_models, cfg_slices,
        #         n_permutations=N_PERMUTATIONS,
        #         seed=abs(hash((sub_str, n_lab))) & 0xFFFFFFFF,
        #         save_diagnostics=SAVE_DIAGNOSTICS,
        #     )
        #     for n_lab, roi, y in neuron_args
        # )
        fit_dt = time.time() - t_fit

        sub_diag = {}
        for (rows, diag), (n_lab, _, _) in zip(results, neuron_args):
            mni_x, mni_y, mni_z = get_neuron_mni(n_lab)
            for row in rows:
                row['subject'] = sub_str
                row['MNI_x'] = mni_x
                row['MNI_y'] = mni_y
                row['MNI_z'] = mni_z
                if row.get('all_coefs_zero'):
                    print(f"  WARNING: all coefficients zero — "
                          f"{n_lab} / model={row['model']}")
            all_rows.extend(rows)
            if diag is not None:
                # Tag each per-model diagnostic with this subject's config list.
                for m_diag in diag.values():
                    m_diag['configs'] = configs_sub
                sub_diag[n_lab] = diag
        if SAVE_DIAGNOSTICS and sub_diag:
            diagnostics_all[sub_str] = sub_diag

        # Per-neuron DSR-family coefficient maps (perm-sig neurons only).
        if PLOT_DSR_COEF_MAPS and sub_diag:
            from mc.plotting.cell_results import plot_dsr_coef_matrix
            dsr_plot_dir = os.path.join(OUT_DIR, 'dsr_coef_maps')
            os.makedirs(dsr_plot_dir, exist_ok=True)
            n_maps = 0
            for n_lab, diag in sub_diag.items():
                for m in DSR_COEF_MAP_FAMILY:
                    if m not in diag or m not in X_models:
                        continue
                    d = diag[m]
                    p_perm = d.get('p_perm', np.nan)
                    if not np.isfinite(p_perm) or p_perm >= DSR_COEF_MAP_ALPHA:
                        continue
                    r_per_fold = np.asarray(d['r_per_fold'], dtype=float)
                    # import pdb; pdb.set_trace()
                    if not np.isfinite(r_per_fold).any():
                        continue
                    best_fold  = int(np.nanargmax(r_per_fold))
                    coefs      = np.asarray(d['coefs'][best_fold], dtype=float)
                    if not np.any(coefs):
                        continue
                    test_slice = cfg_slices[best_fold]
                    X_slice    = X_models[m][:, test_slice]
                    save_path  = os.path.join(
                        dsr_plot_dir,
                        f'sub-{sub_str}_{n_lab}_{m}.png',
                    )
                    plot_dsr_coef_matrix(
                        X_slice, coefs, n_lab, m, save_path,
                        n_phases=N_PHASES,
                        n_clocks_per_phase=N_SUBPATHS_PER_TRIAL,
                        fold_r=float(r_per_fold[best_fold]),
                        p_perm=float(p_perm),
                    )
                    n_maps += 1
            if n_maps:
                print(f"  saved {n_maps} DSR-family coefficient maps "
                      f"→ {dsr_plot_dir}")

        n_fits = len(neurons_used) * len(models)
        per_fit_ms = (fit_dt / max(n_fits, 1)) * 1000
        print(f"  fitting done in {fit_dt:.2f}s  "
              f"({n_fits} (neuron × model) | "
              f"{per_fit_ms:.1f} ms per (neuron × model))")
        print(f"  subject total: {time.time() - sub_t0:.2f}s")
        subjects_processed += 1

    print(f"\n=== All subjects done in {time.time() - T_overall_start:.1f}s ===")


    # ── Save results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, 'encoding_results.csv')
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-(subject, neuron, model) results → {out_csv}")

    # Save diagnostics pickle (test-mode or when SAVE_DIAGNOSTICS=True).
    if SAVE_DIAGNOSTICS and diagnostics_all:
        diag_pkl = os.path.join(OUT_DIR, 'diagnostics.pkl')
        with open(diag_pkl, 'wb') as f:
            pickle.dump(diagnostics_all, f)
        print(f"Saved per-(neuron, model) diagnostics → {diag_pkl}")
else:
    # Reload mode: pull saved encoding results (and diagnostics if any)
    # from disk so the plotting block can run without redoing fits.
    out_csv = os.path.join(OUT_DIR, 'encoding_results.csv')
    results_df = pd.read_csv(out_csv)
    print(f"Loaded {len(results_df)} rows from {out_csv}")
    diag_pkl = os.path.join(OUT_DIR, 'diagnostics.pkl')
    if os.path.exists(diag_pkl):
        with open(diag_pkl, 'rb') as f:
            diagnostics_all = pickle.load(f)
        print(f"Loaded diagnostics for {len(diagnostics_all)} subjects "
              f"from {diag_pkl}")
    else:
        diagnostics_all = {}
        print(f"No diagnostics.pkl in {OUT_DIR}; best-neuron showcase will be skipped.")


from mc.plotting.cell_results import (
    plot_neuron_fit,
    plot_perm_histogram,
    plot_best_neuron_per_roi_model,
    plot_r_distribution_grid,
    plot_significance_proportion,
    plot_significant_cells_glassbrain,
    plot_significant_cells_mesh3d,
)

if PLOT_DIAGNOSTICS and diagnostics_all:
    diag_plot_dir = os.path.join(OUT_DIR, 'diagnostic_plots')
    os.makedirs(diag_plot_dir, exist_ok=True)
    n_plots = 0
    for sub_str, per_neuron in diagnostics_all.items():
        for neuron_label, per_model in per_neuron.items():
            for model_name, diag in per_model.items():
                plot_neuron_fit(diag, save_dir=diag_plot_dir)
                plot_perm_histogram(diag, save_dir=diag_plot_dir)
                n_plots += 1
    plt.show()
    print(f"Saved {n_plots} × 2 diagnostic plots → {diag_plot_dir}")

# Per (ROI, model) summary.
if not results_df.empty:
    summary = (
        results_df.groupby(['roi', 'model'])
        .agg(n_neurons=('mean_r', 'size'),
             mean_r=('mean_r', 'mean'),
             sd_r=('mean_r', 'std'),
             median_r=('mean_r', 'median'),
             prop_sig=('p_perm', lambda s: (s < 0.05).mean()))
        .reset_index()
    )
    out_summary_csv = os.path.join(OUT_DIR, 'encoding_summary_roi_model.csv')
    summary.to_csv(out_summary_csv, index=False)
    print("\n=== Per (ROI, model) summary ===")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 140):
        print(summary.to_string(index=False))
    print(f"\nSaved summary → {out_summary_csv}")
else:
    print("\nNo results to summarise.")


# ── Aggregate result plots ──────────────────────────────────────────
if not results_df.empty:
    summary_plot_dir = os.path.join(OUT_DIR, 'summary_plots')
    os.makedirs(summary_plot_dir, exist_ok=True)

    if diagnostics_all:
        plot_best_neuron_per_roi_model(
            diagnostics_all, results_df,
            save_dir=os.path.join(summary_plot_dir, 'best_neuron'),
        )
    else:
        print("Skipping best-neuron showcase: no diagnostics saved.")

    plot_r_distribution_grid(
        results_df, models,
        save_path=os.path.join(summary_plot_dir, 'r_distribution_grid.png'),
        reg_alpha=ALPHA,
    )
    plot_significance_proportion(
        results_df, models,
        save_path=os.path.join(summary_plot_dir, 'significant_proportion.png'),
        reg_alpha=ALPHA,
    )
    print(f"Saved aggregate result plots → {summary_plot_dir}")

    # ── Glass-brain plots of significant cells per model ────────────
    glass_dir = os.path.join(summary_plot_dir, 'significant_glassbrain')
    os.makedirs(glass_dir, exist_ok=True)
    # results_df already carries MNI_x/y/z and the new `roi` from the table.
    coord_ok = results_df[['MNI_x', 'MNI_y', 'MNI_z']].notna().all(axis=1)
    plot_df = results_df[coord_ok].copy().rename(columns={'roi': 'final_roi'})
    if plot_df.empty:
        print("No cells with MNI coordinates — skipping glass-brain plots.")
    else:
        for m in models:
            sub_df = plot_df[plot_df['model'] == m]
            if sub_df.empty:
                continue
            plot_significant_cells_glassbrain(
                sub_df, model_name=m,
                save_path=os.path.join(glass_dir, f'glassbrain_{m}.png'),
                p_threshold=GLASSBRAIN_P_THRESHOLD,
                title_suffix=f'reg alpha={ALPHA}',
            )
            if PLOT_GLASSBRAIN_3D:
                plot_significant_cells_mesh3d(
                    sub_df, model_name=m,
                    save_path=os.path.join(glass_dir, f'mesh3d_{m}.html'),
                    p_threshold=GLASSBRAIN_P_THRESHOLD,
                    title_suffix=f'reg alpha={ALPHA}',
                )
        print(f"Saved glass-brain plots → {glass_dir}")

    if PLOT_DIAGNOSTICS:
        plt.show()
