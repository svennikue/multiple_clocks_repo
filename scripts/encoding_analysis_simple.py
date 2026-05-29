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

# import pdb; pdb.set_trace()
# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE = os.path.join(DATA_DIR, 'group', 'encoding_analysis_simple')

# Reload mode: set to the run tag of a previous run (e.g.
# '2026-05-18_16-33-05') to skip the heavy elastic-net + permutation loop
# and just re-render the summary plots from the saved
# encoding_results.csv (+ diagnostics.pkl, if present).  None = fresh run.
RELOAD_RUN = '2026-05-28_16-45-09-nopenality-newROIs' # '2026-05-20_22-57-03-GREATONE/' # None # '2026-05-18_16-33-05' # None

# Subset re-plot: take an existing run's cell-wise results, restrict them
# to a subset of subjects, and regenerate every result file + plot into a
# new folder (so the follow-up scripts can point straight at it). Requires
# RELOAD_RUN to name the source run; nothing is re-fitted.
#   None                                  -> off
#   'none_RSA'                            -> drop the DSR-RSA-grouping
#                                            subjects (folder tag
#                                            'none-RSA-subset')
#   {'tag': 'name', 'exclude': ['05',..]} -> drop an explicit subject list
#   {'tag': 'name', 'include': ['01',..]} -> keep only an explicit list
SUBSET_REPLOT = 'none_RSA' #None # 'none_RSA' # None

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
POSITIVE = False
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
     'dsr', 'dsr_only_fut', 'dsr_now_next', 'location', 'midnight',
     'phase', 'state', 'state_phase',
    'bttn_prev', 'bttn_next', 'bttn_curr', 'uncover',
]

# ── Combination-model analysis settings ───────────────────────────────
# Which analyses to run:
#   'singles_only' -> original per-base-model loop only (legacy behaviour).
#   'combos_only'  -> new combination + Δ-comparison loop only.
#   'all'          -> both.
COMBO_ANALYSIS_MODE = 'singles_only'

# Combination models: name -> list of base regressors to concatenate.
# Each base name must be buildable by `build_single_trial_regressors`.
COMBO_MODELS = {
    'M_dsr':                ['dsr'],
    'M_state':              ['state'],
    'M_midnight':           ['midnight'],
    'M_state_dsr':          ['state', 'dsr'],
    'M_location_dsr':       ['midnight', 'dsr'],
    'M_state_midnight':     ['state', 'midnight'],
    'M_state_midnight_dsr': ['state', 'midnight', 'dsr'],
    'M_dsr_nownext': ['dsr_now_next'] ,
    'M_dsr_fut': ['dsr_only_fut'],
    # RSA-style control set: state + location-phase + motor.
    # Mirrors the regressor list in our RSA `MRI_combo-nofdb_midn-state`
    # heatmap so the cell-level Δ-tests answer the same conditional
    # question (DSR unique contribution over the same confound set).
    'M_full_controls':           ['state', 'midnight', 'bttn_curr', 'bttn_next'],
    'M_full_controls_dsr':       ['state', 'midnight', 'bttn_curr', 'bttn_next', 'dsr'],
    'M_full_controls_dsr_fut':   ['state', 'midnight', 'bttn_curr', 'bttn_next', 'dsr_only_fut'],
}

# Paired held-out comparisons.  Δ = score(test) - score(baseline),
# computed fold-wise on the SAME CV splits; one-sided H1: Δ > 0.
COMBO_COMPARISONS = [
    # Q1: does another single model beat M_dsr?
    ('M_state',              'M_dsr'),
    ('M_midnight',           'M_dsr'),
    # Q2: does DSR add unique held-out prediction over the other models?
    ('M_state_dsr',          'M_state'),
    ('M_midnight_dsr',       'M_midnight'),
    ('M_state_midnight_dsr', 'M_state_midnight'),
    # Q3 (RSA-style): with the same control set as the RSA, does the
    # DSR design (and its future-lag-only subset) add unique held-out
    # predictive power?
    #   - full DSR added on top of state + midnight + motor controls
    ('M_full_controls_dsr',     'M_full_controls'),
    #   - future-lag-only DSR (clo - clo_now_next) on top of same controls
    #     answers: "does ACC predict future locations beyond simple
    #     location/phase/motor/state coding?"
    ('M_full_controls_dsr_fut', 'M_full_controls'),
    # Q4 (Visual vs ACC dissociation): how much of DSR's contribution is
    # the lag-0 / now-next part vs the future-lag part?  In Visual the
    # answer should be "almost all the contribution is lag-0/now-next"
    # (current-location coding). In ACC the future-lag part should
    # survive on its own (Q3 above).
    ('M_full_controls_dsr',     'M_full_controls_dsr_fut'),
    # Q 5: are there more future lag or more current lag cells?
    ('M_dsr_nownext',     'M_dsr_fut'),
]

# ROI-level alpha for the binomial + one-sample tests on Δ.
COMBO_ROI_ALPHA = 0.05

# Resolved at module level so the per-subject loop knows what regressors
# to build and which analyses to dispatch.
_combo_base_models = sorted({m for parts in COMBO_MODELS.values() for m in parts})
run_singles = COMBO_ANALYSIS_MODE in ('singles_only', 'all')
run_combos  = COMBO_ANALYSIS_MODE in ('combos_only', 'all')
_models_to_build = sorted(
    (set(models) if run_singles else set())
    | (set(_combo_base_models) if run_combos else set())
)
print(f"COMBO_ANALYSIS_MODE = {COMBO_ANALYSIS_MODE}  "
      f"(run_singles={run_singles}, run_combos={run_combos})")
print(f"  base regressors to build: {_models_to_build}")
if run_combos:
    print(f"  combo models:   {list(COMBO_MODELS)}")
    print(f"  Δ comparisons:  {COMBO_COMPARISONS}")


# Models that get the full publication treatment (FDR table, histogram,
# top-N cell fits). The heatmap + glass-brain still cover every model.
PUB_MODELS = ['dsr', 'dsr_only_fut', 'state']


# ROI labels are taken from the MNI-coordinate-based table produced by
# scripts/cell_to_roi_MNI.py.  Rows are matched to neuron labels via
# (subject, cell_idx) parsed from the label `{sub:02d}_{cell_idx:02d}-...`.
ROI_TABLE_PATH = os.path.join(
    DATA_DIR, 'neurons_with_final_roi_labels.csv'
)

# Which ROI-label column of that table to use:
#   'final_roi'     -> original labelling
#   'alt_final_roi' -> alternative labelling (ACC split by y-cutoff,
#                      OFC11+OFC13+ventral_ACC collapsed into medialOFC)
ROI_LABEL_COLUMN = 'alt_final_roi' #'final_roi'

# Only analyse neurons whose ROI is in this list. None = all ROIs
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
DSR_COEF_MAP_FAMILY   = ('dsr','dsr_only_fut')   # trailing comma is required for 1-tuple


# Per-(neuron, model) diagnostics (y_pred, y_test, perm_rs) are needed for the
# best-neuron showcase plot, so we always save them.
SAVE_DIAGNOSTICS = True


# ── Subset-replot resolution ──────────────────────────────────────────
def _norm_sub(s):
    """Normalise a subject id ('01', '1', 1) to an int for comparison."""
    return int(str(s).strip())


def _normalise_subject_col(df):
    """CSV round-trips parse the zero-padded `subject` strings ('01') as
    ints; restore the '01'-style strings so they match the diagnostics
    dict keys (and a fresh run's in-memory tables)."""
    if df is not None and not df.empty and 'subject' in df.columns:
        df = df.copy()
        df['subject'] = df['subject'].map(lambda s: f'{int(s):02d}')
    return df


def resolve_subset(spec):
    """Resolve a SUBSET_REPLOT spec to (tag, keep_predicate).

    `keep_predicate(subject)` returns True if that subject should be kept.
    """
    if spec in ('none_RSA', 'none-RSA-subset'):
        with open(os.path.join(
                DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json')) as f:
            rsa = {_norm_sub(k) for k in json.load(f).keys()}
        print(f"  subset 'none-RSA-subset': excluding {len(rsa)} "
              f"DSR-RSA subjects.")
        return 'none-RSA-subset', (lambda s: _norm_sub(s) not in rsa)
    if isinstance(spec, dict):
        tag = str(spec.get('tag', 'subset'))
        if 'exclude' in spec:
            excl = {_norm_sub(x) for x in spec['exclude']}
            return tag, (lambda s: _norm_sub(s) not in excl)
        if 'include' in spec:
            incl = {_norm_sub(x) for x in spec['include']}
            return tag, (lambda s: _norm_sub(s) in incl)
    raise ValueError(f"Unknown SUBSET_REPLOT spec: {spec!r}")


# ── Per-run output folder + config dump ──────────────────────────────
# Resolve the subset spec first — it needs RELOAD_RUN as the source run.
SUBSET_TAG, _subset_keep = (None, None)
if SUBSET_REPLOT is not None:
    if RELOAD_RUN is None:
        raise ValueError("SUBSET_REPLOT requires RELOAD_RUN to name the "
                         "source run whose results should be subset.")
    SUBSET_TAG, _subset_keep = resolve_subset(SUBSET_REPLOT)

if RELOAD_RUN is not None:
    SOURCE_DIR = os.path.join(OUT_BASE, RELOAD_RUN)
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"Reload folder not found: {SOURCE_DIR}")
    if SUBSET_TAG is not None:
        RUN_TAG = f'{RELOAD_RUN}-{SUBSET_TAG}'
        OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"SUBSET RE-PLOT — source : {SOURCE_DIR}")
        print(f"                 output: {OUT_DIR}")
    else:
        RUN_TAG = RELOAD_RUN
        OUT_DIR = SOURCE_DIR
        print(f"RELOAD MODE — reading results from: {OUT_DIR}")
else:
    SOURCE_DIR = None
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
    'combo_analysis_mode': COMBO_ANALYSIS_MODE,
    'combo_models':     COMBO_MODELS,
    'combo_comparisons': COMBO_COMPARISONS,
    'combo_roi_alpha':  COMBO_ROI_ALPHA,
    'roi_table_path':   ROI_TABLE_PATH,
    'roi_label_column': ROI_LABEL_COLUMN,
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


def _load_roi_table(path, roi_col):
    """Load the MNI-based ROI table and index it by (subject, cell idx)."""
    df = pd.read_csv(path)
    needed = ['subject', 'cell idx', roi_col,
              'MNI_x', 'MNI_y', 'MNI_z', 'electrode label']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"ROI table {path} is missing columns: {missing}  "
            f"(re-run scripts/cell_to_roi_MNI.py if {roi_col!r} is absent)"
        )
    df = df.copy()
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


ROI_TABLE = _load_roi_table(ROI_TABLE_PATH, ROI_LABEL_COLUMN)
print(f"Loaded ROI table with {len(ROI_TABLE)} cells "
      f"({ROI_TABLE[ROI_LABEL_COLUMN].nunique()} distinct ROIs) "
      f"from {ROI_TABLE_PATH}  [column: {ROI_LABEL_COLUMN}]")


def get_neuron_roi(label):
    """Look up the MNI-based ROI for a neuron label, or None if missing."""
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return None
    try:
        roi = ROI_TABLE.loc[(sub, cell_idx), ROI_LABEL_COLUMN]
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
        # import pdb; pdb.set_trace()
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
            # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
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


# ── Combination-model helpers ─────────────────────────────────────────
def build_combo_X(X_models, base_names):
    """Concatenate base-model regressors into one combo design matrix.

    Each X_models[m] is (P_m, T); stack along feature axis (axis=0) so
    every model uses the same time axis / CV splits.
    """
    return np.concatenate([X_models[m] for m in base_names], axis=0)


def analyse_one_neuron_combo(neuron_label, roi_name, y, X_models, cfg_slices,
                             combo_specs, comparison_specs,
                             n_permutations, seed,
                             save_diagnostics=False):
    """Fit every combo + compute paired held-out Δ comparisons for ONE neuron.

    Mirrors `analyse_one_neuron` but iterates over combination models
    (concatenated base regressors).  All combos share the SAME outer CV
    splits (cfg_slices) and the SAME circular-shift schedule within a
    permutation, so the resulting Δ between two combos is a genuine
    paired contrast against the same null.

    Returns
    -------
    combo_rows : list of dicts — one per (neuron, combo).
        Fields: neuron, roi, combo_model, mean_r, r_per_fold, p_perm,
                all_coefs_zero, n_permutations.
    delta_rows : list of dicts — one per (neuron, comparison).
        Fields: neuron, roi, model_test, model_baseline, mean_delta,
                delta_per_fold, p_perm_delta, n_permutations.
    diagnostics : dict[combo_name] or None.
    """
    rng = np.random.default_rng(seed)
    n_folds = len(cfg_slices)

    # 1. Empirical fit + score per combo on identical CV folds.
    emp = {}
    for combo_name, base_names in combo_specs.items():
        X_combo = build_combo_X(X_models, base_names)
        emp[combo_name] = fit_encoding_cv(X_combo, y, cfg_slices)

    # 2. Permutation r per (combo, perm, fold). Same shift schedule across
    #    every combo within a permutation -> paired null.
    perm_r_per_fold = {name: np.full((n_permutations, n_folds), np.nan)
                       for name in combo_specs}
    if n_permutations > 0:
        ref = next(iter(emp.values()))
        for fold_idx in range(n_folds):
            y_test = ref['y_test_per_fold'][fold_idx]
            if y_test.size <= 1:
                continue
            Y_shifted = make_circular_shifts(y_test, n_permutations, rng)
            for combo_name, d in emp.items():
                y_pred = d['y_pred_per_fold'][fold_idx]
                if y_pred.size == 0 or np.std(y_pred) < 1e-12:
                    continue
                perm_r_per_fold[combo_name][:, fold_idx] = vectorized_pearsonr(
                    Y_shifted, y_pred)

    # 3. Per-combo summary row (p_perm on the combo's own held-out score).
    combo_rows = []
    for combo_name, d in emp.items():
        null = np.nanmean(perm_r_per_fold[combo_name], axis=1)
        emp_r = d['mean_r']
        valid = np.isfinite(null)
        if valid.any() and np.isfinite(emp_r):
            p = (np.sum(null[valid] >= emp_r) + 1) / (valid.sum() + 1)
        else:
            p = np.nan
        coef_flat = (np.concatenate([np.asarray(c) for c in d['coefs']])
                     if d['coefs'] else np.array([]))
        combo_rows.append({
            'neuron': neuron_label,
            'roi': roi_name,
            'combo_model': combo_name,
            'mean_r': emp_r,
            'r_per_fold': d['r_per_fold'].tolist(),
            'p_perm': float(p) if np.isfinite(p) else np.nan,
            'all_coefs_zero': coef_flat.size > 0 and bool(np.all(coef_flat == 0)),
            'n_permutations': int(n_permutations),
        })

    # 4. Paired Δ contrasts with permutation null.
    delta_rows = []
    for test_name, base_name in comparison_specs:
        if test_name not in emp or base_name not in emp:
            continue
        r_test = np.asarray(emp[test_name]['r_per_fold'], dtype=float)
        r_base = np.asarray(emp[base_name]['r_per_fold'], dtype=float)
        delta_per_fold = r_test - r_base
        mean_delta = (float(np.nanmean(delta_per_fold))
                      if np.isfinite(delta_per_fold).any() else np.nan)

        perm_delta_per_fold = (
            perm_r_per_fold[test_name] - perm_r_per_fold[base_name]
        )
        perm_delta_mean = np.nanmean(perm_delta_per_fold, axis=1)
        valid = np.isfinite(perm_delta_mean)
        if valid.any() and np.isfinite(mean_delta):
            p_delta = (np.sum(perm_delta_mean[valid] >= mean_delta) + 1) / (valid.sum() + 1)
        else:
            p_delta = np.nan

        delta_rows.append({
            'neuron': neuron_label,
            'roi': roi_name,
            'model_test': test_name,
            'model_baseline': base_name,
            'mean_delta': mean_delta,
            'delta_per_fold': delta_per_fold.tolist(),
            'p_perm_delta': float(p_delta) if np.isfinite(p_delta) else np.nan,
            'n_permutations': int(n_permutations),
        })

    # 5. Diagnostics — kept as a secondary descriptive output (coef maps).
    diagnostics = None
    if save_diagnostics:
        diagnostics = {}
        for combo_name, d in emp.items():
            diagnostics[combo_name] = {
                'neuron':           neuron_label,
                'roi':              roi_name,
                'combo_model':      combo_name,
                'mean_r':           d['mean_r'],
                'r_per_fold':       d['r_per_fold'].tolist(),
                'y_pred_per_fold':  [yp.tolist() for yp in d['y_pred_per_fold']],
                'y_test_per_fold':  [yt.tolist() for yt in d['y_test_per_fold']],
                'coefs':            [c.tolist() for c in d['coefs']],
                'perm_r_per_fold':  perm_r_per_fold[combo_name].tolist(),
            }

    return combo_rows, delta_rows, diagnostics


def compute_combo_roi_summary(delta_df, alpha=0.05):
    """ROI-level inference on per-neuron Δ values.

    For each (roi, model_test, model_baseline) group:
      - one-sided one-sample t-test of mean_delta against 0  (H1: > 0),
      - one-sided Wilcoxon signed-rank,
      - binomial test of #{p_perm_delta < alpha} against chance level alpha.
    """
    rows = []
    if delta_df is None or delta_df.empty:
        return pd.DataFrame(rows)

    for (roi, test, base), g in delta_df.groupby(
        ['roi', 'model_test', 'model_baseline'], sort=False
    ):
        deltas = g['mean_delta'].dropna().to_numpy(dtype=float)
        p_perms = g['p_perm_delta'].dropna().to_numpy(dtype=float)
        n = int(deltas.size)
        if n == 0:
            continue

        # One-sample t-test, one-sided.
        try:
            t_res = stats.ttest_1samp(deltas, 0.0, alternative='greater')
            t_stat = float(t_res.statistic)
            t_p = float(t_res.pvalue)
        except TypeError:
            # scipy < 1.6 — convert two-sided to one-sided manually.
            t_stat, p_two = stats.ttest_1samp(deltas, 0.0)
            t_stat = float(t_stat)
            t_p = float(p_two / 2 if t_stat > 0 else 1 - p_two / 2)
        except Exception:
            t_stat, t_p = np.nan, np.nan

        # Wilcoxon signed-rank, one-sided.
        try:
            wil = stats.wilcoxon(deltas, alternative='greater',
                                 zero_method='wilcox')
            wil_p = float(wil.pvalue)
        except (ValueError, TypeError):
            wil_p = np.nan

        # Binomial test on fraction of neurons with p_perm_delta < alpha.
        n_p = int(p_perms.size)
        n_sig = int(np.sum(p_perms < alpha))
        frac_sig = (n_sig / n_p) if n_p > 0 else np.nan
        try:
            bin_p = float(stats.binomtest(
                n_sig, n_p, p=alpha, alternative='greater').pvalue)
        except AttributeError:
            try:
                bin_p = float(stats.binom_test(
                    n_sig, n_p, p=alpha, alternative='greater'))
            except Exception:
                bin_p = np.nan

        rows.append({
            'roi':                roi,
            'model_test':         test,
            'model_baseline':     base,
            'n_neurons':          n,
            'mean_delta':         float(np.mean(deltas)),
            'median_delta':       float(np.median(deltas)),
            't_stat':             t_stat,
            't_p_greater':        t_p,
            'wilcoxon_p_greater': wil_p,
            'n_sig':              n_sig,
            'n_with_p_perm':      n_p,
            'frac_sig':           float(frac_sig) if np.isfinite(frac_sig) else np.nan,
            'binomial_p_greater': bin_p,
            'alpha':              float(alpha),
        })
    return pd.DataFrame(rows)


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
    # Combination-analysis accumulators.
    all_combo_rows = []
    all_delta_rows = []
    combo_diagnostics_all = {}  # {sub_str: {neuron_label: {combo_name: diag}}}
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

        # Build design and neuron traces.  Build the UNION of base regressors
        # needed by the active analysis modes; each analysis only references
        # the subset it cares about.
        t_build = time.time()
        X_models, Y, cfg_slices, btn_alphabet = build_design_and_neurons(
            beh, locs_df, buttons_df, neurons_used, configs_sub,
            average_repeats=AVERAGE_REPEATS, models_to_build=_models_to_build,
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

        neuron_args = [
            (n_lab, rois_used[n_lab], Y[n_lab])
            for n_lab in neurons_used
        ]
        n_jobs_effective = 1 if len(neuron_args) == 1 else N_JOBS
        fit_dt = 0.0
        sub_diag = {}

        # ── Base-model loop (legacy / 'singles_only' / 'all') ────────────
        if run_singles:
            # Only score the regressors actually requested in `models`.
            X_models_singles = {m: X_models[m] for m in models if m in X_models}
            print(f"  fitting {len(neurons_used)} neurons × "
                  f"{len(X_models_singles)} single models × "
                  f"{N_PERMUTATIONS} perms ...")
            t_fit = time.time()
            results = Parallel(n_jobs=n_jobs_effective, verbose=0)(
                delayed(analyse_one_neuron)(
                    n_lab, roi, y, X_models_singles, cfg_slices,
                    n_permutations=N_PERMUTATIONS,
                    seed=abs(hash((sub_str, n_lab))) & 0xFFFFFFFF,
                    save_diagnostics=SAVE_DIAGNOSTICS,
                )
                for n_lab, roi, y in neuron_args
            )
            fit_dt += time.time() - t_fit

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
                    for m_diag in diag.values():
                        m_diag['configs'] = configs_sub
                    sub_diag[n_lab] = diag
            if SAVE_DIAGNOSTICS and sub_diag:
                diagnostics_all[sub_str] = sub_diag

        # ── Combo-model loop (new) ──────────────────────────────────────
        if run_combos:
            print(f"  fitting {len(neurons_used)} neurons × "
                  f"{len(COMBO_MODELS)} combos (Δ comparisons "
                  f"{len(COMBO_COMPARISONS)}) × {N_PERMUTATIONS} perms ...")
            t_fit = time.time()
            combo_results = Parallel(n_jobs=n_jobs_effective, verbose=0)(
                delayed(analyse_one_neuron_combo)(
                    n_lab, roi, y, X_models, cfg_slices,
                    COMBO_MODELS, COMBO_COMPARISONS,
                    n_permutations=N_PERMUTATIONS,
                    seed=abs(hash((sub_str, n_lab, 'combo'))) & 0xFFFFFFFF,
                    save_diagnostics=SAVE_DIAGNOSTICS,
                )
                for n_lab, roi, y in neuron_args
            )
            fit_dt += time.time() - t_fit

            sub_combo_diag = {}
            for (combo_rows, delta_rows, diag), (n_lab, _, _) in zip(
                combo_results, neuron_args
            ):
                mni_x, mni_y, mni_z = get_neuron_mni(n_lab)
                for row in combo_rows:
                    row['subject'] = sub_str
                    row['MNI_x'] = mni_x
                    row['MNI_y'] = mni_y
                    row['MNI_z'] = mni_z
                    if row.get('all_coefs_zero'):
                        print(f"  WARNING: all combo coefficients zero — "
                              f"{n_lab} / combo={row['combo_model']}")
                for row in delta_rows:
                    row['subject'] = sub_str
                    row['MNI_x'] = mni_x
                    row['MNI_y'] = mni_y
                    row['MNI_z'] = mni_z
                all_combo_rows.extend(combo_rows)
                all_delta_rows.extend(delta_rows)
                if diag is not None:
                    for combo_diag in diag.values():
                        combo_diag['configs'] = configs_sub
                    sub_combo_diag[n_lab] = diag
            if SAVE_DIAGNOSTICS and sub_combo_diag:
                combo_diagnostics_all[sub_str] = sub_combo_diag

        # Per-neuron DSR-family coefficient maps (perm-sig neurons only).
        # Only meaningful for the single-model branch.
        if PLOT_DSR_COEF_MAPS and run_singles and sub_diag:
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

        n_singles_fits = len(neurons_used) * (len(models) if run_singles else 0)
        n_combo_fits   = len(neurons_used) * (len(COMBO_MODELS) if run_combos else 0)
        n_fits = n_singles_fits + n_combo_fits
        per_fit_ms = (fit_dt / max(n_fits, 1)) * 1000
        print(f"  fitting done in {fit_dt:.2f}s  "
              f"(singles={n_singles_fits}, combos={n_combo_fits} | "
              f"{per_fit_ms:.1f} ms per fit)")
        print(f"  subject total: {time.time() - sub_t0:.2f}s")
        subjects_processed += 1

    print(f"\n=== All subjects done in {time.time() - T_overall_start:.1f}s ===")


    # ── Save results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, 'encoding_results.csv')
    if run_singles:
        results_df.to_csv(out_csv, index=False)
        print(f"\nSaved per-(subject, neuron, model) results → {out_csv}")

    # Save diagnostics pickle (test-mode or when SAVE_DIAGNOSTICS=True).
    if run_singles and SAVE_DIAGNOSTICS and diagnostics_all:
        diag_pkl = os.path.join(OUT_DIR, 'diagnostics.pkl')
        with open(diag_pkl, 'wb') as f:
            pickle.dump(diagnostics_all, f)
        print(f"Saved per-(neuron, model) diagnostics → {diag_pkl}")

    # ── Save combo results ───────────────────────────────────────────────
    combo_results_df     = pd.DataFrame(all_combo_rows)
    combo_comparisons_df = pd.DataFrame(all_delta_rows)
    if run_combos:
        combo_csv = os.path.join(OUT_DIR, 'combo_results.csv')
        combo_results_df.to_csv(combo_csv, index=False)
        print(f"Saved per-(neuron, combo) results → {combo_csv}")

        delta_csv = os.path.join(OUT_DIR, 'combo_comparisons.csv')
        combo_comparisons_df.to_csv(delta_csv, index=False)
        print(f"Saved per-(neuron, comparison) Δ table → {delta_csv}")

        combo_roi_df = compute_combo_roi_summary(
            combo_comparisons_df, alpha=COMBO_ROI_ALPHA)
        combo_roi_csv = os.path.join(
            OUT_DIR, 'combo_comparisons_roi_summary.csv')
        combo_roi_df.to_csv(combo_roi_csv, index=False)
        print(f"Saved ROI-level Δ summary → {combo_roi_csv}")
        if not combo_roi_df.empty:
            print("\n=== ROI × Δ summary (alpha={:.2f}) ===".format(
                COMBO_ROI_ALPHA))
            with pd.option_context('display.max_rows', None,
                                   'display.max_columns', None,
                                   'display.width', 160):
                print(combo_roi_df[['roi', 'model_test', 'model_baseline',
                                    'n_neurons', 'mean_delta',
                                    't_p_greater', 'wilcoxon_p_greater',
                                    'n_sig', 'frac_sig', 'binomial_p_greater']]
                      .to_string(index=False))

        if SAVE_DIAGNOSTICS and combo_diagnostics_all:
            combo_pkl = os.path.join(OUT_DIR, 'combo_diagnostics.pkl')
            with open(combo_pkl, 'wb') as f:
                pickle.dump(combo_diagnostics_all, f)
            print(f"Saved per-(neuron, combo) diagnostics → {combo_pkl}")
else:
    # Reload mode: pull whatever saved tables exist in SOURCE_DIR.  Files
    # produced by a 'combos_only' run won't have the singles CSV/pickle,
    # so be tolerant of missing files.
    out_csv = os.path.join(SOURCE_DIR, 'encoding_results.csv')
    if os.path.exists(out_csv):
        results_df = _normalise_subject_col(pd.read_csv(out_csv))
        print(f"Loaded {len(results_df)} rows from {out_csv}")
    else:
        results_df = pd.DataFrame()
        print(f"No encoding_results.csv in {SOURCE_DIR}; "
              f"single-model plots will be empty.")

    diag_pkl = os.path.join(SOURCE_DIR, 'diagnostics.pkl')
    if os.path.exists(diag_pkl):
        with open(diag_pkl, 'rb') as f:
            diagnostics_all = pickle.load(f)
        print(f"Loaded diagnostics for {len(diagnostics_all)} subjects "
              f"from {diag_pkl}")
    else:
        diagnostics_all = {}
        print(f"No diagnostics.pkl in {SOURCE_DIR}; "
              f"best-neuron showcase will be skipped.")

    # Combo outputs (may be absent if this run only produced singles).
    combo_csv = os.path.join(SOURCE_DIR, 'combo_results.csv')
    if os.path.exists(combo_csv):
        combo_results_df = _normalise_subject_col(pd.read_csv(combo_csv))
        print(f"Loaded {len(combo_results_df)} combo rows from {combo_csv}")
    else:
        combo_results_df = pd.DataFrame()

    delta_csv = os.path.join(SOURCE_DIR, 'combo_comparisons.csv')
    if os.path.exists(delta_csv):
        combo_comparisons_df = _normalise_subject_col(pd.read_csv(delta_csv))
        print(f"Loaded {len(combo_comparisons_df)} Δ rows from {delta_csv}")
    else:
        combo_comparisons_df = pd.DataFrame()

    combo_pkl = os.path.join(SOURCE_DIR, 'combo_diagnostics.pkl')
    if os.path.exists(combo_pkl):
        with open(combo_pkl, 'rb') as f:
            combo_diagnostics_all = pickle.load(f)
        print(f"Loaded combo diagnostics for {len(combo_diagnostics_all)} "
              f"subjects from {combo_pkl}")
    else:
        combo_diagnostics_all = {}


# ── Subset filter: restrict every loaded table to a subject subset ───
# Cell-wise results just get filtered by `subject`; the filtered inputs
# are re-saved into the new OUT_DIR and every downstream summary / plot
# then regenerates for the subset.
if SUBSET_TAG is not None:
    keep = _subset_keep

    def _filter_by_subject(df):
        if df is None or df.empty or 'subject' not in df.columns:
            return df
        return df[df['subject'].apply(keep)].reset_index(drop=True)

    n_subs_before = (results_df['subject'].nunique()
                     if not results_df.empty else len(diagnostics_all))
    n_rows_before = len(results_df)
    results_df            = _filter_by_subject(results_df)
    combo_results_df      = _filter_by_subject(combo_results_df)
    combo_comparisons_df  = _filter_by_subject(combo_comparisons_df)
    diagnostics_all       = {k: v for k, v in diagnostics_all.items()
                             if keep(k)}
    combo_diagnostics_all = {k: v for k, v in combo_diagnostics_all.items()
                             if keep(k)}
    kept_ids = sorted({_norm_sub(s) for s in results_df['subject'].unique()}
                      if not results_df.empty
                      else {_norm_sub(k) for k in diagnostics_all})
    print(f"\nSubset '{SUBSET_TAG}': kept {len(results_df)}/{n_rows_before} "
          f"result rows from {len(kept_ids)}/{n_subs_before} subjects.")

    # Re-save the filtered result files into the new folder.
    if not results_df.empty:
        results_df.to_csv(os.path.join(OUT_DIR, 'encoding_results.csv'),
                          index=False)
    if diagnostics_all:
        with open(os.path.join(OUT_DIR, 'diagnostics.pkl'), 'wb') as f:
            pickle.dump(diagnostics_all, f)
    if not combo_results_df.empty:
        combo_results_df.to_csv(
            os.path.join(OUT_DIR, 'combo_results.csv'), index=False)
    if not combo_comparisons_df.empty:
        combo_comparisons_df.to_csv(
            os.path.join(OUT_DIR, 'combo_comparisons.csv'), index=False)
        compute_combo_roi_summary(
            combo_comparisons_df, alpha=COMBO_ROI_ALPHA
        ).to_csv(os.path.join(OUT_DIR,
                              'combo_comparisons_roi_summary.csv'),
                 index=False)
    if combo_diagnostics_all:
        with open(os.path.join(OUT_DIR, 'combo_diagnostics.pkl'), 'wb') as f:
            pickle.dump(combo_diagnostics_all, f)

    subset_config = dict(run_config)
    subset_config.update({
        'run_tag':     RUN_TAG,
        'out_dir':     OUT_DIR,
        'subset_of':   RELOAD_RUN,
        'subset_tag':  SUBSET_TAG,
        'subset_spec': str(SUBSET_REPLOT),
        'subjects':    kept_ids,
    })
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump(subset_config, f, indent=2)
    print(f"Saved subset result files + config → {OUT_DIR}")


from mc.plotting.cell_results import (
    plot_neuron_fit,
    plot_perm_histogram,
    plot_best_neuron_per_roi_model,
    plot_r_distribution_grid,
    plot_significance_proportion,
    plot_significance_counts_bars,
    plot_significant_cells_glassbrain,
    plot_significant_cells_mesh3d,
    compute_roi_model_tstats,
    bh_fdr,
    plot_roi_model_heatmap,
    plot_publication_r_histogram,
    plot_top_n_cells_per_model,
    plot_top_n_cells_per_roi_model,
    plot_roi_beta_glassbrain,
    CANONICAL_ROI_ORDER,
    CANONICAL_ENC_MODEL_ORDER,
)

# Per-(ROI, model) top-cell targets, resolved at render time against ROIs
# actually present in `results_df`.  None -> every ROI in the data.
PUB_TOP_CELL_TARGETS_SPEC = {
    'dsr':   None,
    'state': ['EC', 'medialOFC'],
}
PUB_TOP_N = 5



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
        use_fdr=True,
        legend_outside=True,
        model_order=CANONICAL_ENC_MODEL_ORDER,
    )
    print(f"Saved aggregate result plots → {summary_plot_dir}")
    
    # import pdb; pdb.set_trace()
    # # ── Glass-brain plots of significant cells per model ────────────
    # glass_dir = os.path.join(summary_plot_dir, 'significant_glassbrain')
    # os.makedirs(glass_dir, exist_ok=True)
    # # results_df already carries MNI_x/y/z and the new `roi` from the table.
    # coord_ok = results_df[['MNI_x', 'MNI_y', 'MNI_z']].notna().all(axis=1)
    # plot_df = results_df[coord_ok].copy().rename(columns={'roi': 'final_roi'})
    # if plot_df.empty:
    #     print("No cells with MNI coordinates — skipping glass-brain plots.")
    # else:
    #     for m in models:
    #         sub_df = plot_df[plot_df['model'] == m]
    #         if sub_df.empty:
    #             continue
    #         plot_significant_cells_glassbrain(
    #             sub_df, model_name=m,
    #             save_path=os.path.join(glass_dir, f'glassbrain_{m}.png'),
    #             p_threshold=GLASSBRAIN_P_THRESHOLD,
    #             title_suffix=f'reg alpha={ALPHA}',
    #         )
    #         if PLOT_GLASSBRAIN_3D:
    #             plot_significant_cells_mesh3d(
    #                 sub_df, model_name=m,
    #                 save_path=os.path.join(glass_dir, f'mesh3d_{m}.html'),
    #                 p_threshold=GLASSBRAIN_P_THRESHOLD,
    #                 title_suffix=f'reg alpha={ALPHA}',
    #             )
    #     print(f"Saved glass-brain plots → {glass_dir}")

    # ── Publication-style figures ───────────────────────────────────
    # 1. ROI × model t-test heatmap (all models).
    # 2. ROI-shaded glass-brain per model (one PNG + SVG per model).
    # 3. BH-FDR-corrected table for PUB_MODELS only.
    # 4. Publication histogram for PUB_MODELS (PNG + SVG + PDF).
    # 5. Top-5 cell fits per PUB_MODEL (PNG + SVG + PDF).
    pub_dir = os.path.join(OUT_DIR, 'publication_figures')
    os.makedirs(pub_dir, exist_ok=True)

    # 1. Per-(ROI, model) t-statistic + p-value.
    stats_df = compute_roi_model_tstats(results_df, models=models)
    stats_csv = os.path.join(pub_dir, 'roi_model_tstats.csv')
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved per-(ROI, model) t-stat table → {stats_csv}")

    plot_roi_model_heatmap(
        stats_df,
        models=CANONICAL_ENC_MODEL_ORDER,
        rois=CANONICAL_ROI_ORDER,
        value_col='t', annot_col='p_t', sig_col='p_t',
        n_col='n_cells', alpha=0.05,
        value_label='t-statistic (mean_r > 0)',
        save_path=os.path.join(pub_dir,
                               'roi_model_tstat_heatmap.png'),
        title='ROI × model — one-sided t-test of mean_r > 0',
        base_fontsize=15,
    )

    # # 2. ROI-shaded glass-brain coloured by t-statistic.
    # coord_ok = results_df[['MNI_x', 'MNI_y', 'MNI_z']].notna().all(axis=1)
    # coord_df = results_df[coord_ok].copy()
    # electrodes_per_roi = {}
    # for roi, g in coord_df.groupby('roi'):
    #     electrodes_per_roi[roi] = (
    #         g.drop_duplicates(subset=['subject', 'neuron'])
    #          [['MNI_x', 'MNI_y', 'MNI_z']]
    #          .to_numpy(dtype=float)
    #     )
    # rois_with_cells = sorted(electrodes_per_roi)

    # if rois_with_cells:
    #     glassbrain_dir = os.path.join(pub_dir, 'tstat_glassbrains')
    #     os.makedirs(glassbrain_dir, exist_ok=True)
    #     for m in models:
    #         sub_stats = stats_df[stats_df['model'] == m]
    #         if sub_stats.empty:
    #             continue
    #         ts = dict(zip(sub_stats['roi'], sub_stats['t']))
    #         ps = dict(zip(sub_stats['roi'], sub_stats['p_t']))
    #         plot_roi_beta_glassbrain(
    #             roi_betas=ts, roi_pvals=ps,
    #             only_rois=rois_with_cells,
    #             roi_cell_coords=electrodes_per_roi,
    #             roi_label_column=ROI_LABEL_COLUMN,
    #             title=f'{m} — t-stat (mean_r > 0)',
    #             save_path=os.path.join(
    #                 glassbrain_dir, f'roi_tstat_glassbrain_{m}.png'),
    #         )
    #         plt.close('all')

    # 3. BH-FDR-corrected table over PUB_MODELS × all ROIs.
    pub_stats = stats_df[stats_df['model'].isin(PUB_MODELS)].copy()
    if not pub_stats.empty:
        pub_stats['p_t_fdr'] = bh_fdr(pub_stats['p_t'].to_numpy())
        pub_stats['fdr_family'] = (
            f"models={PUB_MODELS}; n_tests={len(pub_stats)}"
        )
        pub_stats = pub_stats.sort_values(['model', 'p_t_fdr'])
        pub_fdr_csv = os.path.join(pub_dir,
                                   'roi_pub_models_tstats_fdr.csv')
        pub_stats.to_csv(pub_fdr_csv, index=False)
        n_sig_fdr = int((pub_stats['p_t_fdr'] < 0.05).sum())
        print(f"\n=== BH-FDR over PUB_MODELS={PUB_MODELS} "
              f"(n_tests={len(pub_stats)}) ===")
        print(f"{n_sig_fdr} (ROI, model) pairs significant at q < 0.05")
        with pd.option_context('display.max_rows', None,
                               'display.width', 160):
            print(pub_stats[['model', 'roi', 'n_cells',
                             'mean_r', 't', 'p_t', 'p_t_fdr',
                             'prop_sig_perm']].to_string(index=False))
        print(f"Saved FDR table → {pub_fdr_csv}")

    # 4. Publication-ready histogram for the chosen models. Panel sig
    # uses the FDR-adjusted p from step 3 if available, otherwise an
    # in-panel one-sided t-test.
    p_fdr_for_panels = None
    if not pub_stats.empty:
        p_fdr_for_panels = {}
        for m in PUB_MODELS:
            sub_m = pub_stats[pub_stats['model'] == m]
            p_fdr_for_panels[m] = dict(zip(sub_m['roi'], sub_m['p_t_fdr']))

    plot_publication_r_histogram(
        results_df, models=PUB_MODELS,
        rois=CANONICAL_ROI_ORDER,
        model_order=PUB_MODELS,
        save_path=os.path.join(pub_dir, 'pub_r_histogram.png'),
        alpha=0.05,
        base_fontsize=14,
        p_t_per_roi_model=p_fdr_for_panels,
    )

    # 4b. Counts-bar plot: grey total + coloured significant per ROI.
    plot_significance_counts_bars(
        results_df, models=PUB_MODELS,
        rois=CANONICAL_ROI_ORDER,
        model_order=PUB_MODELS,
        save_path=os.path.join(pub_dir, 'pub_sig_counts_bars.png'),
        alpha=0.05, use_fdr=True,
        base_fontsize=13,
    )

    # 5. Top-N cell fits per (PUB_MODEL, ROI), per the user's targets.
    if diagnostics_all:
        rois_in_data = sorted(results_df['roi'].dropna().unique().tolist())
        targets = []
        for model_name, roi_spec in PUB_TOP_CELL_TARGETS_SPEC.items():
            roi_list = (rois_in_data if roi_spec is None
                        else [r for r in roi_spec if r in rois_in_data])
            for roi in roi_list:
                targets.append((model_name, roi, PUB_TOP_N))
        plot_top_n_cells_per_roi_model(
            diagnostics_all, results_df, targets=targets,
            save_dir=os.path.join(pub_dir, 'top_cells_per_roi_model'),
            base_fontsize=13,
        )
    else:
        print("Skipping top-N cell fits: no diagnostics saved/loaded.")

    print(f"Saved publication figures → {pub_dir}")

    if PLOT_DIAGNOSTICS:
        plt.show()
