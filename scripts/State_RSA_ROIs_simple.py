#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-cell State RSA — harmonised with RSA_DSR_ROIs_simple.py.

Two analysis branches:
  - 'all_cells'  : every cell with a valid alt_final_roi label.  Single-model
                   regression: state.  Uses N_GROUPS=6 pseudo-configs prepped
                   by prep_human_cells_RSA-2026.py (MODE='state').
  - 'rsa_cells'  : only cells from sessions in
                   all_sessions_dsrRSA_grouping_summary.json (28 sessions with
                   matched 8-config layout).  Combo regressions with state +
                   visual/button/DSR controls.  Uses N_DSR_CONFIGS=8 configs
                   prepped by prep_human_cells_RSA-2026.py (MODE='dsr').

Two data-RDM variants (named after the DSR script's conventions):
  - 'split_halves' / '_z' — population matrix interleaves run-1 and run-2;
                            full off-diagonal cells used.
  - 'between_tasks' / '_z' — runs averaged per neuron, optional per-neuron
                             z-score, only between-pseudo-config cells used
                             (within-config block diagonal masked out).

Permutation null: label-shuffle on the model RDMs (cheap; data already
trial-averaged).  One-sided p_perm with Phipson–Smyth (k+1)/(N+1).  BH-FDR
across the 7 substantive ROIs (Visual excluded by default).

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc


# ══════════════════════════════════════════════════════════════════════
# ── Settings ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE = os.path.join(DATA_DIR, 'group', 'State_RSA_simple_ROI')
RELOAD_RUN = '2026-06-15_15-56-53' #None #'2026-06-15_09-55-28' #None     # e.g. '2026-06-12_14-17-48' — RELOAD MODE.
                       # When set, skip ALL heavy compute (no SUBJECT_DATA
                       # load, no empirical RSA fits, no permutation loop)
                       # and instead load every artifact from this prior
                       # run folder: results CSVs, perm null .npz files,
                       # data RDMs, electrode coords, and model RDMs.
                       # Then jump straight to the figure stage so you can
                       # iterate on plots without paying any compute cost.
                       # The prior run MUST have completed (i.e. saved
                       # data_rdms_by_roi.npz, electrode_coords.json,
                       # model_rdms.npz alongside the CSVs/perm npz).

N_BINS    = 360       # raw bin count per half-config / run
N_STATES  = 4
N_PHASES  = 3         # passed to mc.simulation.predictions.model_DSR
LEN_STANDARDISED_PATH = 3   # downsampled bins per state (must divide 90)
BINS_PER_CONFIG = N_STATES * LEN_STANDARDISED_PATH   # 12 by default
N_GROUPS_ALL    = 6   # pseudo-configs for the all_cells branch
N_CONFIGS_RSA   = 8   # actual configs for the rsa_cells branch

# Permutation testing.
N_PERMUTATIONS = 500
FDR_ALPHA      = 0.05

# Which data-RDM variants to run.
TESTS = ['split_halves', 'split_halves_z', 'between_tasks', 'between_tasks_z']
#TESTS = ['between_tasks_z']
PRIMARY_TEST = 'split_halves_z' #'between_tasks_z'

# Branches and their model specs.
#   'all_cells' uses only 'state' (configs differ per subject so visual
#   regressors are not shared).
#   'rsa_cells' runs the four combos below.
# For the rsa_cells branch only — the all_cells branch is fixed at
# state-only (no combos possible, since pseudo-tasks don't share location/
# button structure across subjects). All regressors here are built from
# the across-subject mode walked paths via model_DSR (see
# build_designs_rsa_cells) so that `dsr` is config-specific
# and uncorrelated with `state`.
RSA_COMBOS = {
    'state_loc_l2norm':            ['state', 'location', 'l2_norm'],
    'state_loc_l2norm_dsr':        ['state', 'location', 'l2_norm', 'dsr'],
    'state_loc_l2norm_btn':        ['state', 'location', 'l2_norm',
                                    'btn_curr', 'btn_next'],
    'state_loc_l2norm_btn_dsr':    ['state', 'location', 'l2_norm',
                                    'btn_curr', 'btn_next', 'dsr', 'phase'],
    # Phase / midnight visual controls (built from model_DSR too — every
    # regressor here goes through the canonical compute_*  RDM functions).
    'state_loc_l2norm_phase_midn': ['state', 'location', 'l2_norm',
                                    'phase', 'midnight'],
}
# Which combo we treat as primary for the BH-FDR family on 'state'.
FDR_COMBO     = 'state_loc_l2norm_btn_dsr'
FDR_SUBMODEL  = 'state'

# ROI list (verbatim from the DSR script's alt_final_roi family).
ROI_LABEL_COLUMN = 'alt_final_roi'
ROI_ORDER = [
    'mPFC', 'medial_CC',
    'HC_anterior', 'HC_mid',
    'EC', 'PHC',
    'PCC',
    'mOFC',
    'Visual',
]
# 'Visual' is excluded from the BH-FDR family (sanity-check ROI only).
FDR_ROIS = [r for r in ROI_ORDER if r != 'Visual']

# Plot settings.
PLOT_FIGS = True     # set True to render F1-F4 inline at the end


# ══════════════════════════════════════════════════════════════════════
# ── ROI mapping (same shape as DSR script) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════

ROI_TABLE_PATH = os.path.join(DATA_DIR, 'neurons_with_ROI_labels.csv')


def parse_neuron_label(label):
    """Parse '01_07-07-chan120-EC' or '07_3-1-chan53' → (sub:int, cell_idx:int)."""
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def _load_roi_table(path, roi_col):
    df = pd.read_csv(path)
    for ax in ('x', 'y', 'z'):
        if f'MNI_{ax}_final' in df.columns:
            df[f'MNI_{ax}'] = df[f'MNI_{ax}_final']
    needed = ['subject', 'cell idx', roi_col,
              'MNI_x', 'MNI_y', 'MNI_z', 'electrode label']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"ROI table {path} missing columns: {missing}  "
            f"(re-run scripts/cell_to_roi_MNI.py if {roi_col!r} is absent)")
    df = df.copy()
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


ROI_TABLE = _load_roi_table(ROI_TABLE_PATH, ROI_LABEL_COLUMN)
print(f"Loaded ROI table: {len(ROI_TABLE)} cells, "
      f"{ROI_TABLE[ROI_LABEL_COLUMN].nunique()} ROIs "
      f"(column: {ROI_LABEL_COLUMN}).")


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


# ══════════════════════════════════════════════════════════════════════
# ── Helpers ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _scalar(arr):
    return float(np.asarray(arr, dtype=float).ravel()[0])


def downsample_mean(x, target_len):
    """Average raw bins down to target_len bins (used to collapse 360→12)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    edges = np.linspace(0, n, target_len + 1, dtype=int)
    out = np.empty(target_len)
    for i in range(target_len):
        chunk = x[edges[i]:edges[i + 1]]
        out[i] = np.nan if chunk.size == 0 else np.nanmean(chunk)
    return out


def downsample_array(arr, target_len):
    """Apply downsample_mean along the last axis."""
    src = np.asarray(arr, dtype=float)
    new_shape = src.shape[:-1] + (target_len,)
    out = np.empty(new_shape)
    it = np.ndindex(*src.shape[:-1])
    for idx in it:
        out[idx] = downsample_mean(src[idx], target_len)
    return out


def bh_fdr(pvals):
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
    qok = np.empty(n)
    qok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = qok
    return q


# ══════════════════════════════════════════════════════════════════════
# ── Raw data loaders for circular-shift permutation ───────────────────
# ══════════════════════════════════════════════════════════════════════
# To match the DSR script's permutation (per-trial circular shifts of raw
# firing rates), each perm needs to: (1) apply a per-(cell, trial) shift
# along the 360-bin axis of the raw correct-trial data, (2) re-average
# trials per (config-or-group, run) slot, and (3) rebuild the data RDM.
# To keep the perm loop fast we cache, per subject:
#   - raw_neurons:  (n_cells, n_correct_trials, 360) float32 array
#   - slots:        dict {(group, run): np.array of trial-row indices}
#   - neuron_names: list of cell labels (matches the first axis of raw)
# All operations downstream use these in-place; the cache is built once
# per branch.

BEH_COLS = ['rep_correct', 't_A', 't_B', 't_C', 't_D',
            'loc_A', 'loc_B', 'loc_C', 'loc_D',
            'rep_overall', 'new_grid_onset', 'session_no', 'grid_no',
            'correct']


def _load_raw_session(sub, branch):
    """Returns (raw_cells, slot_indices, neuron_names) or (None, None, None).

    branch == 'all_cells' → state grouping (N_GROUPS=6 pseudo-configs).
    branch == 'rsa_cells' → DSR grouping (8 ordered configs).
    """
    # Path setup.
    if branch == 'all_cells':
        log_path = os.path.join(DATA_DIR, f's{sub}', 'state_avg',
                                  f's{sub}_grouping_log.json')
        n_slots  = N_GROUPS_ALL
        slot_key = 'groups'
        slot_attr = 'group_idx'
    else:
        log_path = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg',
                                  f's{sub}_dsr_grouping_log_two_runs.json')
        n_slots  = N_CONFIGS_RSA
        slot_key = 'configs'
        slot_attr = 'config_idx'
    if not os.path.isfile(log_path):
        return None, None, None
    with open(log_path) as f:
        log = json.load(f)

    # Use the existing load_norm_data helper to pull raw neurons + beh.
    sub_data = mc.analyse.helpers_human_cells.load_norm_data(DATA_DIR, [sub])
    key = f'sub-{sub}'
    if key not in sub_data:
        return None, None, None
    beh = sub_data[key]['beh']
    neurons = sub_data[key]['normalised_neurons']
    names = sorted(neurons.keys())
    if not names:
        return None, None, None
    # Filter to correct trials (matches prep's beh_correct).
    beh_correct = beh[beh['correct'] == 1].reset_index(drop=True)
    # Need original beh index→position-in-correct mapping. The neurons
    # dataframes are indexed identically to beh, so position-in-original
    # is the row position. We must subset trials by `correct == 1`.
    correct_idx = beh.index[beh['correct'] == 1].to_numpy()
    if correct_idx.size == 0:
        return None, None, None

    # Stack neurons into (n_cells, n_correct_trials, 360).
    raw_stack = np.empty(
        (len(names), correct_idx.size, N_BINS), dtype=np.float32)
    for c, n_lab in enumerate(names):
        arr = neurons[n_lab].iloc[correct_idx].to_numpy(dtype=np.float32)
        raw_stack[c] = arr

    # Build slot → trial-row-indices (positions within correct_idx).
    grid_no_correct = beh_correct['grid_no'].to_numpy()
    slot_indices = {}
    for grp in log[slot_key]:
        s_idx = grp[slot_attr]
        for run_no, key_blocks in [(1, 'run1_blocks'), (2, 'run2_blocks')]:
            blocks = grp.get(key_blocks, []) or []
            if not blocks:
                slot_indices[(s_idx, run_no - 1)] = np.zeros(0, dtype=int)
                continue
            mask = np.isin(grid_no_correct, blocks)
            slot_indices[(s_idx, run_no - 1)] = np.where(mask)[0]
    return raw_stack, slot_indices, names


# ══════════════════════════════════════════════════════════════════════
# ── Branch 1 data loader: all_cells, N_GROUPS=6 pseudo-configs ────────
# ══════════════════════════════════════════════════════════════════════

ALL_CELLS_SUBJECTS = [f'{i:02}' for i in range(1, 64)]


def load_state_avg(sub):
    """Returns (arr, neuron_names) or (None, None).

    arr: (n_neurons, N_GROUPS_ALL, 2, N_BINS) — runs split, raw 360 bins.
    """
    sub_dir = os.path.join(DATA_DIR, f's{sub}', 'state_avg')
    npy = os.path.join(sub_dir, f's{sub}_neural_avg.npy')
    meta = os.path.join(sub_dir, f's{sub}_neuron_meta.json')
    if not os.path.isfile(npy) or not os.path.isfile(meta):
        return None, None
    arr = np.load(npy)
    with open(meta) as f:
        m = json.load(f)
    names = m['neuron_names']
    if arr.shape[0] != len(names):
        print(f"  s{sub}: n_neurons mismatch ({arr.shape[0]} vs {len(names)}) — skipped.")
        return None, None
    return arr, names


# ══════════════════════════════════════════════════════════════════════
# ── Branch 2 data loader: rsa_cells, 8 matched configs ────────────────
# ══════════════════════════════════════════════════════════════════════

DSR_SUMMARY_PATH = os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json')
with open(DSR_SUMMARY_PATH) as f:
    _DSR_SUMMARY = json.load(f)
RSA_CELLS_SUBJECTS = list(_DSR_SUMMARY.keys())
print(f"RSA-cells branch: {len(RSA_CELLS_SUBJECTS)} sessions from {DSR_SUMMARY_PATH}.")

# Fixed 8 configs (locA, locB, locC, locD).
DSR_CONFIGS = [
    (3, 7, 9, 5), (8, 2, 6, 7), (1, 9, 5, 8), (4, 8, 1, 3),
    (6, 4, 2, 9), (9, 1, 3, 4), (7, 3, 4, 2), (2, 5, 7, 6),
]
DSR_CONFIG_LABELS = [f'{c[0]}-{c[1]}-{c[2]}-{c[3]}' for c in DSR_CONFIGS]


def load_dsr_avg(sub):
    """Returns (arr, neuron_names) or (None, None).

    arr: (n_neurons, 8, 2, N_BINS).
    """
    sub_dir = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg')
    npy = os.path.join(sub_dir, f's{sub}_dsr_neural_avg_two_runs.npy')
    meta = os.path.join(sub_dir, f's{sub}_dsr_neuron_meta_two_runs.json')
    if not os.path.isfile(npy) or not os.path.isfile(meta):
        return None, None
    arr = np.load(npy)
    with open(meta) as f:
        m = json.load(f)
    names = m['neuron_names']
    if arr.shape[0] != len(names):
        print(f"  s{sub}: n_neurons mismatch ({arr.shape[0]} vs {len(names)}) — skipped.")
        return None, None
    return arr, names


# ══════════════════════════════════════════════════════════════════════
# ── Build pooled per-ROI population arrays for each branch ────────────
# ══════════════════════════════════════════════════════════════════════

def pool_population(subject_arrays, target_roi):
    """Concatenate neurons across subjects that belong to `target_roi`.

    subject_arrays: list of (sub, arr, names) with arr shape
        (n_neurons, n_configs, 2, N_BINS).
    Returns (pop_arr, kept_labels):
        pop_arr.shape = (n_kept_neurons, n_configs, 2, N_BINS)
        kept_labels   = list of str neuron labels.
    """
    kept_arrays, kept_labels = [], []
    for sub, arr, names in subject_arrays:
        if arr is None:
            continue
        for n_idx, name in enumerate(names):
            roi = get_neuron_roi(name)
            if roi != target_roi:
                continue
            kept_arrays.append(arr[n_idx])  # (n_cfg, 2, 360)
            kept_labels.append(name)
    if not kept_arrays:
        return None, []
    return np.stack(kept_arrays, axis=0), kept_labels


# ══════════════════════════════════════════════════════════════════════
# ── Build the 4 data-RDM variants from a pooled array ─────────────────
# ══════════════════════════════════════════════════════════════════════

def _zscore_neurons(m):
    """Z-score per column (per neuron) across the population matrix."""
    mu = np.nanmean(m, axis=0)
    sd = np.nanstd(m, axis=0)
    sd[sd == 0] = 1.0
    return (m - mu) / sd


def _inflate_split_halves(flat, n):
    """Upper-triangle (k=1) flat vec → symmetric n×n square with 0 diag.
    Used for plotting the data RDM in the split_halves variant."""
    mat = np.zeros((n, n), dtype=float)
    ii, jj = np.triu_indices(n, k=1)
    mat[ii, jj] = flat
    mat[jj, ii] = flat
    return mat


def _inflate_between_tasks(flat, n, block_size):
    """Off-block-diagonal flat vec → n×n square with NaN within-config blocks
    (matching the masking convention of the between_tasks RSA)."""
    mat = np.full((n, n), np.nan)
    ii, jj = np.triu_indices(n, k=1)
    within = (ii // block_size) == (jj // block_size)
    off_ii, off_jj = ii[~within], jj[~within]
    mat[off_ii, off_jj] = flat
    mat[off_jj, off_ii] = flat
    return mat


def build_data_rdms(pop_arr):
    """Build the four data-RDM variants for one ROI using the canonical
    functions from `mc.analyse.my_RSA`:

      - split_halves / _z → compute_crosscorr  (input: 2N rows)
      - between_tasks /_z → compute_crosscorr_within (input: N rows,
                                                      block_size = BPC)

    Returns (flat_rdms, square_rdms). `square_rdms` is provided for the
    F1 plotting code; we just inflate the flat upper-tri vector back into
    a symmetric matrix (no extra RDM math).
    """
    n_neurons, n_cfg, n_runs, _ = pop_arr.shape
    assert n_runs == 2, f"expected 2 runs, got {n_runs}"

    ds = downsample_array(pop_arr, BINS_PER_CONFIG)

    # split_halves: stack run-1 || run-2 across configs.
    rows_split = []
    for run_i in range(2):
        for c in range(n_cfg):
            rows_split.append(ds[:, c, run_i, :].T)
    mat_split   = np.vstack(rows_split)
    mat_split_z = _zscore_neurons(mat_split)

    # between_tasks: average runs, stack configs.
    avg_runs = np.nanmean(ds, axis=2)
    mat_between   = np.vstack([avg_runs[:, c, :].T for c in range(n_cfg)])
    mat_between_z = _zscore_neurons(mat_between)

    flat = {}
    flat['split_halves'] = np.asarray(
        mc.analyse.my_RSA.compute_crosscorr(
            mat_split, plotting=False, include_diagonal=False,
            no_tasks=n_cfg, model='data_split_halves')[0], dtype=float)
    flat['split_halves_z'] = np.asarray(
        mc.analyse.my_RSA.compute_crosscorr(
            mat_split_z, plotting=False, include_diagonal=False,
            no_tasks=n_cfg, model='data_split_halves_z')[0], dtype=float)
    _, between, _ = mc.analyse.my_RSA.compute_crosscorr_within(
        mat_between, plotting=False, include_diagonal=False,
        no_tasks=n_cfg, model='data_between_tasks',
        block_size=BINS_PER_CONFIG)
    flat['between_tasks'] = np.asarray(between[0], dtype=float)
    _, between_z, _ = mc.analyse.my_RSA.compute_crosscorr_within(
        mat_between_z, plotting=False, include_diagonal=False,
        no_tasks=n_cfg, model='data_between_tasks_z',
        block_size=BINS_PER_CONFIG)
    flat['between_tasks_z'] = np.asarray(between_z[0], dtype=float)

    # Square form for plotting (inflate the flat vec).
    N = n_cfg * BINS_PER_CONFIG
    sq = {
        'split_halves':    _inflate_split_halves(flat['split_halves'],    N),
        'split_halves_z':  _inflate_split_halves(flat['split_halves_z'],  N),
        'between_tasks':   _inflate_between_tasks(flat['between_tasks'],
                                                    N, BINS_PER_CONFIG),
        'between_tasks_z': _inflate_between_tasks(flat['between_tasks_z'],
                                                    N, BINS_PER_CONFIG),
    }
    return flat, sq


# ══════════════════════════════════════════════════════════════════════
# ── Model RDMs  ────────────────
# ══════════════════════════════════════════════════════════════════════
#
# Designs come from `mc.simulation.predictions.model_DSR` (state, location,
# phase, dsr, midnight, phase_state, dsr_now_next) + the across-subject
# MODE buttons / locations (via `mc.analyse.helpers_human_cells.
# compute_mode_paths_per_config`). RDMs are built exclusively via the
# canonical functions in `mc.analyse.my_RSA`:
#
#   - Hamming for categorical / one-hot regressors:
#       location, dsr, dsr_now_next, btn_curr, btn_next  →
#         compute_hamming_distance      (split_halves variant)
#         compute_hamming_distance_within (between_tasks variant)
#   - Cosine for continuous regressors:
#       state, phase, midnight, phase_state, l2_norm  →
#         compute_crosscorr             (split_halves variant)
#         compute_crosscorr_within      (between_tasks variant)
#
# This matches RSA_DSR_ROIs_simple.py line 636-654 exactly.

# Which regressors use Hamming distance (everything else uses cosine).
HAMMING_MODELS = {
    'location', 'dsr', 'dsr_now_next', 'btn_curr', 'btn_next',
    'btn_prev', 'uncover',
}


def _downsample_360_to_BPC(mat):
    """(n_features, 360) → (BPC, n_features). Each downsampled bin is the
    mean of 30 raw bins. Mirrors what RSA_DSR_ROIs_simple.py does (line
    602-607) when reshaping the model_DSR output to the cell binning."""
    mat = np.asarray(mat, dtype=float)
    n_feat = mat.shape[0]
    bins_per_state = 360 // N_STATES
    chunk = bins_per_state // LEN_STANDARDISED_PATH
    ds = mat.reshape(n_feat, BINS_PER_CONFIG, chunk).mean(axis=2)
    return ds.T


# Cached so we only walk the 28 subjects once per script invocation.
_MODE_PATHS_CACHE = {}


def _get_mode_paths():
    if not _MODE_PATHS_CACHE:
        print("Computing mode paths via "
              "mc.analyse.helpers_human_cells.compute_mode_paths_per_config …")
        mp = mc.analyse.helpers_human_cells.compute_mode_paths_per_config(
            DATA_DIR, RSA_CELLS_SUBJECTS, DSR_CONFIGS)
        _MODE_PATHS_CACHE.update(mp)
    return _MODE_PATHS_CACHE


def build_designs_rsa_cells():
    """Build per-config design matrices for the rsa_cells branch using
    `mc.simulation.predictions.model_DSR(walked, no_phase_neurons=N_PHASES)`
    on the across-subject MODE walked locations (exactly as
    RSA_DSR_ROIs_simple.py:602-607 does). Buttons are one-hots of the MODE
    button per chunk; l2_norm is the multi-column −‖x_curr − x_j‖ regressor
    matching create_fMRI_model_RDMs_on_clean_beh.py:240.

    Returns dict{model_name: (n_cfg*BPC, n_features) ndarray}.
    """
    import mc.simulation.predictions as _preds
    mode_paths = _get_mode_paths()

    grid_coords = {i: np.array([(i - 1) % 3, (i - 1) // 3], dtype=float)
                   for i in range(1, 10)}
    BTN_RAW = ['LeftArrow', 'RightArrow', 'UpArrow', 'DownArrow']
    chunk = 360 // BINS_PER_CONFIG

    designs = {k: [] for k in (
        'state', 'location', 'phase', 'midnight',
        'dsr', 'phase_state', 'dsr_now_next',
        'btn_curr', 'btn_next', 'l2_norm',
    )}

    for cfg in DSR_CONFIGS:
        mp = mode_paths.get(cfg)
        if mp is None:
            continue
        walked = (mp['mode_locs'] - 1).clip(0, 8).tolist()
        (loc_og, phas_og, stat_og, midn_og,
         clo_og, phas_stat_og, clo_nn_og) = _preds.model_DSR(
            locations=walked, no_phase_neurons=N_PHASES)

        designs['state'].append(_downsample_360_to_BPC(stat_og))
        designs['location'].append(_downsample_360_to_BPC(loc_og))
        designs['phase'].append(_downsample_360_to_BPC(phas_og))
        designs['midnight'].append(_downsample_360_to_BPC(midn_og))
        designs['dsr'].append(_downsample_360_to_BPC(clo_og))
        designs['phase_state'].append(_downsample_360_to_BPC(phas_stat_og))
        designs['dsr_now_next'].append(_downsample_360_to_BPC(clo_nn_og))

        # Buttons from the mode-button sequence (one-hot per cardinal key,
        # per BPC chunk).
        mode_btn = mp['mode_btns']
        btn_curr_X = np.zeros((BINS_PER_CONFIG, len(BTN_RAW)))
        for b in range(BINS_PER_CONFIG):
            seg = mode_btn[b * chunk:(b + 1) * chunk]
            seg = [str(v) for v in seg if str(v) in BTN_RAW]
            if seg:
                btn = Counter(seg).most_common(1)[0][0]
                btn_curr_X[b, BTN_RAW.index(btn)] = 1.0
        btn_next_X = np.roll(btn_curr_X, -LEN_STANDARDISED_PATH, axis=0)
        designs['btn_curr'].append(btn_curr_X)
        designs['btn_next'].append(btn_next_X)

        # l2_norm: −‖x_curr − x_j‖ for j=1..9. Multi-column → cosine RDM.
        loc_chunks = mp['mode_locs'].reshape(BINS_PER_CONFIG, chunk)
        l2 = np.zeros((BINS_PER_CONFIG, 9))
        for b in range(BINS_PER_CONFIG):
            curr = int(Counter(loc_chunks[b].tolist())
                       .most_common(1)[0][0])
            for j in range(1, 10):
                l2[b, j - 1] = -np.linalg.norm(
                    grid_coords[curr] - grid_coords[j])
        designs['l2_norm'].append(l2)

    return {k: np.vstack(v) for k, v in designs.items() if v}


def build_designs_all_cells_state_only():
    """All-cells branch — state-only. model_DSR's stat_og output is
    location-agnostic, so we feed any walked path and take just the state
    matrix. We replicate it N_GROUPS_ALL times so the per-pseudo-task layout
    matches the all_cells population matrix (which has 6 pseudo-configs).
    """
    import mc.simulation.predictions as _preds
    dummy = [0] * 360
    _, _, stat_og, _, _, _, _ = _preds.model_DSR(
        locations=dummy, no_phase_neurons=N_PHASES)
    state_per_cfg = _downsample_360_to_BPC(stat_og)   # (BPC, 4)
    return {'state': np.vstack([state_per_cfg] * N_GROUPS_ALL)}


def _model_rdm_canonical(X, variant, n_cfg, model_name):
    """Compute the RDM for a single design matrix X using the canonical
    `mc.analyse.my_RSA` functions:

      - split_halves variant: pass np.vstack([X, X]) (2N rows) to
        compute_crosscorr / compute_hamming_distance — that's the same
        as RSA_DSR_ROIs_simple.py:644-649 does for the cross-halves
        block. Since the model is identical in both halves the cross-
        half block equals the within-half cosine/hamming RDM.
      - between_tasks variant: pass single-half X (N rows) to the
        _within counterpart with block_size = BINS_PER_CONFIG and take
        the off-block-diagonal output (RSA_DSR_ROIs_simple.py:650-654).
    """
    use_hamming = model_name in HAMMING_MODELS
    if variant.startswith('split_halves'):
        doubled = np.vstack([X, X])
        if use_hamming:
            rdm = mc.analyse.my_RSA.compute_hamming_distance(
                doubled, plotting=False, include_diagonal=False,
                no_tasks=n_cfg, model_name=model_name)
        else:
            rdm = mc.analyse.my_RSA.compute_crosscorr(
                doubled, plotting=False, include_diagonal=False,
                no_tasks=n_cfg, model=model_name)
        return np.asarray(rdm[0], dtype=float)
    # between_tasks variant
    if use_hamming:
        _, between, _ = mc.analyse.my_RSA.compute_hamming_distance_within(
            X, plotting=False, include_diagonal=False,
            no_tasks=n_cfg, model_name=model_name,
            block_size=BINS_PER_CONFIG)
    else:
        _, between, _ = mc.analyse.my_RSA.compute_crosscorr_within(
            X, plotting=False, include_diagonal=False,
            no_tasks=n_cfg, model=model_name,
            block_size=BINS_PER_CONFIG)
    return np.asarray(between[0], dtype=float)


def build_model_rdms_state_only(n_cfg, variants):
    """All-cells branch: cosine RDM of the state regressor only."""
    designs = build_designs_all_cells_state_only()
    return {v: {name: _model_rdm_canonical(X, v, n_cfg, name)
                for name, X in designs.items()}
            for v in variants}


def build_model_rdms_rsa_cells(configs, variants):
    """rsa_cells branch: full set of regressors from model_DSR + mode
    buttons + l2_norm, each turned into a flat RDM via the canonical
    functions in `mc.analyse.my_RSA`."""
    n_cfg = len(configs)
    designs = build_designs_rsa_cells()
    out = {v: {name: _model_rdm_canonical(X, v, n_cfg, name)
               for name, X in designs.items()}
           for v in variants}
    # Sanity print: state × dsr RDM correlation should be ≈ 0.
    primary = 'between_tasks_z'
    if primary in out and 'state' in out[primary] and 'dsr' in out[primary]:
        for a, b in (('state', 'dsr'), ('state', 'location'),
                     ('dsr', 'location')):
            if a in out[primary] and b in out[primary]:
                r = float(np.corrcoef(out[primary][a],
                                        out[primary][b])[0, 1])
                print(f"  RDM corr ({primary})  {a:10s} × {b:10s}  r = {r:+.3f}")
    return out


# ══════════════════════════════════════════════════════════════════════
# ── Single-model + combo RSA helpers ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def fit_single_model(model_rdm, data_rdm):
    """Returns (t, beta, p_param) for a single-regressor RSA."""
    res = mc.analyse.my_RSA.evaluate_model(model_rdm, data_rdm)
    return (_scalar(res[0]), _scalar(res[1]), _scalar(res[2]))


def fit_combo_model(model_rdm_dict, combo_list, data_rdm):
    """Returns dict{sub_model: (t, beta, p_param)} from a combo OLS.

    Raises a clean error if any model RDM is all-NaN or constant (otherwise
    statsmodels reports a deep MissingDataError that's hard to trace).
    """
    stacked = np.stack([model_rdm_dict[m] for m in combo_list], axis=1)
    for i, m in enumerate(combo_list):
        col = stacked[:, i]
        if not np.any(np.isfinite(col)):
            raise ValueError(f"model RDM {m!r} is all-NaN")
        if np.nanstd(col) < 1e-12:
            raise ValueError(
                f"model RDM {m!r} is constant — OLS z-score would divide "
                f"by zero. Check the design matrix for an empty / all-zero "
                f"column.")
    res = mc.analyse.my_RSA.evaluate_model(stacked, data_rdm)
    t = np.asarray(res[0], dtype=float).ravel()
    b = np.asarray(res[1], dtype=float).ravel()
    p = np.asarray(res[2], dtype=float).ravel()
    return {m: (float(t[i]), float(b[i]), float(p[i]))
            for i, m in enumerate(combo_list)}


def _prep_ols_design(X_cols):
    """Z-score each column of X, prepend an intercept, drop any NaN rows.

    Returns (X_std_with_intercept, keep_mask, mu, sd).
    """
    X = np.asarray(X_cols, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    keep = ~(np.isnan(X).any(axis=1))
    Xk = X[keep]
    mu = Xk.mean(axis=0)
    sd = Xk.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (Xk - mu) / sd
    Xs = np.column_stack([np.ones(Xs.shape[0]), Xs])
    return Xs, keep, mu, sd


def _prep_ols_y(y_full, keep):
    """Apply the design's NaN mask to Y and z-score (matching evaluate_model)."""
    y = np.asarray(y_full, dtype=float)[keep]
    y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
    return y


def _gen_within_config_perm_indices(n_cfg, n_perms, seed=42):
    """Generate (n_perms, n_cfg*BPC) permutation indices that shuffle bins
    WITHIN each config block independently. This is the same null as the
    original State_RSA_human_cells-2026.py and matches the spirit of the
    DSR script's permutation (preserve config identity; randomise bin →
    state assignment).
    """
    rng = np.random.default_rng(seed)
    N = n_cfg * BINS_PER_CONFIG
    perms = np.tile(np.arange(N), (n_perms, 1))
    for i in range(n_perms):
        for c in range(n_cfg):
            blk = slice(c * BINS_PER_CONFIG, (c + 1) * BINS_PER_CONFIG)
            perms[i, blk] = perms[i, blk][rng.permutation(BINS_PER_CONFIG)]
    return perms


def _extract_indices(variant, n_cfg):
    """Return (ii, jj) flat indices into the N×N square RDM for the given
    variant: full upper triangle for split_halves; off-block-diagonal
    upper triangle for between_tasks."""
    N = n_cfg * BINS_PER_CONFIG
    ii, jj = np.triu_indices(N, k=1)
    if variant.startswith('split_halves'):
        return ii, jj
    within = (ii // BINS_PER_CONFIG) == (jj // BINS_PER_CONFIG)
    return ii[~within], jj[~within]


def within_config_perm_pvalue(model_rdm_flat, sq_data_rdm, variant, n_cfg,
                                perm_indices):
    """One-sided permutation p_perm for a SINGLE-model fit using within-
    config bin permutation. Re-indexing the SQUARE data RDM with the same
    permutation in rows AND cols (= what the original State script does) is
    algebraically equivalent to permuting the model RDM rows AND cols (= the
    rebuild-model-RDM-under-null approach). We use the data-side reindex so
    pinv(X) can be precomputed once.

    Returns (p_perm, null_betas).
    """
    X_std, keep, _, _ = _prep_ols_design(model_rdm_flat)
    ii, jj = _extract_indices(variant, n_cfg)
    emp_y = sq_data_rdm[ii, jj]
    y_std = _prep_ols_y(emp_y, keep)
    pinv_X = np.linalg.pinv(X_std)
    emp_beta = float((pinv_X @ y_std)[1])
    # Build (n_perms, len(ii)) permuted Y matrix.
    Y_perm_full = sq_data_rdm[perm_indices[:, ii], perm_indices[:, jj]]
    # Filter NaNs (same `keep` mask) + z-score each row.
    Y_kept = Y_perm_full[:, keep]
    mu = Y_kept.mean(axis=1, keepdims=True)
    sd = Y_kept.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Y_z = (Y_kept - mu) / sd
    null_betas = (Y_z @ pinv_X.T)[:, 1]
    p_perm = (int(np.sum(null_betas >= emp_beta)) + 1) / (perm_indices.shape[0] + 1)
    return p_perm, null_betas


# ══════════════════════════════════════════════════════════════════════
# ── Per-trial circular-shift permutation (matches DSR exactly) ────────
# ══════════════════════════════════════════════════════════════════════
# Per perm, for each subject's raw firing-rate trials:
#   shifts ~ Uniform[0, 360) per (cell, trial)
#   shifted[c, t, b] = raw[c, t, (b - shifts[c, t]) % 360]
#   re-average per (slot, run) using cached trial-row-indices
# Then downsample 360 → BPC, stack into the per-ROI pop matrix, rebuild
# the data RDM (all 4 variants), and refit OLS to record null β.

def _rebuild_pop_under_shift(raw_cache, slot_cache, n_slots, rng):
    """For each subject, sample per-(cell, trial) shifts, apply, and re-
    average per (slot, run). Returns a dict {sub: pop_arr (n_cells, n_slots,
    2, N_BINS)}.
    """
    out = {}
    for sub, raw in raw_cache.items():
        n_cells, n_trials, _ = raw.shape
        shifts = rng.integers(0, N_BINS, size=(n_cells, n_trials))
        # idx[c, t, b] = (b - shifts[c, t]) % N_BINS
        idx = (np.arange(N_BINS)[None, None, :]
               - shifts[:, :, None]) % N_BINS
        shifted = np.take_along_axis(raw, idx, axis=2)
        pop = np.full((n_cells, n_slots, 2, N_BINS), np.nan,
                       dtype=np.float32)
        for (s_idx, r_idx), trial_rows in slot_cache[sub].items():
            if trial_rows.size == 0:
                continue
            pop[:, s_idx, r_idx, :] = shifted[:, trial_rows, :].mean(axis=1)
        out[sub] = pop
    return out


def _flatten_pop_to_roi(pop_per_sub, cell_subj, cell_idx_in_sub, roi_mask):
    """Stack the cells of one ROI from per-subject pop arrays into one
    (n_roi_cells, n_slots, 2, N_BINS) array."""
    rows = []
    for k in np.where(roi_mask)[0]:
        sub, c_idx = cell_subj[k], cell_idx_in_sub[k]
        rows.append(pop_per_sub[sub][c_idx])
    return np.stack(rows, axis=0)


def _quick_flat_rdms_from_pop(pop_arr, n_cfg):
    """Flat RDMs for all four variants via the canonical functions.

    Used inside the permutation inner loop, where we need the four flat
    vectors but no square form. Re-uses `build_data_rdms` (which is built
    on `mc.analyse.my_RSA.compute_crosscorr` / `compute_crosscorr_within`)
    and discards the square output."""
    flat, _ = build_data_rdms(pop_arr)
    return flat


def circular_shift_perm(branch, raw_cache, slot_cache,
                         cell_subj, cell_idx_in_sub, cell_roi,
                         rois_present, n_cfg, model_rdms_dict,
                         model_specs, n_perms, seed=42):
    """Run circular-shift permutation PER ROI. For each ROI, the per-perm
    rebuild only touches the cells in that ROI — same as the DSR script's
    `subject → config → neuron → perm` loop layout. This is ~5–10× faster
    than the previous version that rebuilt the pop matrix for all 940 cells
    per perm.

    Returns dict {(roi, variant, label, sub_model): null_array(n_perms,)}.
    """
    n_slots = n_cfg

    # Precompute pinv(X) per (variant, label). Constant across the entire
    # run — the model RDMs don't change with the data permutation.
    pinvs = {}
    for variant in TESTS:
        for label, sub_models in model_specs:
            X_cols = np.stack([model_rdms_dict[variant][m] for m in sub_models],
                              axis=1)
            X_std, keep, _, _ = _prep_ols_design(X_cols)
            pinvs[(variant, label)] = (X_std, np.linalg.pinv(X_std), keep)

    null = {}
    for roi in rois_present:
        # 1) Filter the raw cache to only this ROI's cells, per subject.
        roi_raw = {}            # sub → (n_roi_cells_sub, n_trials, 360)
        roi_n_per_sub = {}      # sub → n_roi_cells_sub
        for sub, raw in raw_cache.items():
            mask_in_sub = np.array(
                [cell_subj[k] == sub and cell_roi[k] == roi
                 for k in range(len(cell_subj))])
            if not mask_in_sub.any():
                continue
            idxs = cell_idx_in_sub[mask_in_sub].astype(int)
            roi_raw[sub] = raw[idxs]
            roi_n_per_sub[sub] = len(idxs)
        total_cells = sum(roi_n_per_sub.values())
        if total_cells == 0:
            continue
        print(f"  [{branch}/{roi}] {total_cells} cells across "
              f"{len(roi_raw)} sessions ...")

        # 2) Pre-arrange cell offsets so we can write directly into one
        # contiguous pop_roi array per perm.
        sub_order = sorted(roi_raw.keys())
        offsets = {}
        c_off = 0
        for sub in sub_order:
            offsets[sub] = c_off
            c_off += roi_n_per_sub[sub]

        # 3) Allocate null arrays for this ROI.
        for variant in TESTS:
            for label, sub_models in model_specs:
                for m in sub_models:
                    null[(roi, variant, label, m)] = np.empty(
                        n_perms, dtype=np.float32)

        # 4) Per-perm loop — operates only on this ROI's cells.
        rng = np.random.default_rng(seed)
        pop_roi = np.full(
            (total_cells, n_slots, 2, N_BINS), np.nan, dtype=np.float32)
        for perm_i in range(n_perms):
            for sub in sub_order:
                raw = roi_raw[sub]
                n_cells_sub, n_trials, _ = raw.shape
                shifts = rng.integers(0, N_BINS,
                                       size=(n_cells_sub, n_trials))
                idx = (np.arange(N_BINS)[None, None, :]
                       - shifts[:, :, None]) % N_BINS
                shifted = np.take_along_axis(raw, idx, axis=2)
                off = offsets[sub]
                for (s_idx, r_idx), trial_rows in slot_cache[sub].items():
                    if trial_rows.size == 0:
                        continue
                    pop_roi[off:off + n_cells_sub, s_idx, r_idx, :] = (
                        shifted[:, trial_rows, :].mean(axis=1))
            flat_rdms = _quick_flat_rdms_from_pop(pop_roi, n_cfg)
            for variant in TESTS:
                y = flat_rdms[variant]
                for label, sub_models in model_specs:
                    X_std, pinv_X, keep = pinvs[(variant, label)]
                    y_std = _prep_ols_y(y, keep)
                    betas = (pinv_X @ y_std)[1:]
                    for j, m in enumerate(sub_models):
                        null[(roi, variant, label, m)][perm_i] = betas[j]
            if (perm_i + 1) % 200 == 0:
                print(f"    {roi}: perm {perm_i + 1}/{n_perms}")
    return null


def within_config_perm_combo(model_rdm_dict, combo_list, sq_data_rdm,
                              variant, n_cfg, perm_indices):
    """Combo-OLS variant of within_config_perm_pvalue. Same permutation is
    applied across all sub-models (preserves multicollinearity).
    """
    X_cols = np.stack([model_rdm_dict[m] for m in combo_list], axis=1)
    X_std, keep, _, _ = _prep_ols_design(X_cols)
    ii, jj = _extract_indices(variant, n_cfg)
    emp_y = sq_data_rdm[ii, jj]
    y_std = _prep_ols_y(emp_y, keep)
    pinv_X = np.linalg.pinv(X_std)
    emp_betas = (pinv_X @ y_std)[1:]
    Y_perm_full = sq_data_rdm[perm_indices[:, ii], perm_indices[:, jj]]
    Y_kept = Y_perm_full[:, keep]
    mu = Y_kept.mean(axis=1, keepdims=True)
    sd = Y_kept.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Y_z = (Y_kept - mu) / sd
    null_all = (Y_z @ pinv_X.T)[:, 1:]
    out = {}
    for j, m in enumerate(combo_list):
        emp_b = float(emp_betas[j])
        null_b = null_all[:, j]
        p = (int(np.sum(null_b >= emp_b)) + 1) / (perm_indices.shape[0] + 1)
        out[m] = (p, null_b)
    return out


# ══════════════════════════════════════════════════════════════════════
# ── Output folder ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

os.makedirs(OUT_BASE, exist_ok=True)
if RELOAD_RUN is not None:
    RUN_TAG = RELOAD_RUN
    OUT_DIR = os.path.join(OUT_BASE, RELOAD_RUN)
    if not os.path.isdir(OUT_DIR):
        raise FileNotFoundError(f"Reload folder not found: {OUT_DIR}")
    print(f"RELOAD MODE — reading from: {OUT_DIR}")
else:
    RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
    os.makedirs(OUT_DIR, exist_ok=True)
    run_config = {
        'run_tag':              RUN_TAG,
        'timestamp':            datetime.now().isoformat(timespec='seconds'),
        'data_dir':             DATA_DIR,
        'out_dir':              OUT_DIR,
        'roi_label_column':     ROI_LABEL_COLUMN,
        'roi_order':            ROI_ORDER,
        'fdr_rois':             FDR_ROIS,
        'fdr_test':             PRIMARY_TEST,
        'fdr_combo':            FDR_COMBO,
        'fdr_submodel':         FDR_SUBMODEL,
        'fdr_alpha':            FDR_ALPHA,
        'tests':                TESTS,
        'rsa_combos':           RSA_COMBOS,
        'n_permutations':       N_PERMUTATIONS,
        'n_groups_all_cells':   N_GROUPS_ALL,
        'n_configs_rsa_cells':  N_CONFIGS_RSA,
        'len_standardised_path': LEN_STANDARDISED_PATH,
        'all_cells_subjects':   ALL_CELLS_SUBJECTS,
        'rsa_cells_subjects':   RSA_CELLS_SUBJECTS,
    }
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"Run output: {OUT_DIR}")

# import pdb; pdb.set_trace()
# ══════════════════════════════════════════════════════════════════════
# ── Load all data once per branch ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

# NOTE: legacy load_state_avg / load_dsr_avg (pre-averaged .npy files
# from the prep script) are no longer used. The DSR-style loop below
# (further down) loads RAW trial data via
# `mc.analyse.helpers_human_cells.load_norm_data` per subject, exactly
# the same way RSA_DSR_ROIs_simple.py:297-300 does.


# ══════════════════════════════════════════════════════════════════════
# ── Build model RDMs (once — same for every ROI) ──────────────────────
# ══════════════════════════════════════════════════════════════════════

MODEL_RDMS_PATH = os.path.join(OUT_DIR, 'model_rdms.npz')
if RELOAD_RUN is not None and os.path.isfile(MODEL_RDMS_PATH):
    print(f"\nLoading model RDMs from {MODEL_RDMS_PATH} ...")
    _mr = np.load(MODEL_RDMS_PATH, allow_pickle=False)
    model_rdms_all = {v: {} for v in TESTS}
    model_rdms_rsa = {v: {} for v in TESTS}
    for k in _mr.files:
        branch, variant, name = k.split('__', 2)
        if branch == 'all':
            model_rdms_all.setdefault(variant, {})[name] = _mr[k]
        else:
            model_rdms_rsa.setdefault(variant, {})[name] = _mr[k]
    print(f"  loaded {sum(len(v) for v in model_rdms_all.values())} all-cells "
          f"+ {sum(len(v) for v in model_rdms_rsa.values())} rsa-cells model RDMs")
else:
    print("\nBuilding model RDMs (state-only) for all_cells branch...")
    model_rdms_all = build_model_rdms_state_only(N_GROUPS_ALL, TESTS)
    print("Building model RDMs (combos) for rsa_cells branch...")
    model_rdms_rsa = build_model_rdms_rsa_cells(DSR_CONFIGS, TESTS)


# ══════════════════════════════════════════════════════════════════════
# ── Run RSA + permutation per ROI per branch per variant ──────────────
# ══════════════════════════════════════════════════════════════════════

results_rows = []           # single-model branch (all_cells)
results_combo_rows = []     # combo branch (rsa_cells)
perm_nulls = {              # for plotting later
    'all_cells': {},        # {(roi, test): null_array}
    'rsa_cells': {},        # {(roi, combo, test, sub_model): null_array}
}
electrode_coords = {        # for the glass-brain
    'all_cells': {roi: {} for roi in ROI_ORDER},
    'rsa_cells': {roi: {} for roi in ROI_ORDER},
}
# Per-ROI data RDMs for the variant figure gallery (F1).
data_rdms_by_roi = {        # {branch: {roi: {test: rdm_vec}}}
    'all_cells': {},
    'rsa_cells': {},
}


def _stash_electrode(roi_dict, label):
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return
    key = (sub, cell_idx)
    if key in roi_dict:
        return
    mni = get_neuron_mni(label)
    if all(np.isfinite(mni)):
        roi_dict[key] = mni


# Stash square RDMs alongside flat ones so the figure code can mask /
# triangulate / re-index without re-computing the cosine.
square_rdms_by_roi = {'all_cells': {}, 'rsa_cells': {}}



# ══════════════════════════════════════════════════════════════════════
# ── DSR-style empirical + permutation loop  ──────────────────────────
# Mirrors RSA_DSR_ROIs_simple.py:293-422 exactly — load all subjects with
# `mc.analyse.helpers_human_cells.load_norm_data`, then a SINGLE per-ROI
# loop walks subjects → configs → neurons (filtered by ROI), applying
# per-trial circular shifts on the raw (n_trials, 360) firing rate,
# averaging across trials, and downsampling to BPC bins. The downsampled
# (BPC,) vector is appended into per-(config × perm) accumulators —
# identical to DSR's `acc_neurons_all[conf]` and `perm_ACC_neurons_all[conf]`.
# ══════════════════════════════════════════════════════════════════════

# Cache all subject data once via the canonical loader. Skipped in RELOAD mode.
SUBJECT_DATA = {}
if RELOAD_RUN is None:
    print("\nLoading SUBJECT_DATA via mc.analyse.helpers_human_cells.load_norm_data ...")
    ALL_SUBS_UNION = sorted(set(ALL_CELLS_SUBJECTS) | set(RSA_CELLS_SUBJECTS))
    for sub_str in ALL_SUBS_UNION:
        SUBJECT_DATA[sub_str] = mc.analyse.helpers_human_cells.load_norm_data(
            DATA_DIR, [sub_str])
    print(f"  cached data for {len(SUBJECT_DATA)} subjects.")


def _build_pseudo_cfg_trial_indices(subjects):
    """All_cells branch: from each subject's grouping_log.json, derive
    per-(pseudo-config, run-half) row-index lists into `normalised_neurons`."""
    out = {}
    for sub in subjects:
        if sub not in SUBJECT_DATA:
            continue
        log_path = os.path.join(DATA_DIR, f's{sub}', 'state_avg',
                                  f's{sub}_grouping_log.json')
        if not os.path.isfile(log_path):
            continue
        with open(log_path) as f:
            log = json.load(f)
        beh = SUBJECT_DATA[sub][f'sub-{sub}']['beh'].copy().reset_index(drop=True)
        gid = beh['grid_no'].to_numpy().astype(int)
        correct = (beh['correct'].to_numpy() == 1)
        sub_out = {}
        for g in log['groups']:
            blocks_all = g.get('blocks') or []
            blocks_th1 = g.get('run1_blocks') or []
            blocks_th2 = g.get('run2_blocks') or []
            sub_out[g['group_idx']] = {
                'all': np.where(correct & np.isin(gid, blocks_all))[0].tolist(),
                'th1': np.where(correct & np.isin(gid, blocks_th1))[0].tolist(),
                'th2': np.where(correct & np.isin(gid, blocks_th2))[0].tolist(),
            }
        out[sub] = sub_out
    return out


def _build_rsa_cfg_trial_indices(subjects):
    """rsa_cells branch: per matched DSR config, row-index lists for the
    correct trials. Halves are split by trial index (first / second half),
    matching the DSR script's th1/th2 = trials[0:10] / trials[10:20]."""
    out = {}
    for sub in subjects:
        if sub not in SUBJECT_DATA:
            continue
        beh = SUBJECT_DATA[sub][f'sub-{sub}']['beh'].copy().reset_index(drop=True)
        beh['cfg_tuple'] = list(zip(
            beh['loc_A'].astype(int), beh['loc_B'].astype(int),
            beh['loc_C'].astype(int), beh['loc_D'].astype(int)))
        correct = (beh['correct'].to_numpy() == 1)
        cfg_arr = beh['cfg_tuple'].to_numpy()
        sub_out = {}
        for c_idx, cfg in enumerate(DSR_CONFIGS):
            mask = correct & np.array([t == cfg for t in cfg_arr])
            idx = np.where(mask)[0].tolist()
            if not idx:
                continue
            mid = len(idx) // 2
            sub_out[c_idx] = {
                'all': idx,
                'th1': idx[:mid] if mid > 0 else idx,
                'th2': idx[mid:] if mid > 0 else [],
            }
        out[sub] = sub_out
    return out


if RELOAD_RUN is None:
    pseudo_cfg_idx = _build_pseudo_cfg_trial_indices(ALL_CELLS_SUBJECTS)
    rsa_cfg_idx    = _build_rsa_cfg_trial_indices(RSA_CELLS_SUBJECTS)
    print(f"  pseudo-config trial maps : {len(pseudo_cfg_idx)} subjects")
    print(f"  rsa-config trial maps    : {len(rsa_cfg_idx)} subjects")
else:
    pseudo_cfg_idx, rsa_cfg_idx = {}, {}


def _ds_360_to_BPC(vec_360):
    """1D (360,) → (BPC,) by mean over consecutive chunks (mirrors DSR
    line 399: `.reshape(N_CONDS_PER_CONF, 360/N_CONDS_PER_CONF).mean(axis=1)`)."""
    return np.asarray(vec_360, dtype=float).reshape(
        BINS_PER_CONFIG, 360 // BINS_PER_CONFIG).mean(axis=1)


def _stack_pop_per_config(acc_per_cfg, n_cfg):
    """List of n_cells × (BPC,) per config → (n_cells, n_cfg*BPC) hstack."""
    arrays = []
    for c in range(n_cfg):
        a = np.asarray(acc_per_cfg.get(c, []))
        if a.size == 0:
            return None
        arrays.append(a)
    return np.hstack(arrays)


def _stack_pop_per_cfg_th(acc_per_cfg_th, n_cfg):
    """List of cells × (BPC,) per (config, run-half) → (n_cells, 2*n_cfg*BPC)
    hstacked in (th=1 then th=2, configs in order) — same layout as
    RSA_DSR_ROIs_simple.py:431-443."""
    arrays = []
    for th in (1, 2):
        for c in range(n_cfg):
            a = np.asarray(acc_per_cfg_th.get(c, {}).get(th, []))
            if a.size == 0:
                return None
            arrays.append(a)
    return np.hstack(arrays)


def _build_data_rdms_from_pop(pop_emp_all_mat, pop_emp_th_mat, n_cfg):
    """Build all four flat data RDMs from the (n_cells × N) and
    (n_cells × 2N) population matrices. All four use canonical
    `mc.analyse.my_RSA.compute_crosscorr` / `compute_crosscorr_within`."""
    flat = {}
    # split_halves: pop_emp_th is (n_cells, 2N). Transpose → (2N, n_cells).
    mat_split = pop_emp_th_mat.T
    flat['split_halves'] = np.asarray(mc.analyse.my_RSA.compute_crosscorr(
        mat_split, plotting=False, include_diagonal=False,
        no_tasks=n_cfg, model='data_split_halves')[0], dtype=float)
    mat_split_z = _zscore_neurons(mat_split)
    flat['split_halves_z'] = np.asarray(mc.analyse.my_RSA.compute_crosscorr(
        mat_split_z, plotting=False, include_diagonal=False,
        no_tasks=n_cfg, model='data_split_halves_z')[0], dtype=float)
    # between_tasks: pop_emp_all is (n_cells, N). Transpose → (N, n_cells).
    mat_between = pop_emp_all_mat.T
    _, between, _ = mc.analyse.my_RSA.compute_crosscorr_within(
        mat_between, plotting=False, include_diagonal=False,
        no_tasks=n_cfg, model='data_between_tasks',
        block_size=BINS_PER_CONFIG)
    flat['between_tasks'] = np.asarray(between[0], dtype=float)
    mat_between_z = _zscore_neurons(mat_between)
    _, between_z, _ = mc.analyse.my_RSA.compute_crosscorr_within(
        mat_between_z, plotting=False, include_diagonal=False,
        no_tasks=n_cfg, model='data_between_tasks_z',
        block_size=BINS_PER_CONFIG)
    flat['between_tasks_z'] = np.asarray(between_z[0], dtype=float)
    return flat


def _run_branch_dsr_style(branch, subjects, cfg_idx_per_sub, n_cfg,
                            model_rdms, model_specs):
    """Single ROI-outer loop that mirrors RSA_DSR_ROIs_simple.py:313-422.

    For each ROI:
      - For each subject, for each config, for each neuron filtered by
        `get_neuron_roi(neuron) == roi`:
          - Pull raw (n_trials, 360).
          - Apply per-trial circular shifts (one shift per trial, per
            perm), mean across trials, downsample 360 → BPC.
          - Append the resulting (BPC,) into per-(config × perm)
            accumulators.
      - Empirical: same chain but without shifts.
      - After all subjects/cells: hstack the per-config (n_cells, BPC)
        matrices to build the empirical + per-perm population matrices,
        pass to canonical compute_* RDM functions, fit OLS.
    """
    rng = np.random.default_rng(seed=42)
    results_rows = []
    nulls = {}
    rois = [r for r in ROI_ORDER
            if r in ('mPFC', 'HC_anterior', 'HC_mid', 'EC',
                      'PHC', 'PCC', 'mOFC')]

    for roi_name in rois:
        # Containers (DSR convention).
        acc_neurons_all = {c: [] for c in range(n_cfg)}
        acc_neurons     = {c: {1: [], 2: []} for c in range(n_cfg)}
        perm_ACC_neurons_all = {c: [[] for _ in range(N_PERMUTATIONS)]
                                  for c in range(n_cfg)}
        perm_ACC_neurons     = {c: {1: [[] for _ in range(N_PERMUTATIONS)],
                                      2: [[] for _ in range(N_PERMUTATIONS)]}
                                  for c in range(n_cfg)}
        cells_seen = 0
        n_bins = 360

        for sub_str in subjects:
            if sub_str not in SUBJECT_DATA:
                continue
            curr_neurons = SUBJECT_DATA[sub_str][f'sub-{sub_str}'][
                'normalised_neurons']
            sub_cfg_idx = cfg_idx_per_sub.get(sub_str, {})

            for c_idx in range(n_cfg):
                trials = sub_cfg_idx.get(c_idx)
                if not trials or not trials['all']:
                    continue
                idx_all = trials['all']
                idx_th1 = trials['th1'] or idx_all
                idx_th2 = trials['th2'] or idx_all

                for n_lab in curr_neurons:
                    if get_neuron_roi(n_lab) != roi_name:
                        continue
                    _stash_electrode(electrode_coords[branch][roi_name], n_lab)
                    cells_seen += 1

                    conf_neuron = curr_neurons[n_lab].iloc[idx_all].to_numpy()
                    th1_neuron  = curr_neurons[n_lab].iloc[idx_th1].to_numpy()
                    th2_neuron  = curr_neurons[n_lab].iloc[idx_th2].to_numpy()

                    # Empirical: mean across trials → downsample.
                    acc_neurons_all[c_idx].append(
                        _ds_360_to_BPC(np.nanmean(conf_neuron, axis=0)))
                    acc_neurons[c_idx][1].append(
                        _ds_360_to_BPC(np.nanmean(th1_neuron, axis=0)))
                    acc_neurons[c_idx][2].append(
                        _ds_360_to_BPC(np.nanmean(th2_neuron, axis=0)))

                    # Permutations: per-trial circular shift on raw 360
                    # data (DSR line 384-409), then avg + downsample.
                    for p in range(N_PERMUTATIONS):
                        # "all" trials
                        sh = rng.integers(0, n_bins, size=conf_neuron.shape[0])
                        ni = (np.arange(n_bins) - sh[:, None]) % n_bins
                        pn = np.take_along_axis(conf_neuron, ni, axis=1)
                        perm_ACC_neurons_all[c_idx][p].append(
                            _ds_360_to_BPC(np.nanmean(pn, axis=0)))
                        # th1
                        sh1 = rng.integers(0, n_bins, size=th1_neuron.shape[0])
                        ni1 = (np.arange(n_bins) - sh1[:, None]) % n_bins
                        pn1 = np.take_along_axis(th1_neuron, ni1, axis=1)
                        perm_ACC_neurons[c_idx][1][p].append(
                            _ds_360_to_BPC(np.nanmean(pn1, axis=0)))
                        # th2
                        sh2 = rng.integers(0, n_bins, size=th2_neuron.shape[0])
                        ni2 = (np.arange(n_bins) - sh2[:, None]) % n_bins
                        pn2 = np.take_along_axis(th2_neuron, ni2, axis=1)
                        perm_ACC_neurons[c_idx][2][p].append(
                            _ds_360_to_BPC(np.nanmean(pn2, axis=0)))

        if cells_seen == 0:
            print(f"  [{branch}/{roi_name}] no cells — skipping.")
            continue

        pop_emp_all_mat = _stack_pop_per_config(acc_neurons_all, n_cfg)
        pop_emp_th_mat  = _stack_pop_per_cfg_th(acc_neurons,     n_cfg)
        if pop_emp_all_mat is None or pop_emp_th_mat is None:
            print(f"  [{branch}/{roi_name}] empty per-config slot — skipping.")
            continue
        n_neurons_roi = pop_emp_all_mat.shape[0]
        print(f"  [{branch}/{roi_name}] {n_neurons_roi} cells "
              f"(per-config slots filled across {len(set(subjects))} sessions)")

        # Empirical data RDMs (4 variants) + figure cache.
        data_rdms = _build_data_rdms_from_pop(
            pop_emp_all_mat, pop_emp_th_mat, n_cfg)
        data_rdms_by_roi[branch][roi_name] = dict(data_rdms)
        N_total = n_cfg * BINS_PER_CONFIG
        square_rdms_by_roi[branch][roi_name] = {
            'split_halves':    _inflate_split_halves(
                data_rdms['split_halves'], N_total),
            'split_halves_z':  _inflate_split_halves(
                data_rdms['split_halves_z'], N_total),
            'between_tasks':   _inflate_between_tasks(
                data_rdms['between_tasks'], N_total, BINS_PER_CONFIG),
            'between_tasks_z': _inflate_between_tasks(
                data_rdms['between_tasks_z'], N_total, BINS_PER_CONFIG),
        }

        # Empirical fits.
        for variant in TESTS:
            data_rdm = data_rdms[variant]
            for label, sub_models in model_specs:
                if label == 'state_only':
                    model_rdm = model_rdms[variant]['state']
                    t, beta, p_param = fit_single_model(model_rdm, data_rdm)
                    results_rows.append({
                        'branch': branch, 'roi': roi_name,
                        'n_neurons': n_neurons_roi, 'test': variant,
                        'combo': 'state_only', 'sub_model': 'state',
                        't': t, 'beta': beta, 'p_param': p_param,
                    })
                else:
                    combo_emp = fit_combo_model(
                        model_rdms[variant], sub_models, data_rdm)
                    for m, (t, beta, p_param) in combo_emp.items():
                        results_rows.append({
                            'branch': branch, 'roi': roi_name,
                            'n_neurons': n_neurons_roi, 'test': variant,
                            'combo': label, 'sub_model': m,
                            't': t, 'beta': beta, 'p_param': p_param,
                        })

        # Per-perm fits — precompute pinv(X) once per (variant, label).
        pinvs = {}
        for variant in TESTS:
            for label, sub_models in model_specs:
                if label == 'state_only':
                    X_cols = model_rdms[variant]['state'][:, None]
                else:
                    X_cols = np.stack(
                        [model_rdms[variant][m] for m in sub_models], axis=1)
                X_std, keep, _, _ = _prep_ols_design(X_cols)
                pinvs[(variant, label)] = (X_std, np.linalg.pinv(X_std), keep)

        # Pre-allocate per-perm null arrays.
        null_alloc = {}
        for variant in TESTS:
            for label, sub_models in model_specs:
                if label == 'state_only':
                    null_alloc[(roi_name, variant, label, 'state')] = \
                        np.empty(N_PERMUTATIONS, dtype=np.float32)
                else:
                    for m in sub_models:
                        null_alloc[(roi_name, variant, label, m)] = \
                            np.empty(N_PERMUTATIONS, dtype=np.float32)

        for p_i in range(N_PERMUTATIONS):
            perm_pop_all = _stack_pop_per_config(
                {c: perm_ACC_neurons_all[c][p_i] for c in range(n_cfg)}, n_cfg)
            perm_pop_th  = _stack_pop_per_cfg_th(
                {c: {1: perm_ACC_neurons[c][1][p_i],
                     2: perm_ACC_neurons[c][2][p_i]}
                 for c in range(n_cfg)}, n_cfg)
            if perm_pop_all is None or perm_pop_th is None:
                continue
            perm_data_rdms = _build_data_rdms_from_pop(
                perm_pop_all, perm_pop_th, n_cfg)
            for variant in TESTS:
                y = perm_data_rdms[variant]
                for label, sub_models in model_specs:
                    X_std, pinv_X, keep = pinvs[(variant, label)]
                    y_std = _prep_ols_y(y, keep)
                    betas = (pinv_X @ y_std)[1:]
                    if label == 'state_only':
                        null_alloc[(roi_name, variant, label,
                                     'state')][p_i] = float(betas[0])
                    else:
                        for j, m in enumerate(sub_models):
                            null_alloc[(roi_name, variant, label,
                                         m)][p_i] = float(betas[j])
            if (p_i + 1) % 200 == 0:
                print(f"    [{branch}/{roi_name}] perm "
                      f"{p_i + 1}/{N_PERMUTATIONS}")

        nulls.update(null_alloc)

    return results_rows, nulls


if RELOAD_RUN is None:
    # Run both branches.
    print("\n========== Branch 1: all_cells (state-only) ==========")
    emp_rows_all, null_all_dict = _run_branch_dsr_style(
        'all_cells', ALL_CELLS_SUBJECTS, pseudo_cfg_idx, N_GROUPS_ALL,
        model_rdms_all, [('state_only', None)])

    print("\n========== Branch 2: rsa_cells (combos) ==========")
    emp_rows_rsa, null_rsa_dict = _run_branch_dsr_style(
        'rsa_cells', RSA_CELLS_SUBJECTS, rsa_cfg_idx, N_CONFIGS_RSA,
        model_rdms_rsa, list(RSA_COMBOS.items()))

    emp_results_all = emp_rows_all
    emp_results_rsa = emp_rows_rsa
    null_all = null_all_dict
    null_rsa = null_rsa_dict


    # ── Combine empirical β/t/p with null β to produce p_perm + z_perm.
    # `emp_results_all` / `emp_results_rsa` are now lists of dicts (one row
    # per (roi, test, combo, sub_model)) coming from the DSR-style loop.
    for row in emp_results_all:
        key = (row['roi'], row['test'], 'state_only', row['sub_model'])
        null = null_all.get(key)
        if null is None:
            continue
        beta = row['beta']
        p_perm = (int(np.sum(null >= beta)) + 1) / (N_PERMUTATIONS + 1)
        z_perm = float((beta - null.mean()) / null.std()) \
            if null.std() > 0 else np.nan
        perm_nulls['all_cells'][(row['roi'], row['test'])] = null
        results_rows.append({**row, 'p_perm': p_perm, 'z_perm': z_perm})

    for row in emp_results_rsa:
        key = (row['roi'], row['test'], row['combo'], row['sub_model'])
        null = null_rsa.get(key)
        if null is None:
            continue
        beta = row['beta']
        p_perm = (int(np.sum(null >= beta)) + 1) / (N_PERMUTATIONS + 1)
        z_perm = float((beta - null.mean()) / null.std()) \
            if null.std() > 0 else np.nan
        perm_nulls['rsa_cells'][
            (row['roi'], row['combo'], row['test'], row['sub_model'])] = null
        results_combo_rows.append({**row, 'p_perm': p_perm, 'z_perm': z_perm})


    # ══════════════════════════════════════════════════════════════════════
    # ── BH-FDR family for `state` ─────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    results_df       = pd.DataFrame(results_rows)
    results_combo_df = pd.DataFrame(results_combo_rows)

    # All-cells branch: BH-FDR over the 7 substantive ROIs at PRIMARY_TEST.
    for df_ref, fam_mask in [
        (results_df,
         (results_df['test'] == PRIMARY_TEST)
         & (results_df['roi'].isin(FDR_ROIS))),
        (results_combo_df,
         (results_combo_df['test']      == PRIMARY_TEST)
         & (results_combo_df['combo']   == FDR_COMBO)
         & (results_combo_df['sub_model'] == FDR_SUBMODEL)
         & (results_combo_df['roi'].isin(FDR_ROIS))),
    ]:
        df_ref['in_fdr_family'] = False
        df_ref.loc[fam_mask, 'in_fdr_family'] = True
        df_ref['p_fdr'] = np.nan
        if fam_mask.any():
            df_ref.loc[fam_mask, 'p_fdr'] = bh_fdr(
                df_ref.loc[fam_mask, 'p_perm'].to_numpy())


    # Save CSVs.
    results_df.to_csv(os.path.join(OUT_DIR, 'results_summary.csv'), index=False)
    results_combo_df.to_csv(
        os.path.join(OUT_DIR, 'results_summary_combos.csv'), index=False)
    print(f"\nSaved CSVs: results_summary.csv, results_summary_combos.csv")

    # Confirmatory FDR table.
    fam_all = results_df[results_df['in_fdr_family']].copy().sort_values('p_fdr')
    fam_combo = results_combo_df[results_combo_df['in_fdr_family']].copy().sort_values('p_fdr')
    fam_all.to_csv(os.path.join(OUT_DIR, 'confirmatory_fdr_all_cells.csv'),
                   index=False)
    fam_combo.to_csv(os.path.join(OUT_DIR, 'confirmatory_fdr_rsa_cells.csv'),
                     index=False)

    # Save perm nulls (npz; one entry per key).
    all_cells_nulls = {f'{roi}__{test}': v
                       for (roi, test), v in perm_nulls['all_cells'].items()}
    np.savez(os.path.join(OUT_DIR, 'perm_nulls_all_cells.npz'),
             **all_cells_nulls)
    rsa_cells_nulls = {f'{roi}__{combo}__{test}__{sm}': v
                       for (roi, combo, test, sm), v
                       in perm_nulls['rsa_cells'].items()}
    np.savez(os.path.join(OUT_DIR, 'perm_nulls_rsa_cells.npz'),
             **rsa_cells_nulls)
    print("Saved permutation nulls.")

    # Save data RDMs per ROI (flat vectors; square inflated on reload).
    data_rdm_dict = {}
    for branch in ('all_cells', 'rsa_cells'):
        for roi, vmap in data_rdms_by_roi[branch].items():
            for variant, vec in vmap.items():
                data_rdm_dict[f'{branch}__{roi}__{variant}'] = np.asarray(vec)
    if data_rdm_dict:
        np.savez(os.path.join(OUT_DIR, 'data_rdms_by_roi.npz'), **data_rdm_dict)
        print(f"Saved data RDMs ({len(data_rdm_dict)} entries) "
              "→ data_rdms_by_roi.npz")

    # Save electrode coords (used by F4 glass-brain).
    electrode_save = {b: {r: {f'{s}-{i}': list(map(float, c))
                              for (s, i), c in m.items()}
                          for r, m in roi_map.items()}
                      for b, roi_map in electrode_coords.items()}
    with open(os.path.join(OUT_DIR, 'electrode_coords.json'), 'w') as f:
        json.dump(electrode_save, f, indent=2)
    print("Saved electrode coords → electrode_coords.json")

    # Save model RDMs (so reload mode can skip the mode-paths rebuild).
    model_rdm_dict = {}
    for variant, sub in model_rdms_all.items():
        for name, vec in sub.items():
            model_rdm_dict[f'all__{variant}__{name}'] = np.asarray(vec)
    for variant, sub in model_rdms_rsa.items():
        for name, vec in sub.items():
            model_rdm_dict[f'rsa__{variant}__{name}'] = np.asarray(vec)
    np.savez(os.path.join(OUT_DIR, 'model_rdms.npz'), **model_rdm_dict)
    print(f"Saved model RDMs ({len(model_rdm_dict)} entries) → model_rdms.npz")

    # Pretty-print the confirmatory family.
    print("\n=== Confirmatory FDR family — all_cells × state ===")
    print(fam_all[['roi', 'n_neurons', 'beta', 't', 'z_perm', 'p_perm', 'p_fdr']]
          .to_string(index=False))
    print("\n=== Confirmatory FDR family — rsa_cells × state (combo "
          f"{FDR_COMBO}) ===")
    print(fam_combo[['roi', 'n_neurons', 'beta', 't', 'z_perm', 'p_perm', 'p_fdr']]
          .to_string(index=False))
else:
    # ══════════════════════════════════════════════════════════════════
    # ── RELOAD MODE: read all artifacts from a prior run folder ───────
    # ══════════════════════════════════════════════════════════════════
    print(f"\n========== RELOAD MODE — loading artifacts from {OUT_DIR} ==========")

    # Results CSVs.
    results_df       = pd.read_csv(os.path.join(OUT_DIR, 'results_summary.csv'))
    results_combo_df = pd.read_csv(
        os.path.join(OUT_DIR, 'results_summary_combos.csv'))
    fam_all   = results_df[results_df.get('in_fdr_family', False) == True]\
                    .copy().sort_values('p_fdr')
    fam_combo = results_combo_df[results_combo_df.get('in_fdr_family', False) == True]\
                    .copy().sort_values('p_fdr')
    print(f"  loaded results: {len(results_df)} all-cells rows, "
          f"{len(results_combo_df)} rsa-cells rows")

    # Perm nulls.
    p_all_path = os.path.join(OUT_DIR, 'perm_nulls_all_cells.npz')
    p_rsa_path = os.path.join(OUT_DIR, 'perm_nulls_rsa_cells.npz')
    if os.path.isfile(p_all_path):
        _npz = np.load(p_all_path)
        for k in _npz.files:
            roi, test = k.split('__')
            perm_nulls['all_cells'][(roi, test)] = _npz[k]
    if os.path.isfile(p_rsa_path):
        _npz = np.load(p_rsa_path)
        for k in _npz.files:
            roi, combo, test, sm = k.split('__')
            perm_nulls['rsa_cells'][(roi, combo, test, sm)] = _npz[k]
    print(f"  loaded perm nulls: all_cells={len(perm_nulls['all_cells'])}, "
          f"rsa_cells={len(perm_nulls['rsa_cells'])}")

    # Data RDMs (+ inflate to square for figures).
    rdm_path = os.path.join(OUT_DIR, 'data_rdms_by_roi.npz')
    if os.path.isfile(rdm_path):
        _rdm = np.load(rdm_path)
        for k in _rdm.files:
            branch, roi, variant = k.split('__', 2)
            data_rdms_by_roi[branch].setdefault(roi, {})[variant] = _rdm[k]
        for branch in ('all_cells', 'rsa_cells'):
            n_cfg = N_GROUPS_ALL if branch == 'all_cells' else N_CONFIGS_RSA
            N_total = n_cfg * BINS_PER_CONFIG
            for roi, vmap in data_rdms_by_roi[branch].items():
                sq = {}
                for variant, vec in vmap.items():
                    if variant.startswith('split_halves'):
                        sq[variant] = _inflate_split_halves(vec, N_total)
                    else:
                        sq[variant] = _inflate_between_tasks(
                            vec, N_total, BINS_PER_CONFIG)
                square_rdms_by_roi[branch][roi] = sq
        n_combos = sum(len(v) for v in data_rdms_by_roi.values())
        print(f"  loaded data RDMs for {n_combos} ROI×branch combos.")
    else:
        print(f"  WARNING: {rdm_path} missing — F1 data panels will be empty.")

    # Electrode coords.
    ec_path = os.path.join(OUT_DIR, 'electrode_coords.json')
    if os.path.isfile(ec_path):
        with open(ec_path) as f:
            _ec = json.load(f)
        for b in ('all_cells', 'rsa_cells'):
            for r, m in _ec.get(b, {}).items():
                electrode_coords[b].setdefault(r, {})
                for k, v in m.items():
                    sub, idx = k.split('-')
                    electrode_coords[b][r][(int(sub), int(idx))] = tuple(v)
        n_el = sum(len(m) for b in electrode_coords
                   for m in electrode_coords[b].values())
        print(f"  loaded {n_el} electrode coords.")
    else:
        print(f"  WARNING: {ec_path} missing — F4 glass-brain will be empty.")

    # Re-print confirmatory tables so the reload output mirrors a fresh run.
    print("\n=== Confirmatory FDR family — all_cells × state ===")
    print(fam_all[['roi', 'n_neurons', 'beta', 't', 'z_perm', 'p_perm', 'p_fdr']]
          .to_string(index=False))
    print("\n=== Confirmatory FDR family — rsa_cells × state (combo "
          f"{FDR_COMBO}) ===")
    print(fam_combo[['roi', 'n_neurons', 'beta', 't', 'z_perm', 'p_perm', 'p_fdr']]
          .to_string(index=False))
    print("\nReload complete — jumping to figures.\n")

print(f"\nAll outputs under: {OUT_DIR}")


# ══════════════════════════════════════════════════════════════════════
# ── Figures (Stage 3) ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Showgirl2 palette colours (subset used here).
COL_HIST_BG  = '#BC7E6A'
COL_HIST_SIG = '#9A383C'
COL_BAR_BG   = '#7191A9'
COL_BAR_SIG  = '#2B4159'
COL_ZERO     = '0.25'


# import pdb; pdb.set_trace()

def _model_rdm_square_for_plot(X, n_cfg, model_name):
    """Build the square N×N model RDM for plotting by re-using the
    canonical functions: compute the flat between_tasks RDM via
    `_model_rdm_canonical(...)`, then inflate it back into a square with
    NaN within-config blocks (so the F1 model panel looks like the data
    panel)."""
    flat_between = _model_rdm_canonical(
        X, 'between_tasks', n_cfg, model_name)
    N = n_cfg * BINS_PER_CONFIG
    return _inflate_between_tasks(flat_between, N, BINS_PER_CONFIG)


def _imshow_rdm(ax, sq, variant, n_cfg, label_kind, title=None,
                  vmin=None, vmax=None, cbar=True, fontsize=7):
    """Plot a square RDM with: lower triangle masked, within-config blocks
    masked for between_tasks variants, config-block separator lines, and
    per-config axis ticks (pseudo-task numbers OR config labels)."""
    N = n_cfg * BINS_PER_CONFIG
    mat = sq.astype(float).copy()
    # Hide strict lower triangle (keeps the diagonal visible-but-white where
    # the cosine RDM evaluates to 0).
    li, lj = np.tril_indices(N, k=-1)
    mat[li, lj] = np.nan
    if variant.startswith('between_tasks'):
        # Mask within-config blocks too — these aren't regressed.
        block_idx = np.arange(N) // BINS_PER_CONFIG
        same = block_idx[:, None] == block_idx[None, :]
        mat[same] = np.nan
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(mat), 99)) if np.isfinite(mat).any() else 1.0
    if vmin is None:
        vmin = -vmax
    im = ax.imshow(mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal',
                    interpolation='nearest')
    # Block separators.
    for k in range(1, n_cfg):
        ax.axhline(k * BINS_PER_CONFIG - 0.5, color='black', lw=0.4, alpha=0.5)
        ax.axvline(k * BINS_PER_CONFIG - 0.5, color='black', lw=0.4, alpha=0.5)
    tick_pos = [c * BINS_PER_CONFIG + BINS_PER_CONFIG / 2.0 - 0.5
                for c in range(n_cfg)]
    if label_kind == 'pseudo':
        labels = [f'task {c + 1}' for c in range(n_cfg)]
    else:
        labels = DSR_CONFIG_LABELS
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize - 1)
    ax.set_yticklabels(labels, fontsize=fontsize - 1)
    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize - 1)
    return im


# ── F1: model RDM + per-ROI data RDM gallery (both branches × both variants)
def render_F1():
    for branch, model_design_builder, model_keys, n_cfg, label_kind in [
        ('all_cells',
         lambda: build_designs_all_cells_state_only(),
         ['state'], N_GROUPS_ALL, 'pseudo'),
        ('rsa_cells',
         lambda: build_designs_rsa_cells(),
         None, N_CONFIGS_RSA, 'config'),
    ]:
        designs = model_design_builder()
        if model_keys is None:
            model_keys = list(designs.keys())
        rois_present = list(square_rdms_by_roi[branch].keys())

        for test in TESTS:
            # Build square model RDMs for plotting via the canonical RDM
            # functions (inflated back into a square so the F1 model
            # panel matches the data panel masking convention).
            model_squares = {}
            for m in model_keys:
                X = designs[m]
                model_squares[m] = _model_rdm_square_for_plot(X, n_cfg, m)

            n_models = len(model_keys)
            n_panels = n_models + len(rois_present)
            ncols = 4
            nrows = int(np.ceil(n_panels / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 3.0, nrows * 3.1),
                                     squeeze=False, constrained_layout=True)
            p = 0
            # Model RDMs first.
            for m in model_keys:
                ax = axes[p // ncols, p % ncols]
                _imshow_rdm(ax, model_squares[m], test, n_cfg, label_kind,
                              title=f'model: {m}', fontsize=7)
                p += 1
            # Per-ROI data RDMs, annotated with n_neurons.
            for roi in rois_present:
                ax = axes[p // ncols, p % ncols]
                sq = square_rdms_by_roi[branch][roi][test]
                df_ref = (results_df if branch == 'all_cells'
                          else results_combo_df)
                n_row = df_ref[(df_ref['roi'] == roi)
                                 & (df_ref['test'] == test)]
                n_neur = int(n_row['n_neurons'].iloc[0]) if not n_row.empty else 0
                _imshow_rdm(ax, sq, test, n_cfg, label_kind,
                              title=f'{roi}  (n = {n_neur})', fontsize=7)
                p += 1
            for k in range(p, nrows * ncols):
                axes[k // ncols, k % ncols].axis('off')
            fig.suptitle(f'F1 — {branch} × {test}\n'
                          f'upper triangle only; within-config blocks masked '
                          f'for between_tasks',
                          fontsize=10)
            stem = os.path.join(FIG_DIR, f'F1_{branch}_{test}')
            fig.savefig(stem + '.pdf', bbox_inches='tight')
            fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  wrote {stem}.pdf/.png")


# ── F2: EC permutation histogram (all_cells × state × primary variant)
def render_F2():
    key = ('EC', PRIMARY_TEST)
    if key not in perm_nulls['all_cells']:
        print(f"  F2: no perm null for EC × {PRIMARY_TEST} — skipped.")
        return
    null = perm_nulls['all_cells'][key]
    row = results_df[(results_df['roi'] == 'EC')
                     & (results_df['test'] == PRIMARY_TEST)].iloc[0]
    emp_beta = float(row['beta'])
    p_fdr = float(row['p_fdr']) if np.isfinite(row['p_fdr']) else float('nan')
    sig = ('***' if p_fdr < 0.001 else '**' if p_fdr < 0.01
           else '*' if p_fdr < 0.05 else 'n.s.')

    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    finite = np.concatenate([null, [emp_beta]])
    lim = max(0.02, 1.1 * np.nanmax(np.abs(finite)))
    bins = np.linspace(-lim, lim, 30)
    ax.hist(null, bins=bins, color=COL_HIST_BG, edgecolor='white',
            linewidth=0.4, density=True, label='permutation null')
    ax.axvline(0, color=COL_ZERO, lw=1.0, ls='--')
    ax.axvline(emp_beta, color=COL_HIST_SIG, lw=2.0,
               label=f'empirical β = {emp_beta:+.3f}')
    ax.text(0.97, 0.95, sig, transform=ax.transAxes,
            ha='right', va='top', fontsize=12)
    ax.set_xlabel('β (state)')
    ax.set_ylabel('density')
    ax.set_title(f'F2 — EC × state ({PRIMARY_TEST})\n'
                 f'n_neurons={int(row["n_neurons"])}, '
                 f'q_FDR = {p_fdr:.2g}',
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    stem = os.path.join(FIG_DIR, 'F2_EC_perm_hist')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ── F3: cross-ROI heatmap (state vs l2_norm, primary combo, primary variant)
def render_F3():
    sub = results_combo_df[
        (results_combo_df['test'] == PRIMARY_TEST)
        & (results_combo_df['combo'] == FDR_COMBO)
        & (results_combo_df['sub_model'].isin(['state', 'l2_norm']))
    ].copy()
    rois = [r for r in ROI_ORDER if r in set(sub['roi'])]
    models = ['state', 'l2_norm']
    t_mat = np.full((len(rois), len(models)), np.nan)
    p_mat = np.full_like(t_mat, np.nan)
    n_per_roi = {}
    for i, roi in enumerate(rois):
        row_set = sub[sub['roi'] == roi]
        if not row_set.empty:
            n_per_roi[roi] = int(row_set['n_neurons'].max())
        for j, m in enumerate(models):
            rr = row_set[row_set['sub_model'] == m]
            if rr.empty:
                continue
            t_mat[i, j] = float(rr['t'].iloc[0])
            # Use p_perm directly; FDR was only applied to 'state' family.
            p_mat[i, j] = float(rr['p_perm'].iloc[0])
    # Apply BH-FDR per model column for visual highlight.
    fdr_mat = np.full_like(p_mat, np.nan)
    for j in range(len(models)):
        fdr_mat[:, j] = bh_fdr(p_mat[:, j])

    fig, ax = plt.subplots(figsize=(2.8, 0.45 * len(rois) + 1.3),
                            constrained_layout=True)
    vmax = float(np.nanmax(np.abs(t_mat))) if np.isfinite(t_mat).any() else 1
    im = ax.imshow(t_mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    for i in range(len(rois)):
        for j in range(len(models)):
            q = fdr_mat[i, j]
            if np.isfinite(q) and q < FDR_ALPHA:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor='black',
                                            linewidth=1.5))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois, fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(f'F3 — rsa_cells × {FDR_COMBO}\n{PRIMARY_TEST}',
                 fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label('t-stat', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    stem = os.path.join(FIG_DIR, 'F3_heatmap_state_vs_l2_norm')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ── F4: electrode glass-brain — each electrode coloured by the t-stat of
#        its ROI for the chosen (combo, sub_model). Mimics the DSR script's
#        approach of "projecting the heatmap onto the area where the cells
#        were from" rather than colouring by ROI identity.
def render_F4(branch='rsa_cells', combo=None, sub_model='state',
               test=None, save_stem=None):
    try:
        from nilearn import plotting as nl_plotting
    except Exception as e:
        print(f"  F4: nilearn not available — skipped ({e}).")
        return
    if test is None:
        test = PRIMARY_TEST
    if combo is None and branch == 'rsa_cells':
        combo = FDR_COMBO

    # Build per-ROI t-stat map.
    if branch == 'all_cells':
        sub = results_df[(results_df['test'] == test)
                         & (results_df['sub_model'] == sub_model)]
    else:
        sub = results_combo_df[
            (results_combo_df['test'] == test)
            & (results_combo_df['combo'] == combo)
            & (results_combo_df['sub_model'] == sub_model)
        ]
    roi_to_t = {row['roi']: float(row['t']) for _, row in sub.iterrows()}
    if not roi_to_t:
        print(f"  F4 ({branch}/{combo}/{sub_model}): no rows — skipped.")
        return

    # Symmetric colour scale around 0.
    tmax = max(abs(v) for v in roi_to_t.values())
    norm = mc_colors_Normalize = (
        plt.matplotlib.colors.Normalize(vmin=-tmax, vmax=tmax))
    cmap = plt.cm.RdBu_r

    coords = []
    colours = []
    sizes   = []
    for roi, mp in electrode_coords[branch].items():
        if roi not in roi_to_t:
            continue
        t_val = roi_to_t[roi]
        col = cmap(norm(t_val))
        sz  = 12 + 14 * (abs(t_val) / tmax if tmax > 0 else 0)
        for _, xyz in mp.items():
            coords.append(xyz)
            colours.append(col)
            sizes.append(sz)
    if not coords:
        print(f"  F4 ({branch}/{combo}/{sub_model}): no MNI coords — skipped.")
        return

    title = (f'F4 — electrodes coloured by t({sub_model}); '
             f'{branch}{" × " + combo if combo else ""}, {test}')
    display = nl_plotting.plot_glass_brain(
        None, display_mode='lyrz', title=title,
        black_bg=False, plot_abs=False)
    display.add_markers(np.asarray(coords), marker_color=colours,
                        marker_size=sizes)
    fig = plt.gcf()
    # Add a colorbar.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, fraction=0.02, pad=0.04, shrink=0.6)
    cb.set_label(f't({sub_model})', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if save_stem is None:
        suffix = f'{branch}_{sub_model}'
        if combo:
            suffix += f'_{combo}'
        save_stem = os.path.join(FIG_DIR, f'F4_glassbrain_{suffix}')
    fig.savefig(save_stem + '.pdf', bbox_inches='tight')
    fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {save_stem}.pdf/.png")


# ── F5: combo overview — one figure per combo (heatmap of ROIs × sub-models
#        + a row of permutation-null histograms for the 'state' sub-model
#        per ROI). For proof-checking the combo analyses.
def render_F5_combo_overview():
    for combo_name, combo_list in RSA_COMBOS.items():
        sub = results_combo_df[
            (results_combo_df['test'] == PRIMARY_TEST)
            & (results_combo_df['combo'] == combo_name)
        ]
        rois_here = [r for r in ROI_ORDER if r in set(sub['roi'])]
        if not rois_here:
            continue
        sub_models = combo_list
        # Heatmap matrix.
        t_mat = np.full((len(rois_here), len(sub_models)), np.nan)
        p_mat = np.full_like(t_mat, np.nan)
        b_mat = np.full_like(t_mat, np.nan)
        for i, roi in enumerate(rois_here):
            rs = sub[sub['roi'] == roi]
            for j, m in enumerate(sub_models):
                rr = rs[rs['sub_model'] == m]
                if rr.empty:
                    continue
                t_mat[i, j] = float(rr['t'].iloc[0])
                p_mat[i, j] = float(rr['p_perm'].iloc[0])
                b_mat[i, j] = float(rr['beta'].iloc[0])
        # FDR per column for proof-checking visualisation.
        fdr_mat = np.full_like(p_mat, np.nan)
        for j in range(len(sub_models)):
            fdr_mat[:, j] = bh_fdr(p_mat[:, j])

        n_rois_h = len(rois_here)
        # Heatmap column width and histogram row both scale with
        # max(n_sub_models, n_rois_h) so the largest combo doesn't squash.
        fig_w = max(8.5, 1.2 * n_rois_h + 0.6 * len(sub_models) + 3)
        fig_h = 0.55 * len(rois_here) + 0.5 * len(sub_models) + 3.5
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        gs = fig.add_gridspec(2, n_rois_h, height_ratios=[1.7, 1.0])
        # Top panel (spans all columns): the heatmap.
        ax_hm = fig.add_subplot(gs[0, :])
        vmax = float(np.nanmax(np.abs(t_mat))) if np.isfinite(t_mat).any() else 1
        im = ax_hm.imshow(t_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            aspect='auto')
        for i in range(len(rois_here)):
            for j in range(len(sub_models)):
                q = fdr_mat[i, j]
                if np.isfinite(q) and q < FDR_ALPHA:
                    ax_hm.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor='black', linewidth=1.4))
                if np.isfinite(t_mat[i, j]):
                    ax_hm.text(j, i, f'{t_mat[i, j]:.1f}',
                                 ha='center', va='center', fontsize=7,
                                 color=('white' if abs(t_mat[i, j]) > 0.6 * vmax
                                        else 'black'))
        ax_hm.set_xticks(range(len(sub_models)))
        ax_hm.set_xticklabels(sub_models, rotation=30, ha='right',
                                fontsize=8)
        ax_hm.set_yticks(range(len(rois_here)))
        ax_hm.set_yticklabels([
            f'{r} (n={int(sub[sub["roi"] == r]["n_neurons"].max())})'
            for r in rois_here], fontsize=8)
        ax_hm.set_title(f'rsa_cells × {combo_name} × {PRIMARY_TEST}\n'
                          f'numbers = t-stat; black box = BH-FDR<{FDR_ALPHA} '
                          f'across ROIs within this column',
                          fontsize=9)
        cb = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.02)
        cb.set_label('t-stat', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        # Bottom row: perm-null histograms for the 'state' sub-model per ROI.
        if 'state' in sub_models:
            for j, roi in enumerate(rois_here):
                ax = fig.add_subplot(gs[1, j])
                key = (roi, combo_name, PRIMARY_TEST, 'state')
                null = perm_nulls['rsa_cells'].get(key)
                emp_b = None
                rr = sub[(sub['roi'] == roi)
                         & (sub['sub_model'] == 'state')]
                if not rr.empty:
                    emp_b = float(rr['beta'].iloc[0])
                if null is None or emp_b is None:
                    ax.axis('off'); continue
                lim = max(0.02, 1.1 * np.nanmax(np.abs(
                    np.concatenate([null, [emp_b]]))))
                ax.hist(null, bins=24, color=COL_HIST_BG,
                          edgecolor='white', linewidth=0.3, density=True)
                ax.axvline(0, color=COL_ZERO, lw=0.7, ls='--')
                ax.axvline(emp_b, color=COL_HIST_SIG, lw=1.5)
                p_perm = float(rr['p_perm'].iloc[0])
                z = ((emp_b - null.mean()) / null.std()
                     if null.std() > 0 else float('nan'))
                ax.set_title(f'{roi}\nβ={emp_b:.3f}  z={z:.1f}\n'
                             f'p_perm={p_perm:.3f}',
                             fontsize=7)
                ax.set_xlim(-lim, lim)
                ax.set_xticks([-lim, 0, lim])
                ax.set_xticklabels([f'{-lim:.2f}', '0', f'{lim:.2f}'],
                                     fontsize=6)
                ax.set_yticks([])
                ax.spines[['top', 'right', 'left']].set_visible(False)
        stem = os.path.join(FIG_DIR, f'F5_combo_overview_{combo_name}')
        fig.savefig(stem + '.pdf', bbox_inches='tight')
        fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote {stem}.pdf/.png")


# ── F6: state-only heatmap for the all_cells branch (single-column).
#   Mirrors F3's heatmap style, but for the single-model all_cells fit.
def render_F6_all_cells_state_heatmap():
    sub = results_df[(results_df['test'] == PRIMARY_TEST)
                     & (results_df['sub_model'] == 'state')]
    rois = [r for r in ROI_ORDER if r in set(sub['roi'])]
    if not rois:
        print("  F6: no rows — skipped.")
        return
    t_vals = np.array([float(sub[sub['roi'] == r]['t'].iloc[0]) for r in rois])
    b_vals = np.array([float(sub[sub['roi'] == r]['beta'].iloc[0]) for r in rois])
    z_vals = np.array([float(sub[sub['roi'] == r]['z_perm'].iloc[0]) for r in rois])
    p_fdr = np.array([float(sub[sub['roi'] == r]['p_fdr'].iloc[0]) for r in rois])
    n_per = [int(sub[sub['roi'] == r]['n_neurons'].iloc[0]) for r in rois]

    fig, ax = plt.subplots(figsize=(2.6, 0.5 * len(rois) + 1.4),
                            constrained_layout=True)
    vmax = float(np.nanmax(np.abs(t_vals))) if np.isfinite(t_vals).any() else 1
    im = ax.imshow(t_vals[:, None], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    aspect='auto')
    for i in range(len(rois)):
        if np.isfinite(p_fdr[i]) and p_fdr[i] < FDR_ALPHA:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor='black', linewidth=1.5))
        ax.text(0, i, f'{t_vals[i]:.1f}', ha='center', va='center',
                fontsize=8,
                color=('white' if abs(t_vals[i]) > 0.6 * vmax else 'black'))
    ax.set_xticks([0])
    ax.set_xticklabels(['state'], fontsize=9)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels([f'{r} (n={n})' for r, n in zip(rois, n_per)],
                        fontsize=8)
    ax.set_title(f'F6 — all_cells × state\n{PRIMARY_TEST}; black box = '
                  f'BH-FDR < {FDR_ALPHA}', fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.06)
    cb.set_label('t-stat', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    stem = os.path.join(FIG_DIR, 'F6_all_cells_state_heatmap')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ── F7: per-ROI permutation null vs empirical (all_cells × state).
#   Extends F2 (EC only) to all ROIs in one multi-panel figure.
def render_F7_all_cells_perm_panel():
    sub = results_df[(results_df['test'] == PRIMARY_TEST)
                     & (results_df['sub_model'] == 'state')]
    rois = [r for r in ROI_ORDER if r in set(sub['roi'])]
    if not rois:
        print("  F7: no rows — skipped.")
        return
    n = len(rois)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.6, nrows * 2.2),
                              squeeze=False, constrained_layout=True)
    for i, roi in enumerate(rois):
        ax = axes[i // ncols, i % ncols]
        null = perm_nulls['all_cells'].get((roi, PRIMARY_TEST))
        row = sub[sub['roi'] == roi].iloc[0]
        emp_b = float(row['beta'])
        if null is None:
            ax.axis('off'); continue
        lim = max(0.02, 1.1 * np.nanmax(np.abs(
            np.concatenate([null, [emp_b]]))))
        bins = np.linspace(-lim, lim, 28)
        ax.hist(null, bins=bins, color=COL_HIST_BG, edgecolor='white',
                linewidth=0.3, density=True, label='perm null')
        ax.axvline(0, color=COL_ZERO, lw=0.8, ls='--')
        ax.axvline(emp_b, color=COL_HIST_SIG, lw=1.8,
                   label=f'empirical β = {emp_b:+.3f}')
        z = float(row['z_perm'])
        p_fdr = float(row['p_fdr'])
        p_perm = float(row['p_perm'])
        sig = ('***' if p_fdr < 0.001 else '**' if p_fdr < 0.01
               else '*' if p_fdr < 0.05 else 'n.s.')
        ax.text(0.97, 0.95, sig, transform=ax.transAxes,
                ha='right', va='top', fontsize=11)
        ax.set_title(f'{roi}  (n = {int(row["n_neurons"])})\n'
                     f'β={emp_b:+.3f}  z={z:.1f}  '
                     f'p_perm={p_perm:.3f}  q={p_fdr:.3g}',
                     fontsize=7.5)
        ax.set_xlim(-lim, lim)
        ax.set_xticks([-lim, 0, lim])
        ax.set_xticklabels([f'{-lim:.2f}', '0', f'{lim:.2f}'], fontsize=6)
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper left')
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')
    fig.suptitle(f'F7 — all_cells × state × {PRIMARY_TEST}: '
                  f'permutation null vs empirical β, per ROI', fontsize=10)
    stem = os.path.join(FIG_DIR, 'F7_all_cells_perm_panel')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


print("\n========== Stage 3: Figures ==========")
render_F1()
render_F2()
render_F3()
render_F4(branch='rsa_cells', combo=FDR_COMBO, sub_model='state')
render_F4(branch='all_cells', combo=None, sub_model='state',
            save_stem=os.path.join(FIG_DIR, 'F4_glassbrain_all_cells_state'))
render_F5_combo_overview()
render_F6_all_cells_state_heatmap()
render_F7_all_cells_perm_panel()


# ══════════════════════════════════════════════════════════════════════
# ── Final stats JSON (matches DSR script schema) ──────────────────────
# ══════════════════════════════════════════════════════════════════════

stats_archive = {
    'meta': {
        'run_tag':            RUN_TAG,
        'timestamp':          datetime.now().isoformat(timespec='seconds'),
        'data_dir':           DATA_DIR,
        'out_dir':            OUT_DIR,
        'fdr_test':           PRIMARY_TEST,
        'fdr_combo':          FDR_COMBO,
        'fdr_submodel':       FDR_SUBMODEL,
        'fdr_alpha':          FDR_ALPHA,
        'fdr_rois':           FDR_ROIS,
        'n_permutations':     N_PERMUTATIONS,
        'tests':              TESTS,
        'rsa_combos':         RSA_COMBOS,
        'roi_label_column':   ROI_LABEL_COLUMN,
        'roi_order':          ROI_ORDER,
        'all_cells_subjects': ALL_CELLS_SUBJECTS,
        'rsa_cells_subjects': RSA_CELLS_SUBJECTS,
    },
    'all_cells':  results_df.to_dict(orient='records'),
    'rsa_cells':  results_combo_df.to_dict(orient='records'),
    'confirmatory_fdr_all_cells':  fam_all.to_dict(orient='records'),
    'confirmatory_fdr_rsa_cells':  fam_combo.to_dict(orient='records'),
}
json_path = os.path.join(OUT_DIR, 'state_rsa_results.json')
with open(json_path, 'w') as f:
    json.dump(stats_archive, f, indent=2,
              default=lambda o: float(o) if hasattr(o, 'item') else str(o))
print(f"\nWrote {json_path}")
print(f"Final outputs under: {OUT_DIR}")
