#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:52:57 2026

@author: Svenja Kuchenhoff

RSA for DSR. 
attempt to simplify - now looping over ROIs.

final analysis that i'm using for all

"""


import os
import sys
import io
import json
import contextlib
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import mc
import mc.plotting.dsr_figures as dsr_figs   # shared rodent/human pub figures
from collections import Counter

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')

# import pdb; pdb.set_trace()

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR     = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE     = os.path.join(DATA_DIR, 'group', 'DSR_RSA_simple_ROI')

# Reload mode: set to the run tag of a previous run (e.g.
# '2026-05-18_16-33-05') to skip the heavy RSA + permutation loop and
# just re-render the overview plots from the saved
# results_summary*.csv files in OUT_BASE/<RELOAD_RUN>/.  None = run fresh.
RELOAD_RUN = None  #'2026-05-26_17-54-23' 

# import pdb; pdb.set_trace()
configs = [
    '3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
    '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6',
]


N_CONFIGS = len(configs)
N_CONDS_PER_CONF = 12
LEN_STANDARDISED_PATH = 12
N_PHASES = 3
states           = ['A', 'B', 'C', 'D']
RESOLUTIONx = 1
PLOT_FIGS = False
N_PERMUTATIONS = 100 #None #1000 # 500 # None or 300
SPLIT_UNCV_BUTTONS = True

# Phase-based masking. Phase of a condition at position pos inside a config is
# (pos % N_CONDS_PER_CONF) % N_PHASES; with N_PHASES=3 this cycles
# early–middle–late across the 12 conds per config.
#   'full'         -> no masking (all pairs)
#   'within_phase' -> keep only same-phase pairs (e.g. early ↔ early)
#   'across_phase' -> keep only different-phase pairs (e.g. early ↔ middle)
# Drives the *primary* RSA pipeline (used by FDR, summary tables, glassbrains,
# permutations, pub figures).
PHASE_MASK_MODE = 'full'

# If True, ALSO run the empirical (no-permutation) RSA in all three modes
# and produce comparison heatmaps of the combo betas side-by-side per ROI.
# Cheap; uses the model + data RDMs already built. Does not change the primary
# results.
RUN_PHASE_MODE_COMPARISON = True

assert PHASE_MASK_MODE in ('full', 'within_phase', 'across_phase'), PHASE_MASK_MODE

PLOT_GLASSBRAINS = False
# ── Models / combos to evaluate per ROI this round ────────────────────
# All model RDMs are built each run (cheap). These lists only restrict the
# *expensive* per-ROI evaluation + permutation step.
# - `models`: base models evaluated per ROI. Use `[]` to skip single-model RSA.
# - `combo_models`: combos evaluated per ROI. Sub-models are pulled from the
#   always-built model_RDMs, so combos may reference any model regardless of
#   what's in `models`.
models = ['dsr_old', 'dsr_fmri','state','midnight', 'bttn_curr', 'bttn_next', 'location', 'phase', 'repeat_counter', 'uncover']
#models = ['dsr_old', 'midnight','state', 'repeat_counter', 'uncover', 'bttn_prev', 'bttn_next', 'bttn_curr', 'location', 'phase']

combo_models = {
    'fMRI_midnight_state':          ['bttn_curr', 'bttn_next', 'location', 'state', 'midnight', 'dsr_fmri'],
    'fMRI_state':          ['bttn_curr', 'bttn_next', 'location', 'state', 'dsr_fmri'],
    'fMRIold_state':          ['bttn_curr', 'bttn_next', 'location', 'state', 'dsr_old'],
    'fMRI':          ['bttn_curr', 'bttn_next', 'location','dsr_fmri'],
    'fMRI_phase_rep':          ['bttn_curr', 'bttn_next', 'location','dsr_fmri', 'phase', 'repeat_counter'],
    'fMRI_phase_rep_uncvr':          ['bttn_curr', 'bttn_next', 'location','dsr_fmri', 'phase', 'repeat_counter', 'uncover'],
    }


# combo_models = {
#     'st-cnt-uncvr-bttns-loc-dsr':          ['state', 'repeat_counter', 'uncover', 'bttn_prev', 'bttn_next', 'bttn_curr', 'location', 'dsr_old'], #visual controls plus state plus buttons
#     'cnt-uncvr-bttns-loc-dsr':          ['repeat_counter', 'uncover', 'bttn_prev', 'bttn_next', 'bttn_curr', 'location', 'dsr_old'], #visual controls minus state plus buttons
#     'st-cnt-uncvr-loc-ph-dsr':          ['state', 'repeat_counter', 'uncover', 'location', 'phase', 'dsr_old'], #visual controls plus state and phase
#     'st-cnt-uncvr-loc-dsr':          ['state', 'repeat_counter', 'uncover', 'location', 'dsr_old'], #visual controls plus state
#     'cnt-loc-uncvr-dsr':          ['repeat_counter', 'uncover', 'location', 'dsr_old'], #visual controls minus state
#     'loc-uncvr-dsr':          ['location', 'uncover', 'dsr_old'], # only location uncover and dsr
#     'loc-ph-uncvr-dsr':          ['location', 'phase', 'uncover', 'dsr_old'], # only location uncover phase and dsr
#     'st-loc-cnt-uncvr-buttons-midn-dsr':          ['state', 'location', 'repeat_counter', 'uncover','bttn_prev', 'bttn_next', 'bttn_curr', 'midnight', 'dsr_old'], # visuals plus state plus midnight 
#     'loc-cnt-uncvr-buttons-midn-dsr':          ['location', 'repeat_counter', 'uncover','bttn_prev', 'bttn_next', 'bttn_curr', 'midnight', 'dsr_old'], # visuals minus state plus midnight 
# }

print(f"Base models evaluated this run ({len(models)}): {models}")
print(f"Combos evaluated this run     ({len(combo_models)}): {list(combo_models.keys())}")


# ── Multiple-comparison correction (confirmatory family) ─────────────
# BH-FDR is applied to ONE pre-specified family: the effect(s) of interest
# inside the core combo model(s), for the theory-chosen primary test
# variant, across every ROI tested.  Everything else (single-model RSA,
# control regressors, other combos, other test variants) stays
# uncorrected / exploratory.
#   FDR_COMBOS differ only by `state` -> their dsr_old betas are
#   correlated; set FDR_COMBOS to a single combo if you consider them
#   one hypothesis (a 10-test family rather than 20).
FDR_TEST      = 'between_tasks_z'   # primary variant: all-repeat-averaged
                                    # per-config, z-scored per neuron,
                                    # between-task-config cells only
                                    # (within-config block masked out).
# Confirmatory family: ONE primary combo × the effect of interest × all
# ROIs tested (≈ 7-9 tests). `MRI_combo-nofdb_midn` is treated as a
# robustness check rather than a second confirmatory test, since its
# `dsr_old` beta is highly correlated with the primary combo (the two
# differ only by the `state` regressor). This keeps the FDR family
# consistent with the publication panel (encoding_publication_panels.py).
FDR_COMBOS    = ['fMRI_midnight_state'] # usually 'MRI_combo-nofdb_midn-state'
FDR_SUBMODELS = ['dsr_fmri']  # dsr_fmri dsr_old effect(s) of interest within the combo
FDR_ALPHA     = 0.05


# import pdb; pdb.set_trace()
# ── ROI assignment from MNI-based table (cell_to_roi_MNI.py output) ───
# Each neuron is matched by (subject, cell_idx) parsed from its label.
ROI_TABLE_PATH = os.path.join(
    DATA_DIR, 'neurons_with_final_roi_labels.csv'
)

# Which ROI-label column of that table to use:
#   'final_roi'     -> original labelling
#   'alt_final_roi' -> alternative labelling (ACC split by y-cutoff,
#                      OFC11+OFC13+ventral_ACC collapsed into medialOFC)
ROI_LABEL_COLUMN = 'alt_final_roi' #'final_roi'

# Which ROI values to analyse this run — one list per labelling column.
ROIS_TO_ANALYZE_BY_COLUMN = {
    'final_roi': [
        'EC', 'Parahippocampal',
        'HC_anterior', 'HC_mid',
        'ventral_ACC', 'ACC',
        'posterior_CC',
        'OFC11', 'OFC13', 'Visual',
    ],
    'alt_final_roi': [
        'ACC', 'EC', 'Parahippocampal',
        'HC_anterior', 'HC_mid',
        'medialOFC', 'medial_CC',
        'PCC', 'Visual',
    ],
}
ROIS_TO_ANALYZE = ROIS_TO_ANALYZE_BY_COLUMN[ROI_LABEL_COLUMN]

# ROI that gets the shared rodent-style publication figures (fig 2 + fig 3)
# saved into ``OUT_DIR/pub_figures/``. Set to None to disable, or to another
# ROI name from ROIS_TO_ANALYZE to pick a different reference ROI.
EXAMPLE_ROI_FOR_FIGS = 'ACC'


def parse_neuron_label(label):
    """Parse '01_07-07-chan120-EC' into (sub:int, cell_idx:int)."""
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def _load_roi_table(path, roi_col):
    df = pd.read_csv(path)
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
print(f"Loaded ROI table with {len(ROI_TABLE)} cells "
      f"({ROI_TABLE[ROI_LABEL_COLUMN].nunique()} distinct ROIs) from "
      f"{ROI_TABLE_PATH}  [column: {ROI_LABEL_COLUMN}]")
# the ROI_LABEL_COLUMN column indicates the correct ROI to take.


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


def _make_roi_predicate(target_roi):
    def pred(label):
        return get_neuron_roi(label) == target_roi
    return pred


ROI_RULES = {roi: _make_roi_predicate(roi) for roi in ROIS_TO_ANALYZE}


def downsample_mode(x, target_len=10):
    """Mode-downsample x to ``target_len`` slots without discarding bins.

    Using ``block = len(x) // target_len`` truncated whenever target_len did
    not divide len(x): with len(x)=360 and target_len=144 (n_dsr_neurons or
    n_conds_per_config*len_per_bin) the last 72 raw bins were dropped and
    every cond's downsampled window was misaligned by 6 bins relative to
    the conceptual 30-bin layout. Similarly target_len=12 on a 30-bin
    subpath dropped 6 bins per cond.

    This version distributes input bins evenly across the output slots:
    slot i = x[(i*n)//target_len : ((i+1)*n)//target_len]. All bins are
    used; slot sizes differ by at most 1.
    """
    x = np.asarray(x, dtype=object)
    n = len(x)
    return np.array([
        Counter(x[(i * n) // target_len : ((i + 1) * n) // target_len])
            .most_common(1)[0][0]
        for i in range(target_len)
    ], dtype=object)


def make_phase_masks_for_cells(n_configs, n_conds_per_config, n_phases,
                                include_diagonal=False):
    """Phase-based boolean masks aligned with the cells RSA vector layouts.

    Phase of a condition at position ``pos`` inside a config is
    ``(pos % n_conds_per_config) % n_phases``; with N_PHASES=3 this cycles
    early–middle–late–early–middle–late–… across the 12 conds per config.

    Returns a dict ``{variant: {mode: 1d_bool_array}}`` plus a
    ``'_phase_per_condition'`` vector and a ``'_n'`` size, both for
    diagnostics.

      variants:
        'split_halves'   -> mask aligned with ``compute_crosscorr``'s output:
                            upper-tri (k=1 unless include_diagonal) of a
                            symmetrized cross-half N×N block, where
                            N = n_configs * n_conds_per_config.
        'between_tasks'  -> mask aligned with ``compute_crosscorr_within``'s
                            between-block output: upper-tri of the same N×N
                            RDM, with same-config blocks already excluded.

      modes:
        'full'         -> all True (no masking).
        'within_phase' -> True where the two conds share a phase.
        'across_phase' -> True where the two conds have different phases.
    """
    n = n_configs * n_conds_per_config
    phase = np.tile(np.arange(n_conds_per_config) % n_phases, n_configs)
    k = 0 if include_diagonal else 1
    ii, jj = np.triu_indices(n, k=k)
    same_phase = phase[ii] == phase[jj]
    between_block = (ii // n_conds_per_config) != (jj // n_conds_per_config)
    split_halves = {
        'full':         np.ones_like(same_phase, dtype=bool),
        'within_phase': same_phase,
        'across_phase': ~same_phase,
    }
    same_phase_bt = same_phase[between_block]
    between_tasks = {
        'full':         np.ones_like(same_phase_bt, dtype=bool),
        'within_phase': same_phase_bt,
        'across_phase': ~same_phase_bt,
    }
    return {
        'split_halves':         split_halves,
        'between_tasks':        between_tasks,
        '_phase_per_condition': phase,
        '_n':                   n,
    }


# Layout-only — does not depend on data, so build once.
PHASE_MASKS = make_phase_masks_for_cells(
    N_CONFIGS, N_CONDS_PER_CONF, N_PHASES, include_diagonal=False)

ALL_PHASE_MODES = ('full', 'within_phase', 'across_phase')


def _phase_mask_for(test_name, mode):
    """Mask vector for (test_name, mode). Returns None when no masking is needed.

    test_name may end in '_z' (the permutation loop uses suffixed names);
    that suffix is stripped for the variant lookup.
    """
    if mode == 'full':
        return None
    base = test_name[:-2] if test_name.endswith('_z') else test_name
    return PHASE_MASKS[base][mode]


def _apply_phase_mask(arr, mask):
    """Slice a 1-D RDM vector or a (n_pairs, n_models) stacked design matrix."""
    if mask is None:
        return arr
    a = np.asarray(arr)
    return a[mask] if a.ndim == 1 else a[mask, :]


def _phase_mask_matrix(mode, n_configs, n_conds_per_config, n_phases):
    """Square N×N boolean mask (True = kept) for visualising RDMs.

    Same definition as the 1-D vector masks but expanded to a full N×N matrix
    (no upper-tri restriction) so it can be applied to a square RDM image.
    """
    n = n_configs * n_conds_per_config
    phase = np.tile(np.arange(n_conds_per_config) % n_phases, n_configs)
    same_phase_mat = phase[:, None] == phase[None, :]
    if mode == 'full':
        return np.ones((n, n), dtype=bool)
    if mode == 'within_phase':
        return same_phase_mat
    if mode == 'across_phase':
        return ~same_phase_mat
    raise ValueError(f"unknown phase mode {mode!r}")


def build_mode_path_dsr(mode_vec, n_conds_per_config, len_per_bin):
    """fMRI-style DSR: per-bin mode trajectory, flattened and rolled by bin.

    Mirrors the construction in create_fMRI_model_RDMs_on_clean_beh.py
    (EVs['DSR']): take the mode trajectory, downsample to
    n_conds_per_config * len_per_bin integer location IDs, then for each
    bin roll the flattened vector left by ``pos * len_per_bin`` so the
    "current" bin sits at the front. Returned matrix feeds compute_hamming_distance.
    """
    base = downsample_mode(mode_vec, target_len=n_conds_per_config * len_per_bin)
    return np.stack([np.roll(base, -pos * len_per_bin)
                     for pos in range(n_conds_per_config)], axis=0)


def make_empty(rows, cols, dtype=float):
    return np.zeros((rows, cols), dtype=dtype)

def _scalar(arr):
    """Safely extract a Python float from a size-1 array or scalar."""
    return float(np.asarray(arr, dtype=float).ravel()[0])


def _degenerate(x, atol=1e-12):
    """A 1-D vector is degenerate if it has any non-finite entry or zero variance."""
    x = np.asarray(x, dtype=float).ravel()
    return (not np.isfinite(x).all()) or (np.nanvar(x) <= atol)


def eval_tuple(rdm, data_rdm, label=''):
    """Single-model RSA eval. Returns (t, beta, p) as plain floats.

    Returns (NaN, NaN, NaN) when the model or data vector is degenerate
    (constant after masking, contains NaN/inf, etc.) — phase-mask modes
    can make e.g. the 'phase' regressor constant; we surface that with a
    print rather than letting statsmodels error out.
    """
    if _degenerate(rdm):
        print(f"  [skip-single] {label}: model RDM is constant/non-finite "
              f"under current mask → NaN")
        return (float('nan'), float('nan'), float('nan'))
    if _degenerate(data_rdm):
        return (float('nan'), float('nan'), float('nan'))
    try:
        return tuple(_scalar(v) for v in mc.analyse.my_RSA.evaluate_model(
            rdm, data_rdm))
    except Exception as exc:
        print(f"  [skip-single] {label}: {exc!s} → NaN")
        return (float('nan'), float('nan'), float('nan'))


def build_combo_rdm(rdm_dict, combo_list):
    """Stack several model RDMs into one multi-model design matrix."""
    return np.stack([rdm_dict[m][0] for m in combo_list], axis=1)


def evaluate_combo_safe(stacked, data_vec, sub_models, label=''):
    """Combo RSA eval that drops degenerate regressor columns.

    Each column of ``stacked`` corresponds to one ``sub_models`` entry. Any
    column that is non-finite or zero-variance after masking is removed from
    the design before calling evaluate_model; its slot in the returned arrays
    is set to NaN so the caller still sees one (t, beta, p) per sub-model.

    Returns (t_arr, beta_arr, p_arr) as 1-D numpy arrays of length ``len(sub_models)``.
    """
    X = np.asarray(stacked, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_reg = X.shape[1]
    nan_arr = lambda: np.full(n_reg, np.nan, dtype=float)
    if _degenerate(data_vec):
        return nan_arr(), nan_arr(), nan_arr()

    finite_cols = np.isfinite(X).all(axis=0)
    var_cols = np.nanvar(X, axis=0) > 1e-12
    good = finite_cols & var_cols
    if not good.any():
        print(f"  [skip-combo] {label}: all {n_reg} regressors degenerate "
              f"under current mask → NaN")
        return nan_arr(), nan_arr(), nan_arr()
    if not good.all():
        dropped = [sub_models[i] for i in range(n_reg) if not good[i]]
        print(f"  [drop-combo] {label}: dropping {dropped} "
              f"(constant/non-finite under mask)")

    X_sub = X[:, good]
    try:
        t_sub, beta_sub, p_sub = mc.analyse.my_RSA.evaluate_model(X_sub, data_vec)
    except Exception as exc:
        print(f"  [skip-combo] {label}: {exc!s} → NaN")
        return nan_arr(), nan_arr(), nan_arr()

    t_full, beta_full, p_full = nan_arr(), nan_arr(), nan_arr()
    t_full[good] = np.asarray(t_sub, dtype=float).ravel()
    beta_full[good] = np.asarray(beta_sub, dtype=float).ravel()
    p_full[good] = np.asarray(p_sub, dtype=float).ravel()
    return t_full, beta_full, p_full


with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json'), 'r') as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())

# ── Per-run output folder (timestamped) + config dump ────────────────
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

    run_config = {
        'run_tag':              RUN_TAG,
        'timestamp':            datetime.now().isoformat(timespec='seconds'),
        'data_dir':             DATA_DIR,
        'out_dir':              OUT_DIR,
        'configs':              configs,
        'N_CONFIGS':            N_CONFIGS,
        'N_CONDS_PER_CONF':     N_CONDS_PER_CONF,
        'LEN_STANDARDISED_PATH': LEN_STANDARDISED_PATH,
        'N_PHASES':             N_PHASES,
        'states':               states,
        'RESOLUTIONx':          RESOLUTIONx,
        'N_PERMUTATIONS':       N_PERMUTATIONS,
        'models':               models,
        'combo_models':         combo_models,
        'roi_label_column':     ROI_LABEL_COLUMN,
        'fdr_test':             FDR_TEST,
        'fdr_combos':           FDR_COMBOS,
        'fdr_submodels':        FDR_SUBMODELS,
        'fdr_alpha':            FDR_ALPHA,
        'phase_mask_mode':      PHASE_MASK_MODE,
        'run_phase_mode_comparison': RUN_PHASE_MODE_COMPARISON,
        'rois':                 list(ROI_RULES.keys()),
    }
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"Run output: {OUT_DIR}")


if RELOAD_RUN is None:
    # Cache subject data once; loop ROIs over the cache afterwards.
    print("Loading subject data once for all ROIs...")
    SUBJECT_DATA = {}
    for sub_str in SUBJECTS:
        SUBJECT_DATA[sub_str] = mc.analyse.helpers_human_cells.load_norm_data(DATA_DIR, [sub_str])
    print(f"Cached data for {len(SUBJECT_DATA)} subjects.")


    # Containers for cross-ROI summary
    summary_rows = []
    summary_combo_rows = []
    roi_perm_results = {}
    roi_empirical_results = {}
    # mode -> roi -> test_name -> {raw_combo, z_combo, n_pairs_kept, n_pairs_total}
    # Populated only when RUN_PHASE_MODE_COMPARISON; used by the cross-ROI
    # phase-mode comparison plot at the end of the run.
    roi_mode_comparison = {}
    # Per-ROI electrode coordinates (deduplicated by (sub, cell_idx)) for the
    # schematic glass-brain shown alongside the overview heatmaps.
    roi_electrode_coords = {roi: {} for roi in ROI_RULES}

    # ── Phase-mask diagnostic (once, before ROI loop) ────────────────────
    # Print phase per condition + mask sizes per (variant, mode) so we can
    # verify which RDM cells are kept vs. excluded by each masking mode.
    # Also save a visual of the three mask matrices (96×96 each).
    print("\n=== Phase-mask diagnostic ===")
    print(f"Phase per condition (length {PHASE_MASKS['_n']}) — "
          f"{N_CONFIGS} configs × {N_CONDS_PER_CONF} conds each, "
          f"phase = (pos % {N_CONDS_PER_CONF}) % {N_PHASES} → cycles 0,1,2 = "
          f"early,middle,late within each state's sub-path.")
    _phase_vec = PHASE_MASKS['_phase_per_condition']
    for c_idx in range(N_CONFIGS):
        seg = _phase_vec[c_idx * N_CONDS_PER_CONF:(c_idx + 1) * N_CONDS_PER_CONF]
        print(f"  config {c_idx} ({configs[c_idx]:>8s}): {seg.tolist()}")
    print(f"\nMask sizes per RDM variant × mode "
          f"(kept = pairs entering evaluate_model):")
    print(f"  {'variant':<16s} {'mode':<14s} {'kept':>7s} / {'total':>7s} "
          f"{'(% kept)':>10s}")
    for _variant in ('split_halves', 'between_tasks'):
        for _mode in ALL_PHASE_MODES:
            _mask = PHASE_MASKS[_variant][_mode]
            _k, _t = int(_mask.sum()), int(_mask.size)
            print(f"  {_variant:<16s} {_mode:<14s} {_k:>7d} / {_t:>7d} "
                  f"{100.0*_k/_t:>9.1f}%")

    # Plot the three 96×96 mask matrices once.
    _N = N_CONFIGS * N_CONDS_PER_CONF
    fig_pm, ax_pm = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for _a, _mode in zip(ax_pm, ALL_PHASE_MODES):
        _M = _phase_mask_matrix(_mode, N_CONFIGS, N_CONDS_PER_CONF, N_PHASES)
        _a.imshow(_M.astype(int), cmap='Greys_r', vmin=0, vmax=1, aspect='equal')
        _a.set_title(f"mask: {_mode}\n(white = kept, black = excluded)",
                     fontsize=10)
        # Config boundaries (red) and phase boundaries (faint cyan).
        for c in range(1, N_CONFIGS):
            _a.axvline(c * N_CONDS_PER_CONF - 0.5, color='red', lw=0.7)
            _a.axhline(c * N_CONDS_PER_CONF - 0.5, color='red', lw=0.7)
        _a.set_xticks(np.arange(N_CONFIGS) * N_CONDS_PER_CONF + N_CONDS_PER_CONF / 2)
        _a.set_xticklabels([str(i) for i in range(N_CONFIGS)], fontsize=8)
        _a.set_yticks(np.arange(N_CONFIGS) * N_CONDS_PER_CONF + N_CONDS_PER_CONF / 2)
        _a.set_yticklabels([str(i) for i in range(N_CONFIGS)], fontsize=8)
        _a.set_xlabel('config (12 conds each)', fontsize=9)
    fig_pm.suptitle('Phase masks (96×96) — red lines = config boundaries',
                    fontsize=11)
    fig_pm.tight_layout()
    fig_pm.savefig(os.path.join(OUT_DIR, 'phase_mask_diagnostic.png'),
                   dpi=150, bbox_inches='tight')
    print(f"Saved phase-mask diagnostic figure to "
          f"{os.path.join(OUT_DIR, 'phase_mask_diagnostic.png')}")
    plt.show()
    print(f"\nPrimary pipeline runs with PHASE_MASK_MODE = '{PHASE_MASK_MODE}'.")
    print(f"RUN_PHASE_MODE_COMPARISON = {RUN_PHASE_MODE_COMPARISON} — "
          f"{'will produce cross-mode comparison heatmaps.' if RUN_PHASE_MODE_COMPARISON else 'comparison disabled.'}\n")


    for roi_name, roi_pred in ROI_RULES.items():
        print(f"\n========== ROI: {roi_name} ==========")

        # set up dicts and lists to load data
        acc_neurons, locs, buttons = {}, {}, {}
        acc_neurons_all, locs_all, buttons_all = {}, {}, {}
        perm_ACC_neurons_all, perm_ACC_neurons = {}, {}

        for conf in configs:
            acc_neurons[conf] = {}
            perm_ACC_neurons[conf]= {}
            locs[conf] = {}
            buttons[conf] = {}
            buttons_all[conf] = []
            acc_neurons_all[conf] = []
            perm_ACC_neurons_all[conf] = []
            locs_all[conf] = []
            for th in [1,2]:
                acc_neurons[conf][th] = []
                perm_ACC_neurons[conf][th] = []
                locs[conf][th] = []
                buttons[conf][th] = []


        N_CONFIGS = len(configs)


        for sub_str in SUBJECTS:
            data_dict = SUBJECT_DATA[sub_str]
            # figure out correct config indices

            beh = data_dict[f"sub-{sub_str}"]['beh'].copy().reset_index(drop=True)
            beh['config'] = list(zip(
                beh['loc_A'].astype(int), beh['loc_B'].astype(int),
                beh['loc_C'].astype(int), beh['loc_D'].astype(int),
            ))
            beh['grid_no']    = beh['grid_no'].astype(int)
            beh['config_str'] = beh['config'].apply(
                lambda t: f'{t[0]}-{t[1]}-{t[2]}-{t[3]}')
            curr_neurons = data_dict[f"sub-{sub_str}"]['normalised_neurons']
            
            for conf in configs:
                idx_curr_conf = (beh['config_str'] == conf) & (beh['correct'] == 1)
                all_locs_conf = data_dict[f"sub-{sub_str}"]['locations'][idx_curr_conf].to_numpy()
                # average 10 repeats, respectively
                locs[conf][1].append(all_locs_conf[0:10, :])
                locs[conf][2].append(all_locs_conf[10:20, :])
                locs_all[conf].append(all_locs_conf)
                
                all_buttons_conf = data_dict[f"sub-{sub_str}"]['buttons'][idx_curr_conf].to_numpy()
                
                buttons[conf][1].append(all_buttons_conf[0:10, :])
                buttons[conf][2].append(all_buttons_conf[10:20, :])
                buttons_all[conf].append(all_buttons_conf)
                
                for n_lab in curr_neurons:
                    # if 'MCC' in n_lab:
                    #     print(f"now adding hippocampal neuron with MCC label, in session {sub_str}.")
                    if roi_pred(n_lab):
                        # Record MNI coords (once per cell) for the ROI overview plot.
                        sub_int, cell_int = parse_neuron_label(n_lab)
                        
                        if sub_int is not None:
                            key = (sub_int, cell_int)
                            if key not in roi_electrode_coords[roi_name]:
                                mni = get_neuron_mni(n_lab)
                                if all(np.isfinite(mni)):
                                    roi_electrode_coords[roi_name][key] = mni
                        conf_acc_neurons = curr_neurons[n_lab][idx_curr_conf].to_numpy()


                        # this is where the permutations need to be
                        if N_PERMUTATIONS:

                            rng     = np.random.default_rng()
                            n_bins = conf_acc_neurons.shape[1]

                            for p in range(N_PERMUTATIONS):
                                # generates the shifts for all trials
                                shifts = rng.integers(0, n_bins, size=(conf_acc_neurons.shape[0]))
                                # creates new column indices
                                new_idx = (np.arange(n_bins) - shifts[:, None]) % n_bins
                                # takes values from og neuron along axis 1, using the shifted per-row indices.
                                perm_neuron = np.take_along_axis(conf_acc_neurons, new_idx, axis=1)

                                perm_avg_conf_all = np.nanmean(perm_neuron, axis = 0)
                                perm_avg_downsampled_all = perm_avg_conf_all.reshape(N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 1)
                                perm_ACC_neurons_all[conf].append(perm_avg_downsampled_all)

                                for th in [1,2]:
                                    if th == 1:
                                        start = 0
                                    elif th == 2:
                                        start = 10
                                    perm_avg_config = np.nanmean(perm_neuron[start:start+10, :], axis = 0)
                                    perm_avg_downsampled = perm_avg_config.reshape(N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 1)
                                    perm_ACC_neurons[conf][th].append(perm_avg_downsampled)


                        avg_conf_all = np.nanmean(conf_acc_neurons, axis = 0)
                        avg_downsampled_all = avg_conf_all.reshape(N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 1)
                        acc_neurons_all[conf].append(avg_downsampled_all)
                        for th in [1,2]:
                            if th == 1:
                                start = 0
                            elif th == 2:
                                start = 10
                            avg_config = np.nanmean(conf_acc_neurons[start:start+10, :], axis = 0)
                            avg_downsampled = avg_config.reshape(N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 1)
                            acc_neurons[conf][th].append(avg_downsampled)

        # import pdb; pdb.set_trace()

        n_neurons = len(acc_neurons[conf][th])
        
        if n_neurons == 0:
            print(f"[{roi_name}] no neurons matched — skipping.")
            continue

        print(f"[{roi_name}] {n_neurons} neurons collected.")

        rows = []
        row_labels = []

        # create the long neuron vector.
        for task_half in [1, 2]:
            for config in configs:
                neuron_values = acc_neurons[config][task_half]
                neuron_values = np.asarray(neuron_values)

                rows.append(neuron_values)
                row_labels.append((config, task_half))

        mat = np.hstack(rows)
        data_RDM = mc.analyse.my_RSA.compute_crosscorr(mat.T, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model=f'data in {roi_name}')

        # z-scored ACC neurons.
        mu = np.nanmean(mat, axis=1)      # one mean per neuron
        sd = np.nanstd(mat, axis=1)       # one std per neuron
        mat_z = (mat.T - mu) / sd
        data_RDM_z = mc.analyse.my_RSA.compute_crosscorr(mat_z, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model=f'data in z-scored {roi_name} neurons')



        row_all = []
        row_labels_all = []
        for config in configs:
            all_neuron_values = acc_neurons_all[config]
            row_all.append(all_neuron_values)
            row_labels_all.append(config)
        mat_all = np.hstack(row_all)

        data_RDM_within, data_RDM_across, data_RDM_full = mc.analyse.my_RSA.compute_crosscorr_within(mat_all.T, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model=f'data in {roi_name}', block_size=N_CONDS_PER_CONF)

        # z-scored ACC neurons.
        # z-scored ACC neurons.
        mu_all = np.nanmean(mat_all, axis=1)      # one mean per neuron
        sd_all = np.nanstd(mat_all, axis=1)       # one std per neuron
        mat_all_z = (mat_all.T - mu_all) / sd_all
        #data_RDM_within_z, data_RDM_across_z, data_RDM_full_z = mc.analyse.my_RSA.compute_crosscorr_within(mat_all_z, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model=f'data in z-scored {roi_name} neurons', block_size=N_CONDS_PER_CONF)
        data_RDM_within_z, data_RDM_across_z, data_RDM_full_z = mc.analyse.my_RSA.compute_crosscorr_within(mat_all_z, plotting=True, include_diagonal=False, no_tasks=len(configs), model=f'data in z-scored {roi_name} neurons', block_size=N_CONDS_PER_CONF)
        
        #import pdb; pdb.set_trace()


        # create a mode path.
        mode_locs, mode_locs_all, mode_buttons, mode_buttons_all = {}, {}, {}, {}
        for c in configs:
            mode_locs[c] = {}
            locs_all_per_conf = locs_all[c]
            stacked_all = np.vstack(locs_all_per_conf) # (n_trials_total, 360)
            m_all = stats.mode(stacked_all, axis=0, keepdims=False, nan_policy='omit')
            mode_locs_all[c] = m_all.mode.astype(float)

            mode_buttons[c] = {}
            buttons_all_per_conf = buttons_all[c]
            stacked_all_buttons = np.vstack(buttons_all_per_conf) # (n_trials_total, 360)
            b_m_all = stats.mode(stacked_all_buttons, axis=0, keepdims=False, nan_policy='omit')
            mode_buttons_all[c] = b_m_all.mode
            
            
            for th in [1, 2]:
                loc_per_config = locs[c][th]
                stacked = np.vstack(loc_per_config) # (n_trials_total, 360)
                m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
                mode_locs[c][th] = m.mode.astype(float)
                
                # and for buttons
                button_per_config = buttons[c][th]
                stacked_b = np.vstack(button_per_config) # (n_trials_total, 360)
                m_b = stats.mode(stacked_b, axis=0, keepdims=False, nan_policy='omit')
                mode_buttons[c][th] = m_b.mode


        print("Mode-location and mode-button arrays built.")

        # build model rdms

        print("Building model matrices...")
        
        n_conditions = N_CONFIGS * N_CONDS_PER_CONF
        n_dsr_neurons = LEN_STANDARDISED_PATH * N_PHASES * len(states)
        
        # Initialise matrices for each half
        matrices = {
            run_id: {
                'loc':     make_empty(n_conditions, LEN_STANDARDISED_PATH),
                'dsr':     make_empty(n_conditions, n_dsr_neurons),
                'dsr_fmri': make_empty(n_conditions, N_CONDS_PER_CONF * LEN_STANDARDISED_PATH),
                'bttn_curr': make_empty(n_conditions, LEN_STANDARDISED_PATH, dtype=object),
                'bttn_prev': make_empty(n_conditions, LEN_STANDARDISED_PATH, dtype=object),
                'bttn_next': make_empty(n_conditions, LEN_STANDARDISED_PATH, dtype=object),
            }
            for run_id in (1, 2)
        }
        
        for c_idx, c in enumerate(configs):
            row_start = c_idx * N_CONDS_PER_CONF
            for run_id, mats in matrices.items():
                mode_vec        = mode_locs[c][run_id]      # (360,)
                mode_vec_button = mode_buttons[c][run_id]
                LEN_OG_SUBPATH = int(len(mode_vec)/N_CONDS_PER_CONF)
        
                dsr_base = downsample_mode(mode_vec, target_len=n_dsr_neurons)

                # fMRI-style DSR: integer-ID mode trajectory, rolled per bin
                # (parallel to create_fMRI_model_RDMs_on_clean_beh.py / EVs['DSR']).
                # Same construct as 'dsr' above, just labelled separately so we
                # can compare the L=12-per-bin / fMRI-aligned wiring head-to-head.
                mats['dsr_fmri'][row_start:row_start + N_CONDS_PER_CONF] = (
                    build_mode_path_dsr(mode_vec, N_CONDS_PER_CONF, LEN_STANDARDISED_PATH))

                for n_subpath in range(N_CONDS_PER_CONF):
                    row = row_start + n_subpath
        
                    # --- location ---
                    subpath = mode_vec[n_subpath * LEN_OG_SUBPATH:(n_subpath + 1) * LEN_OG_SUBPATH]
                    mats['loc'][row] = downsample_mode(subpath, target_len=LEN_STANDARDISED_PATH)
        
                    # --- dsr ---
                    mats['dsr'][row] = np.roll(dsr_base, -n_subpath * LEN_STANDARDISED_PATH)
        
                    # --- buttons (current / previous / next), shift by ±1 subpath ---
                    # --- buttons (current / previous / next), wraparound by ±1 subpath ---
                    for key, offset in [('bttn_curr', 0), ('bttn_prev', -1), ('bttn_next', +1)]:
                        # import pdb; pdb.set_trace()
                        shifted_n = (n_subpath + offset) % N_CONDS_PER_CONF
                        s = shifted_n * LEN_OG_SUBPATH
                        mats[key][row] = downsample_mode(mode_vec_button[s : s + LEN_OG_SUBPATH], target_len=LEN_STANDARDISED_PATH)
                        

        # --- state / feedback / phase (unchanged logic) ---
        state_config    = np.zeros((N_CONDS_PER_CONF, len(states)))
        feedback_config = np.zeros((N_CONDS_PER_CONF, len(states)))
        phase_config    = np.zeros((N_CONDS_PER_CONF, N_PHASES))
        
        for s_i, s in enumerate(states):
            start_phase = RESOLUTIONx * s_i * N_PHASES
            state_config[start_phase : RESOLUTIONx * (s_i + 1) * N_PHASES, s_i] = 1
            if s == 'A':
                feedback_config[:RESOLUTIONx, s_i] = 1
            for p_i in range(N_PHASES):
                phase_config[start_phase + p_i * RESOLUTIONx : start_phase + (p_i + 1) * RESOLUTIONx, p_i] = 1
        
        state_half    = np.tile(state_config,    (len(configs), 1))
        feedback_half = np.tile(feedback_config, (len(configs), 1))
        phase_half    = np.tile(phase_config,    (len(configs), 1))
        
        model_concat = {
            'location':   np.concatenate([matrices[1]['loc'],     matrices[2]['loc']],     axis=0),
            'dsr':        np.concatenate([matrices[1]['dsr'],     matrices[2]['dsr']],     axis=0),
            'dsr_fmri':   np.concatenate([matrices[1]['dsr_fmri'], matrices[2]['dsr_fmri']], axis=0),
            'bttn_curr':    np.concatenate([matrices[1]['bttn_curr'], matrices[2]['bttn_curr']], axis=0),
            'bttn_prev':    np.concatenate([matrices[1]['bttn_prev'], matrices[2]['bttn_prev']], axis=0),
            'bttn_next':    np.concatenate([matrices[1]['bttn_next'], matrices[2]['bttn_next']], axis=0),
            'state':      np.tile(state_half, (2, 1)),
            'repeat_counter': np.tile(feedback_half, (2,1))
        }
        
        if SPLIT_UNCV_BUTTONS == True:
            # now, as an additional step, split 'return' buttons from buttons.
            # also replace/forward fill the locations that had 'return' with the
            # previoys buttons that were pressed.
            # model_concat['uncover'] = (model_concat['bttn_curr'] == "Return").astype(int)
            model_concat['uncover'] = np.where(model_concat['bttn_curr'] == "Return","uncover","off")
            for button_type in ['bttn_curr', 'bttn_prev', 'bttn_next']:
                arr = model_concat[button_type].copy().astype(object)
            
                # Replace Return with missing
                arr[arr == "Return"] = np.nan
                # Forward-fill normally; backward-fill only affects leading Returns
                filled = (pd.Series(arr.ravel()).ffill().bfill().to_numpy().reshape(arr.shape))
                model_concat[button_type] = filled
        
        # old way
        dsr_old, midnight, dsr_old_now_next, state_phase, phase, loc_old = [], [], [], [], [], []
        for th in [1,2]:
            for c in configs:
                walked   = mode_locs[c][th]
                walked = [int(w-1) for w in walked]
                loc_og_matrix, phase_og_matrix, stat_matrix, midnight_matrix, dsr_matrix, phas_stat_matrix, dsr_now_next_matrix = mc.simulation.predictions.model_DSR(locations = walked, no_phase_neurons=N_PHASES)
                
                loc_old_downsampled = loc_og_matrix.reshape(loc_og_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis=2)
                state_phase_downsampled = phas_stat_matrix.reshape(phas_stat_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis=2)
                dsr_downsampled = dsr_matrix.reshape(dsr_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis=2)
                midn_downsampled = midnight_matrix.reshape(midnight_matrix.shape[0],N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 2)
                dsr_nownext_downsampled = dsr_now_next_matrix.reshape(dsr_now_next_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 2)
                phase_downsampled = phase_og_matrix.reshape(phase_og_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 2)

                dsr_old.append(dsr_downsampled)
                midnight.append(midn_downsampled)
                dsr_old_now_next.append(dsr_nownext_downsampled)
                state_phase.append(state_phase_downsampled)
                phase.append(phase_downsampled)
                loc_old.append(loc_old_downsampled)
                


        model_concat['dsr_old'] = np.transpose(np.concatenate(dsr_old, axis = 1))
        model_concat['midnight'] = np.transpose(np.concatenate(midnight, axis = 1))
        model_concat['dsr_old_now_next'] = np.transpose(np.concatenate(dsr_old_now_next, axis = 1))
        model_concat['state_phase'] = np.transpose(np.concatenate(state_phase, axis = 1))
        model_concat['phase'] = np.transpose(np.concatenate(phase, axis = 1))
        model_concat['location_old'] = np.transpose(np.concatenate(loc_old, axis = 1))


        model_RDMs = {}
        model_RDMs_within = {}
        model_RDMs_across = {}
        full = {}

        for m in model_concat:

            if m in ('location', 'dsr', 'dsr_fmri', 'bttn_prev', 'bttn_next', 'bttn_curr', 'uncover'):
                model_RDMs[m] = mc.analyse.my_RSA.compute_hamming_distance(
                    model_concat[m], plotting=False, include_diagonal=False,
                    model_name=m, no_tasks=len(configs))

                model_RDMs_within[m], model_RDMs_across[m], full[m] = mc.analyse.my_RSA.compute_hamming_distance_within(
                    model_concat[m][0:len(mat_all_z)], plotting=PLOT_FIGS,
                    include_diagonal=False,
                    model_name=m, no_tasks=len(configs),
                    block_size=N_CONDS_PER_CONF)
            else:
                model_RDMs[m] = mc.analyse.my_RSA.compute_crosscorr(
                    model_concat[m], plotting=False, include_diagonal=False,
                    no_tasks=len(configs), model=m)
                model_RDMs_within[m], model_RDMs_across[m], full[m] = mc.analyse.my_RSA.compute_crosscorr_within(
                    model_concat[m][0:len(mat_all_z)], plotting=PLOT_FIGS,
                    include_diagonal=False,
                    no_tasks=len(configs), model=m,
                    block_size=N_CONDS_PER_CONF)


        # Repeat-counter rows are mostly zero, so the cosine RDM has NaN cells
        # wherever both endpoints are all-zero (norm=0). Replace with 1 (max
        # cosine dissimilarity) across ALL three RDM-vector dicts so combos
        # that include 'repeat_counter' don't blow up evaluate_model when run
        # on the between_tasks / within variants.
        for _rdm_dict in (model_RDMs, model_RDMs_within, model_RDMs_across):
            _vec = _rdm_dict['repeat_counter'][0]
            _vec[np.isnan(_vec)] = 1
        # Defensive guard for the vector RDMs that actually feed evaluate_model.
        # Any remaining NaN/inf in any model would crash statsmodels deep
        # inside evaluate_model; replace with 0 and report which model produced
        # them so we can investigate rather than silently fudge results.
        for _label, _rdm_dict in (('split_halves',  model_RDMs),
                                  ('within',        model_RDMs_within),
                                  ('between_tasks', model_RDMs_across)):
            for _m in _rdm_dict:
                _arr = np.asarray(_rdm_dict[_m][0], dtype=float)
                if not np.isfinite(_arr).all():
                    n_bad = int((~np.isfinite(_arr)).sum())
                    print(f"  [{roi_name}] {_label} / {_m}: {n_bad} non-finite "
                          f"entries -> replaced with 0")
                    _arr[~np.isfinite(_arr)] = 0
                    _rdm_dict[_m][0] = _arr

        # ── Publication figures 2 + 3 (shared with rodent pipeline) ────────
        # Built once for the example ROI: data + per-model activations and
        # RDMs across all 8 task configurations, using the mode-path models
        # (mode_locs_all, which already pools both halves). Mirrors the
        # rodent pipeline so a single change to the plotting helpers
        # propagates to both analyses.
        if roi_name == EXAMPLE_ROI_FOR_FIGS:
            DSR_MODEL_ORDER_HUMAN = ['dsr_old', 'state', 'location_old', 'phase']
            half1_slice = slice(0, N_CONFIGS * N_CONDS_PER_CONF)  # the across-blocks RDM is built from half-1
            fig_model_acts = {m: model_concat[m][half1_slice].T
                              for m in DSR_MODEL_ORDER_HUMAN}
            fig_model_rdms = {m: np.asarray(full[m], dtype=float).copy()
                              for m in DSR_MODEL_ORDER_HUMAN}
            fig_data_act   = mat_all_z.T      # (n_neurons, N_CONFIGS*N_CONDS_PER_CONF)
            fig_data_rdm   = np.asarray(data_RDM_full_z, dtype=float).copy()

            # CONFIRMATORY view: when the primary pipeline masks out pairs,
            # NaN those same cells in the pub figure RDMs so the figure shows
            # ONLY what entered evaluate_model. Diagonal blocks (within-config)
            # are already masked from the between-tasks variant, so for the
            # phase-mask we additionally NaN cross-phase OR within-phase cells
            # depending on the mode.
            if PHASE_MASK_MODE != 'full':
                _keep_phase = _phase_mask_matrix(
                    PHASE_MASK_MODE, N_CONFIGS, N_CONDS_PER_CONF, N_PHASES)
                _cfg_idx = np.repeat(np.arange(N_CONFIGS), N_CONDS_PER_CONF)
                _between_cfg = _cfg_idx[:, None] != _cfg_idx[None, :]
                _keep_for_fig = _keep_phase & _between_cfg
                fig_data_rdm = np.where(_keep_for_fig, fig_data_rdm, np.nan)
                for _m in DSR_MODEL_ORDER_HUMAN:
                    fig_model_rdms[_m] = np.where(_keep_for_fig,
                                                  fig_model_rdms[_m], np.nan)
                print(f"[pub fig] masked pub-figure RDMs to mode "
                      f"'{PHASE_MASK_MODE}' (between-config cells only).")

            figs_dir = os.path.join(OUT_DIR, 'pub_figures')
            os.makedirs(figs_dir, exist_ok=True)

            dsr_figs.pub_figure_example_subject(
                data_activation=fig_data_act, data_rdm=fig_data_rdm,
                model_activations=fig_model_acts, model_rdms=fig_model_rdms,
                model_order=DSR_MODEL_ORDER_HUMAN,
                n_tasks=N_CONFIGS, n_conds_per_task=N_CONDS_PER_CONF,
                recday_label=f'human cells / {roi_name} [{PHASE_MASK_MODE}]',
                save_stem=os.path.join(figs_dir,
                                       f'fig2_human_{roi_name}_{PHASE_MASK_MODE}'))

            # Fig 3 schematics: one example config from the mode trajectory.
            ex_config_str = configs[0]
            ex_walked = np.asarray(mode_locs_all[ex_config_str], dtype=int) - 1   # 0-indexed
            ex_walked = np.clip(ex_walked, 0, 8)
            ex_config_tuple = tuple(int(x) for x in ex_config_str.split('-'))
            dsr_figs.pub_figure_model_schematics(
                walked_path=ex_walked, task_config=ex_config_tuple,
                no_phase_neurons=N_PHASES,
                recday_label=f'human cells / config {ex_config_str}',
                save_stem=os.path.join(figs_dir, 'fig3_human_model_schematics'))

        # import pdb; pdb.set_trace()
        print("Computing RSA...")

        # ── Empirical results ────────────────────────────────────────────────
        empirical_results = {}
        empirical_results_z = {}

        empirical_combo_results = {}
        empirical_combo_results_z = {}


        test_specs = [
            # 'split_halves' = full off-diagonal RDM from a 2-half (10
            # reps each) population matrix; 'between_tasks' = repeats
            # pre-averaged per config, only off-block (between-task-
            # config) cells used.
            ('split_halves',  model_RDMs,        data_RDM[0],        data_RDM_z[0]),
            #('within',       model_RDMs_within, data_RDM_within[0], data_RDM_within_z[0]),
            ('between_tasks', model_RDMs_across, data_RDM_across[0], data_RDM_across_z[0]),
        ]

        for test_name, rdm_dict, raw_data, z_data in test_specs:
            pmask = _phase_mask_for(test_name, PHASE_MASK_MODE)
            raw_m = _apply_phase_mask(raw_data, pmask)
            z_m   = _apply_phase_mask(z_data, pmask)

            empirical_results[test_name] = {
                m: eval_tuple(_apply_phase_mask(rdm_dict[m][0], pmask), raw_m,
                              label=f'[{roi_name}] {PHASE_MASK_MODE}/{test_name}/{m}')
                for m in models
            }
            empirical_results_z[test_name] = {
                m: eval_tuple(_apply_phase_mask(rdm_dict[m][0], pmask), z_m,
                              label=f'[{roi_name}] {PHASE_MASK_MODE}/{test_name}_z/{m}')
                for m in models
            }


            empirical_combo_results[test_name] = {
                combo: evaluate_combo_safe(
                    _apply_phase_mask(build_combo_rdm(rdm_dict, combo_models[combo]), pmask),
                    raw_m, combo_models[combo],
                    label=f'[{roi_name}] {PHASE_MASK_MODE}/{test_name}/{combo}')
                for combo in combo_models}

            empirical_combo_results_z[test_name] = {
                combo: evaluate_combo_safe(
                    _apply_phase_mask(build_combo_rdm(rdm_dict, combo_models[combo]), pmask),
                    z_m, combo_models[combo],
                    label=f'[{roi_name}] {PHASE_MASK_MODE}/{test_name}_z/{combo}')
                for combo in combo_models}

        # ── Mode comparison: empirical-only RSA for all 3 phase modes ────
        # Cheap (no permutations) — uses the SAME model + data RDMs and just
        # applies a different mask per mode. Stored separately from the primary
        # results above and consumed by the cross-ROI comparison heatmap.
        if RUN_PHASE_MODE_COMPARISON:
            for mode in ALL_PHASE_MODES:
                roi_mode_comparison.setdefault(mode, {})[roi_name] = {}
                for test_name, rdm_dict, raw_data, z_data in test_specs:
                    pmask = _phase_mask_for(test_name, mode)
                    raw_m = _apply_phase_mask(raw_data, pmask)
                    z_m   = _apply_phase_mask(z_data, pmask)
                    roi_mode_comparison[mode][roi_name][test_name] = {
                        'raw_single': {
                            m: eval_tuple(
                                _apply_phase_mask(rdm_dict[m][0], pmask), raw_m,
                                label=f'[{roi_name}] {mode}/{test_name}/{m}')
                            for m in models},
                        'z_single': {
                            m: eval_tuple(
                                _apply_phase_mask(rdm_dict[m][0], pmask), z_m,
                                label=f'[{roi_name}] {mode}/{test_name}_z/{m}')
                            for m in models},
                        'raw_combo': {
                            combo: evaluate_combo_safe(
                                _apply_phase_mask(
                                    build_combo_rdm(rdm_dict, combo_models[combo]), pmask),
                                raw_m, combo_models[combo],
                                label=f'[{roi_name}] {mode}/{test_name}/{combo}')
                            for combo in combo_models},
                        'z_combo': {
                            combo: evaluate_combo_safe(
                                _apply_phase_mask(
                                    build_combo_rdm(rdm_dict, combo_models[combo]), pmask),
                                z_m, combo_models[combo],
                                label=f'[{roi_name}] {mode}/{test_name}_z/{combo}')
                            for combo in combo_models},
                        'n_pairs_kept': int(pmask.sum()) if pmask is not None else int(
                            np.asarray(raw_data).size),
                        'n_pairs_total': int(np.asarray(raw_data).size),
                    }



        #tests = ['split_halves', 'split_halves_z', 'within', 'within_z', 'between_tasks', 'between_tasks_z']
        tests = ['split_halves', 'split_halves_z', 'between_tasks', 'between_tasks_z']
        perm_results = {test: {m: [] for m in models} for test in tests}
        perm_results_combo = {
            test: {combo: {'t': [], 'beta': [], 'p': []} for combo in combo_models}
            for test in tests
        }


        if N_PERMUTATIONS:
            for perm_i in range(N_PERMUTATIONS):
                # print(f"\nComputing {N_PERMUTATIONS} circularly-shifted results ...")
                start_perm = n_neurons*perm_i
                end_perm = n_neurons*(perm_i+1)

                rows = []
                row_labels = []

                # create the long neuron vector.
                for task_half in [1, 2]:
                    for config in configs:
                        neuron_values = perm_ACC_neurons[config][task_half][start_perm:end_perm]
                        neuron_values = np.asarray(neuron_values)

                        rows.append(neuron_values)
                        row_labels.append((config, task_half))

                perm_mat = np.hstack(rows)
                perm_data_RDM = mc.analyse.my_RSA.compute_crosscorr(perm_mat.T, plotting=False, include_diagonal=False, no_tasks=len(configs), model=f'permuted data in {roi_name}')

                # z-scored ACC neurons.
                mu = np.nanmean(perm_mat, axis=1)      # one mean per neuron
                sd = np.nanstd(perm_mat, axis=1)       # one std per neuron
                perm_mat_z = (perm_mat.T - mu) / sd
                perm_data_RDM_z = mc.analyse.my_RSA.compute_crosscorr(perm_mat_z, plotting=False, include_diagonal=False, no_tasks=len(configs), model=f'permuted data in z-scored {roi_name} neurons')



                row_all = []
                row_labels_all = []
                for config in configs:
                    all_neuron_values = perm_ACC_neurons_all[config][start_perm:end_perm]
                    row_all.append(all_neuron_values)
                    row_labels_all.append(config)
                perm_mat_all = np.hstack(row_all)

                perm_data_RDM_within, perm_data_RDM_across, perm_data_RDM_full = mc.analyse.my_RSA.compute_crosscorr_within(perm_mat_all.T, plotting=False, include_diagonal=False, no_tasks=len(configs), model=f'data in {roi_name}', block_size=N_CONDS_PER_CONF)

                # z-scored ACC neurons.
                # z-scored ACC neurons.
                mu_all = np.nanmean(perm_mat_all, axis=1)      # one mean per neuron
                sd_all = np.nanstd(perm_mat_all, axis=1)       # one std per neuron
                perm_mat_all_z = (perm_mat_all.T - mu_all) / sd_all
                perm_data_RDM_within_z, perm_data_RDM_across_z, _perm_data_RDM_full_z = mc.analyse.my_RSA.compute_crosscorr_within(perm_mat_all_z, plotting=False, include_diagonal=False, no_tasks=len(configs), model=f'data in z-scored {roi_name} neurons', block_size=N_CONDS_PER_CONF)


                perm_specs = [
                    ('split_halves',    model_RDMs,        perm_data_RDM[0]),
                    ('split_halves_z',  model_RDMs,        perm_data_RDM_z[0]),
                    #('within',         model_RDMs_within, perm_data_RDM_within[0]),
                    #('within_z',       model_RDMs_within, perm_data_RDM_within_z[0]),
                    ('between_tasks',   model_RDMs_across, perm_data_RDM_across[0]),
                    ('between_tasks_z', model_RDMs_across, perm_data_RDM_across_z[0]),
                ]

                for test_name, rdm_dict, data_rdm in perm_specs:
                    pmask = _phase_mask_for(test_name, PHASE_MASK_MODE)
                    data_m = _apply_phase_mask(data_rdm, pmask)

                    for m in models:
                        beta = eval_tuple(
                            _apply_phase_mask(rdm_dict[m][0], pmask), data_m,
                            label=f'perm/{test_name}/{m}')[1]
                        perm_results[test_name][m].append(beta)

                    for combo, combo_list in combo_models.items():
                        stacked = _apply_phase_mask(
                            build_combo_rdm(rdm_dict, combo_list), pmask)
                        res = evaluate_combo_safe(
                            stacked, data_m, combo_list,
                            label=f'perm/{test_name}/{combo}')

                        perm_results_combo[test_name][combo]['t'].append(np.asarray(res[0], dtype=float).ravel())
                        perm_results_combo[test_name][combo]['beta'].append(np.asarray(res[1], dtype=float).ravel())
                        perm_results_combo[test_name][combo]['p'].append(np.asarray(res[2], dtype=float).ravel())

                if (perm_i + 1) % 25 == 0 or perm_i == 0:
                    print(f"  Permutation {perm_i + 1}/{N_PERMUTATIONS} done")



        def plot_perm_hist_grid(
            perm_results,
            empirical_results,
            empirical_results_z,
            #tests=('split_halves', 'split_halves_z', 'within', 'within_z', 'between_tasks', 'between_tasks_z'),
            tests=('split_halves', 'split_halves_z', 'between_tasks', 'between_tasks_z'),
            models=('location', 'dsr', 'state', 'dsr_old', 'midnight', 'dsr_old_now_next'),
            bins=25,
            density=True,
            figsize_per_panel=(2.0, 1.8),
            alpha=0.05,
            suptitle=None,
        ):
            nrows, ncols = len(tests), len(models)

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                sharey=False,
                constrained_layout=True,
            )

            # Force `axes` to a 2-D (nrows, ncols) array so axes[r, c] always works.
            axes = np.asarray(axes).reshape(nrows, ncols)

            for r, test in enumerate(tests):
                emp = empirical_results_z[test[:-2]] if test.endswith('_z') else empirical_results[test]

                # row-specific symmetric x-limits
                row_vals = []
                for model in models:
                    x = np.asarray(perm_results[test][model], dtype=float).ravel()
                    beta = float(np.asarray(emp[model][1]).ravel()[0])
                    row_vals.append(x)
                    row_vals.append(np.array([beta], dtype=float))

                row_vals = np.concatenate(row_vals)
                lim = np.nanmax(np.abs(row_vals))
                lim = 1.0 if (not np.isfinite(lim) or lim == 0) else 1.05 * lim
                edges = np.linspace(-lim, lim, bins + 1)

                for c, model in enumerate(models):
                    ax = axes[r, c]
                    x = np.asarray(perm_results[test][model], dtype=float).ravel()
                    beta = float(np.asarray(emp[model][1]).ravel()[0])

                    # one-sided positive permutation p-value
                    p_one_sided = (np.sum(x >= beta) + 1) / (x.size + 1)

                    ax.hist(
                        x, bins=edges, density=density,
                        color='0.75', edgecolor='white', linewidth=0.6
                    )
                    ax.axvline(0, color='black', lw=0.9)
                    ax.axvline(beta, color='red', lw=1.6)

                    if p_one_sided < 0.1:
                        ax.text(
                            0.04, 0.96, f"p={p_one_sided:.3f}",
                            transform=ax.transAxes,
                            ha='left', va='top',
                            fontsize=8
                        )

                    # add a fat star when significant and positive
                    if (beta > 0) and (p_one_sided < alpha):
                        y0, y1 = ax.get_ylim()
                        ax.set_ylim(y0, y1 * 1.15)
                        ax.text(
                            beta, y1 * 1.08, '★',
                            ha='center', va='bottom',
                            fontsize=16, fontweight='bold',
                            color='black'
                        )
                    else:
                        # still leave a little headroom for consistency
                        y0, y1 = ax.get_ylim()
                        ax.set_ylim(y0, y1 * 1.08)

                    ax.set_xlim(-lim, lim)
                    ax.tick_params(labelsize=8, length=2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    if r == 0:
                        ax.set_title(model, fontsize=9)
                    if c == 0:
                        ax.set_ylabel(test, fontsize=9)

            if suptitle is not None:
                fig.suptitle(suptitle, fontsize=12)

            return fig, axes

        def plot_perm_hist_grid_combo(
            perm_results_combo,
            empirical_combo_results,
            empirical_combo_results_z,
            combo_key,
            combo_models,
            #tests=('split_halves', 'split_halves_z', 'within', 'within_z', 'between_tasks', 'between_tasks_z'),
            tests=('split_halves', 'split_halves_z', 'between_tasks', 'between_tasks_z'),
            bins=25,
            density=True,
            figsize_per_panel=(2.0, 1.8),
            alpha=0.05,
            suptitle=None,
        ):
            cols = combo_models[combo_key]
            nrows, ncols = len(tests), len(cols)

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                sharey=False,
                constrained_layout=True,
            )

            # Force `axes` to a 2-D (nrows, ncols) array so axes[r, c] always works.
            axes = np.asarray(axes).reshape(nrows, ncols)

            for r, test in enumerate(tests):
                emp = empirical_combo_results_z[test[:-2]] if test.endswith('_z') else empirical_combo_results[test]

                x_all = np.asarray(perm_results_combo[test][combo_key]['beta'], dtype=float)
                beta_emp = np.asarray(emp[combo_key][1], dtype=float).ravel()

                if x_all.ndim != 2:
                    raise ValueError(
                        f"{test}/{combo_key}: expected permuted beta array with shape "
                        f"(n_permutations, n_combo_models), got {x_all.shape}. "
                        "Store the full beta vector, not a scalar."
                    )

                row_vals = np.concatenate([x_all.ravel(), beta_emp.ravel()])
                lim = np.nanmax(np.abs(row_vals))
                lim = 1.0 if (not np.isfinite(lim) or lim == 0) else 1.05 * lim
                edges = np.linspace(-lim, lim, bins + 1)

                for c, model_name in enumerate(cols):
                    ax = axes[r, c]
                    x = x_all[:, c]
                    beta = beta_emp[c]

                    p_one_sided = (np.sum(x >= beta) + 1) / (x.size + 1)

                    ax.hist(
                        x, bins=edges, density=density,
                        color='0.75', edgecolor='white', linewidth=0.6
                    )
                    ax.axvline(0, color='black', lw=0.9)
                    ax.axvline(beta, color='red', lw=1.6)

                    if p_one_sided < 0.1:
                        ax.text(
                            0.04, 0.96, f"p={p_one_sided:.3f}",
                            transform=ax.transAxes,
                            ha='left', va='top',
                            fontsize=8
                        )

                    if (beta > 0) and (p_one_sided < alpha):
                        y0, y1 = ax.get_ylim()
                        ax.set_ylim(y0, y1 * 1.15)
                        ax.text(
                            beta, y1 * 1.08, '★',
                            ha='center', va='bottom',
                            fontsize=16, fontweight='bold',
                            color='black'
                        )
                    else:
                        y0, y1 = ax.get_ylim()
                        ax.set_ylim(y0, y1 * 1.08)

                    ax.set_xlim(-lim, lim)
                    ax.tick_params(labelsize=8, length=2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    if r == 0:
                        ax.set_title(model_name, fontsize=9)
                    if c == 0:
                        ax.set_ylabel(test, fontsize=9)

            if suptitle is not None:
                fig.suptitle(suptitle, fontsize=12)

            return fig, axes

        # if len(models) > 0:
        #     fig, axes = plot_perm_hist_grid(
        #         perm_results=perm_results,
        #         empirical_results=empirical_results,
        #         empirical_results_z=empirical_results_z,
        #         models=models,
        #         bins=30,
        #         alpha=0.05,
        #         suptitle=f'ROI: {roi_name} (n={n_neurons} neurons)'
        #     )
        #     fig.savefig(os.path.join(OUT_DIR, f'permutation_grid_{roi_name}.png'), dpi=150)
        #     plt.show()
        # else:
        #     print(f"[{roi_name}] no base models selected — skipping single-model plot.")

        # for combo_key in combo_models:
        #     fig_c, axes_c = plot_perm_hist_grid_combo(
        #         perm_results_combo=perm_results_combo,
        #         empirical_combo_results=empirical_combo_results,
        #         empirical_combo_results_z=empirical_combo_results_z,
        #         combo_key=combo_key,
        #         combo_models=combo_models,
        #         bins=30,
        #         alpha=0.05,
        #         suptitle=f'ROI: {roi_name} – combo {combo_key} (n={n_neurons})'
        #     )
        #     fig_c.savefig(
        #         os.path.join(OUT_DIR, f'permutation_grid_{roi_name}_combo_{combo_key}.png'),
        #         dpi=150,
        #     )
        #     plt.show()

        # ── Cache for cross-ROI summary ──────────────────────────────────────
        roi_perm_results[roi_name] = perm_results
        roi_empirical_results[roi_name] = {
            'raw': empirical_results,
            'z':   empirical_results_z,
            'combo_raw': empirical_combo_results,
            'combo_z':   empirical_combo_results_z,
        }

        # ── Build per-ROI rows for the overview table ────────────────────────
        # tests for raw vs z follow the same naming convention used in perm_results
        test_pairs = [
            ('split_halves',    'raw'), ('split_halves_z',    'z'),
            #('within',         'raw'), ('within_z',          'z'),
            ('between_tasks',   'raw'), ('between_tasks_z',   'z'),
        ]
        for test_name, kind in test_pairs:
            # empirical_results / empirical_results_z are keyed by the BASE
            # test name (no '_z' suffix); perm_results carries the suffix.
            emp_key = test_name[:-2] if test_name.endswith('_z') else test_name
            emp_dict = empirical_results if kind == 'raw' else empirical_results_z
            for m in models:
                t_val, beta_val, p_param = emp_dict[emp_key][m]
                perm_betas = np.asarray(perm_results[test_name][m], dtype=float).ravel()
                if perm_betas.size > 0:
                    p_perm = (np.sum(perm_betas >= beta_val) + 1) / (perm_betas.size + 1)
                else:
                    p_perm = np.nan
                summary_rows.append({
                    'roi':    roi_name,
                    'n_neurons': n_neurons,
                    'test':   test_name,
                    'model':  m,
                    't':      t_val,
                    'beta':   beta_val,
                    'p_param': p_param,
                    'p_perm': p_perm,
                })

            # combo models
            combo_emp_dict = empirical_combo_results if kind == 'raw' else empirical_combo_results_z
            for combo in combo_models:
                t_arr    = np.asarray(combo_emp_dict[emp_key][combo][0], dtype=float).ravel()
                beta_arr = np.asarray(combo_emp_dict[emp_key][combo][1], dtype=float).ravel()
                p_arr    = np.asarray(combo_emp_dict[emp_key][combo][2], dtype=float).ravel()
                for sub_idx, sub_model in enumerate(combo_models[combo]):
                    perm_beta_list = perm_results_combo[test_name][combo]['beta']
                    if len(perm_beta_list) > 0:
                        perm_beta_sub = np.asarray(
                            [np.asarray(b, dtype=float).ravel()[sub_idx] for b in perm_beta_list],
                            dtype=float,
                        )
                        p_perm = (np.sum(perm_beta_sub >= beta_arr[sub_idx]) + 1) / (perm_beta_sub.size + 1)
                    else:
                        p_perm = np.nan
                    summary_combo_rows.append({
                        'roi':       roi_name,
                        'n_neurons': n_neurons,
                        'test':      test_name,
                        'combo':     combo,
                        'sub_model': sub_model,
                        't':         float(t_arr[sub_idx]),
                        'beta':      float(beta_arr[sub_idx]),
                        'p_param':   float(p_arr[sub_idx]),
                        'p_perm':    p_perm,
                    })


    # ── Regressor-collinearity check (model-RDM correlations) ───────────────
    # model_RDMs / _within / _across are identical across ROIs (depend only on
    # mode_locs). Use the last ROI's dict and plot once.
    for rdm_dict, label in [
        (model_RDMs,        'full'),
        #(model_RDMs_within, 'within'),
        (model_RDMs_across, 'between_tasks'),
    ]:
        mc.plotting.results.plot_model_rdm_correlation(
            rdm_dict,
            title=f'Model-RDM correlations ({label})',
            save_path=os.path.join(OUT_DIR, f'model_rdm_correlations_{label}.png'),
        )


    # ── Cross-ROI overview tables ────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_combo_df = pd.DataFrame(summary_combo_rows)

    summary_csv = os.path.join(OUT_DIR, 'results_summary.csv')
    summary_combo_csv = os.path.join(OUT_DIR, 'results_summary_combos.csv')
    summary_df.to_csv(summary_csv, index=False)
    summary_combo_df.to_csv(summary_combo_csv, index=False)
    print(f"\nSaved overview tables:\n  {summary_csv}\n  {summary_combo_csv}")

    # ── Phase-mode comparison: betas across full / within / across ────────
    # Empirical (no-permutation) betas for the primary test variant under each
    # phase-mask mode. One figure per single-model row (1 column wide) AND
    # one figure per combo (n_sub_models wide), so we can see whether a result
    # that looks negative in one mode is positive in another and by how much.
    if RUN_PHASE_MODE_COMPARISON and roi_mode_comparison:
        cmp_dir = os.path.join(OUT_DIR, 'phase_mode_comparison')
        os.makedirs(cmp_dir, exist_ok=True)
        cmp_test_name = FDR_TEST                          # e.g. 'between_tasks_z'
        cmp_base = (cmp_test_name[:-2]
                    if cmp_test_name.endswith('_z') else cmp_test_name)
        cmp_combo_key  = ('z_combo'
                          if cmp_test_name.endswith('_z') else 'raw_combo')
        cmp_single_key = ('z_single'
                          if cmp_test_name.endswith('_z') else 'raw_single')

        # Long-format CSVs: one for single-model rows, one for combo rows.
        single_rows, combo_rows = [], []
        for _mode, per_roi in roi_mode_comparison.items():
            for _roi, per_test in per_roi.items():
                _entry = per_test.get(cmp_base)
                if _entry is None:
                    continue
                # single models
                for m, (t_v, b_v, p_v) in _entry[cmp_single_key].items():
                    single_rows.append({
                        'mode':          _mode,
                        'roi':           _roi,
                        'model':         m,
                        't':             float(t_v) if np.isfinite(t_v) else np.nan,
                        'beta':          float(b_v) if np.isfinite(b_v) else np.nan,
                        'p_param':       float(p_v) if np.isfinite(p_v) else np.nan,
                        'n_pairs_kept':  _entry['n_pairs_kept'],
                        'n_pairs_total': _entry['n_pairs_total'],
                    })
                # combo models
                for combo, res in _entry[cmp_combo_key].items():
                    _t    = np.asarray(res[0], dtype=float).ravel()
                    _beta = np.asarray(res[1], dtype=float).ravel()
                    _p    = np.asarray(res[2], dtype=float).ravel()
                    for sub_idx, sub_model in enumerate(combo_models[combo]):
                        combo_rows.append({
                            'mode':       _mode,
                            'roi':        _roi,
                            'combo':      combo,
                            'sub_model':  sub_model,
                            't':          float(_t[sub_idx]),
                            'beta':       float(_beta[sub_idx]),
                            'p_param':    float(_p[sub_idx]),
                            'n_pairs_kept':  _entry['n_pairs_kept'],
                            'n_pairs_total': _entry['n_pairs_total'],
                        })

        single_df = pd.DataFrame(single_rows)
        combo_df  = pd.DataFrame(combo_rows)
        single_csv = os.path.join(cmp_dir,
                                  f'phase_mode_comparison_singles_{cmp_test_name}.csv')
        combo_csv  = os.path.join(cmp_dir,
                                  f'phase_mode_comparison_combos_{cmp_test_name}.csv')
        single_df.to_csv(single_csv, index=False)
        combo_df.to_csv(combo_csv, index=False)
        print(f"\nSaved phase-mode comparison CSVs:\n  {single_csv}\n  {combo_csv}")

        def _draw_mode_heatmaps(df, columns, col_axis, suptitle, save_stem):
            """Three side-by-side ROI x ``columns`` β heatmaps, one per mode.

            ``col_axis`` is the column name in ``df`` used to pivot horizontally
            ('model' for singles, 'sub_model' for combos). ``columns`` is the
            ordered list of column values to display.
            """
            _rois = sorted(df['roi'].unique(),
                           key=lambda r: list(ROI_RULES).index(r)
                           if r in ROI_RULES else 999)
            _grid = {}
            for _mode in ALL_PHASE_MODES:
                _g = (df[df['mode'] == _mode]
                      .pivot_table(index='roi', columns=col_axis,
                                   values='beta'))
                _g = _g.reindex(index=_rois, columns=columns)
                _grid[_mode] = _g
            _stack = np.concatenate([g.to_numpy().ravel() for g in _grid.values()])
            _lim = float(np.nanmax(np.abs(_stack)))
            if not np.isfinite(_lim) or _lim == 0:
                _lim = 1.0

            n_cols = max(len(columns), 1)
            fig, axes = plt.subplots(
                1, 3,
                figsize=(max(2.6, 0.7 * n_cols + 1.6) * 3,
                         0.55 * len(_rois) + 1.6),
                sharey=True, constrained_layout=True)
            for ax, _mode in zip(axes, ALL_PHASE_MODES):
                G = _grid[_mode].to_numpy()
                im = ax.imshow(G, cmap='RdBu_r', vmin=-_lim, vmax=_lim,
                               aspect='auto')
                ax.set_xticks(np.arange(n_cols))
                ax.set_xticklabels(columns, rotation=40, ha='right', fontsize=8)
                ax.set_yticks(np.arange(len(_rois)))
                ax.set_yticklabels(_rois, fontsize=8)
                ax.set_title(f'{_mode}', fontsize=10)
                for i in range(G.shape[0]):
                    for j in range(G.shape[1]):
                        v = G[i, j]
                        if np.isfinite(v):
                            ax.text(j, i, f'{v:.2f}',
                                    ha='center', va='center', fontsize=7,
                                    color='white' if abs(v) > 0.55 * _lim
                                    else 'black')
            cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
            cbar.set_label('empirical β', fontsize=9)
            fig.suptitle(suptitle, fontsize=11)
            png = f'{save_stem}.png'
            fig.savefig(png, dpi=150, bbox_inches='tight')
            plt.show()
            return png

        # Single-model heatmap (all single models in one figure x 3 modes).
        if not single_df.empty:
            png = _draw_mode_heatmaps(
                single_df, columns=list(models), col_axis='model',
                suptitle=f'Phase-mode comparison — single models  ({cmp_test_name})',
                save_stem=os.path.join(
                    cmp_dir, f'phase_mode_comparison_singles_{cmp_test_name}'))
            print(f"  saved {png}")

        # One figure per combo.
        if not combo_df.empty:
            for combo, sub_models in combo_models.items():
                _cdf = combo_df[combo_df['combo'] == combo]
                if _cdf.empty:
                    continue
                png = _draw_mode_heatmaps(
                    _cdf, columns=list(sub_models), col_axis='sub_model',
                    suptitle=f'Phase-mode comparison — combo {combo}  ({cmp_test_name})',
                    save_stem=os.path.join(
                        cmp_dir,
                        f'phase_mode_comparison_combo_{combo}_{cmp_test_name}'))
                print(f"  saved {png}")
else:
    # Reload mode: pull summary CSVs (and electrode coords if saved) from disk.
    summary_df = pd.read_csv(os.path.join(OUT_DIR, 'results_summary.csv'))
    summary_combo_df = pd.read_csv(
        os.path.join(OUT_DIR, 'results_summary_combos.csv'))
    coords_csv = os.path.join(OUT_DIR, 'roi_electrode_coords.csv')
    if os.path.exists(coords_csv):
        ec_df = pd.read_csv(coords_csv)
        roi_electrode_coords = {
            roi: {(int(r['subject']), int(r['cell_idx'])):
                  (float(r['MNI_x']), float(r['MNI_y']), float(r['MNI_z']))
                  for _, r in g.iterrows()}
            for roi, g in ec_df.groupby('roi')
        }
    else:
        # Fall back to deriving electrode positions from the ROI table.
        roi_electrode_coords = {}
        for roi in ROI_RULES:
            sub_tbl = ROI_TABLE[ROI_TABLE[ROI_LABEL_COLUMN] == roi]
            roi_electrode_coords[roi] = {
                (int(s), int(c)): (float(row['MNI_x']),
                                   float(row['MNI_y']),
                                   float(row['MNI_z']))
                for (s, c), row in sub_tbl.iterrows()
            }
    print(f"Loaded summary_df ({len(summary_df)} rows) and "
          f"summary_combo_df ({len(summary_combo_df)} rows) from {OUT_DIR}.")


# ── BH-FDR correction over the confirmatory family ───────────────────
def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Returns an array the same length as `pvals`; NaN inputs stay NaN.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # step-up monotone
    q_ok = np.empty(n)
    q_ok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = q_ok
    return q


_fam_cols = {'test', 'combo', 'sub_model', 'p_perm'}
if not summary_combo_df.empty and _fam_cols.issubset(summary_combo_df.columns):
    # Flag the confirmatory family and BH-correct only those p_perm values.
    summary_combo_df['in_fdr_family'] = (
        summary_combo_df['test'].eq(FDR_TEST)
        & summary_combo_df['combo'].isin(FDR_COMBOS)
        & summary_combo_df['sub_model'].isin(FDR_SUBMODELS)
    )
    summary_combo_df['p_fdr'] = np.nan
    _fam = summary_combo_df['in_fdr_family']
    if _fam.any():
        summary_combo_df.loc[_fam, 'p_fdr'] = benjamini_hochberg(
            summary_combo_df.loc[_fam, 'p_perm'].to_numpy())
        _fam_df = (summary_combo_df.loc[_fam,
                   ['roi', 'combo', 'sub_model', 'n_neurons',
                    'beta', 'p_perm', 'p_fdr']]
                   .sort_values(['combo', 'p_fdr'])
                   .reset_index(drop=True))
        n_sig = int((_fam_df['p_fdr'] < FDR_ALPHA).sum())
        print(f"\n=== BH-FDR confirmatory family "
              f"(test={FDR_TEST}, combos={FDR_COMBOS}, "
              f"effect={FDR_SUBMODELS}) ===")
        print(f"{len(_fam_df)} tests; {n_sig} significant at q < {FDR_ALPHA}")
        with pd.option_context('display.max_rows', None,
                               'display.width', 160):
            print(_fam_df.to_string(index=False))
        _fam_df.to_csv(os.path.join(OUT_DIR, 'confirmatory_fdr.csv'),
                       index=False)
    else:
        print("\nBH-FDR: confirmatory family is empty — check the FDR_* "
              "settings against the combo / test names actually run.")
    # Re-save the combo table so it carries p_fdr / in_fdr_family.
    summary_combo_df.to_csv(
        os.path.join(OUT_DIR, 'results_summary_combos.csv'), index=False)
else:
    print("\nBH-FDR: no combo results available to correct.")


def _render_overview_plots(summary_df, summary_combo_df,
                           roi_electrode_coords, out_dir,
                           models, combo_models,
                           heatmap_test='across_z'):
    """Render the pivot tables, per-ROI heatmaps and electrode glass-brain.

    Pulled out into a function so both the fresh-run and the
    reload-only paths can produce the same set of overview figures.
    """
    # Pivoted view: beta per (roi, test) x model
    if not summary_df.empty:
        pivot_beta = summary_df.pivot_table(
            index=['roi', 'test'], columns='model', values='beta'
        )
        print("\n=== Beta overview (rows: ROI x test, cols: model) ===")
        print(pivot_beta.to_string())

        pivot_pperm = summary_df.pivot_table(
            index=['roi', 'test'], columns='model', values='p_perm'
        )
        print("\n=== Permutation p-value overview ===")
        print(pivot_pperm.to_string())

    # base models heatmap (only if any base models were evaluated)
    from mc.plotting.cell_results import (
        plot_roi_model_heatmap, CANONICAL_ROI_ORDER,
        CANONICAL_RSA_MODEL_ORDER,
    )
    if len(models) > 0 and not summary_df.empty:
        sub_df_base = summary_df[summary_df['test'] == heatmap_test].copy()
        if sub_df_base.empty:
            print(f"No rows for test={heatmap_test}; skipping base-model heatmap.")
        else:
            fig_h, ax_h = plot_roi_model_heatmap(
                sub_df_base,
                models=CANONICAL_RSA_MODEL_ORDER,
                rois=CANONICAL_ROI_ORDER,
                value_col='beta', annot_col='p_perm', sig_col='p_perm',
                n_col='n_neurons', alpha=0.05,
                value_label='empirical beta',
                title=f'ROIs x base models — {heatmap_test}',
                save_path=os.path.join(
                    out_dir, f'heatmap_roi_models_{heatmap_test}.png'),
                base_fontsize=15,
            )
            if fig_h is not None:
                plt.show()
    else:
        print("No base models evaluated this run — skipping base-model heatmap.")

    # combo models heatmap
    if not summary_combo_df.empty:
        for combo_key, sub_models in combo_models.items():
            sub_df = summary_combo_df[
                (summary_combo_df['combo'] == combo_key)
                & (summary_combo_df['test'] == heatmap_test)
            ].copy()
            if sub_df.empty:
                continue
            sub_df['model'] = sub_df['sub_model']

            fig_hc, ax_hc = plot_roi_model_heatmap(
                sub_df,
                models=sub_models,
                rois=CANONICAL_ROI_ORDER,
                value_col='beta', annot_col='p_perm', sig_col='p_perm',
                n_col='n_neurons', alpha=0.05,
                value_label='empirical beta',
                title=f'ROIs x {combo_key} sub-models — {heatmap_test}',
                save_path=os.path.join(
                    out_dir,
                    f'heatmap_roi_combo_{combo_key}_{heatmap_test}.png'),
                base_fontsize=15,
            )
            if fig_hc is not None:
                plt.show()

    # ROI electrode schematic
    if PLOT_GLASSBRAINS == True:
        from mc.plotting.cell_results import (
            plot_roi_electrodes_glassbrain, plot_roi_beta_glassbrain,
        )
        electrodes_per_roi = {
            roi: np.array(list(coords.values()), dtype=float)
            for roi, coords in roi_electrode_coords.items()
            if coords
        }
        if electrodes_per_roi:
            plot_roi_electrodes_glassbrain(
                electrodes_per_roi,
                save_path=os.path.join(out_dir, 'roi_electrodes_glassbrain.png'),
                title='ROI electrode locations (one panel per ROI)',
                per_roi_panels=True,
            )
            plot_roi_electrodes_glassbrain(
                electrodes_per_roi,
                save_path=os.path.join(
                    out_dir, 'roi_electrodes_glassbrain_combined.png'),
                title='ROI electrode locations (all ROIs combined)',
                per_roi_panels=False,
            )
        else:
            print("No electrode coordinates collected — skipping ROI glass-brain.")

        # ── ROI-shaded glass-brain (heatmap colours on a brain) ──────────
        # One figure per (model, heatmap_test): each anatomical ROI mask is
        # shaded by its beta value, using the same RdBu_r palette as the
        # ROI x model heatmap.  Restricted to ROIs that actually have cells.
        rois_with_cells = sorted(electrodes_per_roi)
        if rois_with_cells:
            glassbrain_dir = os.path.join(out_dir, 'roi_beta_glassbrains')
            os.makedirs(glassbrain_dir, exist_ok=True)
    
            if len(models) > 0 and not summary_df.empty:
                sub_t = summary_df[summary_df['test'] == heatmap_test]
                for m in models:
                    rows = sub_t[sub_t['model'] == m]
                    if rows.empty:
                        continue
                    betas = dict(zip(rows['roi'], rows['beta']))
                    pvals = dict(zip(rows['roi'], rows['p_perm']))
                    plot_roi_beta_glassbrain(
                        roi_betas=betas, roi_pvals=pvals,
                        only_rois=rois_with_cells,
                        roi_cell_coords=electrodes_per_roi,
                        roi_label_column=ROI_LABEL_COLUMN,
                        title=f'{m} beta — {heatmap_test}',
                        save_path=os.path.join(
                            glassbrain_dir,
                            f'roi_beta_glassbrain_{m}_{heatmap_test}.png'),
                    )
                    plt.show()

        if not summary_combo_df.empty:
            sub_t = summary_combo_df[summary_combo_df['test'] == heatmap_test]
            for combo_key, sub_models in combo_models.items():
                for sm in sub_models:
                    rows = sub_t[(sub_t['combo'] == combo_key)
                                 & (sub_t['sub_model'] == sm)]
                    if rows.empty:
                        continue
                    betas = dict(zip(rows['roi'], rows['beta']))
                    pvals = dict(zip(rows['roi'], rows['p_perm']))
                    plot_roi_beta_glassbrain(
                        roi_betas=betas, roi_pvals=pvals,
                        only_rois=rois_with_cells,
                        roi_cell_coords=electrodes_per_roi,
                        roi_label_column=ROI_LABEL_COLUMN,
                        title=f'{combo_key} | {sm} beta — {heatmap_test}',
                        save_path=os.path.join(
                            glassbrain_dir,
                            f'roi_beta_glassbrain_{combo_key}_{sm}_{heatmap_test}.png'),
                    )
                    plt.show()


# ── Cross-ROI heatmap (rows=ROI, cols=model) ─────────────────────────────
# Heatmap rendering now lives in mc.plotting.cell_results
# (plot_roi_model_heatmap), shared with encoding_analysis_simple.py so
# fonts, ROI/model order and significance styling stay in sync.


HEATMAP_TEST = 'between_tasks_z'  # one of: split_halves, split_halves_z,
                                  # within, within_z, between_tasks,
                                  # between_tasks_z

# Save the per-ROI electrode coordinates so that the reload path can
# reproduce the schematic glass-brain without rerunning the analysis.
_coord_rows = [
    {'roi': roi, 'subject': sub_int, 'cell_idx': cell_int,
     'MNI_x': x, 'MNI_y': y, 'MNI_z': z}
    for roi, d in roi_electrode_coords.items()
    for (sub_int, cell_int), (x, y, z) in d.items()
]
if _coord_rows:
    pd.DataFrame(_coord_rows).to_csv(
        os.path.join(OUT_DIR, 'roi_electrode_coords.csv'), index=False
    )

_render_overview_plots(
    summary_df, summary_combo_df, roi_electrode_coords, OUT_DIR,
    models=models, combo_models=combo_models, heatmap_test=HEATMAP_TEST,
)
