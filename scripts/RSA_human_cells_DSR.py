#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:27:09 2026

@author: Svenja Kuchenhoff

DSR RSA.
Sessions with shared DSR configs across subjects:
  ['s27','s28','s31','s32','s33','s34','s35','s36','s37','s38','s40',
   's43','s44','s45','s46','s49','s50','s51','s53','s55','s56','s57',
   's58','s59','s60','s61','s62','s63']

  configs:
    0  3-7-9-5     4  6-4-2-9
    1  8-2-6-7     5  9-1-3-4
    2  1-9-5-8     6  7-3-4-2
    3  4-8-1-3     7  2-5-7-6

Pipeline:
  1. Load neural averages (prep_human_cells_RSA-2026.py, DSR mode).
  2. Build mode-location arrays from raw behaviour (per config × run).
  3. Build model matrices (location, DSR, state, feedback).
  4. Compute model RDMs (Hamming for location/DSR, crosscorr for state/feedback).
  5. Build data RDMs for whole-brain and each ROI.
  6. Evaluate: unique regression per model + combined regression.
  7. Plot summary heatmap.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR     = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR      = os.path.join(DATA_DIR, 'group', 'DSR_RSA')
N_BINS       = 360
INCLUDE_DIAG = False

# Feature 2: Downsample flag
DOWNSAMPLE_BINS = 120   # set to None to keep 360 bins; set to int to average down

# Feature 3: Phase mask flag
MASK_CROSS_PHASE = True   # if True, mask RDM entries comparing different phases

states           = ['A', 'B', 'C', 'D']
rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY', 'R-WHITE-MATTER', 'OCCIP']
models           = ['location', 'dsr', 'state', 'feedback']

# DSR subpath parameters: 12 subpaths × 30 bins = 360
LEN_STANDARDISED_PATH = 30
N_SUBPATHS = N_BINS // LEN_STANDARDISED_PATH   # 12

# Effective number of bins after optional downsampling
N_BINS_USED = DOWNSAMPLE_BINS if DOWNSAMPLE_BINS else N_BINS

# Fixed config order (same axis in all saved .npy files)
configs = [
    '3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
    '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6',
]
N_CONFIGS = len(configs)

with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json'), 'r') as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())

os.makedirs(OUT_DIR, exist_ok=True)


# ── Load neural averages into a DataFrame ────────────────────────────
# neurons columns: session | neuron_label | roi | electrode_label |
#                  neuron_data (2, 360) | config
rows = []
for sub in SUBJECTS:
    SUB_DIR = f"{DATA_DIR}/s{sub}/dsr_avg"
    sesh_neurons = np.load(os.path.join(SUB_DIR, f's{sub}_dsr_neural_avg.npy'))
    with open(os.path.join(SUB_DIR, f's{sub}_dsr_neuron_meta.json'), 'r') as f:
        nmeta = json.load(f)

    for n_idx, n in enumerate(sesh_neurons):
        for conf in config_summary[sub]:
            rows.append({
                'session':        sub,
                'neuron_idx':     n_idx,
                'neuron_label':   nmeta['neuron_names'][n_idx],
                'roi':            nmeta['cell_labels'][n_idx],
                'electrode_label': nmeta['electrode_labels'][n_idx],
                'neuron_data':    n[conf['config_idx'], :, :],   # (2, 360)
                'config':         conf['config'],
            })

neurons = pd.DataFrame(rows)
print(f"Loaded {len(neurons['neuron_label'].unique())} unique neurons "
      f"from {len(SUBJECTS)} sessions.")


# ── Build mode-location arrays from behaviour ─────────────────────────
# For each (config, run), stack all correct-trial location arrays (360 bins)
# across sessions and take the per-bin mode.
# mode_locs[config][run_id] → ndarray (360,), values 1..9

print("Building mode-location arrays...")
accum = {c: {1: [], 2: []} for c in configs}

for sub in SUBJECTS:
    glog_path = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg',
                             f's{sub}_dsr_grouping_log.json')
    if not os.path.exists(glog_path):
        continue
    with open(glog_path) as f:
        glog = json.load(f)

    data_dict, _ = mc.analyse.helpers_human_cells.get_data(int(sub))
    key = f'sub-{int(sub):02}'
    if key not in data_dict:
        continue
    beh  = data_dict[key]['beh'].copy()
    locs = data_dict[key]['locations']   # DataFrame (n_trials, 360)

    def _cfg_str(row):
        return (f"{int(row['loc_A'])}-{int(row['loc_B'])}-"
                f"{int(row['loc_C'])}-{int(row['loc_D'])}")
    beh['config_str'] = beh.apply(_cfg_str, axis=1)
    beh['grid_no']    = beh['grid_no'].astype(int)

    for cfg_entry in glog['configs']:
        c = cfg_entry['config']
        for run_id, blocks in [(1, cfg_entry['run1_blocks']),
                               (2, cfg_entry['run2_blocks'])]:
            mask = (
                (beh['config_str'] == c) &
                (beh['correct'] == 1) &
                (beh['grid_no'].isin(blocks))
            )
            idx = beh.index[mask]
            if len(idx) == 0:
                continue
            accum[c][run_id].append(locs.loc[idx].values.astype(float))

mode_locs = {}
for c in configs:
    mode_locs[c] = {}
    for run_id in [1, 2]:
        parts = accum[c][run_id]
        if not parts:
            mode_locs[c][run_id] = np.full(N_BINS, np.nan)
        else:
            stacked = np.vstack(parts)   # (n_trials_total, 360)
            m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
            mode_locs[c][run_id] = m.mode.astype(float)

print("Mode-location arrays built.")


# ── Build model matrices ───────────────────────────────────────────────
# All matrices shape: (2 * N_CONFIGS * N_BINS, n_features)
# run-1 block (N_CONFIGS * N_BINS rows) followed by run-2 block.
# This matches the data matrix layout used in compute_crosscorr.
#
# location (Hamming):
#   Row for bin b = the 30-element mode-path of the subpath b belongs to.
#   All 30 bins within a subpath share the same row vector.
#   → n_features = LEN_STANDARDISED_PATH = 30
#
# dsr (Hamming):
#   Base vector = mode_vec (360,).
#   Row for bin b = np.roll(base_vec, -(b // 30) * 30).
#   → n_features = N_BINS = 360
#
# state (crosscorr):
#   One-hot over 4 states, 90-bin blocks, tiled across configs.
#   → n_features = 4
#
# feedback (crosscorr):
#   Brief signal (bins 0-9 of state A), tiled across configs.
#   → n_features = 4

print("Building model matrices...")

loc_th1 = np.zeros((N_CONFIGS * N_BINS, LEN_STANDARDISED_PATH), dtype=float)
loc_th2 = np.zeros((N_CONFIGS * N_BINS, LEN_STANDARDISED_PATH), dtype=float)
dsr_th1 = np.zeros((N_CONFIGS * N_BINS, N_BINS),                dtype=float)
dsr_th2 = np.zeros((N_CONFIGS * N_BINS, N_BINS),                dtype=float)

for c_idx, c in enumerate(configs):
    row_start = c_idx * N_BINS
    for run_id, loc_mat, dsr_mat in [(1, loc_th1, dsr_th1),
                                     (2, loc_th2, dsr_th2)]:
        mode_vec = mode_locs[c][run_id]   # (360,)

        # location: each bin gets the 30-element subpath sequence it belongs to
        for k in range(N_SUBPATHS):
            # 30 bins = 1 subpath
            subpath_seq = mode_vec[k * LEN_STANDARDISED_PATH:(k + 1) * LEN_STANDARDISED_PATH]
            loc_mat[
                row_start + k * LEN_STANDARDISED_PATH:
                row_start + (k + 1) * LEN_STANDARDISED_PATH, :
            ] = subpath_seq   # broadcasts same (30,) to each of the 30 rows

        # dsr: roll base vector left by whole-subpath steps
        for b in range(N_BINS):
            roll_by = -(b // LEN_STANDARDISED_PATH) * LEN_STANDARDISED_PATH
            dsr_mat[row_start + b, :] = np.roll(mode_vec, roll_by)

# state and feedback: one-hot over states, tiled
state_half    = np.zeros((N_BINS, len(states)))
feedback_half = np.zeros((N_BINS, len(states)))
for s_i, s in enumerate(states):
    state_half[s_i * 90:(s_i + 1) * 90, s_i] = 1
    if s == 'A':
        feedback_half[0:10, s_i] = 1

# ── Feature 2: Downsample model matrices if requested ─────────────────
if DOWNSAMPLE_BINS is not None:
    factor = N_BINS // DOWNSAMPLE_BINS
    print(f"Downsampling model matrices from {N_BINS} to {DOWNSAMPLE_BINS} bins (factor={factor})...")

    # loc_th: (N_CONFIGS * N_BINS, 30) → (N_CONFIGS, N_BINS, 30) → avg → (N_CONFIGS, DOWNSAMPLE_BINS, 30)
    #         → (N_CONFIGS * DOWNSAMPLE_BINS, 30)
    loc_th1 = (loc_th1.reshape(N_CONFIGS, N_BINS, LEN_STANDARDISED_PATH)
               .reshape(N_CONFIGS, DOWNSAMPLE_BINS, factor, LEN_STANDARDISED_PATH)
               .mean(axis=2)
               .reshape(N_CONFIGS * DOWNSAMPLE_BINS, LEN_STANDARDISED_PATH))
    loc_th2 = (loc_th2.reshape(N_CONFIGS, N_BINS, LEN_STANDARDISED_PATH)
               .reshape(N_CONFIGS, DOWNSAMPLE_BINS, factor, LEN_STANDARDISED_PATH)
               .mean(axis=2)
               .reshape(N_CONFIGS * DOWNSAMPLE_BINS, LEN_STANDARDISED_PATH))

    # dsr_th: (N_CONFIGS * N_BINS, N_BINS) → same row downsampling, keep feature dim
    dsr_th1 = (dsr_th1.reshape(N_CONFIGS, N_BINS, N_BINS)
               .reshape(N_CONFIGS, DOWNSAMPLE_BINS, factor, N_BINS)
               .mean(axis=2)
               .reshape(N_CONFIGS * DOWNSAMPLE_BINS, N_BINS))
    dsr_th2 = (dsr_th2.reshape(N_CONFIGS, N_BINS, N_BINS)
               .reshape(N_CONFIGS, DOWNSAMPLE_BINS, factor, N_BINS)
               .mean(axis=2)
               .reshape(N_CONFIGS * DOWNSAMPLE_BINS, N_BINS))

    # state/feedback: (N_BINS, 4) → (DOWNSAMPLE_BINS, factor, 4).mean(axis=1) → (DOWNSAMPLE_BINS, 4)
    state_half    = state_half.reshape(DOWNSAMPLE_BINS, factor, len(states)).mean(axis=1)
    feedback_half = feedback_half.reshape(DOWNSAMPLE_BINS, factor, len(states)).mean(axis=1)

state_one_config    = np.vstack([state_half,    state_half])
feedback_one_config = np.vstack([feedback_half, feedback_half])

model_concat = {
    'location': np.concatenate([loc_th1, loc_th2], axis=0),           # (2*8*N_BINS_USED, 30)
    'dsr':      np.concatenate([dsr_th1, dsr_th2], axis=0),            # (2*8*N_BINS_USED, 360)
    'state':    np.tile(state_one_config,    (N_CONFIGS, 1)),           # (2*8*N_BINS_USED, 4)
    'feedback': np.tile(feedback_one_config, (N_CONFIGS, 1)),           # (2*8*N_BINS_USED, 4)
}

for m, mat in model_concat.items():
    print(f"  {m:12s}: {mat.shape}")


# ── Compute model RDMs ─────────────────────────────────────────────────
print("Computing model RDMs...")

model_RDMs = {}
for m in models:
    if m in ('location', 'dsr'):
        model_RDMs[m] = mc.analyse.my_RSA.compute_hamming_distance(
            model_concat[m], plotting=False, include_diagonal=INCLUDE_DIAG)
    else:
        model_RDMs[m] = mc.analyse.my_RSA.compute_crosscorr(
            model_concat[m], plotting=False, include_diagonal=INCLUDE_DIAG)

# Patch feedback NaNs (bins where only state A is defined → undefined elsewhere)
nan_mask_feedback = np.isnan(model_RDMs['feedback'][0])
model_RDMs['feedback'][0][nan_mask_feedback] = 1

for m in models:
    print(f"  {m:12s} RDM: {model_RDMs[m][0].shape}")

# Stacked model matrix for combined regression: shape (n_rdm_entries, n_models)
stacked_models = np.stack([model_RDMs[m][0] for m in models], axis=1)


# ── Feature 3: Phase mask ──────────────────────────────────────────────

def make_phase_mask(n_bins_used, n_configs, include_diagonal=False):
    """
    Generate a boolean mask (True = keep, False = exclude) for the full RDM.

    Phase structure: bins repeat in 3 phases within each 90-bin reward segment.
    When downsampled, the segment length scales proportionally.
      segment_len = n_bins_used // 4
      phase_len   = segment_len // 3

    Returns a 2D boolean array of shape (total_bins, total_bins)
    where True means 'same phase' (keep).
    """
    total_bins  = 2 * n_configs * n_bins_used
    segment_len = n_bins_used // 4
    phase_len   = segment_len // 3

    # Assign a phase to each global row/column index
    # global bin within a config block = row_idx % n_bins_used
    def _phase(idx):
        bin_within_config  = idx % n_bins_used
        bin_within_segment = bin_within_config % segment_len
        return bin_within_segment // phase_len

    phases = np.array([_phase(i) for i in range(total_bins)])

    # same_phase[i, j] = True if phases match
    same_phase = phases[:, None] == phases[None, :]
    return same_phase


# ── Helper: assemble neuron matrix for a neuron subset ────────────────

def build_neuron_matrix(neuron_subset: pd.DataFrame):
    """
    Returns tuple: (matrix, n_excluded)
      matrix: (2 * N_CONFIGS * N_BINS_USED, n_valid_neurons)
      n_excluded: number of neuron columns dropped due to NaN.

    Run-1 block on top, run-2 block below.
    If DOWNSAMPLE_BINS is not None, bins are averaged down after NaN exclusion.
    """
    uniq = neuron_subset['neuron_label'].unique()
    n    = len(uniq)
    th1  = np.zeros((N_CONFIGS * N_BINS, n))
    th2  = np.zeros((N_CONFIGS * N_BINS, n))
    for n_idx, nl in enumerate(uniq):
        for c_idx, c in enumerate(configs):
            rows = neuron_subset[
                (neuron_subset['neuron_label'] == nl) &
                (neuron_subset['config'] == c)
            ]
            if len(rows) == 0:
                continue
            nd = rows['neuron_data'].to_numpy()[0]   # (2, 360)
            th1[c_idx * N_BINS:(c_idx + 1) * N_BINS, n_idx] = nd[0, :]
            th2[c_idx * N_BINS:(c_idx + 1) * N_BINS, n_idx] = nd[1, :]

    # Feature 1: exclude neurons with NaN in either run
    valid_cols = ~(np.isnan(th1).any(axis=0) | np.isnan(th2).any(axis=0))
    n_excluded = int(np.sum(~valid_cols))
    th1 = th1[:, valid_cols]
    th2 = th2[:, valid_cols]

    # Feature 2: downsample bins if requested
    if DOWNSAMPLE_BINS is not None:
        factor     = N_BINS // DOWNSAMPLE_BINS
        n_valid    = th1.shape[1]
        th1 = (th1.reshape(N_CONFIGS, N_BINS, n_valid)
               .reshape(N_CONFIGS, DOWNSAMPLE_BINS, factor, n_valid)
               .mean(axis=2)
               .reshape(N_CONFIGS * DOWNSAMPLE_BINS, n_valid))
        th2 = (th2.reshape(N_CONFIGS, N_BINS, n_valid)
               .reshape(N_CONFIGS, DOWNSAMPLE_BINS, factor, n_valid)
               .mean(axis=2)
               .reshape(N_CONFIGS * DOWNSAMPLE_BINS, n_valid))

    return np.concatenate([th1, th2], axis=0), n_excluded


# ── RSA: whole-brain + per ROI ─────────────────────────────────────────
print("Running RSA...")

# Apply phase mask to model RDMs if requested
if MASK_CROSS_PHASE:
    # compute_crosscorr produces an (n x n) RDM where n = N_CONFIGS * N_BINS_USED
    # (cross-corr between run1 and run2 halves; NOT the full 2*n matrix).
    # Build the phase mask for this single-half size only.
    n_rdm = N_CONFIGS * N_BINS_USED
    phase_mask_2d = make_phase_mask(N_BINS_USED, N_CONFIGS, include_diagonal=INCLUDE_DIAG)
    # make_phase_mask is built for 2*n_configs*n_bins; slice the top-left n_rdm×n_rdm
    # block (same phase structure as the cross-half quadrant, which is periodic).
    phase_mask_2d = phase_mask_2d[:n_rdm, :n_rdm]
    # Extract upper-triangle to match compute_crosscorr's triu_indices output
    if INCLUDE_DIAG:
        tri_rows, tri_cols = np.triu_indices(n_rdm, k=0)
    else:
        tri_rows, tri_cols = np.triu_indices(n_rdm, k=1)
    phase_mask_flat = phase_mask_2d[tri_rows, tri_cols]
    n_masked = int(np.sum(~phase_mask_flat))
    print(f"  Phase mask: {n_masked} RDM entries masked out of {len(phase_mask_flat)} "
          f"({100.0 * n_masked / len(phase_mask_flat):.1f}%)")

    for m in models:
        rdm_vec = model_RDMs[m][0].copy().astype(float)
        rdm_vec[~phase_mask_flat] = np.nan
        model_RDMs[m][0][:] = rdm_vec

    # Rebuild stacked models after masking
    stacked_models = np.stack([model_RDMs[m][0] for m in models], axis=1)


def run_rsa(data_mat: np.ndarray) -> dict:
    """
    Run RSA for one neural population matrix.
    Returns dict with 'data_rdm', 'unique' (per model), 'combined'.
    """

    data_rdm = mc.analyse.my_RSA.compute_crosscorr(
        data_mat, plotting=True, include_diagonal=INCLUDE_DIAG)

    # Apply phase mask to data RDM if requested
    if MASK_CROSS_PHASE:
        data_rdm_vec = data_rdm[0].copy().astype(float)
        data_rdm_vec[~phase_mask_flat] = np.nan
        data_rdm[0][:] = data_rdm_vec

    # unique regression per model
    unique = {}
    for m in models:
        res = mc.analyse.my_RSA.evaluate_model(model_RDMs[m][0], data_rdm[0])
        unique[m] = (float(res[0]), float(res[1]), float(res[2]))

    # combined regression (all models simultaneously)
    res_combo = mc.analyse.my_RSA.evaluate_model(stacked_models, data_rdm[0])
    t_combo = np.asarray(res_combo[0], dtype=float).ravel()
    b_combo = np.asarray(res_combo[1], dtype=float).ravel()
    p_combo = np.asarray(res_combo[2], dtype=float).ravel()
    combined = {'t': t_combo, 'beta': b_combo, 'p': p_combo}

    return {'data_rdm': data_rdm, 'unique': unique, 'combined': combined}


# whole brain
wb_mat, wb_excluded = build_neuron_matrix(neurons)
print(f"  whole_brain: excluded {wb_excluded} neurons with NaN.")
wb_res = run_rsa(wb_mat)
wb_res['n_neurons'] = len(neurons['neuron_label'].unique()) - wb_excluded

roi_results = {'whole_brain': wb_res}
print(f"  whole_brain (n={wb_res['n_neurons']}) — unique t: "
      + ', '.join(f"{m}={wb_res['unique'][m][0]:.2f}" for m in models))

for roi in rois_of_interest:
    subset = neurons[neurons['roi'] == roi]
    n_roi_raw = len(subset['neuron_label'].unique())
    if n_roi_raw == 0:
        print(f"  {roi}: no neurons, skipping.")
        continue
    roi_mat, roi_excluded = build_neuron_matrix(subset)
    n_roi = n_roi_raw - roi_excluded
    print(f"  {roi}: excluded {roi_excluded} neurons with NaN.")
    if n_roi == 0:
        print(f"  {roi}: no valid neurons after NaN exclusion, skipping.")
        continue
    res     = run_rsa(roi_mat)
    res['n_neurons'] = n_roi
    roi_results[roi] = res
    print(f"  {roi} (n={n_roi}) — unique t: "
          + ', '.join(f"{m}={res['unique'][m][0]:.2f}" for m in models))


# ── Plot summary heatmap ───────────────────────────────────────────────
present_rois = ['whole_brain'] + [r for r in rois_of_interest if r in roi_results]

fig = mc.plotting.cell_results.plot_rsa_heatmap(
    results  = roi_results,
    models   = models,
    rois     = present_rois,
    title    = 'DSR RSA — t-values per model × ROI',
    save_path= os.path.join(OUT_DIR, 'DSR_RSA_heatmap.png'),
)
plt.show()

print('\nAll done.')
