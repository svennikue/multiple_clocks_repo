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
import io
import json
import contextlib
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import mc

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')

# import pdb; pdb.set_trace()

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR     = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR      = os.path.join(DATA_DIR, 'group', 'DSR_RSA')
# N_BINS       = 360 # this is determined by the downsampled data. better: define goal downsampled bins
INCLUDE_DIAG = False

PLOT_FIGS = True

KEEP_RUNS_SEPRATE = False

# ── Permutation settings ───────────────────────────────────────────────
RUN_PERMUTATIONS = False    # set False to skip permutation testing
N_PERMS          = 300     # number of circular-shift permutations
PERM_SEED        = 42


# currently assuming 3 phases per state: min 3*4 = 12. 5*3*4 = 60
# i.e. each subpath 
# set to None to keep 360 bins; set to int to average down
# DSR subpath parameters: 12 subpaths × 30 bins = 360
LEN_STANDARDISED_PATH = 10

# normally, n conditions in an RDM per config is 1x -> state*phases = e.g 12 conditions per config.
# I can increase this easily to e.g. 24, such that I have finer detail of locations.
RESOLUTIONx = 2

# Feature 3: Phase mask flag
MASK_CROSS_PHASE = False   # if True, mask RDM entries comparing different phases

# Within-run RSA: if True, average both runs and compute within-run RDMs
# instead of across-run (cross-correlating run1 vs run2)
WITHIN_RUN = False

# Neuron analysis: if True, print diagnostics about each neuron's reliability
# across configs (signal dropout, cross-config variance) and exclude unreliable cells
NEURON_ANALYSIS = False

N_PHASES = 3
states           = ['A', 'B', 'C', 'D']

N_CONDS_PER_CONF = N_PHASES * len(states) * RESOLUTIONx


rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY', 'R-WHITE-MATTER', 'OCCIP']
models           = ['location', 'dsr', 'state', 'feedback', 'clocks', 'phase', 'midnight', 'loc_og']

combo_models = ['state', 'feedback', 'clocks', 'phase', 'midnight', 'loc_og']



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
    if KEEP_RUNS_SEPRATE == True:
        sesh_neurons = np.load(os.path.join(f"{SUB_DIR}_runs_separate", f's{sub}_dsr_neural_avg.npz'))
        with open(os.path.join(SUB_DIR, f's{sub}_dsr_neuron_meta.json'), 'r') as f:
            nmeta = json.load(f)
        
        configs = sesh_neurons.files
        for conf in configs:
            for n_idx, n in enumerate(sesh_neurons[conf]):
                rows.append({
                    'session':        sub,
                    'neuron_idx':     n_idx,
                    'neuron_label':   nmeta['neuron_names'][n_idx],
                    'roi':            nmeta['cell_labels'][n_idx],
                    'electrode_label': nmeta['electrode_labels'][n_idx],
                    'neuron_data':    n,   # n_neurons, n_runs, N_BINS
                    'n_reps': n.shape[0],
                    'config':         conf
                })
            
    if KEEP_RUNS_SEPRATE == False:
        sesh_neurons = np.load(os.path.join(SUB_DIR, f's{sub}_dsr_neural_avg_two_runs.npy'))
        with open(os.path.join(SUB_DIR, f's{sub}_dsr_neuron_meta_two_runs.json'), 'r') as f:
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
                    'config':         conf['config']
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
    if KEEP_RUNS_SEPRATE == True:
        glog_path = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg_runs_separate', f's{sub}_dsr_grouping_log.json')
    else:
        glog_path = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg', f's{sub}_dsr_grouping_log_two_runs.json')
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
    
    N_BINS = locs.shape[1]
    # LEN_OG_SUBPATH = int(N_BINS/N_PHASES/len(states))
    
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


# for categorical variables (e.g. location/dsr encoding)
def downsample_mode(x, target_len=10):
    x = np.asarray(x)
    block = len(x) // target_len
    return np.array([
        stats.mode(x[i*block:(i+1)*block], keepdims=False).mode
        for i in range(target_len)
    ])

# for continous variables (e.g. firing rate of neurons)
def downsample_mean(x, target_len):
    x = np.asarray(x)
    n = len(x)
    
    edges = np.linspace(0, n, target_len + 1, dtype=int)
    
    out = []
    for i in range(target_len):
        chunk = x[edges[i]:edges[i+1]]
        
        if len(chunk) == 0:
            out.append(np.nan)  # empty bin
        else:
            out.append(np.nanmean(chunk))
    
    return np.array(out)



print("Building model matrices...")

# matrices will be n_configs * n_subpaths, LEN_STANDARDISED_PATH for locations.
loc_th1 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH), dtype=float)
loc_th2 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH), dtype=float)

# matrices will be n_configs * n_subpaths, LEN_STANDARDISED_PATH*n_subpaths for dsr.
dsr_th1 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH*N_PHASES*len(states)), dtype=float)
dsr_th2 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH*N_PHASES*len(states)), dtype=float)

for c_idx, c in enumerate(configs):
    row_curr_config_start = c_idx * N_CONDS_PER_CONF
    for run_id, loc_mat, dsr_mat in [(1, loc_th1, dsr_th1),
                                     (2, loc_th2, dsr_th2)]:
        mode_vec = mode_locs[c][run_id]   # (360,)
        LEN_OG_SUBPATH = int(len(mode_vec)/N_CONDS_PER_CONF)
        # the DSR of the first step is the mode_vec downsampled 
        dsr_first_step = downsample_mode(mode_vec, target_len = LEN_STANDARDISED_PATH*N_PHASES*len(states))
        
        for n_subpath in range(N_CONDS_PER_CONF):
            # depending on how many subpaths I want, divide the path up
            subpath = mode_vec[n_subpath*LEN_OG_SUBPATH:(n_subpath+1)*LEN_OG_SUBPATH]
            # this step is not absolutely necessary
            downsampled_subpath = downsample_mode(subpath, target_len = LEN_STANDARDISED_PATH)
            loc_mat[row_curr_config_start + n_subpath,:] = downsampled_subpath
            
            # dsr: roll base vector left by whole-subpath steps
            roll_by = -(n_subpath) * LEN_STANDARDISED_PATH
            dsr_mat[row_curr_config_start + n_subpath, :] = np.roll(dsr_first_step, roll_by)

# state and feedback: one-hot over states, tiled
state_config    = np.zeros((N_CONDS_PER_CONF, len(states)))
feedback_config = np.zeros((N_CONDS_PER_CONF, len(states)))
phase_config = np.zeros((N_CONDS_PER_CONF, N_PHASES))

for s_i, s in enumerate(states):
    start_phase = RESOLUTIONx * s_i * N_PHASES
    state_config[start_phase: RESOLUTIONx * (s_i + 1) *N_PHASES, s_i] = 1
    if s == 'A':
        feedback_config[0:RESOLUTIONx, s_i] = 1
    for p_i in range(N_PHASES):
        phase_config[start_phase + p_i*RESOLUTIONx : start_phase+ (p_i+1)*RESOLUTIONx, p_i] = 1
        
        #phase_config[s_i*N_PHASES+p_i: s_i*N_PHASES+p_i+RESOLUTIONx, p_i] = 1

feedback_half = np.tile(feedback_config, (len(configs), 1))
state_half = np.tile(state_config, (len(configs), 1))
phase_half = np.tile(phase_config, (len(configs), 1))
    


# ── Clocks model matrices ──────────────────────────────────────────────
# For each config × run, call set_clocks_ephys using the mode location
# path (0-indexed, matching the ephys_validation.py convention).
# clocks_matrix shape: (n_fields*phases*n_neurons, N_BINS) = (324, 360)
# We take the mean clock-neuron activity per subpath as the feature vector,
# giving one row per condition (N_PHASES × len(states) conditions per config).

_n_clock_neurons = 9 * N_PHASES * (N_PHASES * len(states))  # 324

# # phase x location x future lag neurons
# clocks_th1 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, _n_clock_neurons), dtype=float)
# clocks_th2 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, _n_clock_neurons), dtype=float)

# # phase x location neurons
# midnight_th1 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9 * N_PHASES), dtype = float)
# midnight_th2 = np.zeros((N_CONFIGS * N_PHASES * len(states), 9 * N_PHASES), dtype = float)

# # location neurons
# loc_og_th1 = np.zeros((N_CONFIGS * N_PHASES * len(states), 9), dtype = float)
# loc_og_th2 = np.zeros((N_CONFIGS * N_PHASES * len(states), 9), dtype = float)

prep_models = {}
models_prepping = ['clocks', 'midnight', 'loc_og']
for m in models_prepping:
    prep_models[m] = {}
    for th in ([1,2]):
        if m == 'clocks':
            prep_models[m][th] = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, _n_clock_neurons), dtype=float)
        if m == 'midnight':
            prep_models[m][th] = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9 * N_PHASES), dtype = float)
        if m == 'loc_og':
            prep_models[m][th] = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9), dtype = float)
            

for c_idx, c in enumerate(configs):
    row_start  = c_idx * N_CONDS_PER_CONF
    # task config: 0-indexed, same convention as ephys_validation.py
    task_config = [int((int(x) - 1)) for x in c.split('-')]

    for run_id, clocks_mat in [(1, prep_models['clocks'][1]), (2, prep_models['clocks'][2])]:
        curr_task = mode_locs[c][run_id].copy()   # (N_BINS,), values 1-9 or NaN
        # forward-fill NaN entries before 0-indexing
        for i in range(len(curr_task)):
            if np.isnan(curr_task[i]):
                curr_task[i] = curr_task[i - 1] if i > 0 else float(task_config[0] + 1)
        # fields need to be between 0 and 8, as integers (ephys_validation.py)
        curr_task = [int((field_no - 1)) for field_no in curr_task]

        with contextlib.redirect_stdout(io.StringIO()):
            location_og_matrix = mc.simulation.predictions.set_location_ephys(curr_task, task_config, grid_size = 3, plotting = False)
            clocks_matrix, midnight_matrix = mc.simulation.predictions.set_clocks_ephys(
                curr_task, task_config,
                grid_size=3, phases=N_PHASES, plotting=False)

        # mean clock-neuron activity per subpath → one feature row per condition
        for n_subpath in range(N_CONDS_PER_CONF):
            t0 = n_subpath * LEN_OG_SUBPATH
            t1 = (n_subpath + 1) * LEN_OG_SUBPATH
            subpath_clocks = np.nanmean(clocks_matrix[:, t0:t1], axis=1)
            # replace nan with 0 
            prep_models['clocks'][run_id][row_start + n_subpath, :] = np.where(np.isnan(subpath_clocks), 0.0, subpath_clocks)
            #clocks_mat[row_start + n_subpath, :] = np.where(np.isnan(subpath_clocks), 0.0, subpath_clocks)
            
            # same for midnight model
            subpath_midnight = np.nanmean(midnight_matrix[:, t0:t1], axis=1)
            prep_models['midnight'][run_id][row_start + n_subpath, :] = np.where(np.isnan(subpath_midnight), 0.0, subpath_midnight)
            
            # same for location model
            subpath_location = np.nanmean(location_og_matrix[:, t0:t1], axis=1)
            prep_models['loc_og'][run_id][row_start + n_subpath, :] = np.where(np.isnan(subpath_location), 0.0, subpath_location)
            

if WITHIN_RUN:
    model_concat = {
        'location': (loc_th1 + loc_th2) / 2,
        'dsr':      (dsr_th1 + dsr_th2) / 2,
        'state':    state_half,
        'feedback': feedback_half,
        'clocks':   (prep_models['clocks'][1] + prep_models['clocks'][2]) / 2,
        'loc_og':   (prep_models['loc_og'][1] + prep_models['loc_og'][2]) / 2,
        'midnight': (prep_models['midnight'][1] + prep_models['midnight'][2]) / 2,
        'phase':    phase_half,
    }
else:
    model_concat = {
        'location': np.concatenate([loc_th1,    loc_th2],    axis=0),
        'dsr':      np.concatenate([dsr_th1,    dsr_th2],    axis=0),
        'state':    np.tile(state_half,    (2, 1)),
        'feedback': np.tile(feedback_half, (2, 1)),
        'clocks':   np.concatenate([prep_models['clocks'][1], prep_models['clocks'][2]], axis=0),
        'loc_og':   np.concatenate([prep_models['loc_og'][1], prep_models['loc_og'][2]], axis=0),
        'midnight': np.concatenate([prep_models['midnight'][1], prep_models['midnight'][2]], axis=0),
        'phase':    np.tile(phase_half,    (2, 1)),
    }

for m, mat in model_concat.items():
    print(f"  {m:12s}: {mat.shape}")



# ── Compute model RDMs ─────────────────────────────────────────────────
print("Computing model RDMs...")


# RDMs will be n_configs * n_subpaths * 4

model_RDMs = {}
for m in models:
    if m in ('location', 'dsr'):
        if WITHIN_RUN:
            model_RDMs[m] = mc.analyse.my_RSA.compute_hamming_distance_within(
                model_concat[m], plotting=PLOT_FIGS, include_diagonal=INCLUDE_DIAG,
                model_name=m, no_tasks=len(configs))
        else:
            model_RDMs[m] = mc.analyse.my_RSA.compute_hamming_distance(
                model_concat[m], plotting=PLOT_FIGS, include_diagonal=INCLUDE_DIAG,
                model_name=m, no_tasks=len(configs))
    else:
        if WITHIN_RUN:
            model_RDMs[m] = mc.analyse.my_RSA.compute_crosscorr_within(
                model_concat[m], plotting=PLOT_FIGS, include_diagonal=INCLUDE_DIAG,
                no_tasks=len(configs), model=m)
        else:
            model_RDMs[m] = mc.analyse.my_RSA.compute_crosscorr(
                model_concat[m], plotting=PLOT_FIGS, include_diagonal=INCLUDE_DIAG,
                no_tasks=len(configs), model=m)


# Patch feedback NaNs (bins where only state A is defined → undefined elsewhere)
nan_mask_feedback = np.isnan(model_RDMs['feedback'][0])
model_RDMs['feedback'][0][nan_mask_feedback] = 1

for m in models:
    print(f"  {m:12s} RDM: {model_RDMs[m][0].shape}")



# ── Feature 3: Phase mask ──────────────────────────────────────────────

def make_phase_mask(n_phases, n_configs, n_states = 4, include_diagonal=True):
    """
    Generates a boolean mask that is True for within-phase and false 
    for between phase
    """
    
    len_one_config = n_phases * n_states

    # within one config: same phase -> True
    idx = np.arange(len_one_config)
    phase_id = idx % n_phases
    single_config_rdm_mask = phase_id[:, None] == phase_id[None, :]

    # scale up to all configs.
    all_config_rdm_mask_2d = np.tile(single_config_rdm_mask, (n_configs, n_configs))
    
    # extract the 1d!
    n = all_config_rdm_mask_2d.shape[1]
    
    if include_diagonal:
        all_config_rdm_mask_1d = all_config_rdm_mask_2d[np.triu_indices(n, k=0)]
    else:
        all_config_rdm_mask_1d = all_config_rdm_mask_2d[np.triu_indices(n, k=1)]

    plt.figure();
    plt.imshow(all_config_rdm_mask_2d)

    for i in range(0,n,int(n/n_configs)):
        plt.axvline(i-0.5, color='white', ls = 'dashed')
    for i in range(0,n,int(n/n_configs)):
        plt.axhline(i-0.5, color='white', ls = 'dashed')
    for i in range(0,n,len_one_config):
        plt.axvline(i-0.5, color='white', ls = 'dashed')
    for i in range(0,n,len_one_config):
        plt.axhline(i-0.5, color='white', ls = 'dashed')
    if n_configs == 8:
        labels = ['3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3', '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6']
    plt.yticks(np.arange(2, all_config_rdm_mask_2d.shape[1], int(n/n_configs)), labels)
    
    
    return all_config_rdm_mask_1d



# ── Helper: assemble neuron matrix for a neuron subset ────────────────

def build_neuron_matrix(neuron_subset: pd.DataFrame, roi_name=None,
                        shift_dict=None):
    """
    Build the condition × neuron matrix for RSA.

    shift_dict : optional {neuron_label: int}
        Circular shift (0..N_BINS-1) applied to each neuron's raw 360-bin
        timecourse before downsampling.  Same shift for every config and
        both task halves of that neuron.

    Returns (matrix, n_excluded).
    Each neuron column is z-scored across conditions before return so that
    the RDM reflects pattern variation, not firing-rate amplitude.
    """
    uniq_neurons = neuron_subset['neuron_label'].unique()
    n_neurons    = len(uniq_neurons)

    th1 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, n_neurons))
    th2 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, n_neurons))

    for n_idx, n_label in enumerate(uniq_neurons):
        shift = shift_dict.get(n_label, 0) if shift_dict else 0
        for c_idx, c in enumerate(configs):
            rows = neuron_subset[
                (neuron_subset['neuron_label'] == n_label) &
                (neuron_subset['config'] == c)
            ]
            if len(rows) == 0:
                continue
            n_data = rows['neuron_data'].to_numpy()[0]   # (2, 360)
            import pdb; pdb.set_trace()
            raw0 = np.roll(n_data[0, :], shift) if shift else n_data[0, :]
            raw1 = np.roll(n_data[1, :], shift) if shift else n_data[1, :]
            th1[c_idx * N_CONDS_PER_CONF:(c_idx + 1) * N_CONDS_PER_CONF, n_idx] = downsample_mean(raw0, target_len=N_CONDS_PER_CONF)
            th2[c_idx * N_CONDS_PER_CONF:(c_idx + 1) * N_CONDS_PER_CONF, n_idx] = downsample_mean(raw1, target_len=N_CONDS_PER_CONF)

    # exclude neurons with NaN in either run
    valid_cols = ~(np.isnan(th1).any(axis=0) | np.isnan(th2).any(axis=0))
    n_excluded_nan = int(np.sum(~valid_cols))
    if shift_dict is None:
        print(f"  Excluding {n_excluded_nan} neurons with NaN.")

    # ── Neuron analysis (observed data only) ──────────────────────────
    if NEURON_ANALYSIS and shift_dict is None:
        label = roi_name or 'population'
        config_means = np.zeros((N_CONFIGS, n_neurons))
        for c_idx in range(N_CONFIGS):
            sl = slice(c_idx * N_CONDS_PER_CONF, (c_idx + 1) * N_CONDS_PER_CONF)
            config_means[c_idx, :] = np.nanmean(
                (th1[sl, :] + th2[sl, :]) / 2, axis=0)

        mu = np.nanmean(config_means, axis=0)
        sd = np.nanstd(config_means, axis=0)
        cv = np.where(np.abs(mu) > 1e-12, sd / np.abs(mu), 0.0)

        near_zero_frac = np.zeros((N_CONFIGS, n_neurons))
        for c_idx in range(N_CONFIGS):
            sl = slice(c_idx * N_CONDS_PER_CONF, (c_idx + 1) * N_CONDS_PER_CONF)
            avg = (th1[sl, :] + th2[sl, :]) / 2
            near_zero_frac[c_idx, :] = np.mean(np.abs(avg) < 1e-6, axis=0)

        max_cfg_mean = np.max(config_means, axis=0)
        dropout_mask = (
            (np.any(near_zero_frac > 0.5, axis=0)) &
            (max_cfg_mean > 0.1) & valid_cols
        )
        n_dropout = int(np.sum(dropout_mask))

        n_valid = int(valid_cols.sum())
        print(f"\n  ── Neuron analysis ({label}) ──")
        print(f"    Total neurons: {n_neurons}  |  valid (no NaN): {n_valid}")
        print(f"    Cross-config CV: mean={cv[valid_cols].mean():.3f}, "
              f"max={cv[valid_cols].max():.3f}")
        print(f"    Neurons with CV > 1.0: {int(np.sum(cv[valid_cols] > 1.0))}")
        print(f"    Neurons with signal dropout pattern: {n_dropout}")
        if n_dropout > 0:
            drop_names = uniq_neurons[dropout_mask]
            print(f"    Dropout neuron labels: {list(drop_names[:10])}"
                  + (" ..." if n_dropout > 10 else ""))

        valid_idx = np.where(valid_cols)[0]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                                 constrained_layout=True)
        fig.suptitle(f'Neuron diagnostics — {label}', fontsize=12,
                     weight='bold')

        ax = axes[0, 0]
        im = ax.imshow(config_means[:, valid_idx], aspect='auto',
                        cmap='viridis')
        ax.set_xlabel('Neuron (valid only)'); ax.set_ylabel('Config')
        ax.set_yticks(range(N_CONFIGS))
        ax.set_yticklabels(configs, fontsize=7)
        ax.set_title('Mean activity per config x neuron')
        plt.colorbar(im, ax=ax)

        ax = axes[0, 1]
        ax.hist(cv[valid_idx], bins=30, color='steelblue', alpha=0.7)
        ax.axvline(1.0, color='crimson', ls='--', label='CV=1')
        ax.set_xlabel('CV (across configs)'); ax.set_ylabel('Count')
        ax.set_title('Cross-config CV per neuron'); ax.legend(fontsize=7)

        ax = axes[1, 0]
        im = ax.imshow(near_zero_frac[:, valid_idx], aspect='auto',
                        cmap='Reds', vmin=0, vmax=1)
        ax.set_xlabel('Neuron (valid only)'); ax.set_ylabel('Config')
        ax.set_yticks(range(N_CONFIGS))
        ax.set_yticklabels(configs, fontsize=7)
        ax.set_title('Fraction near-zero bins per config x neuron')
        plt.colorbar(im, ax=ax)

        ax = axes[1, 1]
        for c_idx in range(N_CONFIGS):
            ax.plot(config_means[c_idx, valid_idx], alpha=0.5,
                    label=configs[c_idx], lw=0.8)
        ax.set_xlabel('Neuron (valid only)'); ax.set_ylabel('Mean activity')
        ax.set_title('Config-level mean activity per neuron')
        ax.legend(fontsize=6, ncol=2)

        diag_path = os.path.join(OUT_DIR, f'neuron_diagnostics_{label}.png')
        fig.savefig(diag_path, dpi=150, bbox_inches='tight')
        print(f"    Diagnostic plot saved → {diag_path}")
        plt.show()

        if n_dropout > 0:
            print(f"    Excluding {n_dropout} dropout neurons.")
            valid_cols &= ~dropout_mask

    n_excluded = int(np.sum(~valid_cols))
    th1 = th1[:, valid_cols]
    th2 = th2[:, valid_cols]

    if WITHIN_RUN:
        mat = (th1 + th2) / 2
    else:
        mat = np.concatenate([th1, th2], axis=0)

    # z-score each neuron across conditions so the RDM reflects
    # pattern variation, not amplitude differences between neurons
    col_mu = mat.mean(axis=0)
    col_sd = mat.std(axis=0)
    col_sd[col_sd == 0] = 1   # avoid div-by-zero for constant neurons
    mat = (mat - col_mu) / col_sd

    return mat, n_excluded



def perm_pvalue(null_vals, obs_val):
    """One-tailed permutation p-value with +1 correction."""
    null = np.asarray(null_vals, dtype=float)
    return float((1 + np.sum(null >= obs_val)) / (len(null) + 1))


def _compute_data_rdm(mat, plot=False, label=None):
    """Compute data RDM, dispatching to within- or across-run."""
    if WITHIN_RUN:
        return mc.analyse.my_RSA.compute_crosscorr_within(
            mat, plotting=plot, include_diagonal=INCLUDE_DIAG,
            no_tasks=len(configs), model=label)
    else:
        return mc.analyse.my_RSA.compute_crosscorr(
            mat, plotting=plot, include_diagonal=INCLUDE_DIAG,
            no_tasks=len(configs), model=label)



# ── RSA: whole-brain + per ROI ─────────────────────────────────────────
print("Running RSA...")
     
# create a mask that is ones if you compare the same state.
if MASK_CROSS_PHASE:
    # compute_crosscorr produces an (n x n) RDM where n = N_CONFIGS * N_PHASES*len(states)
    # (cross-corr between run1 and run2 halves; NOT the full 2*n matrix).
    # Build the phase mask for this single-half size only.
    phase_mask = make_phase_mask(N_PHASES, N_CONFIGS, len(states), include_diagonal=INCLUDE_DIAG)
    n_masked = int(np.sum(~phase_mask))
    print(f"  Phase mask: {n_masked} RDM entries masked out of {len(phase_mask)} "
          f"({100.0 * n_masked / len(phase_mask):.1f}%)")


def _scalar(arr):
    """Safely extract a Python float from a size-1 array or scalar."""
    return float(np.asarray(arr, dtype=float).ravel()[0])


def run_rsa(data_mat: np.ndarray, plot: bool = False, roi_name = None) -> dict:
    """
    Run RSA for one neural population matrix.
    Returns dict with 'data_rdm', 'unique' (per model), 'combined'.

    unique[m] = (t, beta, p)  — all plain Python floats
    combined  = {'t': array, 'beta': array, 'p': array}  — length n_models
    """
    if WITHIN_RUN:
        data_rdm = mc.analyse.my_RSA.compute_crosscorr_within(
            data_mat, plotting=PLOT_FIGS, include_diagonal=INCLUDE_DIAG,
            no_tasks=len(configs), model=roi_name)
    else:
        data_rdm = mc.analyse.my_RSA.compute_crosscorr(
            data_mat, plotting=plot, include_diagonal=INCLUDE_DIAG, no_tasks=len(configs), model=roi_name)

    
    # ── unique regression per model ────────────────────────────────────
    unique = {}
    for m in models:
        if m == 'dsr' and MASK_CROSS_PHASE:
            data_masked  = data_rdm[0].copy().astype(float)
            data_masked[~phase_mask] = np.nan
            model_masked = model_RDMs[m][0].copy().astype(float)
            model_masked[~phase_mask] = np.nan
            res = mc.analyse.my_RSA.evaluate_model(model_masked, data_masked)
        else:
            res = mc.analyse.my_RSA.evaluate_model(model_RDMs[m][0], data_rdm[0])

        unique[m] = (_scalar(res[0]), _scalar(res[1]), _scalar(res[2]))

    # ── combined regression (all models simultaneously) ────────────────
    if MASK_CROSS_PHASE:
        mdict = {}
        for m in model_RDMs:
            tmp = model_RDMs[m][0].copy().astype(float)
            tmp[~phase_mask] = np.nan
            mdict[m] = tmp
        data_masked = data_rdm[0].copy().astype(float)
        data_masked[~phase_mask] = np.nan
        stacked_models = np.stack([mdict[m] for m in combo_models], axis=1)
        print(f"these models are in combo model: {combo_models}")
        res_combo = mc.analyse.my_RSA.evaluate_model(stacked_models, data_masked)
    else:
        stacked_models = np.stack([model_RDMs[m][0] for m in combo_models], axis=1)
        print(f"these models are in combo model: {combo_models}")
        res_combo = mc.analyse.my_RSA.evaluate_model(stacked_models, data_rdm[0])

    combined = {
        't':    np.asarray(res_combo[0], dtype=float).ravel(),
        'beta': np.asarray(res_combo[1], dtype=float).ravel(),
        'p':    np.asarray(res_combo[2], dtype=float).ravel(),
    }

    return {'data_rdm': data_rdm, 'unique': unique, 'combined': combined}


# whole brain
wb_mat, wb_excluded = build_neuron_matrix(neurons, roi_name='whole_brain')
print(f"  whole_brain: excluded {wb_excluded} neurons with NaN.")
wb_res = run_rsa(wb_mat, plot=PLOT_FIGS, roi_name='All neurons')
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
    roi_mat, roi_excluded = build_neuron_matrix(subset, roi_name=roi)
    n_roi = n_roi_raw - roi_excluded
    print(f"  {roi}: excluded {roi_excluded} neurons with NaN.")
    if n_roi == 0:
        print(f"  {roi}: no valid neurons after NaN exclusion, skipping.")
        continue
    res     = run_rsa(roi_mat, plot=PLOT_FIGS, roi_name = roi)
    res['n_neurons'] = n_roi
    roi_results[roi] = res
    print(f"  {roi} (n={n_roi}) — unique t: "
          + ', '.join(f"{m}={res['unique'][m][0]:.2f}" for m in models))


# ── Collect present ROIs ───────────────────────────────────────────────
present_rois = [r for r in rois_of_interest if r in roi_results]



# ── Circular-shift permutation testing ────────────────────────────────
#
# Null: circularly shift each neuron's raw 360-bin timecourse by a random
# amount (0..N_BINS-1), then downsample + z-score as usual.
# Same shift for all configs and both task halves of a given neuron,
# but a different shift per neuron.
# Model RDMs stay fixed — only the data RDM changes.

if RUN_PERMUTATIONS:
    print(f"\nRunning {N_PERMS} circular-shift permutations ...")
    rng = np.random.default_rng(seed=PERM_SEED)

    # Neuron labels present in each ROI (after NaN/dropout exclusion)
    _roi_neuron_labels = {}
    _roi_neuron_labels['whole_brain'] = neurons['neuron_label'].unique()
    for roi in present_rois:
        if roi == 'whole_brain':
            _roi_neuron_labels[roi] = neurons['neuron_label'].unique()
        else:
            _roi_neuron_labels[roi] = neurons[neurons['roi'] == roi]['neuron_label'].unique()

    all_neuron_labels = neurons['neuron_label'].unique()
    n_all_neurons = len(all_neuron_labels)

    # Storage for null betas
    null_unique   = {roi: {m: [] for m in models} for roi in present_rois}
    null_combined = {roi: {m: [] for m in combo_models} for roi in present_rois}

    pm = phase_mask if MASK_CROSS_PHASE else None

    for perm_i in range(N_PERMS):
        # draw one random shift per neuron (shared across configs & halves)
        shifts = rng.integers(0, N_BINS, size=n_all_neurons)
        shift_dict = dict(zip(all_neuron_labels, shifts))

        for roi in present_rois:
            if roi == 'whole_brain':
                subset = neurons
            else:
                subset = neurons[neurons['roi'] == roi]

            perm_mat, _ = build_neuron_matrix(subset, roi_name=roi,
                                              shift_dict=shift_dict)

            # print the randomised data RDM on the first permutation (whole_brain)
            plot_first = (perm_i == 0 and roi == 'whole_brain')
            perm_data_rdm = _compute_data_rdm(
                perm_mat, plot=plot_first,
                label=f'perm0_{roi}' if plot_first else None)

            data_vec = perm_data_rdm[0].copy().astype(float)
            if pm is not None:
                data_vec[~pm] = np.nan

            # unique regressions
            for m in models:
                model_vec = model_RDMs[m][0].copy().astype(float)
                if pm is not None:
                    model_vec[~pm] = np.nan
                res = mc.analyse.my_RSA.evaluate_model(model_vec, data_vec)
                null_unique[roi][m].append(
                    float(np.asarray(res[1], dtype=float).ravel()[0]))

            # combined regression
            combo_vecs = []
            for m in combo_models:
                model_vec = model_RDMs[m][0].copy().astype(float)
                if pm is not None:
                    model_vec[~pm] = np.nan
                combo_vecs.append(model_vec)
            stacked = np.stack(combo_vecs, axis=1)
            res_c = mc.analyse.my_RSA.evaluate_model(stacked, data_vec)
            betas_c = np.asarray(res_c[1], dtype=float).ravel()
            for m_i, m in enumerate(combo_models):
                null_combined[roi][m].append(float(betas_c[m_i]))

        if (perm_i + 1) % 25 == 0 or perm_i == 0:
            print(f"  Permutation {perm_i + 1}/{N_PERMS} done")

    print(f"  All {N_PERMS} permutations done.")

    # ── Permutation p-values ──────────────────────────────────────────
    perm_p_unique   = {roi: {} for roi in present_rois}
    perm_p_combined = {roi: {} for roi in present_rois}

    print("\n  Permutation p-values (beta):")
    for roi in present_rois:
        for m in models:
            null_u = np.asarray(null_unique[roi][m], dtype=float)
            obs_u  = float(roi_results[roi]['unique'][m][1])
            p_u    = perm_pvalue(null_u, obs_u)
            perm_p_unique[roi][m] = p_u
            print(f"    {roi:20s}  {m:10s}  unique p={p_u:.4f}"
                  f"  (obs={obs_u:.4f}, null_mean={np.nanmean(null_u):.4f},"
                  f" null_sd={np.nanstd(null_u):.4f})")

        for m_i, m in enumerate(combo_models):
            null_c = np.asarray(null_combined[roi][m], dtype=float)
            obs_c  = float(roi_results[roi]['combined']['beta'][m_i])
            p_c    = perm_pvalue(null_c, obs_c)
            perm_p_combined[roi][m] = p_c
            print(f"    {roi:20s}  {m:10s}  combined p={p_c:.4f}"
                  f"  (obs={obs_c:.4f}, null_mean={np.nanmean(null_c):.4f},"
                  f" null_sd={np.nanstd(null_c):.4f})")

    # Save null distributions
    np.save(os.path.join(OUT_DIR, 'perm_null_unique.npy'),
            {roi: {m: np.asarray(null_unique[roi][m]) for m in models}
             for roi in present_rois})
    np.save(os.path.join(OUT_DIR, 'perm_null_combined.npy'),
            {roi: {m: np.asarray(null_combined[roi][m]) for m in combo_models}
             for roi in present_rois})
    print(f"  Null distributions saved to {OUT_DIR}")

    # ── Plot permutation distributions ────────────────────────────────
    def _perm_star(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'n.s.'

    for reg_label, null_dict, perm_p_dict, obs_key, model_list in [
        ('unique',   null_unique,   perm_p_unique,   'unique',   models),
        ('combined', null_combined, perm_p_combined, 'combined', combo_models),
    ]:
        n_rois_plot   = len(present_rois)
        n_models_plot = len(model_list)

        fig_p, axes_p = plt.subplots(
            n_models_plot, n_rois_plot,
            figsize=(max(4, n_rois_plot * 2.5), max(3, n_models_plot * 2.5)),
            sharex=False, sharey=False, constrained_layout=True)
        fig_p.suptitle(f'Permutation distributions — beta ({reg_label})',
                       fontsize=12, weight='bold')

        if n_models_plot == 1:
            axes_p = axes_p[np.newaxis, :]
        if n_rois_plot == 1:
            axes_p = axes_p[:, np.newaxis]

        for r_idx, roi in enumerate(present_rois):
            for m_idx, m in enumerate(model_list):
                ax = axes_p[m_idx, r_idx]
                null_vals = np.asarray(null_dict[roi][m], dtype=float)

                if obs_key == 'unique':
                    obs_val = float(roi_results[roi]['unique'][m][1])
                else:
                    obs_val = float(roi_results[roi]['combined']['beta'][
                        model_list.index(m)])

                p_val     = perm_p_dict[roi][m]
                null_mean = np.nanmean(null_vals)

                ax.hist(null_vals, bins=40, color='steelblue', alpha=0.7,
                        density=True)
                ax.axvline(null_mean, color='black', lw=1, ls='--',
                           label=f'null mean={null_mean:.3f}')
                ax.axvline(obs_val, color='crimson', lw=2,
                           label=f'obs={obs_val:.3f}')
                ax.set_title(
                    f'{roi} (n={roi_results[roi]["n_neurons"]})\n'
                    f'{m}  {_perm_star(p_val)} (p={p_val:.3f})',
                    fontsize=7, pad=3)
                ax.tick_params(labelsize=6)
                if r_idx == 0:
                    ax.set_ylabel('density', fontsize=6)
                if m_idx == n_models_plot - 1:
                    ax.set_xlabel('beta', fontsize=6)
                ax.legend(fontsize=5, loc='upper left')

        save_perm = os.path.join(
            OUT_DIR, f'DSR_RSA_perm_distributions_{reg_label}.png')
        fig_p.savefig(save_perm, dpi=150, bbox_inches='tight')
        print(f'  Saved → {save_perm}')
        plt.show()

    print("Permutation testing complete.")

    # ── Summary heatmap with permutation significance ─────────────────
    fig_unique, fig_combined = mc.plotting.cell_results.plot_rsa_heatmap(
        results         = roi_results,
        models          = models,
        combo_models    = combo_models,
        rois            = present_rois,
        title           = 'DSR RSA — beta per model × ROI',
        save_path       = os.path.join(OUT_DIR, 'DSR_RSA_heatmap.png'),
        perm_p_unique   = perm_p_unique,
        perm_p_combined = perm_p_combined,
    )
    plt.show()

else:
    fig_unique, fig_combined = mc.plotting.cell_results.plot_rsa_heatmap(
        results      = roi_results,
        models       = models,
        combo_models = combo_models,
        rois         = present_rois,
        title        = 'DSR RSA — beta per model × ROI',
        save_path    = os.path.join(OUT_DIR, 'DSR_RSA_heatmap.png'),
    )
    plt.show()

print('\nAll done.')
