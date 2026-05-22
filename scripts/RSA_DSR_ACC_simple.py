#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:52:57 2026

@author: a desparate attempt to simplify.
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
OUT_DIR      = os.path.join(DATA_DIR, 'group', 'DSR_RSA_simple')


configs = [
    '3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
    '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6',
]

N_CONFIGS = len(configs)
N_CONDS_PER_CONF = 12
LEN_STANDARDISED_PATH = 10
N_PHASES = 3
states           = ['A', 'B', 'C', 'D']
RESOLUTIONx = 1
PLOT_FIGS = False 
N_PERMUTATIONS = 500 # None or 300

models = ['location', 'dsr', 'state', 'dsr_old', 'midnight', 'dsr_old_now_next']
combo_models = {'loc_dsr': ['location', 'dsr_old'],
                'loc_st_dsr': ['location', 'state', 'dsr_old'],
                'loc_nownext_dsr': ['location', 'dsr_old_now_next']}


# ── ROI labelling ─────────────────────────────────────────────────────
# Which ROI-label column of the MNI table (scripts/cell_to_roi_MNI.py)
# decides which cells count as ACC:
#   'final_roi'     -> original labelling
#   'alt_final_roi' -> alternative labelling (ACC split by y-cutoff,
#                      OFC11+OFC13+ventral_ACC collapsed into medialOFC)
ROI_LABEL_COLUMN = 'final_roi'
ACC_ROI_LABELS   = ('ACC',)   # ROI value(s) treated as ACC for this RSA
ROI_TABLE_PATH   = os.path.join(DATA_DIR, 'neurons_with_final_roi_labels.csv')



# for categorical variables (e.g. location/dsr encoding)
def downsample_mode(x, target_len=10):
    x = np.asarray(x)
    block = len(x) // target_len
    return np.array([
        stats.mode(x[i*block:(i+1)*block], keepdims=False).mode
        for i in range(target_len)
    ])

def eval_tuple(rdm, data_rdm):
    """Return (t, beta, p) as plain Python floats."""
    return tuple(_scalar(v) for v in mc.analyse.my_RSA.evaluate_model(rdm, data_rdm))


def build_combo_rdm(rdm_dict, combo_list):
    """Stack several model RDMs into one multi-model design matrix."""
    return np.stack([rdm_dict[m][0] for m in combo_list], axis=1)


def _scalar(arr):
    """Safely extract a Python float from a size-1 array or scalar."""
    return float(np.asarray(arr, dtype=float).ravel()[0])


# ── ROI-table lookup ─────────────────────────────────────────────────
def parse_neuron_label(label):
    """'01_07-07-chan120-EC' -> (subject:int, cell_idx:int), else (None, None)."""
    try:
        sub_str, rest = label.split('_', 1)
        return int(sub_str), int(rest.split('-', 1)[0])
    except (ValueError, IndexError):
        return None, None


def _load_roi_table(path, roi_col):
    """Load the MNI-based ROI table, indexed by (subject, cell idx)."""
    df = pd.read_csv(path)
    for c in ['subject', 'cell idx', roi_col]:
        if c not in df.columns:
            raise ValueError(
                f"ROI table {path} is missing column {c!r}  "
                f"(re-run scripts/cell_to_roi_MNI.py if {roi_col!r} is absent)")
    df = df.copy()
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


def get_neuron_roi(label):
    """ROI label (ROI_LABEL_COLUMN) for a neuron, or None if not in the table."""
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return None
    try:
        roi = ROI_TABLE.loc[(sub, cell_idx), ROI_LABEL_COLUMN]
    except KeyError:
        return None
    if isinstance(roi, pd.Series):           # duplicate rows — first non-null
        roi = roi.dropna().iloc[0] if roi.notna().any() else None
    return None if (roi is None or pd.isna(roi)) else str(roi)


ROI_TABLE = _load_roi_table(ROI_TABLE_PATH, ROI_LABEL_COLUMN)
print(f"Loaded ROI table: {len(ROI_TABLE)} cells, column '{ROI_LABEL_COLUMN}', "
      f"ACC = {ACC_ROI_LABELS}")


# set up dicts and lists to load data
acc_neurons, locs = {}, {}
acc_neurons_all, locs_all = {}, {}
perm_ACC_neurons_all, perm_ACC_neurons = {}, {}

for conf in configs:
    acc_neurons[conf] = {}
    perm_ACC_neurons[conf]= {}
    locs[conf] = {}
    acc_neurons_all[conf] = []
    perm_ACC_neurons_all[conf] = []
    locs_all[conf] = []
    for th in [1,2]:
        acc_neurons[conf][th] = []
        perm_ACC_neurons[conf][th] = []
        locs[conf][th] = []
        
    
 
    
N_CONFIGS = len(configs)

with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json'), 'r') as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())

os.makedirs(OUT_DIR, exist_ok=True)


for sub_str in SUBJECTS:
    data_dict = mc.analyse.helpers_human_cells.load_norm_data(DATA_DIR, [sub_str])
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
        
        for n_lab in curr_neurons:
            if get_neuron_roi(n_lab) in ACC_ROI_LABELS:
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
data_RDM = mc.analyse.my_RSA.compute_crosscorr(mat.T, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model='data in ACC')

# z-scored ACC neurons.
mu = np.nanmean(mat, axis=1)      # one mean per neuron
sd = np.nanstd(mat, axis=1)       # one std per neuron
mat_z = (mat.T - mu) / sd
data_RDM_z = mc.analyse.my_RSA.compute_crosscorr(mat_z, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model='data in z-scored ACC neurons')



row_all = []
row_labels_all = []
for config in configs:
    all_neuron_values = acc_neurons_all[config]
    row_all.append(all_neuron_values)
    row_labels_all.append(config)
mat_all = np.hstack(row_all)

data_RDM_within, data_RDM_across, data_RDM_full = mc.analyse.my_RSA.compute_crosscorr_within(mat_all.T, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model='data in ACC', block_size=N_CONDS_PER_CONF)

# z-scored ACC neurons.
# z-scored ACC neurons.
mu_all = np.nanmean(mat_all, axis=1)      # one mean per neuron
sd_all = np.nanstd(mat_all, axis=1)       # one std per neuron
mat_all_z = (mat_all.T - mu_all) / sd_all
data_RDM_within_z, data_RDM_across_z, data_RDM_full_z = mc.analyse.my_RSA.compute_crosscorr_within(mat_all_z, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model='data in z-scored ACC neurons', block_size=N_CONDS_PER_CONF)



# create a mode path.
mode_locs = {}
mode_locs_all = {}
for c in configs:
    mode_locs[c] = {}
    locs_all_per_conf = locs_all[c]
    stacked_all = np.vstack(locs_all_per_conf) # (n_trials_total, 360)
    m_all = stats.mode(stacked_all, axis=0, keepdims=False, nan_policy='omit')
    mode_locs_all[c] = m_all.mode.astype(float)
    
    for th in [1, 2]:
        loc_per_config = locs[c][th]
        stacked = np.vstack(loc_per_config) # (n_trials_total, 360)
        m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
        mode_locs[c][th] = m.mode.astype(float)
        
        
print("Mode-location arrays built.")

# build model rdms

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


model_concat = {
    'location': np.concatenate([loc_th1,    loc_th2],    axis=0),
    'dsr':      np.concatenate([dsr_th1,    dsr_th2],    axis=0),
    'state':    np.tile(state_half,    (2, 1))
    #'feedback': np.tile(feedback_half, (2, 1))
}


# old way 
dsr_old, midnight, dsr_old_now_next = [], [], []
for th in [1,2]:
    for c in configs:
        walked   = mode_locs[c][th]
        walked = [int(w-1) for w in walked]
        loc_og_matrix, phase_og_matrix, stat_matrix, midnight_matrix, dsr_matrix, phas_stat_matrix, dsr_now_next_matrix = mc.simulation.predictions.model_DSR(locations = walked, no_phase_neurons=N_PHASES)
        

        dsr_downsampled = dsr_matrix.reshape(dsr_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis=2)
        midn_downsampled = midnight_matrix.reshape(midnight_matrix.shape[0],N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 2)
        dsr_nownext_downsampled = dsr_now_next_matrix.reshape(dsr_now_next_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 2)
        
        dsr_old.append(dsr_downsampled)
        midnight.append(midn_downsampled)
        dsr_old_now_next.append(dsr_nownext_downsampled)


model_concat['dsr_old'] = np.transpose(np.concatenate(dsr_old, axis = 1))
model_concat['midnight'] = np.transpose(np.concatenate(midnight, axis = 1))
model_concat['dsr_old_now_next'] = np.transpose(np.concatenate(dsr_old_now_next, axis = 1))




model_RDMs = {}
model_RDMs_within = {}
model_RDMs_across = {}
full = {}

for m in model_concat:

    if m in ('location', 'dsr'):
        model_RDMs[m] = mc.analyse.my_RSA.compute_hamming_distance(
            model_concat[m], plotting=PLOT_FIGS, include_diagonal=False,
            model_name=m, no_tasks=len(configs))
        
        model_RDMs_within[m], model_RDMs_across[m], full[m] = mc.analyse.my_RSA.compute_hamming_distance_within(
            model_concat[m][0:len(mat_all_z)], plotting=False,
            include_diagonal=False,
            model_name=m, no_tasks=len(configs),
            block_size=N_CONDS_PER_CONF)
    else:
        model_RDMs[m] = mc.analyse.my_RSA.compute_crosscorr(
            model_concat[m], plotting=PLOT_FIGS, include_diagonal=False,
            no_tasks=len(configs), model=m)
        model_RDMs_within[m], model_RDMs_across[m], full[m] = mc.analyse.my_RSA.compute_crosscorr_within(
            model_concat[m][0:len(mat_all_z)], plotting=False,
            include_diagonal=False,
            no_tasks=len(configs), model=m,
            block_size=N_CONDS_PER_CONF)
        

print("Computing RSA...")

# ── Empirical results ────────────────────────────────────────────────
empirical_results = {}
empirical_results_z = {}

empirical_combo_results = {}
empirical_combo_results_z = {}


test_specs = [
    ('crossval', model_RDMs, data_RDM[0], data_RDM_z[0]),
    ('within',   model_RDMs_within, data_RDM_within[0], data_RDM_within_z[0]),
    ('across',   model_RDMs_across, data_RDM_across[0], data_RDM_across_z[0]),
]

for test_name, rdm_dict, raw_data, z_data in test_specs:
    empirical_results[test_name] = {
        m: eval_tuple(rdm_dict[m][0], raw_data) for m in models
    }
    empirical_results_z[test_name] = {
        m: eval_tuple(rdm_dict[m][0], z_data) for m in models
    }


    empirical_combo_results[test_name] = {combo: mc.analyse.my_RSA.evaluate_model(
        build_combo_rdm(rdm_dict, combo_models[combo]), raw_data) for combo in combo_models}
    
    empirical_combo_results_z[test_name] = {combo: mc.analyse.my_RSA.evaluate_model(
            build_combo_rdm(rdm_dict, combo_models[combo]), z_data)for combo in combo_models}



tests = ['crossval', 'crossval_z', 'within', 'within_z', 'across', 'across_z']
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
        perm_data_RDM = mc.analyse.my_RSA.compute_crosscorr(perm_mat.T, plotting=False, include_diagonal=False, no_tasks=len(configs), model='permuted data in ACC')

        # z-scored ACC neurons.
        mu = np.nanmean(perm_mat, axis=1)      # one mean per neuron
        sd = np.nanstd(perm_mat, axis=1)       # one std per neuron
        perm_mat_z = (perm_mat.T - mu) / sd
        perm_data_RDM_z = mc.analyse.my_RSA.compute_crosscorr(perm_mat_z, plotting=False, include_diagonal=False, no_tasks=len(configs), model='permuted data in z-scored ACC neurons')



        row_all = []
        row_labels_all = []
        for config in configs:
            all_neuron_values = perm_ACC_neurons_all[config][start_perm:end_perm]
            row_all.append(all_neuron_values)
            row_labels_all.append(config)
        perm_mat_all = np.hstack(row_all)

        perm_data_RDM_within, perm_data_RDM_across, perm_data_RDM_full = mc.analyse.my_RSA.compute_crosscorr_within(perm_mat_all.T, plotting=False, include_diagonal=False, no_tasks=len(configs), model='data in ACC', block_size=N_CONDS_PER_CONF)

        # z-scored ACC neurons.
        # z-scored ACC neurons.
        mu_all = np.nanmean(perm_mat_all, axis=1)      # one mean per neuron
        sd_all = np.nanstd(perm_mat_all, axis=1)       # one std per neuron
        perm_mat_all_z = (perm_mat_all.T - mu_all) / sd_all
        perm_data_RDM_within_z, perm_data_RDM_across_z, _perm_data_RDM_full_z = mc.analyse.my_RSA.compute_crosscorr_within(perm_mat_all_z, plotting=False, include_diagonal=False, no_tasks=len(configs), model='data in z-scored ACC neurons', block_size=N_CONDS_PER_CONF)

        
        perm_specs = [
            ('crossval',   model_RDMs,        perm_data_RDM[0]),
            ('crossval_z', model_RDMs,        perm_data_RDM_z[0]),
            ('within',     model_RDMs_within,  perm_data_RDM_within[0]),
            ('within_z',   model_RDMs_within,  perm_data_RDM_within_z[0]),
            ('across',     model_RDMs_across,  perm_data_RDM_across[0]),
            ('across_z',   model_RDMs_across,  perm_data_RDM_across_z[0]),
        ]

        for test_name, rdm_dict, data_rdm in perm_specs:
            for m in models:
                beta = eval_tuple(rdm_dict[m][0], data_rdm)[1]
                perm_results[test_name][m].append(beta)

            for combo, combo_list in combo_models.items():
                stacked = build_combo_rdm(rdm_dict, combo_list)
                res = mc.analyse.my_RSA.evaluate_model(stacked, data_rdm)
            
                perm_results_combo[test_name][combo]['t'].append(np.asarray(res[0], dtype=float).ravel())
                perm_results_combo[test_name][combo]['beta'].append(np.asarray(res[1], dtype=float).ravel())
                perm_results_combo[test_name][combo]['p'].append(np.asarray(res[2], dtype=float).ravel())
                



        if (perm_i + 1) % 25 == 0 or perm_i == 0:
            print(f"  Permutation {perm_i + 1}/{N_PERMUTATIONS} done")
        # # print("Computing perRSA...")
        # for m in model_RDMs:
        #     # ['crossval', 'crossval_z', 'within', 'within_z', 'across', 'across_z']
        #     res = mc.analyse.my_RSA.evaluate_model(model_RDMs[m][0], perm_data_RDM[0])
        #     perm_results['crossval'][m].append(res[1])

        #     z_res = mc.analyse.my_RSA.evaluate_model(model_RDMs[m][0], perm_data_RDM_z[0])
        #     perm_results['crossval_z'][m].append(z_res[1])

        #     # WITHIN CONFIGS
        #     w_res = mc.analyse.my_RSA.evaluate_model(model_RDMs_within[m][0], perm_data_RDM_within[0])
        #     perm_results['within'][m].append(w_res[1])

        #     w_z_res = mc.analyse.my_RSA.evaluate_model(model_RDMs_within[m][0], perm_data_RDM_within_z[0])
        #     perm_results['within_z'][m].append(w_z_res[1])

        #     # ACROSS CONFIGS
        #     a_res = mc.analyse.my_RSA.evaluate_model(model_RDMs_across[m][0], perm_data_RDM_across[0])
        #     perm_results['across'][m].append(a_res[1])

        #     a_z_res = mc.analyse.my_RSA.evaluate_model(model_RDMs_across[m][0], perm_data_RDM_across_z[0])
        #     perm_results['across_z'][m].append(a_z_res[1])

        # for combo in combo_models:
        #     stacked = np.stack([model_RDMs[m][0] for m in combo_models[combo]], axis=1)
        #     # crossval
        #     results[combo] = mc.analyse.my_RSA.evaluate_model(stacked, perm_data_RDM[0])
        #     results_z[combo] = mc.analyse.my_RSA.evaluate_model(stacked, perm_data_RDM_z[0])
        #     # within
        #     stacked_w = np.stack([model_RDMs_within[m][0] for m in combo_models[combo]], axis=1)
        #     results_within[combo] = mc.analyse.my_RSA.evaluate_model(stacked_w, perm_data_RDM_within[0])
        #     results_within_z[combo] = mc.analyse.my_RSA.evaluate_model(stacked_w, perm_data_RDM_within_z[0])
        #     # across
        #     stacked_a = np.stack([model_RDMs_across[m][0] for m in combo_models[combo]], axis=1)
        #     results_across[combo] =  mc.analyse.my_RSA.evaluate_model(stacked_a, perm_data_RDM_across[0])
        #     results_across_z[combo] = mc.analyse.my_RSA.evaluate_model(stacked_a, perm_data_RDM_across_z[0])
                

    



def plot_perm_hist_grid_combo(
    perm_results_combo,
    empirical_combo_results,
    empirical_combo_results_z,
    combo_key,
    combo_models,
    tests=('crossval', 'crossval_z', 'within', 'within_z', 'across', 'across_z'),
    bins=25,
    density=True,
    figsize_per_panel=(2.0, 1.8),
    alpha=0.05,
):
    cols = combo_models[combo_key]
    nrows, ncols = len(tests), len(cols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharey=False,
        constrained_layout=True,
    )

    if nrows == 1:
        axes = np.array([axes])

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

    return fig, axes

for combo_key in combo_models:
    fig2, axes2 = plot_perm_hist_grid_combo(
        perm_results_combo=perm_results_combo,
        empirical_combo_results=empirical_combo_results,
        empirical_combo_results_z=empirical_combo_results_z,
        combo_key=combo_key,
        combo_models=combo_models,
        bins=30,
        alpha=0.05
    )
    plt.show()
