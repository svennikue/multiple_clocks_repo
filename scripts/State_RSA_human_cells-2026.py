#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:45:17 2026

@author: Svenja Kuchenhoff

using averages created in prep_human_cells_State_RSA-2026.py and computing a
ROI-wise state RSA.

the cells are in the following format:
    - Group configs into N_GROUPS=6 pseudo-configs (merging if >6 exist)
    - Assign grid_no blocks INTACT to run1 / run2 (never split a block)
    - Average correct-trial firing rates within each (group × run)


1. per ROI (enth, hipp, amyg, OFC, PCC, ACC, mixed)
2. collapsed across the brain
3. for temporal lobe (ent+hipp) vs. frontal lobe (ACC+OFC)
        
        
"""

import os
import sys
import json
import numpy as np
import mc
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR  = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
# OUT_DIR   = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group'
N_GROUPS  = 6
N_BINS    = 360
INCLUDE_DIAG = False
SUBJECTS  = [f'{i:02}' for i in range(1, 64)]
EXCLUDE   = []   # add session numbers to skip, e.g. [19, 23]
SUBJECTS  = [s for s in SUBJECTS if int(s) not in EXCLUDE]
states = ['A', 'B', 'C', 'D']
rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY', 'R-WHITE-MATTER', 'OCCIP']


# ── Load all sessions ──────────────────────────────────────────────────
# put them into an neuron df
rows = []
for sub in SUBJECTS:
    SUB_DIR = f"{DATA_DIR}/s{sub}/state_avg"
    sesh_neurons = np.load(os.path.join(SUB_DIR, f's{sub}_neural_avg.npy'))
    with open(os.path.join(SUB_DIR, f's{sub}_neuron_meta.json'), 'r') as file:
        neuron_details = json.load(file)
    
    for n_idx, n in enumerate(sesh_neurons):
        row = {
            "session": sub,
            "neuron_idx": n_idx,
            "neuron_label": neuron_details["neuron_names"][n_idx],
            "roi": neuron_details["cell_labels"][n_idx],
            "electrode_label": neuron_details["electrode_labels"][n_idx],
            "neuron_data": n  # optional (can be large!)
        }
        rows.append(row)
        
neurons = pd.DataFrame(rows)

# create model RDMs
# format is time x neurons
state_one_half = np.zeros((360, len(states)))
feedback_one_half = np.zeros((360, len(states)))

for s_i, s in enumerate(states):
    if s == 'A':
        feedback_one_half[0:10, s_i] = 1
    state_start = s_i * 90
    state_end = (s_i+1) * 90
    print(state_start, state_end)
    state_one_half[state_start:state_end, s_i] = 1
    # import pdb; pdb.set_trace()
state_one_config = np.vstack((state_one_half,state_one_half))
feedback_one_config = np.vstack((feedback_one_half,feedback_one_half))
state = np.tile(state_one_config, (6,1))
feedback = np.tile(feedback_one_config, (6,1))

# input has to be time x neurons
state_RDM = mc.analyse.my_RSA.compute_crosscorr(state, plotting= True, include_diagonal=INCLUDE_DIAG)
feedback_RDM = mc.analyse.my_RSA.compute_crosscorr(feedback, plotting= True, include_diagonal=INCLUDE_DIAG)

A_state_mask = ~np.isnan(feedback_RDM[0])
# import pdb; pdb.set_trace()

# first make sure to not do this for the 'correct nans'
nan_mask = np.isnan(state_RDM[0])
# then turn all nans into 1s
nan_mask_other_states = np.isnan(feedback_RDM[0])
feedback_RDM[0][nan_mask_other_states] = 1



# then, per setting, compute the RSA.

# 1) WHOLE BRAIN 
# format is time x neurons
neurons_th1 = np.zeros((6*360 ,len(neurons)))
neurons_th2 = np.zeros((6*360 ,len(neurons)))

for i, n in enumerate(neurons['neuron_data']):
    th1 = n[:, 0, :]   # (6, 360)
    th2 = n[:, 1, :]   # (6, 360)
    
    neurons_th1[:, i] = th1.reshape(-1)  # or .flatten()
    neurons_th2[:, i] = th2.reshape(-1)

whole_brain = np.concatenate((neurons_th1, neurons_th2), axis = 0)
data_RDM = mc.analyse.my_RSA.compute_crosscorr(whole_brain, plotting= True, include_diagonal=INCLUDE_DIAG)

whole_brain_RSA_state = mc.analyse.my_RSA.evaluate_model(state_RDM[0], data_RDM[0])
if hasattr(whole_brain_RSA_state[0], '__len__'):
    whole_brain_RSA_state = (float(whole_brain_RSA_state[0]), float(whole_brain_RSA_state[1]), float(whole_brain_RSA_state[2]))
print(f"done with whole brain RSA. can predict state with t = {whole_brain_RSA_state[0]}; beta  = {whole_brain_RSA_state[1]}, p = {whole_brain_RSA_state[2]} ")

stacked_model_RDMs = np.stack((state_RDM[0], feedback_RDM[0]), axis=1)
whole_brain_RSA_combo = mc.analyse.my_RSA.evaluate_model(stacked_model_RDMs, data_RDM[0])
if hasattr(whole_brain_RSA_combo[0], '__len__'):
    whole_brain_RSA_combo = (np.array(whole_brain_RSA_combo[0], dtype=float), np.array(whole_brain_RSA_combo[1], dtype=float), np.array(whole_brain_RSA_combo[2], dtype=float))

print(f"if controlling for feedback, can predict state t = {whole_brain_RSA_combo[0][0]}; beta  = {whole_brain_RSA_combo[1][0]}, p = {whole_brain_RSA_combo[2][0]}; /n while controlling for 'new rep feedback', with t = {whole_brain_RSA_combo[0][1]}; beta  = {whole_brain_RSA_combo[1][1]}, p = {whole_brain_RSA_combo[2][1]} ")

# collect whole-brain results
roi_results = {
    'whole_brain': {
        'data_rdm': data_RDM,
        'n_neurons': len(neurons),
        'state': whole_brain_RSA_state,
        'state_controlled': whole_brain_RSA_combo
    }
}

# 2) per ROI
ROIs = neurons['roi'].unique()
rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY', 'R-WHITE-MATTER', 'OCCIP']

for roi in rois_of_interest:
    print(f"now computing RSA for {roi}")
    neurons_curr_roi = neurons[neurons['roi'] == roi]
    
    neurons_th1 = np.zeros((6*360 ,len(neurons_curr_roi)))
    neurons_th2 = np.zeros((6*360 ,len(neurons_curr_roi)))

    for i, n in enumerate(neurons_curr_roi['neuron_data']):
        th1 = n[:, 0, :]   # (6, 360)
        th2 = n[:, 1, :]   # (6, 360)
        
        neurons_th1[:, i] = th1.reshape(-1)  # or .flatten()
        neurons_th2[:, i] = th2.reshape(-1)

    roi_brain = np.concatenate((neurons_th1, neurons_th2), axis = 0)
    data_RDM = mc.analyse.my_RSA.compute_crosscorr(roi_brain, plotting= True, include_diagonal=INCLUDE_DIAG)

    roi_RSA_state = mc.analyse.my_RSA.evaluate_model(state_RDM[0], data_RDM[0])
    if hasattr(roi_RSA_state[0], '__len__'):
        roi_RSA_state = (float(roi_RSA_state[0]), float(roi_RSA_state[1]), float(roi_RSA_state[2]))
    print(f"done with ROI {roi} RSA. can predict state with t = {roi_RSA_state[0]}; beta  = {roi_RSA_state[1]}, p = {roi_RSA_state[2]} ")

    stacked_model_RDMs = np.stack((state_RDM[0], feedback_RDM[0]), axis=1)
    roi_RSA_combo = mc.analyse.my_RSA.evaluate_model(stacked_model_RDMs, data_RDM[0])
    if hasattr(roi_RSA_combo[0], '__len__'):
        roi_RSA_combo = (np.array(roi_RSA_combo[0], dtype=float), np.array(roi_RSA_combo[1], dtype=float), np.array(roi_RSA_combo[2], dtype=float))

    print(f"if controlling for feedback, can predict state t = {roi_RSA_combo[0][0]}; beta  = {roi_RSA_combo[1][0]}, p = {roi_RSA_combo[2][0]}; \n while controlling for 'new rep feedback', with t = {roi_RSA_combo[0][1]}; beta  = {roi_RSA_combo[1][1]}, p = {roi_RSA_combo[2][1]} ")

    roi_results[roi] = {
        'data_rdm': data_RDM,
        'n_neurons': len(neurons_curr_roi),
        'state': roi_RSA_state,
        'state_controlled': roi_RSA_combo
    }

    
    

# plotting
# Overview plot section requested by user
rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY', 'R-WHITE-MATTER', 'OCCIP']

# 1) Data RDM comparison (lower triangle) for the selected ROIs
plt.figure(figsize=(15, 12))
for idx, roi_name in enumerate(rois_of_interest, start=1):
    if roi_name not in roi_results:
        print(f'ROI {roi_name} was not found; skipping plot.')
        continue
    rdm = roi_results[roi_name]['data_rdm']
    n = rdm.shape[0]
    mask = np.triu(np.ones_like(rdm, dtype=bool), k=1)
    plot_rdm = np.copy(rdm)
    plot_rdm[mask] = np.nan

    ax = plt.subplot(3, 3, idx)
    im = ax.imshow(plot_rdm, aspect='auto', cmap='RdBu_r', vmin=0, vmax=2)
    ax.set_title(f'{roi_name} (n={roi_results[roi_name]["n_neurons"]})')
    # demarcate 6 config blocks of 360 bins
    for boundary in range(1, 6):
        c = boundary * 360
        if c < n:
            ax.axvline(c - 0.5, color='k', linestyle='--', linewidth=1)
            ax.axhline(c - 0.5, color='k', linestyle='--', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 0.96, 1])
plt.colorbar(im, ax=plt.gcf().axes, shrink=0.7, location='right')
plt.suptitle('Data RDM lower triangle: whole brain + selected ROIs', y=0.99, fontsize=14)

# 2) and 3) t-values with significance for state and state+feedback
labels = []
state_t = []
state_p = []
state_ctrl_t = []
state_ctrl_p = []
neuron_counts = []

for roi_name in rois_of_interest:
    if roi_name not in roi_results:
        continue
    labels.append(roi_name)
    r = roi_results[roi_name]
    state_t.append(r['state'][0])
    state_p.append(r['state'][2])
    state_ctrl_t.append(r['state_controlled'][0][0])
    state_ctrl_p.append(r['state_controlled'][2][0])
    neuron_counts.append(r['n_neurons'])

# helper for star annotation

def p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''

x = np.arange(len(labels))
width = 0.35

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

axes[0].bar(x - width/2, state_t, width, label='state')
axes[0].set_ylabel('t value')
axes[0].set_title('State RSA (no control)')
for i, (t_val, p_val) in enumerate(zip(state_t, state_p)):
    stars = p_to_stars(p_val)
    axes[0].text(i - width/2, t_val + 0.05 * np.sign(t_val), f'{stars}\nn={neuron_counts[i]}', ha='center', va='bottom', fontsize=8)

axes[1].bar(x - width/2, state_ctrl_t, width, label='state controlled feedback', color='tab:orange')
axes[1].set_ylabel('t value')
axes[1].set_title('State RSA controlling for feedback')
for i, (t_val, p_val) in enumerate(zip(state_ctrl_t, state_ctrl_p)):
    stars = p_to_stars(p_val)
    axes[1].text(i - width/2, t_val + 0.05 * np.sign(t_val), f'{stars}\nn={neuron_counts[i]}', ha='center', va='bottom', fontsize=8)

axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=45, ha='right')
axes[1].legend()
plt.tight_layout(rect=[0, 0, 1, 0.96])

print('\nOverview plotting complete: data RDM + state and state-controlled t values with stars.')

print('\nAll done.')