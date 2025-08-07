#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATTEMPT OF PLOTTING STATE CELLS AS CLOVER PLOT.

@author: xpsy1114
"""

##Importing libraries
from joblib import dump, load
import os, sys, pickle, time, re, csv
from collections import defaultdict#
import mc
from operator import itemgetter

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d



# these are the models you can choose from.
results_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/cells_per_model-11-07-2025"

list_of_all_results = []
state_results = []
# List all files and print their basenames
for file in os.listdir(results_path):
    if 'state' in file or 'stat_model' in file:
        print(file) 
        state_results.append(file)
    list_of_all_results.append(file)

# first, focus on the state ones.
print(f"now loading {state_results[3]} ")
state_df = pd.read_csv(f"{results_path}/{state_results[3]}")
# filter for significant location cells
state_df_sig = state_df[state_df['p_val_time'] < 0.05].reset_index(drop=True)
print(f"based on {state_results[3]}, there are {len(state_df_sig)} significant cells/")
# these are the cells that you will want to plot.
#add a column of session names
sessions = []
for idx, row in state_df_sig.iterrows():
    sessions.append('sub-'+row['cell'].split('_')[2])
state_df_sig['sessions'] = sessions 

# sessions to load
sig_sessions = np.unique(sessions)


# focus on this entorhinal cell: 06-chan119_sesh_01_REC

# identify the session, from there load the data. 
# start with session as there seem to be a few location cells.
sesh = '01'
target_cells = []
avg_corr_target_cells = []
for idx, row in state_df_sig.iterrows():
    if f"sesh_{sesh}" in row['cell']:
        target_cells.append(row['cell'])
        avg_corr_target_cells.append(row['average_corr'])

print(target_cells)

data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
subjects = [sesh]
data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)

# transform neuron per task into 360 timebins
n_bins = 360
warped_trials = []

# go through 'electrode_labels' and figure out the correct index of the cell array.
# take the first target cell.
target_idx = [
    idx for idx, label_cell in enumerate(data[f"sub-{sesh}"]['electrode_labels'])
    if any(target in label_cell or label_cell in target for target in target_cells)
]


def smooth_circular(x, sigma=2):
    return gaussian_filter1d(np.hstack((x, x, x)), sigma)[len(x):2*len(x)]

# Parameters
n_bins = 72
sigma = 2
repeated = False

# Colors for quadrants A, B, C, D
colors = ['#F15A29', '#F7931E', '#C7C6E2', '#6B60AA']
quadrants = ['A', 'B', 'C', 'D']
quarter_size = n_bins // 4

# Warp trials

# Task 1, 01-mLF2Ca01
# CAREFUL WITH RUNNING THIS
# OPENS LOADS OF FIGURES


warped_trials = []
for t_idx, target_cell in enumerate(target_idx):
    avg_corr = state_df_sig['average_corr'][state_df_sig['cell'] == target_cells[t_idx]].to_list()
    for task_i, task in enumerate(data[f"sub-{sesh}"]['reward_configs']):
        timing_curr_task = data[f"sub-{sesh}"]['timings'][task_i]
        firing_rate = data[f"sub-{sesh}"]['neurons'][task_i][target_cell,:]
    
        for trial in timing_curr_task:
            if np.any(np.isnan(trial)) == True:
                continue
            start = int(trial[0])
            end = int(trial[-1]) + 1
            segment = firing_rate[start:end]
        
            original_time = np.linspace(0, 1, len(segment))
            new_time = np.linspace(0, 1, n_bins)
        
            interp_func = interp1d(original_time, segment, kind='linear')
            warped = interp_func(new_time)
            warped_trials.append(warped)
    
        Neuron_norm = np.vstack(warped_trials)
        
        # Compute mean and sem
        mean_ = np.nanmean(Neuron_norm, axis=0)
        mean_smooth = smooth_circular(mean_, sigma=sigma)
        
        sem_ = st.sem(Neuron_norm, axis=0, nan_policy='omit')
        sem_smooth = smooth_circular(sem_, sigma=sigma)
        
        upperx = mean_smooth
        lowerx = mean_smooth
        
        # Plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
        theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    
        # Plot each colored quadrant
        for i in range(4):
            idx_start = i * quarter_size
            idx_end = (i + 1) * quarter_size
            ax.plot(theta[idx_start:idx_end], mean_smooth[idx_start:idx_end], color=colors[i], linewidth=2)
    
        # Overlay full outline
        ax.plot(theta, mean_smooth, color='black', linewidth=1.5, alpha=0.5)
    
        # Add quadrant labels
        label_r = ax.get_rmax() * 1.1
        for i, label in enumerate(quadrants):
            angle = (i + 0.5) * (np.pi / 2)
            ax.text(angle, label_r, label,
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color=colors[i])
        
        ax.set_title(f"task {task_i}, {data[f'sub-{sesh}']['electrode_labels'][target_cell]} \n average corr = {avg_corr[0]}", va='bottom', fontsize=14)
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    
    
