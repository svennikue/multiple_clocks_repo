#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

this script allows to plot single neurons, if you know which ones to plot.
also implements some start of a phase-tuning and a future-locaiton tuning analysis.

PLOTTING STATE CELLS AS CLOVER PLOT.
PLOTTING LOCATION CELLS ON A SPATIAL RATE MAP
PLOTTING PEAKS OF FUTURE LOCATION ENCODING
PLOTTING PROGRESS-TO-GOAL CODING FOR ONE SESSION

@author: xpsy1114
"""

##Importing libraries
from joblib import dump, load
import os, sys, pickle, time, re, csv
from collections import defaultdict#
import mc
from operator import itemgetter
from itertools import product
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

#####
# SETTINGS
plot_state = False
plot_spatial_corr = False
plot_future_spatial_corr = False
plot_goal_progress_tuning = False
plot_goal_progress_tuning_half_split = True

#import pdb; pdb.set_trace()
# 

####
# HELPER FUNCTIONS

# sig_sessions_test = sig_sessions[0:3]
x=(0,1,2)
Task_grid=np.asarray(list(product(x, x)))
Task_grid_plotting=np.column_stack((Task_grid[:,1],Task_grid[:,0]))




def smooth_circular(arr, sigma=2):
    """
    Apply Gaussian smoothing to circular data (like 360-bin polar plots).
    Wraps data before smoothing to avoid edge artifacts.
    """
    extended = np.concatenate([arr, arr, arr])  # tripled to allow wrap-around
    smoothed = gaussian_filter1d(extended, sigma=sigma)
    return smoothed[len(arr):2*len(arr)]  # return center part


def plot_state_polar(firing_across_states, title_string):
    # Colors for quadrants A, B, C, D
    colors = ['#F15A29', '#F7931E', '#C7C6E2', '#6B60AA']
    quadrants = ['A', 'B', 'C', 'D']
    n_bins = len(firing_across_states)
    quarter_size = n_bins // 4

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    # Plot each colored quadrant
    for i in range(4):
        idx_start = i * quarter_size
        idx_end = (i + 1) * quarter_size
        ax.plot(theta[idx_start:idx_end], firing_across_states[idx_start:idx_end], color=colors[i], linewidth=2)

    # Overlay full outline
    ax.plot(theta, firing_across_states, color='black', linewidth=1.5, alpha=0.5)

    # Add quadrant labels
    label_r = ax.get_rmax() * 1.1
    for i, label in enumerate(quadrants):
        angle = (i + 0.5) * (np.pi / 2)
        ax.text(angle, label_r, label,
                ha='center', va='center',
                fontsize=14, fontweight='bold', color=colors[i])

    ax.set_title(title_string, va='bottom', fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
                
def scale_per_neuron(mat, method="minmax", eps=1e-9):
    """
    Scale each row independently.
    mat: (n_neurons, n_timebins)
    method: 'minmax' or 'z'
    Returns scaled matrix with same shape, NaNs preserved.
    """
    out = mat.astype(float).copy()
    if method == "minmax":
        row_min = np.nanmin(out, axis=1, keepdims=True)
        row_max = np.nanmax(out, axis=1, keepdims=True)
        denom = np.where(np.isfinite(row_max - row_min), row_max - row_min, np.nan)
        out = (out - row_min) / (denom + eps)  # [0,1], flat rows -> ~0
    elif method == "z":
        row_mean = np.nanmean(out, axis=1, keepdims=True)
        row_std  = np.nanstd(out, axis=1, keepdims=True)
        out = (out - row_mean) / (row_std + eps)
    else:
        raise ValueError("method must be 'minmax' or 'z'")
    return out


def avg_phase_from_trials(df_trials_360):
    """
    df_trials_360: DataFrame shape (n_trials, 360)
    Returns: 1x90 vector = mean across trials & cycles (folded by 90)
    """
    # average across trials -> (360,)
    trace = df_trials_360.mean(axis=0, skipna=True).to_numpy()
    # fold into cycles x 90 and average across cycles -> (90,)
    return trace.reshape(cycles, period).mean(axis=0)



# these are the models you can choose from.
results_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/cells_per_model-11-07-2025"

list_of_all_results = []


if plot_state == True:
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
    
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, subjects)
    
    
    # the same script but for data norm
    target_idx = [
        idx for idx, label_cell in enumerate(data_norm[f"sub-{sesh}"]['electrode_labels'])
        if any(target in label_cell or label_cell in target for target in target_cells)
    ]


    import pdb; pdb.set_trace()
    # CONTINUE HERE
    # find out which grids are unique
    unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(data_norm[f"sub-{sesh}"]['reward_configs'], axis=0,
                                                            return_index=True,
                                                            return_inverse=True,
                                                            return_counts=True)
    # AVERAGE OVER THE SAME GRIDS!!
    # for i, unique_task_idx in enumerate(idx_unique_grid): 
    for t_idx, target_cell in enumerate(target_cells):
        avg_corr = state_df_sig['average_corr'][state_df_sig['cell'] == target_cells[t_idx]].to_list()
        for curr_neuron in data_norm[f"sub-{sesh}"]['normalised_neurons']:
            if target_cell.startswith(curr_neuron):
                avg_corr_across_tasks = list(np.mean(data_norm[f"sub-{sesh}"]['normalised_neurons'][curr_neuron]))
                smoothed_corr_across_tasks = smooth_circular(avg_corr_across_tasks, sigma=4) 
                plot_state_polar(smoothed_corr_across_tasks, f"across all tasks, {target_cell} \n average corr = {avg_corr[0]}")
    
                for task_idx in range(0, np.max(data_norm[f"sub-{sesh}"]['beh']['grid_no'])):
                    avg_corr_curr_grid = list(np.mean(data_norm[f"sub-{sesh}"]['normalised_neurons'][curr_neuron][data_norm[f"sub-{sesh}"]['beh']['grid_no'] == task_idx]))
                    smoothed_corr = smooth_circular(avg_corr_curr_grid, sigma=4)  # tweak sigma as needed
                    plot_state_polar(smoothed_corr, f"task {task_idx}, {target_cell} \n average corr = {avg_corr[0]}")
                    
             
                
    
    
 
if plot_spatial_corr == True:
    # these are the models you can choose from.
    results_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/cells_per_model-11-07-2025"
    
    list_of_all_results = []
    loc_results = []
    # List all files and print their basenames
    for file in os.listdir(results_path):
        if 'location' in file or 'loc_model' in file:
            print(file) 
            loc_results.append(file)
        list_of_all_results.append(file)

    print(f"now loading {loc_results[4]} ")
    loc_df = pd.read_csv(f"{results_path}/{loc_results[4]}")
    # filter for significant location cells
    loc_df_sig = loc_df[loc_df['p_val_time'] < 0.05].reset_index(drop=True)
    print(f"based on {loc_results[4]}, there are {len(loc_df_sig)} significant cells/")
    # these are the cells that you will want to plot.
    #add a column of session names
    sessions = []
    for idx, row in loc_df_sig.iterrows():
        sessions.append('sub-'+row['cell'].split('_')[2])
    loc_df_sig['sessions'] = sessions 
    
    # sessions to load
    sig_sessions = np.unique(sessions)
    loc_df_sig.head()

    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    

    for sub in sig_sessions:
        if sub == 'sub-18':
            continue
        sesh = sub.split('-')[1]
        print(sesh)
        target_cells = []
        avg_corr_target_cells = []
        for idx, row in loc_df_sig.iterrows():
            if f"sesh_{sesh}" in row['cell']:
                target_cells.append(row['cell'])
                avg_corr_target_cells.append(row['average_corr'])
                
                
        data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, [sesh])
        data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [sesh])

        
        # go through 'electrode_labels' and figure out the correct index of the cell array.
        # take the first target cell.
        target_idx = [
            idx for idx, label_cell in enumerate(data_norm[f"sub-{sesh}"]['electrode_labels'])
            if any(target in label_cell or label_cell in target for target in target_cells)
        ]
        
        # find out which grids are unique
        grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
        grid_configs = data_norm[f"sub-{sesh}"]['beh'][grid_cols].to_numpy()
        
        # Apply np.unique just like in your example
        unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(
            grid_configs,
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True
        )

    
        print(f"unique grids are {unique_grids}")
        
        firing_rate_cells = []
        for t_idx, target_cell in enumerate(target_cells):
            # initialise mean firing rates array for all target neurons
            mean_firing_rates_locs = np.full((9, len(unique_grids)), np.nan, dtype=float)
            # then go through all unique grids and concatenate the neurons to compute the firing rate for each location per grid.
            for curr_neuron in data_norm[f"sub-{sesh}"]['normalised_neurons']:
                if target_cell.startswith(curr_neuron):
                    for task_idx, _ in enumerate(unique_grids):
                    # COMPUTING FIRING RATES PER LOCATION FOR EACH GRID
                        # Boolean mask for trials in the current grid
                        mask_curr_grid = (idx_same_grids == task_idx)
                        
                        # Extract all matching trials (shape: [n_trials, 360])
                        neuron_curr_grid = data_norm[f"sub-{sesh}"]['normalised_neurons'][curr_neuron].loc[mask_curr_grid].to_numpy()
                        locations_curr_grid = data_norm[f"sub-{sesh}"]['locations'][mask_curr_grid].to_numpy()
                        # Flatten all 360-bin trials into one long 1D array
                        all_reps_neuron_concat = neuron_curr_grid.reshape(-1)  # shape: (n_trials * 360,)
                        all_reps_locs_concat = locations_curr_grid.reshape(-1)

                        for loc in range(1,10):
                            mean_firing_rates_locs[loc-1, task_idx] = all_reps_neuron_concat[all_reps_locs_concat == loc].mean()
                            
                firing_rate_cells.append(mean_firing_rates_locs)
                        
                        
                
                # COMPUTING LOCATION CONSISTENCY ACROSS GRIDS
                # compute mean correlation between all grids
                mean_grid_corr = []
                for neuron_i, FR_maps_neuron in enumerate(firing_rate_cells):
                    # Compute pairwise Pearson correlation between columns (i.e., grids)
                    corr_matrix = np.corrcoef(FR_maps_neuron.T)  # transpose: shape (n_grids, 9) --> correlation of grids
            
                    # Mask upper triangle excluding diagonal
                    upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
                    mean_corr = np.nanmean(corr_matrix[upper_triangle_indices])  # nanmean in case of NaNs
                    mean_grid_corr.append(mean_corr)
                    print(f"Neuron {neuron_i}: Mean grid correlation = {mean_corr:.3f}")
            
                    
            
                # PLOTTING
                # firing_rate_cells is a list of (9 x N) arrays, one per neuron
                for neuron_i, FR_maps_neuron in enumerate(firing_rate_cells):
                    max_rate = np.nanmax(FR_maps_neuron)
                    min_rate = np.nanmin(FR_maps_neuron)
            
                    fig1, f1_axes = plt.subplots(figsize=(20, 5), ncols=FR_maps_neuron.shape[1], nrows=1, constrained_layout=True)
            
                    # If only one grid, f1_axes won't be iterable – wrap in list
                    if FR_maps_neuron.shape[1] == 1:
                        f1_axes = [f1_axes]
            
                    for task_ind in np.arange(FR_maps_neuron.shape[1]):
                        FR_map_neuron_task = FR_maps_neuron[:, task_ind].reshape(3, 3)
                        ax1 = f1_axes[task_ind]
            
                        im = ax1.matshow(FR_map_neuron_task, cmap='coolwarm', vmin=min_rate, vmax=max_rate)
                        ax1.axis('off')
            
                        for i_r, reward in enumerate(unique_grids[task_ind]):
                            if i_r == 0:
                                write = 'A'
                            elif i_r == 1:
                                write = 'B'
                            elif i_r == 2:
                                write = 'C'
                            elif i_r == 3:
                                write = 'D'
                            pos_idx = int(reward) - 1  # assuming rewards are 1-based location indices
                            row, col = Task_grid_plotting[pos_idx]
                            ax1.text(col, row, write, ha='center', va='center',
                                     color='black', fontsize=16, fontweight='bold')
            
                    fig1.suptitle(f'Neuron {target_cells[neuron_i]} \n avg encoding corr = {avg_corr_target_cells[neuron_i]} \n mean grid corr = {mean_grid_corr[neuron_i]}', fontsize=20)
                    plt.show()
                    
                    import pdb; pdb.set_trace()
            

if plot_future_spatial_corr == True:
    # COMPUTING MEAN PLACE ENCODING FOR FUTURE
    
    temporal_shift = 30
    shifts = np.arange(0, 360, temporal_shift)  # 12 shifts
    all_shift_curves = {}  # neuron -> [mean_grid_corr at each shift]
    best_shift_maps = {}            # neuron -> 9xN FR map for best shift
    best_shift_values = {}          # neuron -> (best_shift_bins, best_mean_corr)
    firing_rate_cells_per_neuron_best = []  # keep your old structure if you want later reuse
    
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    sesh = '25'
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [sesh])
    
    # find out which grids are unique
    grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
    grid_configs = data_norm[f"sub-{sesh}"]['beh'][grid_cols].to_numpy()
    
    # Apply np.unique just like in your example
    unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(
        grid_configs,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True
    )
    
    #for t_idx, target_cell in enumerate(target_cells):
    for t_idx, target_cell in enumerate(data_norm[f"sub-{sesh}"]['normalised_neurons']):
        mean_corr_per_shift = []
        fr_maps_by_shift = []
        
        for shift in shifts:
            # --- your existing per-neuron/per-grid mean FR by location, with a key change: shift LOCATIONS ---
            # --- build the 9 x N map for this shift (your logic + shifted LOCATIONS) ---
            mean_firing_rates_locs = np.full((9, len(unique_grids)), np.nan, dtype=float)
    
            for curr_neuron in data_norm[f"sub-{sesh}"]['normalised_neurons']:
                if target_cell.startswith(curr_neuron):
                    for task_idx, _ in enumerate(unique_grids):
                        mask_curr_grid = (idx_same_grids == task_idx)
    
                        # Shapes: [n_trials, 360]
                        neuron_curr_grid = data_norm[f"sub-{sesh}"]['normalised_neurons'][curr_neuron].loc[mask_curr_grid].to_numpy()
                        locations_curr_grid = data_norm[f"sub-{sesh}"]['locations'].loc[mask_curr_grid].to_numpy()
    
                        # Shift locations **forward** by `shift` bins so that location[t] -> location[t+shift]
                        # Roll negative to align future location with current firing time t
                        if shift != 0:
                            locations_curr_grid = np.roll(locations_curr_grid, shift=-shift, axis=1)
    
                        # Flatten trials × time
                        fr_all  = neuron_curr_grid.reshape(-1)
                        loc_all = locations_curr_grid.reshape(-1)
    
                        # (Optional) keep valid bins: loc 1..9, finite FR
                        valid = np.isfinite(fr_all) & (loc_all >= 1) & (loc_all <= 9)
                        if not np.any(valid):
                            continue
    
                        fr_all  = fr_all[valid]
                        loc_all = loc_all[valid].astype(int)
    
                        # Your original per-location loop (kept for clarity)
                        for loc in range(1, 10):
                            sel = (loc_all == loc)
                            if np.any(sel):
                                mean_firing_rates_locs[loc-1, task_idx] = fr_all[sel].mean()
                            else:
                                mean_firing_rates_locs[loc-1, task_idx] = np.nan
    
            fr_maps_by_shift.append(mean_firing_rates_locs)
            # --- mean grid correlation for this shift ---
            if np.all(np.isnan(mean_firing_rates_locs)):
                mean_corr_per_shift.append(np.nan)
            else:
                corr_matrix = np.corrcoef(mean_firing_rates_locs.T)  # grids x grids
                upper = np.triu_indices_from(corr_matrix, k=1)
                mean_corr_per_shift.append(np.nanmean(corr_matrix[upper]))
    
    
    
        # pick best shift
        best_idx = np.nanargmax(mean_corr_per_shift) if np.any(np.isfinite(mean_corr_per_shift)) else None
        if best_idx is None:
            # fallback: no data
            continue
    
        best_shift = shifts[best_idx]
        best_map   = fr_maps_by_shift[best_idx]
        best_corr  = float(mean_corr_per_shift[best_idx])
    
        all_shift_curves[target_cell] = np.array(mean_corr_per_shift, dtype=float)
        best_shift_maps[target_cell] = best_map
        best_shift_values[target_cell] = (int(best_shift), best_corr)
        firing_rate_cells_per_neuron_best.append(best_map)  # optional: mirrors your old list
    
        # -------- PLOT ONLY THE BEST-SHIFT MAP FOR THIS NEURON --------
        FR_maps_neuron = best_map  # shape (9, N_grids)
        max_rate = np.nanmax(FR_maps_neuron)
        min_rate = np.nanmin(FR_maps_neuron)
    
        fig1, f1_axes = plt.subplots(figsize=(20, 5), ncols=FR_maps_neuron.shape[1], nrows=1, constrained_layout=True)
        if FR_maps_neuron.shape[1] == 1:
            f1_axes = [f1_axes]
    
        for task_ind in range(FR_maps_neuron.shape[1]):
            FR_map_neuron_task = FR_maps_neuron[:, task_ind].reshape(3, 3)
            ax1 = f1_axes[task_ind]
    
            im = ax1.matshow(FR_map_neuron_task, cmap='coolwarm', vmin=min_rate, vmax=max_rate)
            ax1.axis('off')
    
            # Overlay reward letters (A/B/C/D) at their locations
            for i_r, reward in enumerate(unique_grids[task_ind]):
                write = "ABCD"[i_r]  # compact mapping
                pos_idx = int(reward) - 1  # rewards are 1-based location indices
                row, col = Task_grid_plotting[pos_idx]  # your mapping from index -> (row, col)
                ax1.text(col, row, write, ha='center', va='center',
                         color='black', fontsize=16, fontweight='bold')
    
        fig1.suptitle(
            f'Neuron {target_cell}\n'
            f'Best shift = {best_shift} bins | Mean grid corr = {best_corr:.3f}',
            fontsize=20
        )
        plt.show()
    
        # ---- Optional: also plot mean grid correlation vs shift (diagnostic) ----
        plt.figure(figsize=(7,4))
        for neuron, curve in all_shift_curves.items():
            plt.plot(shifts, curve, marker='o', label=neuron)
        plt.xlabel('Temporal shift (bins of 30)')
        plt.ylabel('Mean grid correlation')
        plt.title(f"Neuron {target_cell}\n Spatial encoding consistency vs temporal shift")
        plt.hlines(0, shifts[0], shifts[-1], 'k', '-')
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    
    
    
if plot_goal_progress_tuning == True:
    # NOTE: tthis isn't very conclusive...
    
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    sesh = '04'
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [sesh])
    
    period = 90  # bins per task cycle
    timebins_total = 360
    cycles = timebins_total // period
    assert timebins_total % period == 0, "360 must be divisible by 90."
    
    neuron_names = list(data_norm[f"sub-{sesh}"]['normalised_neurons'].keys())

    # --- inputs from your structure ---
    # Each neuron is a DataFrame (145 x 360)
    # neurons_dict = {neuron_name: df145x360, ...}  # 6 entries
    # grid_no series aligns with the 145 trial rows:
    grid_no = data_norm[f"sub-{sesh}"]['beh']['grid_no']   # pandas Series length 145
    n_neurons = len(neuron_names)

    # --- ALL-TASKS average: build (n_neurons x 90), get peak order ---
    avg_phase_all = np.vstack([
        avg_phase_from_trials(data_norm[f"sub-{sesh}"]['normalised_neurons'][name])
        for name in neuron_names
    ])  # (n_neurons, 90)
    
    scaled_all = scale_per_neuron(avg_phase_all)
    peak_bins_all = np.nanargmax(scaled_all, axis=1)
    order_all = np.argsort(peak_bins_all)
    names_sorted = [neuron_names[i] for i in order_all]
    scaled_all_sorted = scaled_all[order_all]
    
    # --- PER-TASK averages: list of (n_neurons x 90), scaled and sorted using order_all ---
    scaled_tasks_sorted = []
    task_labels = []
    for t in range(np.max(grid_no)):
        mask = (grid_no == t).to_numpy()
        phase_t = np.vstack([
            avg_phase_from_trials(data_norm[f"sub-{sesh}"]['normalised_neurons'][name].iloc[mask])
            for name in neuron_names
        ])  # (n_neurons, 90)
    
        scaled_t = scale_per_neuron(phase_t)
        scaled_tasks_sorted.append(scaled_t[order_all])
        task_labels.append(f"Task {int(t)}")
    
    # --- PLOT: columns = [All tasks | each task] ---
    cols = 1 + len(scaled_tasks_sorted)
    fig, axes = plt.subplots(1, cols, figsize=(3.8*cols, 6), constrained_layout=True)
    
    # Ensure axes is iterable
    if cols == 1:
        axes = [axes]
    
    # Column 0: all-tasks
    im = axes[0].imshow(scaled_all_sorted, aspect='auto', interpolation='none', vmin=0, vmax=1)
    axes[0].set_title("All tasks")
    axes[0].set_xlabel("Phase bin (0–89)")
    axes[0].set_ylabel("Neurons (sorted by all-tasks peak)")
    axes[0].set_yticks(np.arange(n_neurons))
    axes[0].set_yticklabels(names_sorted)
    
    # Other columns: each task (sorted with same row order)
    for i, mat in enumerate(scaled_tasks_sorted, start=1):
        axes[i].imshow(mat, aspect='auto', interpolation='none', vmin=0, vmax=1)
        axes[i].set_title(task_labels[i-1])
        axes[i].set_xlabel("Phase bin (0–89)")
        axes[i].set_yticks([])  # keep labels only on first column
    
    # one shared colorbar
    cbar = fig.colorbar(im, ax=axes, location='right', fraction=0.025, pad=0.02)
    cbar.set_label("Scaled firing (per neuron, min–max)")
    
    plt.show()




if plot_goal_progress_tuning_half_split == True:
    sessions=list(range(1,60))
    # source_dir = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    all_neurons = []
    COLUMNS = ["session_id", "neuron_id", "roi", "first_half_avg", "second_half_avg"]
    results = []
    for sesh in sessions:
        # load data
        data = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [f"{sesh:02}"])
        if not data:
            continue
        for neuron in data[f"sub-{sesh:02}"]['normalised_neurons']:
            roi = neuron.split('-')[-1]
            all_repeats = len(data[f"sub-{sesh:02}"]['normalised_neurons'][neuron])
            import pdb; pdb.set_trace()
            first_half_split = np.mean(data[f"sub-{sesh:02}"]['normalised_neurons'][neuron][int(all_repeats/2):]).to_numpy()
            second_half_split = np.mean(data[f"sub-{sesh:02}"]['normalised_neurons'][neuron][0:int(all_repeats/2)]).to_numpy()
            results.append({"session_id": sesh,
                            "neuron_id": neuron,
                            "roi": roi,
                            "first_half_avg": first_half_split,
                            "second_half_avg": second_half_split})
    
    results_df = pd.DataFrame(results, columns = COLUMNS)

    # df columns: ["session_id", "neuron_id", "first_half_avg", "second_half_avg"]
    # first_half_avg / second_half_avg: length-360 arrays (list or np.array)
    
    # --- helpers ---
    # ---- helpers ----
    def gaussian_kernel(window=5):
        if window % 2 == 0:
            window += 1
        sigma = window / 6.0                      # ~99.7% mass in window
        r = window // 2
        x = np.arange(-r, r+1, dtype=float)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        return k
    
    def circ_gauss_smooth_1d(row, kernel):
        row  = np.asarray(row, dtype=float)
        mask = np.isfinite(row).astype(float)
        x    = np.where(mask == 1.0, row, 0.0)
        pad  = len(kernel) // 2
        x_p  = np.pad(x,    (pad, pad), mode="wrap")
        m_p  = np.pad(mask, (pad, pad), mode="wrap")
        num  = np.convolve(x_p, kernel, mode="valid")
        den  = np.convolve(m_p, kernel, mode="valid")
        out  = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        return out
    
    def smooth_matrix_rows(M, window=5):
        k = gaussian_kernel(window)
        return np.vstack([circ_gauss_smooth_1d(row, k) for row in M])
    
    def zscore_rows(M):
        mu = np.nanmean(M, axis=1, keepdims=True)
        sd = np.nanstd(M, axis=1, keepdims=True)
        sd = np.where(sd == 0, np.nan, sd)
        return (M - mu) / sd
    
    def stack_column(df, colname, length=360):
        arrs = []
        for x in df[colname].values:
            a = np.asarray(x, dtype=float)
            if a.shape != (length,):
                raise ValueError(f"{colname} entry has shape {a.shape}, expected ({length},)")
            arrs.append(a)
        return np.vstack(arrs)
    
    def block_nanmean_1d(row, block=4):
        """Downsample by non-overlapping blocks (length must be divisible)."""
        row = np.asarray(row, dtype=float)
        L = row.shape[0]
        if L % block != 0:
            raise ValueError(f"length {L} not divisible by block {block}")
        return np.nanmean(row.reshape(-1, block), axis=1)
    
    def block_nanmean_rows(M, block=4):
        return np.vstack([block_nanmean_1d(r, block=block) for r in M])



    # ---- build matrices (N, 360) ----
    filtered_df = results_df[results_df['roi'].isin(['LACC','LPCC', 'ACC','LdACC','LmOFC','LpgACC','LvCC','RACC', 'RPCC', 'RPvmPFC',  'RmOFC', 'RpgACC'])]
    first  = stack_column(filtered_df, "first_half_avg",  360)
    second = stack_column(filtered_df, "second_half_avg", 360)
    
    first  = stack_column(results_df, "first_half_avg",  360)
    second = stack_column(results_df, "second_half_avg", 360)
    
    
    # ---- smooth → z-score → downsample (360→90 via mean of every 4 bins) ----
    first_s  = smooth_matrix_rows(first,  window=10)
    second_s = smooth_matrix_rows(second, window=10)
    
    first_z  = zscore_rows(first_s)
    second_z = zscore_rows(second_s)
    
    first_90  = block_nanmean_rows(first_z,  block=4)   # (N, 90)
    second_90 = block_nanmean_rows(second_z, block=4)   # (N, 90)
    
    # ---- sort by peak index in FIRST (on 90-length signals) ----
    has_finite = np.isfinite(first_90).any(axis=1)
    peak_idx   = np.where(has_finite, np.nanargmax(first_90, axis=1), np.inf)
    order      = np.argsort(peak_idx)
    
    first_sorted_90  = first_90[order]
    second_sorted_90 = second_90[order]
    
    # ---- shared color scale (robust) ----
    vmin, vmax = np.nanpercentile(np.vstack([first_sorted_90, second_sorted_90]), [1, 99])
    
    # ---- plot ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    im0 = axes[0].imshow(first_sorted_90,  aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("ONLY PFC First half (smoothed, z-scored, downsampled to 90; sorted by peak index)")
    axes[0].set_ylabel("Neurons (sorted)")
    
    im1 = axes[1].imshow(second_sorted_90, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title("ONLY PFC Second half (same neuron order)")
    axes[1].set_xlabel("Quarter-averaged bin (0…89)")
    axes[1].set_ylabel("Neurons (sorted)")





    

