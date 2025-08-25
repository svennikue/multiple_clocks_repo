#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:41:15 2025

@author: Svenja Küchenhoff

This script loads the cells normalised in 360 bins and computes their spatial peaks.
It does so using cross-validation: testing the spatial specificity across grids,
leaving one out, respectively. 

"""

import mc
import fire
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from scipy.stats import pearsonr
from scipy.stats import mode
import scipy.stats as st 
import pandas as pd
import copy
from pathlib import Path
from matplotlib.patches import Patch
       
import itertools
from collections import Counter

# import pdb; pdb.set_trace()



def get_data(sub):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [f"{sub:02}"])
    return data_norm, data_folder


    
def filter_data(data, session, rep_filter):
    # filter can be 'all', 'all_correct', 'early', 'late'
    filtered_data = copy.deepcopy(data)
    if rep_filter =='all_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['correct']==1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['correct']==1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['correct']==1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['correct']==1].reset_index(drop = True)    
    elif rep_filter == 'early':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4])].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4])].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4])].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4])].reset_index(drop = True)    
    elif rep_filter == 'late':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([5,6,7,8,9])].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([5,6,7,8,9])].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([5,6,7,8,9])].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([5,6,7,8,9])].reset_index(drop = True)    


    # import pdb; pdb.set_trace()
    return filtered_data

def plot_spatial_shifts_and_rate_maps(FR_maps_neuron, unique_grids, cell_name, all_shift_curves, best_idx):
    
    temporal_shift = 30
    shifts = np.arange(0, 360, temporal_shift)  # 12 shifts
    best_shift = shifts[best_idx]
    best_corr  = float(all_shift_curves[best_idx])


    x=(0,1,2)
    Task_grid=np.asarray(list(product(x, x)))
    Task_grid_plotting=np.column_stack((Task_grid[:,1],Task_grid[:,0]))
    
    max_rate = np.nanmax(FR_maps_neuron)
    min_rate = np.nanmin(FR_maps_neuron)
    # max_rate = 23
    # min_rate = 6
    
        
        
    fig1, f1_axes = plt.subplots(figsize=(20, 5), ncols=FR_maps_neuron.shape[1], nrows=1, constrained_layout=True)
    if FR_maps_neuron.shape[1] == 1:
        f1_axes = [f1_axes]

    for task_ind in range(FR_maps_neuron.shape[1]):
        FR_map_neuron_task = FR_maps_neuron[:, task_ind].reshape(3, 3)
        ax1 = f1_axes[task_ind]

        im = ax1.matshow(FR_map_neuron_task, cmap='coolwarm', vmin=min_rate, vmax=max_rate)
        ax1.axis('off')

        # # Overlay reward letters (A/B/C/D) at their locations
        # for i_r, reward in enumerate(unique_grids[task_ind]):
        #     write = "ABCD"[i_r]  # compact mapping
        #     pos_idx = int(reward) - 1  # rewards are 1-based location indices
        #     row, col = Task_grid_plotting[pos_idx]  # your mapping from index -> (row, col)
        #     ax1.text(col, row, write, ha='center', va='center',
        #              color='black', fontsize=16, fontweight='bold')

    fig1.suptitle(
        f'Neuron {cell_name}\n'
        f'Best shift = {best_shift} bins | Mean grid corr = {best_corr:.3f}',
        fontsize=20
    )
    plt.show()

    # ---- Optional: also plot mean grid correlation vs shift (diagnostic) ----
    plt.figure(figsize=(7,4))
    plt.plot(shifts, all_shift_curves, marker='o', label={cell_name})
    plt.xlabel('Temporal shift (bins of 30)')
    plt.ylabel('Mean grid correlation')
    plt.title(f"Neuron {cell_name}\n Spatial encoding consistency vs temporal shift \n Peak shift at {best_shift} degrees")
    plt.hlines(0, shifts[0], shifts[-1], 'k', '-')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()
        
    
   
def comp_peak_spatial_tuning(neurons, locs, beh, cell_name, idx_same_grids, plotting=False, perm_no = None, combine_two_grids = False):
    temporal_shift = 30
    shifts = np.arange(0, 360, temporal_shift)  # 12 shifts
    unique_grids = np.unique(idx_same_grids)
        
    mean_corr_per_shift = []
    # mean_corr_per_shift_test_pd = []
    fr_maps_by_shift = []
    
    for shift in shifts:
        mean_firing_rates_locs = np.full((9, len(unique_grids)), np.nan, dtype=float)
        for task_idx, _ in enumerate(unique_grids):
            mask_curr_grid = (idx_same_grids == task_idx)
            neurons_curr_grid = neurons[mask_curr_grid]

            locs_shifted = np.roll(locs, shift=-shift, axis=1)
            locs_curr_grid = locs_shifted[mask_curr_grid]
            
            if perm_no:
                # circular shift of location time series by random bins along time_axis
                T = locs_curr_grid.shape[1]
                k = int(perm_no.integers(1, T)) if T > 1 else 0
                locs_curr_grid = np.roll(locs_curr_grid, -k, axis=1)
            
            # Flatten trials × time
            fr_all_reps  = neurons_curr_grid.reshape(-1).astype(float)
            loc_all_reps = locs_curr_grid.reshape(-1).astype(float)
            nan_mask = np.isfinite(fr_all_reps) & np.isfinite(loc_all_reps)
            
            fr_clean  = fr_all_reps[nan_mask]
            loc_clean = loc_all_reps[nan_mask].astype(int)  # back to int if you need integers
    
            for loc in range(1, 10):
                #if loc == 9:
                    #import pdb; pdb.set_trace()
                sel = (loc_clean == loc)
                if np.sum(sel) < 25:
                    # very short samples of the same location will yield imprecise firing rates.
                    # better set to nan and ignore this location!
                    # will be biased otherwise.
                    mean_firing_rates_locs[loc-1, task_idx] = np.nan
                elif np.any(sel):
                    # print(f"length of selection of loc {loc} is {np.sum(sel)} and mean is {fr_clean[sel].mean()} for shift {shift} and first bin is {np.where(sel)[0][0]}")
                    mean_firing_rates_locs[loc-1, task_idx] = fr_clean[sel].mean()
                else:
                    mean_firing_rates_locs[loc-1, task_idx] = np.nan
      
        fr_maps_by_shift.append(mean_firing_rates_locs)

        # --- mean grid correlation for this shift ---
        if np.all(np.isnan(mean_firing_rates_locs)):
            mean_corr_per_shift.append(np.nan)
        else:
            cm = pd.DataFrame(mean_firing_rates_locs).corr()              
            upper = np.triu(np.ones(cm.shape, bool), k=1)
            mean_corr_per_shift.append(np.nanmean(cm.values[upper]))

    # pick best shift
    best_idx = np.nanargmax(mean_corr_per_shift) if np.any(np.isfinite(mean_corr_per_shift)) else None
    peak_shift = shifts[best_idx]
    best_map = fr_maps_by_shift[best_idx]
    # import pdb; pdb.set_trace()
    
    # if cell_name == '07-07-mRF2Ca08-RACC':
    #     import pdb; pdb.set_trace()
        
    if plotting == True:
        if not perm_no:
            plot_spatial_shifts_and_rate_maps(best_map, unique_grids, cell_name, mean_corr_per_shift, best_idx)
            plot_for_each_shift = False
            if plot_for_each_shift == True:
                for shift_idx, shift in enumerate(shifts):
                    plot_spatial_shifts_and_rate_maps(fr_maps_by_shift[shift_idx], unique_grids, cell_name, mean_corr_per_shift, shift_idx)
                    
    # if np.isnan(best_map).any():
    #     import pdb; pdb.set_trace()

    return peak_shift, best_map



def compute_fr_at_spatial_lag(best_shift, neurons, locs):
    # import pdb; pdb.set_trace()
    mean_firing_rates_locs = np.full((9), np.nan, dtype=float)
    locs_shifted = np.roll(locs, shift=-best_shift, axis=1)
    
    fr_all_reps  = neurons.reshape(-1).astype(float)
    loc_all_reps = locs_shifted.reshape(-1).astype(float)
    nan_mask = np.isfinite(fr_all_reps) & np.isfinite(loc_all_reps)
    
    fr_clean  = fr_all_reps[nan_mask]
    loc_clean = loc_all_reps[nan_mask].astype(int)  # back to int if you need integers
    
    # firing rate per location
    for loc in range(1, 10):
        sel = (loc_clean == loc)
        if np.sum(sel) < 25:
            # very short samples of the same location will yield imprecise firing rates.
            # better set to nan and ignore this location!
            # will be biased otherwise.
            mean_firing_rates_locs[loc-1] = np.nan
        elif np.any(sel):
            mean_firing_rates_locs[loc-1] = fr_clean[sel].mean()
        else:
            mean_firing_rates_locs[loc-1] = np.nan
    
    return mean_firing_rates_locs


# --- Helpers for stats ---
def one_tailed_ttest_greater_than_zero(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan  # t, p, mean
    t_stat, p_two = st.ttest_1samp(x, 0.0, nan_policy='omit')
    p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
    return float(t_stat), float(p_one), float(np.mean(x))

def stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'
    

    
def plot_results_per_roi(df, title_string_add):

    roi_label = []
    for i, row in df.iterrows():
        cell_label = row['neuron_id']
        if 'ACC' in cell_label or 'vCC' in cell_label or 'AMC' in cell_label:
            roi = 'ACC'
        elif 'PCC' in cell_label:
            roi = 'PCC'
        elif 'OFC' in cell_label:
            roi = 'OFC'
        elif 'MCC' in cell_label or 'HC' in cell_label:
            roi = 'hippocampal'
        elif 'EC' in cell_label:
            roi = 'entorhinal'
        elif 'AMYG' in cell_label:
            roi = 'amygdala'
        else:
            roi = 'mixed'
        roi_label.append(roi)
    df['roi'] = roi_label
    
    # next, plot per ROI
    bins=20

    rois = df['roi'].unique().tolist()
    n_roi = len(rois)

    results_dir = {}
    results_dir['zero lag (330, 0, 30)'] = df[df["mode_peak_shift"].isin([0, 30, 330])]
    results_dir['future lags'] = df[~df["mode_peak_shift"].isin([0, 30, 330])]
    results_dir['future_reward_times'] = df[df["mode_peak_shift"].isin([90, 180, 240])]
    results_dir['all lags'] = df


    early_color = '#00BFC4'      # turquoise-blue
    late_color = '#E07B39'       # terracotta-orange
    all_color = 'lightcoral'
    all_correct_color = 'teal'

    for result_name in results_dir:
        # Create subplots: one row, n_roi columns
        fig, axes = plt.subplots(1, n_roi, figsize=(n_roi*5, 5), sharey=True)
        # In case there is only one ROI, wrap axes in a list for consistency.
        for ax, roi in zip(axes, rois):
            corrs_allneurons = results_dir[result_name][results_dir[result_name]['roi'] == roi]['avg_consistency_at_peak']
            nan_filter = np.isnan(corrs_allneurons)
            # Remove any NaN values
            valid_corrs = corrs_allneurons[~np.isnan(corrs_allneurons)]
            mean_sample = np.mean(valid_corrs)
            # Perform a two-tailed one-sample t-test
            ttest_result = st.ttest_1samp(valid_corrs, 0)
            t_stat = ttest_result.statistic
            p_two = ttest_result.pvalue
            
            # Convert to a one-tailed p-value for H1: mean > 0.
            if t_stat > 0:
                p_value = p_two / 2
            else:
                p_value = 1 - (p_two / 2)
            # Determine significance level based on p-value
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'n.s.'
            
            # Plot the histogram
            ax.hist(valid_corrs, bins=bins, color='teal', alpha = 0.5, edgecolor='teal')
            ax.axvline(0, color='black', linestyle='dashed', linewidth=2)
            
            # Set subplot title and labels
            ax.set_title(f"{roi} \n for {len(valid_corrs)} neurons \n {result_name} {title_string_add}", fontsize=12)
            ax.set_xlabel("Correlation coefficient", fontsize=14)
            ax.tick_params(axis='both', labelsize=12, width=2, length=6)
            
            # Only set y-label on the first subplot (or adjust as desired)
            ax.set_ylabel("Frequency", fontsize=12)
            
            # Annotate with significance and p-value
            ax.text(0.95, 0.95, f"Significance: {significance}\n(p = {p_value:.3e})\n mean = {mean_sample:.2f}",
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        
        plt.tight_layout()
        plt.show()




#
#
# SOME HELPERS
# neuron = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy()
# means_per_grid = []
# plt.figure(); plt.imshow(neuron, aspect =- 'auto')

# for grididx, grid in enumerate(unique_grids):
#     mask_curr_grid = (grididx == idx_same_grids)
#     data_grid = neuron[mask_curr_grid]
#     mean_firing_rate = np.nanmean(data_grid)
#     means_per_grid.append(mean_firing_rate)
    
    
# plt.figure(); plt.plot(range(0, len(unique_grids)), means_per_grid); plt.ylim(0,np.max(means_per_grid)+0.5); plt.axhline(np.nanmean(neuron),color='black', linestyle='dashed'); plt.show()



#
#
#



def plot_results_per_roi_and_future_lag(df, title_string_add, bins=20):
    # --- Derive ROI labels (kept from your code) ---
    roi_label = []
    for _, row in df.iterrows():
        cell_label = row['neuron_id']
        if 'ACC' in cell_label or 'vCC' in cell_label or 'AMC' in cell_label:
            roi = 'ACC'
        elif 'PCC' in cell_label:
            roi = 'PCC'
        elif 'OFC' in cell_label:
            roi = 'OFC'
        elif 'MCC' in cell_label or 'HC' in cell_label:
            roi = 'hippocampal'
        elif 'EC' in cell_label:
            roi = 'entorhinal'
        elif 'AMYG' in cell_label:
            roi = 'amygdala'
        else:
            roi = 'mixed'
        roi_label.append(roi)

    df = df.copy()
    df['roi'] = roi_label

    # --- Define mutually-exclusive groups ---
    mask_zero   = df["mode_peak_shift"].isin([0, 30, 330])
    mask_reward = df["mode_peak_shift"].isin([90, 180, 240])
    mask_future = (~mask_zero) & (~mask_reward)

    # Build global bin edges so all ROIs use the same binning
    all_vals = df.loc[mask_zero | mask_reward | mask_future, "avg_consistency_at_peak"].dropna().to_numpy()
    if all_vals.size == 0:
        print("No data to plot.")
        return
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    # --- Figure and axes (one subplot per ROI) ---
    rois = df['roi'].unique().tolist()
    n_roi = len(rois)
    fig, axes = plt.subplots(1, n_roi, figsize=(max(5, n_roi*4.5), 5), sharey=True)
    if n_roi == 1:
        axes = [axes]  # make iterable

    # Colors and legend patches
    colors = {
        "zero lag": "lightcoral",
        "future reward lags": "mediumpurple",
        "future lags": "teal",
    }
    legend_patches = [
        Patch(facecolor=colors["zero lag"], edgecolor='black', label="zero lag"),
        Patch(facecolor=colors["future reward lags"], edgecolor='black', label="future reward lags"),
        Patch(facecolor=colors["future lags"], edgecolor='black', label="future lags"),
    ]

    for ax, roi in zip(axes, rois):
        df_roi = df[df['roi'] == roi]

        x_zero   = df_roi.loc[mask_zero.loc[df_roi.index],   "avg_consistency_at_peak"].dropna().to_numpy()
        x_reward = df_roi.loc[mask_reward.loc[df_roi.index], "avg_consistency_at_peak"].dropna().to_numpy()
        x_future = df_roi.loc[mask_future.loc[df_roi.index], "avg_consistency_at_peak"].dropna().to_numpy()

        # Stacked histogram for this ROI
        data_list = [x_zero, x_reward, x_future]
        labels = [
            f"zero lag (n={len(x_zero)})",
            f"future reward lags (n={len(x_reward)})",
            f"future lags (n={len(x_future)})"
        ]
        ax.hist(
            data_list,
            bins=bin_edges,
            stacked=True,
            color=[colors["zero lag"], colors["future reward lags"], colors["future lags"]],
            edgecolor='black',
            alpha=0.9
        )
        ax.axvline(0, color='black', linestyle='dashed', linewidth=2)

        total_n = len(x_zero) + len(x_reward) + len(x_future)
        ax.set_title(f"{roi}\n(n={total_n}) {title_string_add}", fontsize=12)
        ax.set_xlabel("Correlation coefficient", fontsize=12)
        ax.tick_params(axis='both', labelsize=11, width=2, length=6)

        # y-label on left-most subplot only
        if ax is axes[0]:
            ax.set_ylabel("Count", fontsize=12)

    # One shared legend on top
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()

    
def permute_locations(locs, beh_df, no_perms):
    permuted_locs = 1
    import pdb; pdb.set_trace()
    
    return permuted_locs
    

def store_p_vals_perms(true_df, perm_df, out_path):
    # merge obs with all its nulls, compute p for each group
    obs = true_df[["session_id","neuron_id","avg_consistency_at_peak"]].rename(columns={"avg_consistency_at_peak":"obs"})
    nulls = perm_df[["session_id","neuron_id","avg_consistency_at_peak"]].rename(columns={"avg_consistency_at_peak":"perm"})
    merged = obs.merge(nulls, on=["session_id","neuron_id"], how="left")
    
    def perm_p_one_sided(g):
        M = g["perm"].notna().sum()
        if M == 0 or not np.isfinite(g["obs"].iat[0]):
            return pd.Series({"n_perms": M, "p_perm": np.nan})
        k = (g["perm"] >= g["obs"].iat[0]).sum()       # how many nulls >= obs
        p = (k + 1) / (M + 1)
        return pd.Series({"n_perms": M, "p_perm": p})
    
    stats = merged.groupby(["session_id","neuron_id"], sort=False).apply(perm_p_one_sided).reset_index()

    # attach p-values back to the true table
    out = true_df.merge(stats, on=["session_id","neuron_id"], how="left")
    
    # ---------- FDR (Benjamini–Hochberg) ----------
    def bh_reject(pvals, alpha=0.05):
        p = np.asarray(pvals, float)
        mask = np.isfinite(p); m = mask.sum()
        sig = np.zeros_like(p, bool)
        if m == 0: return sig
        order = np.argsort(p[mask]); p_sorted = p[mask][order]
        thresh = alpha * (np.arange(1, m+1) / m)
        passed = p_sorted <= thresh
        if passed.any():
            cutoff = p_sorted[np.nonzero(passed)[0].max()]
            sig[mask] = p[mask] <= cutoff
        return sig

    out["sig_FDR_all"] = bh_reject(out["p_perm"].values, alpha=0.05)
    
    # ---------- save ----------
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")



    
def test_firing_rate_consistency(neuron, criteria = 0.2):
    mean_firing_rate = np.nanmean(neuron)
    sparse_repeats = 0
    
    for repeat in neuron:
        if np.nanmean(repeat) < criteria * mean_firing_rate:
            sparse_repeats = sparse_repeats+1
    if sparse_repeats > 9:
        import pdb; pdb.set_trace()
    # DO THIS PER GRID!
    # mark grid as low if FR grid / FR session < 0.2
    # also mark grid as low in FR if FR grid < 0.1 Hz
    # say that there have to be at least 3 grids that I have good data for for a cell
    # to be included!
    return sparse_repeats, mean_firing_rate




def extract_consistent_grids(neuron, beh, idx_same_grids):
    # per grid, identify firing rate
    # exclude grid if firing rate lower than 20% of mean firing.
    # also make sure to leave at least 3 unique grids.
    
    beh = beh.copy()  # we'll append a column and also return a filtered copy
    beh['new_grid_idx'] = idx_same_grids
    # --- per-grid firing rates ---
    mean_firing_rate = np.nanmean(neuron)
    grid_nos = np.unique(idx_same_grids)

    # FR per grid_no
    grid_fr = {}
    for g in grid_nos:
        mask_g = (idx_same_grids == g)
        grid_fr[g] = np.nanmean(neuron[mask_g])

    # attach column (row-wise) for convenience/inspection
    beh['grid_FR'] = beh['new_grid_idx'].map(grid_fr)

    # --- tentative exclusion: FR < 20% of overall mean (treat NaN FR as low) ---
    excluded_grid_nos = []
    thresh = 0.2 * mean_firing_rate if not np.isnan(mean_firing_rate) else np.nan
    for g in grid_nos:
        fr = grid_fr[g]
        if np.isnan(fr) or (not np.isnan(thresh) and fr < thresh):
            excluded_grid_nos.append(g)
    
    # 3) Tentative keep-mask with low-rate grids removed
    tentative_keep_mask = ~beh['new_grid_idx'].isin(excluded_grid_nos)

    
    # next test based on this new selection, how many unique grids?
    # at least 3 so cross-validation is possible.
    kept_identities = beh['new_grid_idx'][tentative_keep_mask].to_numpy()
    no_unique_good_grids = np.unique(kept_identities)
    target_unique_min = 3
    if no_unique_good_grids < target_unique_min:
        # stepwise add best bad grid in
        # sort excluded grids by FR descending (NaN treated as -inf)
        def fr_key(g):
            fr = grid_fr[g]
            return -np.inf if np.isnan(fr) else fr
        excluded_sorted = sorted(excluded_grid_nos, key=fr_key, reverse=True)

        # prefer adding grids that increase identity diversity
        
        import pdb; pdb.set_trace()
        
        
        # CONTINUE HERE
        # ADD GRIDS BACK IN
        for g in excluded_sorted:
            if len(kept_identities) >= target_unique_min:
                break
            gid = id_by_grid[g]
            if gid not in kept_identities:
                kept_identities.add(gid)
                add_back_grids.add(g)



        
    # I THINK THIS CAN GO
    # GDUJCKSK WRITTEN BY CHAT
    
    
    # Columns that define a grid identity (adjust if yours differ)
    grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']

    # helper: identity tuple for each grid_no (grab from first occurrence)
    def identity_tuple_from_g(g):
        row = beh.loc[beh['grid_no'] == g, grid_cols].iloc[0]
        return tuple(row.values.tolist())
    

    id_by_grid = {g: identity_tuple_from_g(g) for g in grid_nos}
    
    # compute unique identities among kept rows
    kept_identities = set(map(tuple, beh.loc[tentative_keep_mask, grid_cols].to_numpy()))
    n_unique_after_excl = len(kept_identities)
    
    # total unique identities possible
    all_unique_identities = set(map(tuple, beh[grid_cols].to_numpy()))
    target_unique = min(3, len(all_unique_identities))
    
    # in case the CV target is not met
    add_back_grids = set()
    if n_unique_after_excl < target_unique:
        # sort excluded grids by FR descending (NaN treated as -inf)
        def fr_key(g):
            fr = grid_fr[g]
            return -np.inf if np.isnan(fr) else fr
        excluded_sorted = sorted(excluded_grid_nos, key=fr_key, reverse=True)

        # prefer adding grids that increase identity diversity
        for g in excluded_sorted:
            if len(kept_identities) >= target_unique:
                break
            gid = id_by_grid[g]
            if gid not in kept_identities:
                kept_identities.add(gid)
                add_back_grids.add(g)


        # if still short (e.g., identity overlap), add best remaining anyway
        if len(kept_identities) < target_unique:
            for g in excluded_sorted:
                if g not in add_back_grids:
                    add_back_grids.add(g)
                    
    # final per-row keep decision (BUT DO NOT FILTER beh)
    final_keep_mask = tentative_keep_mask | beh['grid_no'].isin(add_back_grids)

    # --- build outputs without filtering beh ---
    # unique grids among kept rows only
    identities_all = beh[grid_cols].to_numpy()
    unique_grids, inv_kept = np.unique(
        identities_all[final_keep_mask],
        axis=0,
        return_inverse=True
    )

    # per-row idx mapping into unique_grids; -1 where not kept
    idx_same_grids = np.full(len(beh), -1, dtype=int)
    idx_same_grids[np.where(final_keep_mask)[0]] = inv_kept

    return unique_grids, idx_same_grids, beh


    
    
def pair_grids_to_increase_spatial_coverage(locs, same_grids, min_coverage=100, min_groups=3):
    """
    Merge original grids into groups to maximize coverage per combo:
      1) more locations with >= min_coverage
      2) higher minimum per-location total
    Returns a new index array (0..n_groups-1) aligned with same_grids.
    """
    loc_vals = np.unique(locs)
    # --- 1) build coverage dict for each original grid ---
    
    grid_cvg = {}
    for grid_idx in np.unique(same_grids):
        grid_cvg[grid_idx] = {}
        all_locs_curr_grid = locs[same_grids == grid_idx]
        for loc in np.unique(locs):
            grid_cvg[grid_idx][loc] = np.count_nonzero(all_locs_curr_grid == loc)
     
    # first pass: test if there are enough grids to build 3 pairs. If not, 
    # identify which grid has the best coverage.
    pairs = {}
    used_grids = []
    if len(grid_cvg) < min_groups*2:
        best_cvg = []
        for g, loc_dict in grid_cvg.items():
            # 1. find the grid with best coverage
            best_cvg.append(len([l for l, v in loc_dict.items() if v < min_coverage]))
            best_cvg = np.array(best_cvg)
            pairs[np.where(best_cvg == min(best_cvg))[0][0]] = np.where(best_cvg == min(best_cvg))[0][0]
            used_grids.append(np.where(best_cvg == min(best_cvg))[0][0])
    
    for g, loc_dict in grid_cvg.items():
        # 1. find weak locations
        low_locs = [l for l, v in loc_dict.items() if v < min_coverage]
        if not low_locs:
            continue
        if g in used_grids:
            continue
        best = None
        for h, other in grid_cvg.items():
            if h == g:
                continue
            if h in used_grids:
                continue
            # 2. combine coverage
            combined = {l: loc_dict[l] + other[l] for l in loc_dict}
            # 3. score: how many weak locations get fixed, then how much boost on the weakest one
            fixed = sum(combined[l] >= min_coverage for l in low_locs)
            boost = min(combined[l] for l in low_locs)   # worst case after merge
            score = (fixed, boost)
            if best is None or score > best[0]:
                best = (score, h)
                
        if best:
            pairs[g] = best[1]
            # also include the vice-versa pair
            used_grids.append(g)
            used_grids.append(best[1])
    
    # lastly, map the new grid indices back to the old unique grid index.
    same_grid_idx_new = np.empty(same_grids.shape)
    for new_same_grid, old_grid_idx in enumerate(pairs):
        same_grid_idx_new[same_grids == old_grid_idx] = new_same_grid
        same_grid_idx_new[same_grids == pairs[old_grid_idx]] = new_same_grid
        
    # # brief check if the coverage is better now
    # new_grid_cvg = {}
    # for grid_idx in np.unique(same_grid_idx_new):
    #     new_grid_cvg[grid_idx] = {}
    #     all_locs_curr_grid = locs[same_grid_idx_new == grid_idx]
    #     for loc in np.unique(locs):
    #         new_grid_cvg[grid_idx][loc] = np.count_nonzero(all_locs_curr_grid == loc)
            
    return same_grid_idx_new



def compute_fut_spatial_tunings(sessions, trials = 'all_correct', plotting = False, no_perms = None, combine_two_grids = False, sparsity_c = None):  
    # trials can be 'all', 'all_correct', 'early', 'late'
    
    # determine results table
    COLUMNS = [
    "session_id", "neuron_id",
    "mode_peak_shift",                     # peak_shift_validated
    "avg_consistency_at_peak",        # avg_consistency_at_peak
    "count_mode_peak",                      # frequency_peak_shift_validated
    "perm_idx", 
    "mean_firing_rate",
    "sparse_repeats"
    ]
    results = []
    included_neurons = []
    for sesh in sessions:
        # load data
        data_raw, source_dir = get_data(sesh)
        group_dir_fut_spat = f"{source_dir}/group/spatial_peaks"
        # import pdb; pdb.set_trace()
        # if this session doesn't exist, skip
        if not data_raw:
            continue
    
        # filter data for only those repeats that were 1) correct and 2) not the first one
        data = filter_data(data_raw, sesh, trials)
        
        # determine identical grids
        grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
        unique_grids, _, idx_same_grids, _ = np.unique(
            data[f"sub-{sesh:02}"]['beh'][grid_cols].to_numpy(),
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True
        )
        
        if no_perms:
            no_perms = np.random.default_rng(123)
            # perms = permute_locations(data[f"sub-{sesh:02}"]['locations'], data[f"sub-{sesh:02}"]['beh'], no_perms = no_perms)
            perms = 200
            include_these_cells = Path(f"{group_dir_fut_spat}/included_cells_{trials}_reps_excl_{sparsity_c}_pct.txt").read_text().splitlines()
        else:
            perms = 1
            
        # build some sort of loop where you store all permutations
        # for each cell, cross-validate the peak task-lag shift for spatial consistency.
        for neuron_idx, curr_neuron in enumerate(data[f"sub-{sesh:02}"]['normalised_neurons']):
            avg_firing = None
            sparse_reps = None
            if curr_neuron == '05-05-mLF3aOFC05-LOFC':
                continue
            #if curr_neuron in ['05-05-mRF2Ca03-RACC', '20-20-mRT2dHbEb08-REC', '22-22-mRT2dHbEb08-REC']:
            if curr_neuron in ['05-05-mRF2Ca03-RACC']:
                import pdb; pdb.set_trace()
                # 05-05-mRF2Ca03-RACC is supposed be be really inconsistent, but highly firing
                # 20-20-mRT2dHbEb08-REC is supposed to be really consistent
                # 22-22-mRT2dHbEb08-REC is supposed to be really consistent but very sparse
            # if curr_neuron in ['16-16-mRT2bHb03-RHC', '23-23-mRT2bHb05-RHC', '08-08-mRT2bHb01-RHC', '19-19-mRT2bHb04-RHC', '03-03-mRF3C07-RpgACC', '21-21-mRT2bHb04-RHC', '18-18-mRT2bHb04-RHC']:
            #     import pdb; pdb.set_trace()
            
            if perms > 1:
                if curr_neuron not in include_these_cells:
                    continue
            else: 
                included_neurons.append(curr_neuron)
            
            # clean the data such that I don't consider 'bad' blocks of repeats
            # with super low firing 
            if sparsity_c:
                unique_grids, idx_same_grids, _ = extract_consistent_grids(data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy(), data[f"sub-{sesh:02}"]['beh'], idx_same_grids)
                # sparse_reps, avg_firing = test_firing_rate_consistency(data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy(),  criteria = sparsity_c)
                # if sparse_reps > 9:
                #     print(f"skipping {curr_neuron} because it fired less than {sparsity_c*100}% of its average firing rate ({avg_firing}) in {sparse_reps} of the repeats")
                #     continue
            
            # instead of doing grid-by-grid, I will now do 2 grids combined.
            if combine_two_grids == True:
                idx_same_grids = pair_grids_to_increase_spatial_coverage(data[f"sub-{sesh:02}"]['locations'], idx_same_grids)
                unique_grids = np.unique(idx_same_grids)
            # like this, I'm hoping to compute a more stable firing rate map because
            # there will be more locations that will have been covered.
            
            import pdb; pdb.set_trace()
            # CONTINUE HERE:
                # next, if the exclusion of bad grids and then the averaging of different grids works,
                # work on weighting the different grids by how much they were visited.
            
            
            
            # loop through n-1 grids, respectively
            # import pdb; pdb.set_trace()
            
            for perm_idx in range(0,perms):
                validate_peak_lag = np.full((len(unique_grids),2), np.nan, dtype=float)
                for idx_test_task, _ in enumerate(unique_grids):
                    mask_test_task = (idx_same_grids == idx_test_task)
                    neurons_test_task = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].loc[mask_test_task].to_numpy()
                    locs_test_task = data[f"sub-{sesh:02}"]['locations'].loc[mask_test_task].to_numpy()
                    
                    mask_train_task = (idx_same_grids != idx_test_task)
                    # create subset of df and neurons.
                    beh_train_task = data[f"sub-{sesh:02}"]['beh'].loc[mask_train_task].reset_index()
                    neurons_train_task = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].loc[mask_train_task].to_numpy()
                    locs_train_task = data[f"sub-{sesh:02}"]['locations'].loc[mask_train_task].to_numpy()
                    unique_idx_train = idx_same_grids[mask_train_task]
                    import pdb; pdb.set_trace()
                    # and compute the peak-spatial tuning rotation
                    peak_task_lag_train, fr_maps_train_at_best_lag = comp_peak_spatial_tuning(neurons_train_task, locs_train_task, beh_train_task, curr_neuron, unique_idx_train, plotting=plotting, perm_no = no_perms, combine_two_grids=combine_two_grids)
                    # careful: sometimes, in the fr_maps, there are nans because the locaitons
                    # have not been visited. Account for that in the next correlatioN!
                    
                    # validate: compute the correlation between rate-map of held-out task at this lag
                    fr_map_test_at_best_lag = compute_fr_at_spatial_lag(peak_task_lag_train, neurons_test_task, locs_test_task)
                    
                    # # ignore nans 
                    consistency_train_test = []
                    for train_grid in fr_maps_train_at_best_lag.T:
                        nan_mask = np.isfinite(train_grid) & np.isfinite(fr_map_test_at_best_lag)
                        consistency_train_test.append(np.corrcoef(train_grid[nan_mask], fr_map_test_at_best_lag[nan_mask])[0][1])
                    spatial_consistency_validation = np.nanmean(consistency_train_test)
    
                    # store the average correlation value for lag-rotation for current test-task
                    validate_peak_lag[idx_test_task,0] = spatial_consistency_validation
                    validate_peak_lag[idx_test_task,1] = peak_task_lag_train
                    # consider storing this???
                
                m = mode(validate_peak_lag[:,1], keepdims=False)
                peak_shift_validated, frequency_peak_shift_validated = m.mode, m.count
                avg_consistency_at_peak = np.mean(validate_peak_lag[:,0])
                
                results.append({
                "session_id": sesh,
                "neuron_id": curr_neuron,
                "mode_peak_shift": peak_shift_validated,
                "avg_consistency_at_peak": avg_consistency_at_peak,
                "count_mode_peak": f"{frequency_peak_shift_validated} out of {len(validate_peak_lag[:,1])}",
                "perm_idx": perm_idx,
                "mean_firing_rate":avg_firing,
                "sparse_repeats": sparse_reps
                })
                
                if not no_perms:
                    print(f"average spatial consistency at lag {peak_shift_validated} for neuron {curr_neuron} is {avg_consistency_at_peak}, peak occuring {frequency_peak_shift_validated} times")
                if no_perms:
                    if perm_idx % 100 == 0:
                        print(f"now computing permutation {perm_idx} for neuron {curr_neuron}...")
                    
                    
    
    # import pdb; pdb.set_trace()
    results_df = pd.DataFrame(results, columns = COLUMNS)
    
    if not os.path.isdir(group_dir_fut_spat):
        os.mkdir(group_dir_fut_spat)
    
    if perm_idx > 1:
        name_result = f"{group_dir_fut_spat}/perms200_spatial_consistency_{trials}_repeats.csv"
        # ADD SOME SORT OF PERM PLOTTING FUNCTION!!!
        empirical_result = pd.read_csv(f"{group_dir_fut_spat}/spatial_consistency_{trials}_repeats.csv")
        import pdb; pdb.set_trace()
        name_result_stats = f"{group_dir_fut_spat}/pval_for_perms200_spatial_consistency_{trials}_repeats.csv"
        store_p_vals_perms(true_df = empirical_result, perm_df = results_df, out_path=name_result_stats)
        mc.plotting.results.plot_perm_spatial_consistency(results_df, empirical_result, name_result_stats)
    else:
        name_result = f"{group_dir_fut_spat}/spatial_consistency_{trials}_repeats.csv"
        plot_results_per_roi(results_df, title_string_add = f'{trials}_repeats')
        plot_results_per_roi_and_future_lag(results_df, title_string_add = f'{trials}_repeats')
        if sparsity_c:
            name_result = f"{group_dir_fut_spat}/spatial_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons.csv"
            print(f"included {len(included_neurons)} neurons.")
            Path(f"{group_dir_fut_spat}/included_cells_{trials}_reps_excl_{sparsity_c}_pct.txt").write_text("\n".join(included_neurons))
          
    results_df.to_csv(name_result)
    print(f"saved cross-validated spatial tuning values in {name_result}")
    
      
    import pdb; pdb.set_trace()
    
    


 # if running from command line, use this one!   
# if __name__ == "__main__":
#     fire.Fire(plot_all)
#     # call this script like
#     # python wrapper_plot_elnetreg_results.py --model_name_string='w_partial_musicboxes_excl_rep1-2', --models_I_want='['withoutnow', 'only2and3future','onlynowandnext']' --exclude_x_repeats='[1,2]' --randomised_reward_locations=False --save_regs=True


if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    # trials can be 'all', 'all_correct', 'early', 'late'
    compute_fut_spatial_tunings(sessions=list(range(15,16)), trials = 'all_correct', no_perms = None, combine_two_grids = True, sparsity_c = 'gridwise_qc')
    # compute_fut_spatial_tunings(sessions=list(range(0,60)), trials = 'all', no_perms = 200, combine_two_grids = True)
    # compute_fut_spatial_tunings(sessions=[31], trials = 'all', plotting = False, no_perms = None, combine_two_grids = True)
    
    
    
    