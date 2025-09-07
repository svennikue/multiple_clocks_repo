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
import re 
from scipy.stats import binomtest, norm

# import pdb; pdb.set_trace()



def get_data(sub):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [f"{sub:02}"])
    return data_norm, data_folder



def weighted_pearson(x, y, w):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    nan_mask = np.isfinite(x) & np.isfinite(y) & (w > 0)
    if nan_mask.sum() < 2: 
        return np.nan
    x = x[nan_mask]; y = y[nan_mask]; w = w[nan_mask]
    mx = np.average(x, weights=w); my = np.average(y, weights=w)
    cov = np.average((x-mx)*(y-my), weights=w)
    vx  = np.average((x-mx)**2,     weights=w)
    vy  = np.average((y-my)**2,     weights=w)
    return cov/np.sqrt(vx*vy) if vx>0 and vy>0 else np.nan


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
        
    
   
def comp_peak_spatial_tuning(neurons, locs, beh, cell_name, idx_same_grids, plotting=False, perm_no = None, weighted = False):
    temporal_shift = 30
    shifts = np.arange(0, 360, temporal_shift)  # 12 shifts
    unique_grids = np.unique(idx_same_grids)
        
    mean_corr_per_shift,fr_maps_by_shift, dwell_by_shift = [], [], []
    # import pdb; pdb.set_trace()
    for shift in shifts:
        mean_firing_rates_locs = np.full((9, len(unique_grids)), np.nan, dtype=float)
        dwell_time_at_locs = np.full((9, len(unique_grids)), np.nan, dtype=float)
        
        for task_count, task_id in enumerate(unique_grids):
            mask_curr_grid = (idx_same_grids == task_id)
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
            loc_clean = loc_all_reps[nan_mask].astype(int)
            dt_clean = np.ones_like(fr_clean, dtype=float)

            # import pdb; pdb.set_trace()
            for loc in range(1, 10):
                sel = (loc_clean == loc)
                dwell_time_at_locs[loc-1, task_count] = dt_clean[sel].sum()
                if weighted == True:
                    if not np.any(sel):
                        mean_firing_rates_locs[loc-1, task_count] = np.nan
                        continue
                    else:
                        mean_firing_rates_locs[loc-1, task_count] = fr_clean[sel].mean()
                else:
                    if np.sum(sel) < 25:
                        # very short samples of the same location will yield imprecise firing rates.
                        # better set to nan and ignore this location!
                        # will be biased otherwise.
                        mean_firing_rates_locs[loc-1, task_count] = np.nan
                    elif np.any(sel):
                        # print(f"length of selection of loc {loc} is {np.sum(sel)} and mean is {fr_clean[sel].mean()} for shift {shift} and first bin is {np.where(sel)[0][0]}")
                        mean_firing_rates_locs[loc-1, task_count] = fr_clean[sel].mean()
                    else:
                        mean_firing_rates_locs[loc-1, task_count] = np.nan
      
        fr_maps_by_shift.append(mean_firing_rates_locs)
        dwell_by_shift.append(dwell_time_at_locs)

        # --- mean grid correlation for this shift ---
        if np.all(np.isnan(mean_firing_rates_locs)):
            mean_corr_per_shift.append(np.nan)
        else:
            if weighted == True:
                no_grids = mean_firing_rates_locs.shape[1]
                vals = []
                for i in range(no_grids):
                    for j in range(i+1, no_grids):
                        w_ij = dwell_time_at_locs[:, i] + dwell_time_at_locs[:, j]  # more trust where either grid spent more time
                        vals.append(weighted_pearson(mean_firing_rates_locs[:, i], mean_firing_rates_locs[:, j], w_ij))
                mean_corr_per_shift.append(np.nanmean(vals))
            else:
                cm = pd.DataFrame(mean_firing_rates_locs).corr()              
                upper = np.triu(np.ones(cm.shape, bool), k=1)
                mean_corr_per_shift.append(np.nanmean(cm.values[upper]))

    # pick best shift
    best_idx = np.nanargmax(mean_corr_per_shift) if np.any(np.isfinite(mean_corr_per_shift)) else None
    peak_shift = shifts[best_idx]
    best_map = fr_maps_by_shift[best_idx]
    best_dwell = dwell_by_shift[best_idx]
        
    if plotting == True:
        # import pdb; pdb.set_trace()
        if not perm_no:
            plot_spatial_shifts_and_rate_maps(best_map, unique_grids, cell_name, mean_corr_per_shift, best_idx)
            plot_for_each_shift = False
            if plot_for_each_shift == True:
                for shift_idx, shift in enumerate(shifts):
                    plot_spatial_shifts_and_rate_maps(fr_maps_by_shift[shift_idx], unique_grids, cell_name, mean_corr_per_shift, shift_idx)
                    
    # if np.isnan(best_map).any():
    #     import pdb; pdb.set_trace()

    return peak_shift, best_map, best_dwell



def compute_fr_at_spatial_lag(best_shift, neurons, locs):
    # import pdb; pdb.set_trace()
    mean_firing_rates_locs = np.full((9), np.nan, dtype=float)
    dwell_time_at_locs     = np.zeros((9), dtype=float)
    
    locs_shifted = np.roll(locs, shift=-best_shift, axis=1)
    
    fr_all_reps  = neurons.reshape(-1).astype(float)
    loc_all_reps = locs_shifted.reshape(-1).astype(float)
    nan_mask = np.isfinite(fr_all_reps) & np.isfinite(loc_all_reps)
    
    fr_clean  = fr_all_reps[nan_mask]
    loc_clean = loc_all_reps[nan_mask].astype(int)  # back to int if you need integers
    dwell_clean = np.ones_like(fr_clean, dtype=float)     
    # firing rate per location
    for loc in range(1, 10):
        sel = (loc_clean == loc)
        dwell_time_at_locs[loc-1] = dwell_clean[sel].sum()
        if dwell_time_at_locs[loc-1] < 25:
            # very short samples of the same location will yield imprecise firing rates.
            # better set to nan and ignore this location!
            # will be biased otherwise.
            mean_firing_rates_locs[loc-1] = np.nan
        elif np.any(sel):
            mean_firing_rates_locs[loc-1] = fr_clean[sel].mean()
        else:
            mean_firing_rates_locs[loc-1] = np.nan
    
    return mean_firing_rates_locs, dwell_time_at_locs


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
    

    
def plot_results_per_roi(df, title_string_add, plot_by_pfc = False, plot_by_cingulate_and_MTL = False):
    # import pdb; pdb.set_trace()
    df['roi'] = rename_rois(df, collapse_pfc = plot_by_pfc, plot_by_cingulate_and_MTL = plot_by_cingulate_and_MTL)

    # --- rows  ---
    results_list = [
        ('zero lag (330, 0, 30)', df[df["mode_peak_shift"].isin([0, 30, 330])]),
        ('future lags',            df[~df["mode_peak_shift"].isin([0, 30, 330])]),
        ('all lags',               df),
    ]
    
    rois   = df['roi'].unique().tolist()
    n_rows = len(results_list)
    n_cols = len(rois)
    bins   = 20
    title_string_add = title_string_add if 'title_string_add' in locals() else ''
    
    plt.rcParams.update({'font.size': 11})
    
    # Make rows closer: use small hspace. Don't share y → allow 2 different y-scales.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8.27, 12.0),     # keep big; height matters more than hspace for readability
        sharex=True, sharey=False,
        gridspec_kw={'hspace': 0.18, 'wspace': 0.3}  # <<< tighter vertical spacing
    )
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    
    # Precompute per-row histogram maxima for two y-scales
    row_max_top3, row_max_all = 0, 0
    cache = [[None]*n_cols for _ in range(n_rows)]
    for r, (result_name, df_res) in enumerate(results_list):
        for c, roi in enumerate(rois):
            vals = df_res.loc[df_res['roi'] == roi, 'avg_consistency_at_peak'].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            counts, _ = np.histogram(vals, bins=bins)
            cache[r][c] = (vals, counts)
            mx = counts.max() if counts.size else 0
            if result_name == 'all lags':
                row_max_all = max(row_max_all, mx)
            else:
                row_max_top3 = max(row_max_top3, mx)
    
    # Plot
    for r, (result_name, df_res) in enumerate(results_list):
        for c, roi in enumerate(rois):
            ax = axes[r, c]
            vals, counts = cache[r][c]
    
            ax.hist(vals, bins=bins, color='teal', alpha=0.5, edgecolor='teal')
            ax.axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    
            # stats box (use helpers)
            t_stat, p_one, mval = one_tailed_ttest_greater_than_zero(vals)
            sig = stars(p_one)
            # keep inside but compact; doesn’t add row spacing
            ax.text(0.98, 0.96,
                    f"n={vals.size}\nmean={mval:.2f}\n{sig} (p={p_one:.1e})",
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
                    fontsize=11)
    
            # Column headers (top row only)
            if r == 0:
                ax.set_title(roi, fontsize=11, pad=4)  # smaller pad -> tighter rows
    
            # Row labels (first col only)
            if c == 0:
                ax.text(-0.16, 0.5, result_name, transform=ax.transAxes,
                        ha='right', va='center', rotation=90, fontsize=11)
    
            # Apply the two y-scales
            ylim_max = row_max_all if result_name == 'all lags' else row_max_top3
            ax.set_ylim(0, max(1, int(ylim_max * 1.04)))  # slight squash
    
            ax.tick_params(axis='both', labelsize=11, width=1.1, length=4)
    
    # Shared labels (don’t repeat on each subplot)
    fig.supxlabel("Correlation coefficient", fontsize=11)
    fig.supylabel("Frequency", fontsize=11)
    
    # Tighten outer margins; keep headroom for the suptitle
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.9, hspace=0.18, wspace=0.3)
    fig.suptitle(f"Mean spatial consistency (cross val) at peak lag per cell, split by ROI \n {title_string_add}",
                 fontsize=12, fontweight='bold', y=0.965)
    
    plt.show()





#
#
# SOME HELPERS
# neuron = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy()
# means_per_grid = []
# plt.figure(); plt.imshow(neuron, aspect = 'auto')

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
    df['roi'] = rename_rois(df, collapse_pfc = False)
    # --- Define mutually-exclusive groups ---
    mask_zero   = df["mode_peak_shift"].isin([0, 30, 330])
    mask_future = (~mask_zero)

    # Build global bin edges so all ROIs use the same binning
    all_vals = df.loc[mask_zero | mask_future, "avg_consistency_at_peak"].dropna().to_numpy()
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
        x_future = df_roi.loc[mask_future.loc[df_roi.index], "avg_consistency_at_peak"].dropna().to_numpy()

        # Stacked histogram for this ROI
        data_list = [x_zero, x_future]
        labels = [
            f"zero lag (n={len(x_zero)})",
            f"future lags (n={len(x_future)})"
        ]
        ax.hist(
            data_list,
            bins=bin_edges,
            stacked=True,
            color=[colors["zero lag"],colors["future lags"]],
            edgecolor='black',
            alpha=0.9
        )
        ax.axvline(0, color='black', linestyle='dashed', linewidth=2)

        total_n = len(x_zero)  + len(x_future)
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
    

def add_state_cell_vals(df, incl_trials):
    results_path_state = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/cells_per_model-11-07-2025"
    state_results = {}
    for file in os.listdir(results_path_state):
        if 'state_reg' in file:
            print(file) 
            if 'only_reps_1-5' in file:
                state_results['early'] = file
            if 'only_reps_1-10' in file:
                state_results['all_correct'] = file
            if 'only_reps_6-10' in file:
                state_results['late'] = file
    state_df = pd.read_csv(f"{results_path_state}/{state_results[incl_trials]}")
      # out_df.neuron_id like: "02-02-chan101-LOFC" or "14-14-mLT2aHaE04-LEC"
    def parse_out_neuron_id(s):
        m = re.match(r'^\s*(\d+)-\d+-(.+)-([A-Za-z0-9][A-Za-z0-9_-]*)\s*$', str(s))
        if m:
            unit_no, unit, roi = m.group(1), m.group(2), m.group(3).upper()
            return unit_no, unit, roi
        return (np.nan, np.nan, np.nan)
    
    # state_df.cell like: "02-chan101_sesh_01_LOFC" or "14-mLT2aHaE04_sesh_12_LEC"
    def parse_state_cell(s):
        m = re.match(r'^\s*(\d+)-(.+?)_sesh_(\d+)_([A-Za-z0-9][A-Za-z0-9_-]*)\s*$', str(s))
        if m:
            unit_no, unit, sesh, roi = m.group(1), m.group(2), int(m.group(3)), m.group(4).upper()
            return unit_no, unit, sesh, roi
        return (np.nan, np.nan, np.nan, np.nan)
    
    out = df.copy()
    st  = state_df.copy()

    out[['unit_no','unit','region']] = out['neuron_id'].apply(lambda x: pd.Series(parse_out_neuron_id(x)))
    out['session_id'] = out['session_id'].astype(int)

    st[['unit_no','unit','session_id','region']] = st['cell'].apply(lambda x: pd.Series(parse_state_cell(x)))
    st['session_id'] = st['session_id'].astype(int)

    # drop duplicate rows in state (keep first) to get a 1–1 key
    st_key = st.drop_duplicates(subset=['subj','unit','session_id','region'])

    merged = out.merge(
        st_key[['subj','unit','session_id','region','average_corr','p_val_time']],
        on=['subj','unit','session_id','region'],
        how='left',
        validate='m:1'  # warn if state has duplicates for a key
    )
    # if you don’t want helper cols kept:
    merged = merged.drop(columns=['subj','unit','region'])
    merged['average_corr_state'] = merged['average_corr']
    merged['p_val_time_state'] = merged['p_val_time']
    return merged
           
        
    


def store_p_vals_perms(true_df, perm_df, out_path, trials, consider_state = False):
    # first, per cell and session, store the respective mean permuted value. 
    # import pdb; pdb.set_trace()
    
    unique_cells = np.unique(perm_df["neuron_id"].to_numpy())
    unique_sessions = np.unique(perm_df["session_id"].to_numpy())
    perm_df['mean_perm_consistency'] = (perm_df.groupby(['session_id', 'neuron_id'])['avg_consistency_at_peak'].transform('mean'))   # or .transform(np.mean))
    # merge obs with all its nulls, compute p for each group
    obs = true_df[["session_id","neuron_id","avg_consistency_at_peak"]].rename(columns={"avg_consistency_at_peak":"obs"})
    nulls = perm_df[["session_id","neuron_id","avg_consistency_at_peak", 'mean_perm_consistency']].rename(columns={"avg_consistency_at_peak":"perm"})
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
    

    if consider_state == True: 
        out_with_state = add_state_cell_vals(out, trials)
        sig_series = pd.Series(pd.NA, index=out_with_state.index, dtype='boolean')

        sig_state_mask = (out_with_state['p_val_time_state'] > 0.05) | (out_with_state['p_val_time_state'].isna())
        bh_without_state = bh_reject(out_with_state.loc[sig_state_mask, 'p_perm'].to_numpy(), alpha=0.05)  # -> 1D bool array
        sig_series.loc[sig_state_mask] = bh_without_state  # fill only where we computed BH
        out['sig_FDR_without_state'] = sig_series.reindex(out.index).astype('boolean')

    
    # 4) write back onto the original table (align by index; fill others with NA)
    out['sig_FDR_all'] = bh_reject(out['p_perm'].to_numpy(), alpha=0.05)

    # ---------- save ----------
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    return out





def rename_rois(df, collapse_pfc = False, plot_by_cingulate_and_MTL = False):
    roi_label = []
    if collapse_pfc == True:
        for _, row in df.iterrows():
            cell_label = row['neuron_id']
            if 'ACC' in cell_label or 'vCC' in cell_label or 'AMC' in cell_label or 'vmPFC' in cell_label or 'OFC' in cell_label or 'PCC' in cell_label:
                roi = 'PFC'
            elif 'MCC' in cell_label or 'HC' in cell_label:
                roi = 'hippocampal'
            elif 'EC' in cell_label:
                roi = 'entorhinal'
            elif 'AMYG' in cell_label:
                roi = 'amygdala'
            else:
                roi = 'mixed'
            roi_label.append(roi)
    elif plot_by_cingulate_and_MTL == True:
        for _, row in df.iterrows():
            cell_label = row['neuron_id']
            if 'ACC' in cell_label or 'vCC' in cell_label or 'AMC' in cell_label or 'vmPFC' in cell_label or 'PCC' in cell_label:
                roi = 'Cingulate'
            elif 'MCC' in cell_label or 'HC' in cell_label or 'EC' in cell_label or 'AMYG' in cell_label:
                roi = 'MTL'
            elif 'OFC' in cell_label:
                roi = 'OFC'
            else:
                roi = 'mixed'
            roi_label.append(roi)
    else:
        for _, row in df.iterrows():
            cell_label = row['neuron_id']
            if 'ACC' in cell_label or 'vCC' in cell_label or 'AMC' in cell_label or 'vmPFC' in cell_label:
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
    return roi_label

def plt_binomial_per_roi(out_df, title_string, alpha=0.05, cmap_name="tab20", collapse_pfc=False):

    p_emp   = np.asarray(out_df['p_perm'].to_numpy(), float)
    regions = np.asarray(rename_rois(out_df, collapse_pfc=collapse_pfc))

    def stars(p):
        if not np.isfinite(p): return "n/a"
        return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'

    def wilson_ci(k, n, conf=0.95):
        if n == 0: return (np.nan, np.nan)
        z = norm.ppf(1 - (1 - conf) / 2)
        phat = k / n
        denom = 1 + z**2 / n
        center = (phat + z**2/(2*n)) / denom
        half = z * np.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
        return center - half, center + half

    # --- summarize per region ---
    rows = []
    for r in np.unique(regions):
        m = (regions == r)
        x = p_emp[m]
        x = x[np.isfinite(x)]
        n = x.size
        k = int((x <= alpha).sum())
        p_binom = binomtest(k, n, p=alpha, alternative="greater").pvalue if n else np.nan
        lo, hi  = wilson_ci(k, n, conf=0.95) if n else (np.nan, np.nan)
        frac    = (k / n) if n else np.nan
        rows.append((r, n, k, frac, lo, hi, p_binom))

    if not rows:
        print("No regions to plot.")
        return

    # sort by region frequency (n desc), tie-break by name
    rows.sort(key=lambda t: (-t[1], t[0]))
    labels, ns, ks, fracs, los, his, ps = map(list, zip(*rows))

    # error bars from Wilson CI
    fracs_arr = np.array(fracs, float)
    los_arr   = np.array(los,   float)
    his_arr   = np.array(his,   float)
    yerr_low  = np.nan_to_num(fracs_arr - los_arr, nan=0.0, posinf=0.0, neginf=0.0)
    yerr_high = np.nan_to_num(his_arr   - fracs_arr, nan=0.0, posinf=0.0, neginf=0.0)
    yerr = np.vstack([yerr_low, yerr_high])

    # --- dynamic y-limit with headroom for text ---
    tops = fracs_arr + yerr_high
    data_top = float(np.nanmax(tops)) if np.isfinite(tops).any() else 0.1
    text_pad = 0.03  # vertical gap above error bar for the annotation
    min_head = 0.06  # minimum headroom from highest top to y_max
    y_max = min(1.0, max(0.15, data_top + max(text_pad + 0.01, min_head)))

    # style
    plt.rcParams.update({
        'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12
    })

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = np.arange(len(labels))
    cmap   = plt.get_cmap(cmap_name, max(3, len(labels)))
    colors = cmap(range(len(labels)))

    ax.bar(x, fracs_arr, width=0.65, color=colors, edgecolor='none', alpha=0.95)
    ax.errorbar(x, fracs_arr, yerr=yerr, fmt='none', capsize=4, lw=1.6, color='black', zorder=3)

    # reference line at alpha
    ax.axhline(alpha, color='gray', lw=1.2, ls='--', label=f'Expected under null (α={alpha:g})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, y_max)  # <<< tight y-axis with headroom
    ax.set_ylabel("Fraction significant")
    ax.set_title(title_string)

    # annotations placed above each bar’s error bar, with guaranteed padding
    for i, (k, n, p) in enumerate(zip(ks, ns, ps)):
        top_i = fracs_arr[i] + yerr_high[i]
        y_i = min(y_max - 0.01, top_i + text_pad)  # never spill beyond y_max
        ax.text(x[i], y_i, f"{k}/{n}\n{stars(p)} (p={p:.2g})",
                ha='center', va='bottom', fontsize=12, zorder=4)

    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    plt.show()

    

def compute_fut_spatial_tunings(sessions, trials = 'all_minus_explore', plotting = False, no_perms = None, combine_two_grids = False, sparsity_c = None, weighted = False, save_all = False):  
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
        data = mc.analyse.helpers_human_cells.filter_data(data_raw, sesh, trials)
        beh_df = data[f"sub-{sesh:02}"]['beh'].copy()
        # determine identical grids
        grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']

        
        if no_perms:
            no_perms = np.random.default_rng(123)
            # perms = permute_locations(data[f"sub-{sesh:02}"]['locations'], data[f"sub-{sesh:02}"]['beh'], no_perms = no_perms)
            perms = 200
            include_these_cells = Path(f"{group_dir_fut_spat}/included_cells_{trials}_reps_{sparsity_c}_pct.txt").read_text().splitlines()
        else:
            perms = 1
            
            
        # for each cell, cross-validate the peak task-lag shift for spatial consistency.
        for neuron_idx, curr_neuron in enumerate(data[f"sub-{sesh:02}"]['normalised_neurons']):
            # resetting unique tasks for each neuron.
            unique_grids, _, beh_df['idx_same_grids'], _ = np.unique(
                beh_df[grid_cols].to_numpy(),
                axis=0,
                return_index=True,
                return_inverse=True,
                return_counts=True
            )

            if perms > 1:
                if curr_neuron not in include_these_cells:
                    continue
            else: 
                included_neurons.append(curr_neuron)
            
            # clean the data such that I don't consider 'bad' blocks of repeats
            # with super low firing 
            idx_same_grids = beh_df['idx_same_grids'].to_numpy()
            if sparsity_c:
                beh_df = mc.analyse.helpers_human_cells.extract_consistent_grids(data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy(), curr_neuron, beh_df)
                consistent_grids_mask = beh_df[f'consistent_FR_{curr_neuron}'].to_numpy()
                # set grid indexes I want to ignore to -1
                idx_same_grids[~consistent_grids_mask] = -1

            if sparsity_c and len(unique_grids) < 3:
                print(f"excluding {curr_neuron} in sesh {sesh} because there were not enough grids with consistent FR!")
                continue
            
                
            # instead of doing grid-by-grid, I will now do 2 grids combined
            # I am maximising spatial coverage in this pairing.
            # like this, I'm hoping to compute a more stable firing rate map because
            # there will be more locations that will have been covered.
            if combine_two_grids == True:
                beh_df = mc.analyse.helpers_human_cells.pair_grids_to_increase_spatial_coverage(data[f"sub-{sesh:02}"]['locations'], beh_df, curr_neuron)
                mask = beh_df[f'paired_grid_idx_{curr_neuron}'].map(lambda x: x is not False)
                unique_grids = np.unique(beh_df[f'paired_grid_idx_{curr_neuron}'][mask].dropna().astype(int))
                idx_same_grids = beh_df[f'paired_grid_idx_{curr_neuron}'].to_numpy()
            else:
                if not idx_same_grids:
                    idx_same_grids = beh_df['idx_same_grids'].to_numpy()


            # loop through n-1 grids, respectively
            # import pdb; pdb.set_trace()
            for perm_idx in range(0,perms):
                validate_peak_lag = np.full((len(unique_grids),2), np.nan, dtype=float)
                for count_test_task, test_task_id in enumerate(unique_grids):
                    mask_test_task = (idx_same_grids == test_task_id)
                    neurons_test_task = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].loc[mask_test_task].to_numpy()
                    locs_test_task = data[f"sub-{sesh:02}"]['locations'].loc[mask_test_task].to_numpy()
                    
                    mask_train_task = (idx_same_grids != test_task_id)
                    # create subset of df and neurons.
                    beh_train_task = data[f"sub-{sesh:02}"]['beh'].loc[mask_train_task].reset_index()
                    neurons_train_task = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].loc[mask_train_task].to_numpy()
                    locs_train_task = data[f"sub-{sesh:02}"]['locations'].loc[mask_train_task].to_numpy()
                    unique_idx_train = idx_same_grids[mask_train_task]
                    
                    # and compute the peak-spatial tuning rotation
                    peak_task_lag_train, fr_maps_train_at_best_lag, dwell_at_train_best_lag = comp_peak_spatial_tuning(neurons_train_task, locs_train_task, beh_train_task, curr_neuron, unique_idx_train, plotting=plotting, perm_no = no_perms, weighted=weighted)

                    # import pdb; pdb.set_trace()
                    # validate: compute the correlation between rate-map of held-out task at this lag
                    fr_map_test_at_best_lag, dwell_test_at_best_lag = compute_fr_at_spatial_lag(peak_task_lag_train, neurons_test_task, locs_test_task)
                    
                    # # ignore nans 
                    consistency_train_test = []
                    eps = 1e-12
                    if weighted == True:
                        # Inverse-variance weights per location:
                        # w = 1 / ( r_tr/T_tr + r_te/T_te )
                        for i, train_grid in enumerate(fr_maps_train_at_best_lag.T):
                            r_tr = np.asarray(train_grid, float)                 # (9,)
                            r_te = np.asarray(fr_map_test_at_best_lag, float)    # (9,)
                    
                            T_tr = np.asarray(dwell_at_train_best_lag[:, i], float)   # (9,)
                            T_te = np.asarray(dwell_test_at_best_lag, float)           # (9,)
                    
                            w = 1.0 / (r_tr/(T_tr + eps) + r_te/(T_te + eps) + eps)    # inverse-variance proxy
                    
                            # weighted Pearson over finite, positive-weight locations
                            consistency_train_test.append(
                                weighted_pearson(r_tr, r_te, w)
                            )
                        spatial_consistency_validation = np.nanmean(consistency_train_test)

                    else:
                        for train_grid in fr_maps_train_at_best_lag.T:
                            nan_mask = np.isfinite(train_grid) & np.isfinite(fr_map_test_at_best_lag)
                            consistency_train_test.append(np.corrcoef(train_grid[nan_mask], fr_map_test_at_best_lag[nan_mask])[0][1])
                        spatial_consistency_validation = np.nanmean(consistency_train_test)
        
                    # store the average correlation value for lag-rotation for current test-task
                    validate_peak_lag[count_test_task,0] = spatial_consistency_validation
                    validate_peak_lag[count_test_task,1] = peak_task_lag_train


                m = mode(validate_peak_lag[:,1], keepdims=False)
                peak_shift_validated, frequency_peak_shift_validated = m.mode, m.count
                avg_consistency_at_peak = np.mean(validate_peak_lag[:,0])
                # import pdb; pdb.set_trace()
                results.append({
                "session_id": sesh,
                "neuron_id": curr_neuron,
                "mode_peak_shift": peak_shift_validated,
                "avg_consistency_at_peak": avg_consistency_at_peak,
                "count_mode_peak": f"{frequency_peak_shift_validated} out of {len(validate_peak_lag[:,1])}",
                "perm_idx": perm_idx,
                "mean_firing_rate": beh_df[f'mean_FR_{curr_neuron}'].to_numpy()[0],
                "sparse_repeats": sum(~beh_df[f'consistent_FR_{curr_neuron}'])
                })
                
                if not no_perms:
                    print(f"average spatial consistency at lag {peak_shift_validated} for neuron {curr_neuron} is {avg_consistency_at_peak}, peak occuring {frequency_peak_shift_validated} times")
                if no_perms:
                    if perm_idx % 100 == 0:
                        print(f"now computing permutation {perm_idx} for neuron {curr_neuron}...")
                    
                    
    
    # import pdb; pdb.set_trace()
    results_df = pd.DataFrame(results, columns = COLUMNS)
    if save_all == True:
        if not os.path.isdir(group_dir_fut_spat):
            os.mkdir(group_dir_fut_spat)
        
        if perm_idx > 1:
            result_string = f"spatial_consistency_{trials}_repeats.csv"
            if sparsity_c:
                result_string = f"spatial_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons.csv"
            if weighted == True and sparsity_c:
                result_string =f"spatial_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons_weighted.csv"
            empirical_result = pd.read_csv(f"{group_dir_fut_spat}/{result_string}")
            perm_string = f"pval_for_perms200_{result_string}"
            name_result_stats = f"{group_dir_fut_spat}/{perm_string}"
            perm_pval_result = store_p_vals_perms(true_df = empirical_result, perm_df = results_df, out_path=name_result_stats, trials=trials, consider_state=False)
            mc.plotting.results.plot_perm_spatial_consistency(results_df, empirical_result, name_result_stats, group_dir_fut_spat)
            title_string = f'Binomial test after single-cell permutations, {trials} repeats'
            plt_binomial_per_roi(perm_pval_result, title_string)
            
            import pdb; pdb.set_trace()

        else:
            name_result = f"{group_dir_fut_spat}/spatial_consistency_{trials}_repeats.csv"
            plot_results_per_roi(results_df, title_string_add = f'{trials}_repeats',plot_by_pfc=False,plot_by_cingulate_and_MTL=True)
            plot_results_per_roi_and_future_lag(results_df, title_string_add = f'{trials}_repeats')
            if sparsity_c:
                name_result = f"{group_dir_fut_spat}/spatial_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons.csv"
                print(f"included {len(included_neurons)} neurons.")
                Path(f"{group_dir_fut_spat}/included_cells_{trials}_reps_{sparsity_c}_pct.txt").write_text("\n".join(included_neurons))
            if weighted == True and sparsity_c:
                name_result = f"{group_dir_fut_spat}/spatial_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons_weighted.csv"
                 
        results_df.to_csv(name_result)
        print(f"saved cross-validated spatial tuning values in {name_result}")  
        import pdb; pdb.set_trace()
    
    
# # helper load state results and merge
# state_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/state_tuning/pval_for_perms200_state_consistency_all_minus_explore_repeats_excl_gridwise_qc_pct_neurons.csv"
# state_df = pd.read_csv(state_path)
# merged = results_df.merge(state_df,on=["session_id","neuron_id"], how="left")
# results_without_state = merged[merged["p_perm"]>0.05]

# if running from command line, use this one!   
# if __name__ == "__main__":
#     fire.Fire(plot_all)
#     # call this script like
#     # python wrapper_plot_elnetreg_results.py --model_name_string='w_partial_musicboxes_excl_rep1-2', --models_I_want='['withoutnow', 'only2and3future','onlynowandnext']' --exclude_x_repeats='[1,2]' --randomised_reward_locations=False --save_regs=True


if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    # trials can be 'all', 'all_correct', 'early', 'late', 'all_minus_explore'
    # compute_fut_spatial_tunings(sessions=[3], trials = 'all', plotting=False, no_perms = None, combine_two_grids = True, sparsity_c = 'gridwise_qc', weighted = True, save_all=False)
    compute_fut_spatial_tunings(sessions=list(range(0,61)), trials = 'all_minus_explore', no_perms = 200, combine_two_grids = True, sparsity_c = 'gridwise_qc', weighted = True, save_all=True)
    # compute_fut_spatial_tunings(sessions=list(range(0,60)), trials = 'all', no_perms = 200, combine_two_grids = True)
    # compute_fut_spatial_tunings(sessions=[31], trials = 'all', plotting = False, no_perms = None, combine_two_grids = True)
    
    
    
    