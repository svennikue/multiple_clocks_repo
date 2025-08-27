#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:41:15 2025

@author: Svenja Küchenhoff

This script is to plot FR maps for neurons based on the computations I make
with the future_spatial_peaks computation.

# Per neuron, I want a figure where: 
# at the top I have a rate map across grids.
# next row is the mean grid correlation plot per temporal shift
# Then, per rotation, I want a rate-map of all grids that are 'summarised'
# this shall be together with a map in greys where I can see the spatial coverage (no of samples) per location


This is based on loading the cells normalised in 360 bins and computing their spatial peaks.

It does so using cross-validation: testing the spatial specificity across grids,
leaving one out, respectively. 

"""

import mc
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        

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
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4,5]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4,5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4,5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4,5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'late':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6,7,8,9,10])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6,7,8,9,10])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6,7,8,9,10])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6,7,8,9,10])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    

    #import pdb; pdb.set_trace()
    return filtered_data
   
    
   
def comp_peak_spatial_tuning(neurons, locs, beh, cell_name, idx_same_grids, plotting=False):
    temporal_shift = 30
    shifts = np.arange(0, 360, temporal_shift)  # 12 shifts
    unique_grids = np.unique(idx_same_grids)
        
    mean_corr_per_shift = []
    # mean_corr_per_shift_test_pd = []
    fr_maps_by_shift = []
    
    for shift in shifts:
        mean_firing_rates_locs = np.full((9, len(unique_grids)), np.nan, dtype=float)
        for task_count, task_id in enumerate(unique_grids):
            mask_curr_grid = (idx_same_grids == task_id)
            neurons_curr_grid = neurons[mask_curr_grid]

            locs_shifted = np.roll(locs, shift=-shift, axis=1)
            locs_curr_grid = locs_shifted[mask_curr_grid]
            
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
                    mean_firing_rates_locs[loc-1, task_count] = np.nan
                elif np.any(sel):
                    # print(f"length of selection of loc {loc} is {np.sum(sel)} and mean is {fr_clean[sel].mean()} for shift {shift} and first bin is {np.where(sel)[0][0]}")
                    mean_firing_rates_locs[loc-1, task_count] = fr_clean[sel].mean()
                else:
                    mean_firing_rates_locs[loc-1, task_count] = np.nan
      
        fr_maps_by_shift.append(mean_firing_rates_locs)
        
        # --- mean grid correlation for this shift ---
        if np.all(np.isnan(mean_firing_rates_locs)):
            mean_corr_per_shift.append(np.nan)
        else:
            cm = pd.DataFrame(mean_firing_rates_locs).corr()              
            upper = np.triu(np.ones(cm.shape, bool), k=1)
            mean_corr_per_shift.append(np.nanmean(cm.values[upper]))

    # import pdb; pdb.set_trace()
    
    # if cell_name == '07-07-mRF2Ca08-RACC':
    #     import pdb; pdb.set_trace()
    # if np.isnan(best_map).any():
    #     import pdb; pdb.set_trace()

    return fr_maps_by_shift, mean_corr_per_shift





def extract_consistent_grids(neuron, cell_name, beh):
    # DIFFERENCE BETWEEN grid-blocks AND unique grids
    # goal: kick out grid-blocks that are unreliable.
    
    # per grid-block, identify firing rate
    # exclude grid if firing rate lower than 20% of mean firing.
    # also make sure to leave at least 3 unique grids.

    beh[f'mean_FR_{cell_name}'] = np.nanmean(neuron)
    # identify firing rate per grid-block.
    grid_nos = np.unique(beh['grid_no'].to_numpy())
    # FR per grid_no
    grid_fr = {}
    for g in grid_nos:
        mask_g = (beh['grid_no'] == g)
        grid_fr[g] = np.nanmean(neuron[mask_g])
        
    # attach column (row-wise) for convenience/inspection
    beh[f'grid_FR_{cell_name}'] = beh['grid_no'].map(grid_fr)
    

    # --- tentative exclusion: FR < 20% of overall mean (treat NaN FR as low) ---
    excluded_grid_nos = []
    thresh = 0.2 * np.nanmean(neuron) if not np.isnan(np.nanmean(neuron)) else np.nan
    for g in grid_nos:
        fr = grid_fr[g]
        if np.isnan(fr) or (not np.isnan(thresh) and fr < thresh):
            excluded_grid_nos.append(g)
    
    # 3) Tentative keep-mask with low-rate grids removed
    tentative_keep_mask = ~beh['grid_no'].isin(excluded_grid_nos)
    # tentative_keep_mask = ~beh['new_grid_idx'].isin(excluded_grid_nos)

    
    # next test based on this new selection, how many UNIQUE GRIDS are left?
    # at least 3 so cross-validation is possible.
    kept_identities = beh['idx_same_grids'][tentative_keep_mask].to_numpy()
    no_unique_good_grids = len(np.unique(kept_identities))
    target_unique_min = 3
    add_back_grids = []
    if no_unique_good_grids < target_unique_min:
        # stepwise add best bad grid in
        # sort excluded grids by FR descending (NaN treated as -inf)
        def fr_key(g):
            fr = grid_fr[g]
            return -np.inf if np.isnan(fr) else fr
        excluded_sorted = sorted(excluded_grid_nos, key=fr_key, reverse=True)

        # prefer adding grids that increase identity diversity
        # ADD GRIDS THAT FIRE MOST BACK IN
        for g in excluded_sorted:
            if len(kept_identities) >= target_unique_min:
                break
            unique_id = np.unique(beh['idx_same_grids'][beh['grid_no'] == g].to_numpy())
            if unique_id not in kept_identities:
                kept_identities.add(unique_id)
                add_back_grids.append(g)
        # if still short (e.g., identity overlap), add best remaining anyway
        if len(kept_identities) < target_unique_min:
            for g in excluded_sorted:
                if g not in add_back_grids:
                    add_back_grids.append(g)


    # final per-row keep decision (BUT DO NOT FILTER beh)
    final_keep_mask = tentative_keep_mask | beh['grid_no'].isin(add_back_grids)
    
    beh[f'consistent_FR_{cell_name}'] = final_keep_mask
   
    return beh


    
    
def pair_grids_to_increase_spatial_coverage(locs, beh, cell_name, min_coverage=100, min_groups=3,max_groups=5):
    """
    Merge original grids into groups to maximize coverage per combo:
      1) more locations with >= min_coverage
      2) higher minimum per-location total
    Returns a new index array (0..n_groups-1) aligned with same_grids.
    (Optionally) filters rows by a consistent_FR_{cell_name} boolean column.
    # Builds grid_cvg: for each original grid id, counts how often each location occurs.
    # Greedily proposes pairs: for grid g, it finds a partner h that (1) fixes the most weak locations (< min_coverage) and (2) maximizes the weakest-location boost.
    # Maps those pairs back to a new index per row (same_grid_idx_new) and writes it into beh[f'paired_grid_idx_{cell_name}'].
    """
    # depending on what filertering happened before, use different grids_nos.
    # import pdb; pdb.set_trace()
    if f"consistent_FR_{cell_name}" in beh:
        # first filter locations and same_grids for grids that are reliable.
        reliable_FR_mask = beh[f"consistent_FR_{cell_name}"].to_numpy()
        locs = locs[reliable_FR_mask]
        same_grids = beh['idx_same_grids'][beh[f'consistent_FR_{cell_name}'] == True].to_numpy()
    else:
        same_grids = beh['idx_same_grids'].to_numpy()
        
    # --- 1) build coverage dict for each original grid ---
    grid_cvg = {}
    # import pdb; pdb.set_trace()
    for grid_idx in np.unique(same_grids):
        grid_cvg[grid_idx] = {}
        all_locs_curr_grid = locs[same_grids == grid_idx]
        for loc in range(1,10):
            grid_cvg[grid_idx][loc] = np.count_nonzero(all_locs_curr_grid == loc)
     
    # first pass: test if there are enough grids to build 3 pairs. If not, 
    # identify which grid has the best coverage and take that as a single grid.
    pairs = {}
    used_grids = []
    if len(grid_cvg) < min_groups*2:
        best_cvg = []
        for g, loc_dict in grid_cvg.items():
            # 1. find the grid with best coverage
            # to do so, first count how many grids are below the cut-off per grid
            weak_counts = {g: sum(v < min_coverage for v in loc_dict.values())
                               for g, loc_dict in grid_cvg.items()}
            # pick the grid with the smallest number of missing coverage
            best_grid = min(weak_counts, key=weak_counts.get)
            pairs[best_grid] = best_grid          # single grid acts as its own "pair"
            used_grids.append(best_grid)

    # next pass: only pairing grids that have bad spatial coverage somewhere.
    for g, loc_dict in grid_cvg.items():
        # 1. find weak locations
        low_locs = [l for l, v in loc_dict.items() if v < min_coverage]
        if not low_locs:
            # import pdb; pdb.set_trace()
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
    
    # next, pair the ones that have not been used (i.e. had good coverage)
    left_over = []
    for i in np.unique(same_grids):
        if i not in used_grids:
            left_over.append(i)
    
    if len(left_over)%2 == 1:
        # if an uneven number is left, keep the best coverage one alone.
        # best coverage defines as highest minimum coverage.
        mins = {g: min(grid_cvg[g].values()) for g in left_over}
        best_grid = np.random.default_rng().choice([g for g, m in mins.items() if m == max(mins.values())])
        pairs[best_grid] = best_grid
        used_grids.append(best_grid)
    
    left_over = []
    for i in np.unique(same_grids):
        if i not in used_grids:
            left_over.append(i)

    # just pair randomly
    for idx, unused_g in enumerate(left_over):
        if unused_g in used_grids:
            continue
        pairs[unused_g] = left_over[idx+1]
        used_grids.append(unused_g)
        used_grids.append(left_over[idx+1])

    # delete if this works
    for i in np.unique(same_grids):
        if i not in used_grids:
            # IF THERE ARE STILL LEFT OVERS, SOMETHING IS WRONG!!
            # next, test how many pairs have been built and how many grids go unused.
            import pdb; pdb.set_trace()



    # lastly, map the new grid indices back to the old unique grid index.
    same_grid_idx_new = np.full(same_grids.shape, -1, dtype=int)
    
    for new_same_grid, old_grid_idx in enumerate(pairs):
        same_grid_idx_new[same_grids == old_grid_idx] = int(new_same_grid)
        same_grid_idx_new[same_grids == pairs[old_grid_idx]] = int(new_same_grid)
    

    # # brief check if the coverage is better now
    # new_grid_cvg = {}
    # for grid_idx in np.unique(same_grid_idx_new):
    #     new_grid_cvg[grid_idx] = {}
    #     all_locs_curr_grid = locs[same_grid_idx_new == grid_idx]
    #     for loc in np.unique(locs):
    #         new_grid_cvg[grid_idx][loc] = np.count_nonzero(all_locs_curr_grid == loc)
    if f"consistent_FR_{cell_name}" in beh:
        beh[f'paired_grid_idx_{cell_name}'] = np.full(reliable_FR_mask.shape, False, dtype=object)
        beh[f'paired_grid_idx_{cell_name}'][reliable_FR_mask] = same_grid_idx_new
    else:
        beh[f'paired_grid_idx_{cell_name}'] = same_grid_idx_new
    return beh




def plot_neuron_overview(neuron_name, beh, neuron, locs, FR_maps_neuron, spatial_corr_per_shift, same_grids, title_string):
    # --- Inputs assumed already defined:
    # neuron : 2D array (repeats x timebins)
    # spatial_corr_per_shift : 1D array length = len(shifts)
    # neuron_name : str
    # shifts : e.g. np.arange(0,360,30)  # 12 values
    # FR_maps_neuron : shape (len(shifts), 9, n_grids)  -> each map reshapes to (3,3)
    # same_grids, locs : arrays to build grid_cvg (coverage)
    # unique_grids : np.unique(same_grids)
    
    # import pdb; pdb.set_trace()
    unique_grids = np.unique(same_grids)               # flexible number of grids
    print(f"{neuron_name} has {len(unique_grids)} grids.")
    # ----- Build spatial coverage (grey column) -----
    grid_cvg = {}
    for g in unique_grids:
        grid_cvg[g] = np.zeros(9, dtype=int)
        glocs = locs[same_grids == g]
        for loc in range(1, 10):
            grid_cvg[g][loc-1] = np.count_nonzero(glocs == loc)
    
    cov_min = min(np.min(v) for v in grid_cvg.values())
    cov_max = max(np.max(v) for v in grid_cvg.values())
    
    # ----- Global FR map color limits -----
    rate_min = np.nanmin(FR_maps_neuron)
    rate_max = np.nanmax(FR_maps_neuron)
    
    # ----- Which grids to show as rows  -----
    n_grids = len(unique_grids)
    grid_order = sorted(grid_cvg.keys())[:n_grids]  # align to n grid rows
    
    # ----- Find best shift for outline -----
    temporal_shift = 30
    shifts = np.arange(0, 360, temporal_shift)  # 12 shifts
    best_idx = np.nanargmax(spatial_corr_per_shift) if np.any(np.isfinite(spatial_corr_per_shift)) else None
    best_shift = shifts[best_idx] if best_idx is not None else None
    
    # ===== FIGURE LAYOUT =====
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(nrows=3, ncols=1, height_ratios=[2.2, 1.8, 7.0], hspace=0.35, figure=fig)
    
    # --- Row 1: Firing rate (full width)
    ax_fr = fig.add_subplot(gs[0, 0])
    im_fr = ax_fr.imshow(neuron, aspect='auto', cmap='Reds')
    ax_fr.vlines([90, 180, 270], 0, neuron.shape[0]-1, linestyles='dotted', linewidth=0.8, colors='black')
    ax_fr.set_title('Firing rate per bin (x) and repeat (y)')
    ax_fr.set_xlabel('Time (bins)')
    ax_fr.set_ylabel('Repeats')
    cbar_fr = fig.colorbar(im_fr, ax=ax_fr, fraction=0.025, pad=0.02)
    cbar_fr.set_label("Rate (spike/s)")
    
    # --- Row 2: Spatial encoding consistency vs temporal shift (full width)
    ax_cons = fig.add_subplot(gs[1, 0])
    ax_cons.plot(shifts, spatial_corr_per_shift, marker='o', color='black', label=str(neuron_name))
    ax_cons.hlines(0, shifts[0], shifts[-1], colors='k', linestyles='-')
    ax_cons.set_xlabel('Temporal shift (bins of 30)')
    ax_cons.set_ylabel('Mean grid correlation')
    ax_cons.set_title(f"Neuron {neuron_name} — Spatial encoding consistency vs. temporal shift")
    ax_cons.legend(loc='best', fontsize=8)
    if best_shift is not None:
        ax_cons.axvline(best_shift, color='tab:red', linestyle='--', linewidth=1)
    
    # --- Rows 3–5: 3 rows × (1 coverage + len(shifts)) columns
    ncols_bottom = 1 + len(shifts)  # 1 coverage col + shifts
    gs_bot = GridSpecFromSubplotSpec(
        nrows=n_grids, ncols=ncols_bottom, subplot_spec=gs[2, 0], wspace=0.05, hspace=0.08
    )
    
    axs = np.empty((n_grids, ncols_bottom), dtype=object)
    
    # Column headers
    for c in range(ncols_bottom):
        ax_title = fig.add_subplot(gs_bot[0, c])
        fig.delaxes(ax_title)  # we'll place real axes shortly; title texts go on real axes
    # we’ll set titles on real axes below (top row only)
    
    # Fill axes
    for r, g in enumerate(grid_order):
        # Coverage column (col 0)
        ax_cov = fig.add_subplot(gs_bot[r, 0])
        axs[r, 0] = ax_cov
        spatial_cov = grid_cvg[g].reshape(3, 3)
        im_cov = ax_cov.imshow(spatial_cov, cmap='Greys', vmin=cov_min, vmax=cov_max)
        ax_cov.set_xticks([]); ax_cov.set_yticks([])
        if r == 0:
            ax_cov.set_title("Coverage")
        # annotate coverage numbers
        for (i, j), v in np.ndenumerate(spatial_cov):
            ax_cov.text(j, i, f"{int(v)}", ha='center', va='center', color='black', fontsize=8)
    
        # Row label (left of coverage cell)
        ax_cov.text(-0.45, 0.5, f"Grid combo {r+1}",
                    transform=ax_cov.transAxes, va='center', ha='right', fontsize=10)
    
        # Shift columns (1..len(shifts))
        for si, sh in enumerate(shifts):
            ax = fig.add_subplot(gs_bot[r, si+1])
            axs[r, si+1] = ax
            spatial = FR_maps_neuron[si][:, r].reshape(3, 3)
            im = ax.imshow(spatial, cmap='coolwarm', vmin=rate_min, vmax=rate_max)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{sh}°", fontsize=9)
    
    # Outline the best-shift column across all three rows
    if best_shift is not None and best_shift in shifts:
        col_idx = 1 + int(np.where(shifts == best_shift)[0][0])  # +1 because col 0 = coverage
        for r in range(n_grids):
            axs[r, col_idx].add_patch(Rectangle((0, 0), 1, 1,
                               transform=axs[r, col_idx].transAxes, fill=False,
                               linewidth=2.5, edgecolor='black', zorder=10))
    
    # Shared colorbar for rate maps (all shift columns)
    # Use the last 'im' as mappable (safe because vmin/vmax consistent)
    cbar_rate = fig.colorbar(im, ax=axs[:, 1:].ravel().tolist(), fraction=0.02, pad=0.01)
    cbar_rate.set_label("Avg. rate per field")
    
    # Optional colorbar for coverage column
    # cbar_cov = fig.colorbar(im_cov, ax=axs[:, 0].ravel().tolist(), fraction=0.02, pad=0.04)
    #cbar_cov.set_label("Coverage (samples)")
    
    # Big y-label for bottom block
    # fig.text(0.06, 0.21, "3×3 spatial maps", rotation=90, va='center', fontsize=11)
    fig.suptitle(title_string, fontsize=18, fontweight="bold", y=0.995)

    plt.show()
 
 

def compute_fut_spatial_tunings(sessions, trials = 'all_correct', combine_two_grids = False, sparsity_c = None, save_all = False):  
    # trials can be 'all', 'all_correct', 'early', 'late'
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
        beh_df = data[f"sub-{sesh:02}"]['beh']
        # determine identical grids
        grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
            
        # for each cell, cross-validate the peak task-lag shift for spatial consistency.
        for neuron_idx, curr_neuron in enumerate(data[f"sub-{sesh:02}"]['normalised_neurons']):

            if curr_neuron not in ['07-07-chan104-LOFC', '16-16-chan118-LHC', '01-01-mLF2aCa07-LACC']:
                continue    
            # if curr_neuron not in ['03-03-chan110-REC', '07-07-chan104-LOFC', '16-16-chan118-LHC', '01-01-mLF2aCa07-LACC']:
            #     continue
            # resetting unique tasks for each neuron.
            unique_grids, _, beh_df['idx_same_grids'], _ = np.unique(
                beh_df[grid_cols].to_numpy(),
                axis=0,
                return_index=True,
                return_inverse=True,
                return_counts=True
            )
            
            
            # clean the data such that I don't consider 'bad' blocks of repeats
            # with super low firing 
            idx_same_grids = beh_df['idx_same_grids'].to_numpy()
            if sparsity_c:
                beh_df = extract_consistent_grids(data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy(), curr_neuron, beh_df)
                idx_same_grids = idx_same_grids[beh_df[f'consistent_FR_{curr_neuron}']]
                
            # instead of doing grid-by-grid, I will now do 2 grids combined
            # I am maximising spatial coverage in this pairing.
            # like this, I'm hoping to compute a more stable firing rate map because
            # there will be more locations that will have been covered.
            if combine_two_grids == True:
                beh_df = pair_grids_to_increase_spatial_coverage(data[f"sub-{sesh:02}"]['locations'], beh_df, curr_neuron)
                idx_same_grids = beh_df[f'paired_grid_idx_{curr_neuron}'].to_numpy()
            else:
                if idx_same_grids:
                    continue
                else:
                    idx_same_grids = beh_df['idx_same_grids'].to_numpy()

            # THIS IS NOW WITHOUT CROSS-VALIDATION NOR PERMS, JUST FOR PLOTTING PURPOSES.
            neuron_data = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy()
            locations = data[f"sub-{sesh:02}"]['locations'].to_numpy()
            behaviour = data[f"sub-{sesh:02}"]['beh']
            fr_by_shift, corr_per_shift = comp_peak_spatial_tuning(neuron_data, locations, behaviour, curr_neuron, idx_same_grids)

            file_name = f"spatial_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons.csv"
            df = pd.read_csv(f"{group_dir_fut_spat}/{file_name}")
            cross_val = df['avg_consistency_at_peak'][df['neuron_id'] == curr_neuron].to_list()

            title = f"overview for {curr_neuron} \n including {trials} repeats. \n cross-val value: {round(cross_val[0], 3)}"
            plot_neuron_overview(curr_neuron, beh_df, data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron], data[f"sub-{sesh:02}"]['locations'], fr_by_shift, corr_per_shift, idx_same_grids, title)
            
            
    
def loop_through_trial_combos(trial_list):
    for incl_trials in trial_list:
        compute_fut_spatial_tunings(sessions=[1,6,9], trials = incl_trials, combine_two_grids = True, sparsity_c = 'gridwise_qc', save_all=False)


if __name__ == "__main__":
    # trials can be 'all', 'all_correct', 'early', 'late'
    # loop through these different trials to compare a single cell.
    loop_through_trial_combos(trial_list = ['all_correct', 'early', 'late'])
    #loop_through_trial_combos(trial_list = ['all', 'all_correct', 'early', 'late'])

    
    