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

def normalize_maps(FR_maps, mode="zscore", qclip=98):
    """
    FR_maps: (n_shifts, 9, n_grids)
    Returns: FR_maps_norm, (vmin, vmax)
    """
    FR_maps = np.asarray(FR_maps, float)
    if mode == "zscore":
        out = np.empty_like(FR_maps, dtype=float)
        for s in range(FR_maps.shape[0]):
            for g in range(FR_maps.shape[2]):
                v = FR_maps[s, :, g]
                m = np.nanmean(v)
                sd = np.nanstd(v)
                out[s, :, g] = (v - m) / (sd if sd > 1e-12 else 1.0)
        lim = np.nanpercentile(np.abs(out), qclip)
        return out, (-float(lim), float(lim))
    elif mode == "global":
        # one scale for all raw maps (keeps absolute magnitude)
        vmin = np.nanpercentile(FR_maps, 2)
        vmax = np.nanpercentile(FR_maps, 98)
        return FR_maps, (float(vmin), float(vmax))
    elif mode == "per_grid":
        # per-grid row scales (list of (vmin,vmax) per grid)
        clims = []
        for g in range(FR_maps.shape[2]):
            row_vals = FR_maps[:, :, g].ravel()
            vmin = np.nanpercentile(row_vals, 2)
            vmax = np.nanpercentile(row_vals, 98)
            clims.append((float(vmin), float(vmax)))
        return FR_maps, clims
    else:
        raise ValueError("mode must be 'zscore', 'global', or 'per_grid'")

    
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
    
   
def comp_peak_spatial_tuning(neurons, locs, beh, cell_name, idx_same_grids, plotting=False, weighted = False):
    temporal_shift = 30
    shifts = np.arange(0, 360, temporal_shift)  # 12 shifts
    unique_grids = np.unique(idx_same_grids)
        
    mean_corr_per_shift,fr_maps_by_shift, dwell_by_shift = [], [], []
    
    for shift in shifts:
        mean_firing_rates_locs = np.full((9, len(unique_grids)), np.nan, dtype=float)
        dwell_time_at_locs = np.full((9, len(unique_grids)), np.nan, dtype=float)
        
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
            dt_clean = np.ones_like(fr_clean, dtype=float)

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

    # import pdb; pdb.set_trace()
    
    # if cell_name == '07-07-mRF2Ca08-RACC':
    #     import pdb; pdb.set_trace()
    # if np.isnan(best_map).any():
    #     import pdb; pdb.set_trace()

    return fr_maps_by_shift, mean_corr_per_shift, dwell_by_shift



def plot_neuron_overview(neuron_name, beh, neuron, locs, FR_maps_neuron,
                         spatial_corr_per_shift, same_grids, title_string):

    unique_grids = np.unique(same_grids)
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

    # ----- Collapsed coverage (sum across grids)  # NEW
    cov_collapsed = sum(grid_cvg.values())
    cov_max_all = max(cov_max, np.max(cov_collapsed))       # scale includes collapsed  # NEW

    # # ----- Global FR map color limits -----
    # rate_min = np.nanmin(FR_maps_neuron)
    # rate_max = np.nanmax(FR_maps_neuron)
    # ----- Collapsed FR maps (mean over grids)
    FR_maps_collapsed = np.nanmean(FR_maps_neuron, axis=2)          # (n_shifts, 9)
    
    # ----- Z-score ALL maps (per 3x3), get one symmetric color scale for pattern comparison
    # stack collapsed as an extra "grid" so limits include it too
    FR_all = np.concatenate([FR_maps_neuron, FR_maps_collapsed[:, :, None]], axis=2)  # (S, 9, G+1)
    FR_all_Z, (rate_min, rate_max) = normalize_maps(FR_all, qclip=98)
    
    # split back
    n_grids = len(unique_grids)
    FR_maps_Z = FR_all_Z[:, :, :n_grids]          # (S, 9, G)
    FR_coll_Z = FR_all_Z[:, :, -1]                # (S, 9)

    # ----- Rows to show -----
    n_grids = len(unique_grids)
    grid_order = sorted(grid_cvg.keys())[:n_grids]

    # ----- Collapsed FR maps (mean over grids)  # NEW
    # FR_maps_neuron: (n_shifts, 9, n_grids) -> (n_shifts, 9)
    FR_maps_collapsed = np.nanmean(FR_maps_neuron, axis=2)

    # ----- Best shift for outline -----
    temporal_shift = 30
    shifts = np.arange(0, 360, temporal_shift)
    best_idx = np.nanargmax(spatial_corr_per_shift) if np.any(np.isfinite(spatial_corr_per_shift)) else None
    best_shift = shifts[best_idx] if best_idx is not None else None

    # ===== FIGURE LAYOUT =====
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(nrows=3, ncols=1, height_ratios=[2.2, 1.8, 7.0], hspace=0.35, figure=fig)

    # Row 1: firing rate
    ax_fr = fig.add_subplot(gs[0, 0])
    im_fr = ax_fr.imshow(neuron, aspect='auto', cmap='Reds')
    ax_fr.vlines([90, 180, 270], 0, neuron.shape[0]-1, linestyles='dotted', linewidth=0.8, colors='black')
    ax_fr.set_title('Firing rate per bin (x) and repeat (y)')
    ax_fr.set_xlabel('Time (bins)')
    ax_fr.set_ylabel('Repeats')
    cbar_fr = fig.colorbar(im_fr, ax=ax_fr, fraction=0.025, pad=0.02)
    cbar_fr.set_label("Rate (spike/s)")

    # Row 2: spatial encoding vs shift
    ax_cons = fig.add_subplot(gs[1, 0])
    ax_cons.plot(shifts, spatial_corr_per_shift, marker='o', color='black', label=str(neuron_name))
    ax_cons.hlines(0, shifts[0], shifts[-1], colors='k', linestyles='-')
    ax_cons.set_xlabel('Temporal shift (bins of 30)')
    ax_cons.set_ylabel('Mean grid correlation')
    ax_cons.set_title(f"Neuron {neuron_name} — Spatial encoding consistency vs. temporal shift")
    ax_cons.legend(loc='best', fontsize=8)
    if best_shift is not None:
        ax_cons.axvline(best_shift, color='tab:red', linestyle='--', linewidth=1)

    # Bottom block: (n_grids + 1) rows  # CHANGED
    ncols_bottom = 1 + len(shifts)  # 1 coverage + shifts
    gs_bot = GridSpecFromSubplotSpec(
        nrows=n_grids + 1, ncols=ncols_bottom, subplot_spec=gs[2, 0], wspace=0.05, hspace=0.08
    )

    axs = np.empty((n_grids + 1, ncols_bottom), dtype=object)

    # Column headers placeholders (titles go on real axes)
    for c in range(ncols_bottom):
        ax_title = fig.add_subplot(gs_bot[0, c])
        fig.delaxes(ax_title)

    # --- Per-grid rows ---
    for r, g in enumerate(grid_order):
        # Coverage
        ax_cov = fig.add_subplot(gs_bot[r, 0]); axs[r, 0] = ax_cov
        spatial_cov = grid_cvg[g].reshape(3, 3)
        im_cov = ax_cov.imshow(spatial_cov, cmap='Greys', vmin=cov_min, vmax=cov_max_all)  # CHANGED vmax
        ax_cov.set_xticks([]); ax_cov.set_yticks([])
        if r == 0: ax_cov.set_title("Coverage")
        for (i, j), v in np.ndenumerate(spatial_cov):
            ax_cov.text(j, i, f"{int(v)}", ha='center', va='center', color='black', fontsize=8)
        ax_cov.text(-0.45, 0.5, f"Grid combo {r+1}", transform=ax_cov.transAxes,
                    va='center', ha='right', fontsize=10)

        # Shifts
        for si, sh in enumerate(shifts):
            ax = fig.add_subplot(gs_bot[r, si+1]); axs[r, si+1] = ax
            #spatial = FR_maps_neuron[si][:, r].reshape(3, 3)
            #im = ax.imshow(spatial, cmap='coolwarm', vmin=rate_min, vmax=rate_max)
            spatial = FR_maps_Z[si][:, r].reshape(3, 3)   # z-scored map
            im = ax.imshow(spatial, cmap='coolwarm', vmin=rate_min, vmax=rate_max)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{sh}°", fontsize=9)

    # --- Collapsed row (last)  # NEW ---
    r_coll = n_grids
    ax_covC = fig.add_subplot(gs_bot[r_coll, 0]); axs[r_coll, 0] = ax_covC
    im_covC = ax_covC.imshow(cov_collapsed.reshape(3, 3), cmap='Greys',
                             vmin=cov_min, vmax=cov_max_all)
    ax_covC.set_xticks([]); ax_covC.set_yticks([])
    ax_covC.text(-0.45, 0.5, "Collapsed grids", transform=ax_covC.transAxes,
                 va='center', ha='right', fontsize=10)
    for (i, j), v in np.ndenumerate(cov_collapsed.reshape(3, 3)):
        ax_covC.text(j, i, f"{int(v)}", ha='center', va='center', color='black', fontsize=8)

    for si, sh in enumerate(shifts):
        ax = fig.add_subplot(gs_bot[r_coll, si+1]); axs[r_coll, si+1] = ax
        spatialC = FR_coll_Z[si].reshape(3, 3)        # z-scored collapsed map
        im = ax.imshow(spatialC, cmap='coolwarm', vmin=rate_min, vmax=rate_max)

        
        #spatialC = FR_maps_collapsed[si].reshape(3, 3)
        #im = ax.imshow(spatialC, cmap='coolwarm', vmin=rate_min, vmax=rate_max)
        ax.set_xticks([]); ax.set_yticks([])
        # (titles already on top row)

    # Outline the best-shift column across all rows (including collapsed)  # CHANGED
    if best_shift is not None and best_shift in shifts:
        col_idx = 1 + int(np.where(shifts == best_shift)[0][0])
        for r in range(n_grids + 1):
            axs[r, col_idx].add_patch(Rectangle((0, 0), 1, 1,
                               transform=axs[r, col_idx].transAxes, fill=False,
                               linewidth=2.5, edgecolor='black', zorder=10))

    # Shared colorbar for all rate maps (incl. collapsed)
    cbar_rate = fig.colorbar(im, ax=axs[:, 1:].ravel().tolist(), fraction=0.02, pad=0.01)
    cbar_rate.set_label("Avg. rate per field")

    fig.suptitle(title_string, fontsize=18, fontweight="bold", y=0.995)
    plt.show()



#
#



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
    Merge original grids into groups to maximize spatial coverage.
    Rules:
      - <3 unique grids  -> discard neuron (output False).
      - 3 -> keep as 3 singletons.
      - 4 -> pair the two worst; others singleton (3 groups).
      - 5 -> pair worst 4 (2 pairs), keep best alone (3 groups).
      - 6 -> 3 pairs (maximize coverage).
      - 7 -> 3 pairs + best alone (4 groups).
      - 8 -> 4 pairs (maximize coverage).
      - 9 -> 4 pairs + best alone (5 groups).
      - 10 -> 5 pairs.
      - >10 -> make 5 groups: start with 5 best pairs, then add the rest to
               existing groups (triplets) maximizing coverage gain.
    Writes integer labels to beh[f'paired_grid_idx_{cell_name}'].
    """
    # --- get same_grids (optionally filter by consistent_FR) ---
    if f"consistent_FR_{cell_name}" in beh:
        # first filter locations and same_grids for grids that are reliable.
        reliable_FR_mask = beh[f"consistent_FR_{cell_name}"].to_numpy()
        locs_used = locs[reliable_FR_mask]
        same_grids = beh['idx_same_grids'][reliable_FR_mask].to_numpy()
    else:
        reliable_FR_mask = None
        locs_used = locs
        same_grids = beh['idx_same_grids'].to_numpy()
    
    # if cell_name == '05-05-mRF3cVPF04-RPvmPFC':
    #     import pdb; pdb.set_trace()
    #     # WHY DOES THIS CLUMP MORE THAN 3 GRIDS TOEGTEHR????
    
    unique_grids = np.unique(same_grids)
    

    # discard if fewer than 3 unique grids (column = False)
    if len(unique_grids) < 3:
        col = f'paired_grid_idx_{cell_name}'
        if reliable_FR_mask is not None:
            out = np.full(reliable_FR_mask.shape, False, dtype=object)
            beh[col] = out
        else:
            beh[col] = np.full(same_grids.shape, False, dtype=object)
        return beh
    

    # --- 1) build coverage dict for each original grid ---
    grid_cvg = {}
    
    for grid_idx in unique_grids:
        grid_cvg_vec = np.zeros(9, dtype=int)
        all_locs_curr_grid = locs_used[same_grids == grid_idx]
        for loc in range(1,10):
            grid_cvg_vec[loc-1] = np.count_nonzero(all_locs_curr_grid == loc)
        grid_cvg[grid_idx] = grid_cvg_vec
        
    
    # --- helpers for scoring coverage and choosing pairs ---
    def group_score(grids):
        """
        Score to maximize:
          1) # of locations with coverage >= min_coverage (higher is better)
          2) minimum coverage across locations (higher is better)
        """
        tot = sum((grid_cvg[g] for g in grids), np.zeros(9, dtype=int))
        n_cov_good = int(np.sum(tot >= min_coverage)) 
        min_cvg = int(np.min(tot)) if tot.size else 0
        return (n_cov_good, min_cvg)
    
    def worst_sort_key(g):
        """
        Higher = worse:
          1) more weak locations (< min_coverage)  -> worse
          2) lower minimum coverage                -> worse
        """
        v = grid_cvg[g]
        weak = int(np.sum(v < min_coverage))
        minv = int(np.min(v))
        return (weak, -minv)   # more weak first; for ties, lower min (i.e., -minv higher)

    def best_sort_key(g):
        """
        Higher = better:
          1) fewer weak locations (< min_coverage) -> better
          2) higher minimum coverage               -> better
        """
        v = grid_cvg[g]
        weak = -int(np.sum(v < min_coverage))  # fewer weak -> larger value
        minv = int(np.min(v))                  # higher min -> larger value
        return (weak, minv)

    def best_pair(rem):
        """Pick (a,b) maximizing group_score({a,b}) with the above priority."""
        rem = list(rem)
        best = None
        for i in range(len(rem)):
            for j in range(i+1, len(rem)):
                a, b = rem[i], rem[j]
                # Comparisons are lexicographic on (n_good, min_cvg), 
                # so “passing the threshold” dominates, and “raising the floor” is second.
                sc = group_score([a, b]) # (n_good, min_cvg)
                if best is None or sc > best[0]:
                    best = (sc, (a, b))
        # best[1] = pair of best fitting grids.
        return best[1] if best else None


     # --- 2) build groups according to n and your rules ---
    groups = []  # list of lists of grid ids
    remaining = set(unique_grids.tolist())


    # choose a single "best" to leave alone in cases 5,7,9
    def pick_best_single(rem):
        # builds scores out of neg 'passes coverage' and minimum coverage
        # takes max(weak, minv) per grid
        return max(rem, key=best_sort_key)
    
    # choose k pairs greedily from remaining
    def add_k_pairs(rem, k):
        for _ in range(k):
            if len(rem) < 2: break
            a_b = best_pair(rem) # a_b = pair of best fitting grids.
            if a_b is None: break
            a, b = a_b
            groups.append([a, b])
            # then remove the pairs that have just been added
            rem.remove(a); rem.remove(b)
    
    if len(unique_grids) == 3:
        # 3 singles
        groups = [[g] for g in unique_grids]
        remaining.clear()
    
    elif len(unique_grids) == 4:
        # pair worst 2; others singleton
        worst2 = sorted(remaining, key=worst_sort_key, reverse=True)[:2]
        groups.append(list(worst2))
        for g in worst2: remaining.remove(g)
        for g in sorted(remaining): groups.append([g])
        remaining.clear()

        
    elif len(unique_grids) in [5,7,9]:
        # best alone; pair worst 4 (2 pairs)
        best_single = pick_best_single(remaining)
        remaining.remove(best_single)
        # after removing the best grid, pair the remaining ones up as usual
        add_k_pairs(remaining, int((len(unique_grids)-1)/2))
        # in the end, add the single grid to the groups and delete all used grids
        groups.append([best_single])
        remaining.clear()
    
    elif len(unique_grids) in [6,8,10]:
        add_k_pairs(remaining, int(len(unique_grids)/2))
    
    else:
        # n > 10  →  make exactly 5 groups max
        add_k_pairs(remaining, max_groups)  # start with 5 best pairs
        # add leftovers to existing groups to maximize coverage gain
        while remaining:
            g = remaining.pop()
            best_gain, best_idx = None, None
            for idx, grp in enumerate(groups):
                # find out which group improves most if left-over is added
                base = group_score(grp)
                new  = group_score(grp + [g])
                gain = tuple(np.array(new) - np.array(base))
                if best_gain is None or gain > best_gain:
                    best_gain, best_idx = gain, idx
            groups[best_idx].append(g)

    # --- 3) map old -> new labels and scatter back to DataFrame ---
    label = {}
    for new_id, grp in enumerate(groups):
        for g in grp:
            label[int(g)] = int(new_id)

    same_grid_idx_new = np.array([label[int(g)] for g in same_grids], dtype=int)

    if reliable_FR_mask is not None:
        # put back into shape with False for grids that have insufficient firing
        out = np.full(reliable_FR_mask.shape, False, dtype=object)  
        out[reliable_FR_mask] = same_grid_idx_new
        beh[f'paired_grid_idx_{cell_name}'] = out
    else:
        beh[f'paired_grid_idx_{cell_name}'] = same_grid_idx_new
        
    return beh




# def plot_neuron_overview(neuron_name, beh, neuron, locs, FR_maps_neuron, spatial_corr_per_shift, same_grids, title_string):
#     # --- Inputs assumed already defined:
#     # neuron : 2D array (repeats x timebins)
#     # spatial_corr_per_shift : 1D array length = len(shifts)
#     # neuron_name : str
#     # shifts : e.g. np.arange(0,360,30)  # 12 values
#     # FR_maps_neuron : shape (len(shifts), 9, n_grids)  -> each map reshapes to (3,3)
#     # same_grids, locs : arrays to build grid_cvg (coverage)
#     # unique_grids : np.unique(same_grids)
    
#     # import pdb; pdb.set_trace()
#     unique_grids = np.unique(same_grids)               # flexible number of grids
#     print(f"{neuron_name} has {len(unique_grids)} grids.")
#     # ----- Build spatial coverage (grey column) -----
#     grid_cvg = {}
#     for g in unique_grids:
#         grid_cvg[g] = np.zeros(9, dtype=int)
#         glocs = locs[same_grids == g]
#         for loc in range(1, 10):
#             grid_cvg[g][loc-1] = np.count_nonzero(glocs == loc)
    
#     cov_min = min(np.min(v) for v in grid_cvg.values())
#     cov_max = max(np.max(v) for v in grid_cvg.values())
    
#     # ----- Global FR map color limits -----
#     rate_min = np.nanmin(FR_maps_neuron)
#     rate_max = np.nanmax(FR_maps_neuron)
    
#     # ----- Which grids to show as rows  -----
#     n_grids = len(unique_grids)
#     grid_order = sorted(grid_cvg.keys())[:n_grids]  # align to n grid rows
    
#     # ----- Find best shift for outline -----
#     temporal_shift = 30
#     shifts = np.arange(0, 360, temporal_shift)  # 12 shifts
#     best_idx = np.nanargmax(spatial_corr_per_shift) if np.any(np.isfinite(spatial_corr_per_shift)) else None
#     best_shift = shifts[best_idx] if best_idx is not None else None
    
#     # ===== FIGURE LAYOUT =====
#     fig = plt.figure(figsize=(22, 14))
#     gs = GridSpec(nrows=3, ncols=1, height_ratios=[2.2, 1.8, 7.0], hspace=0.35, figure=fig)
    
#     # --- Row 1: Firing rate (full width)
#     ax_fr = fig.add_subplot(gs[0, 0])
#     im_fr = ax_fr.imshow(neuron, aspect='auto', cmap='Reds')
#     ax_fr.vlines([90, 180, 270], 0, neuron.shape[0]-1, linestyles='dotted', linewidth=0.8, colors='black')
#     ax_fr.set_title('Firing rate per bin (x) and repeat (y)')
#     ax_fr.set_xlabel('Time (bins)')
#     ax_fr.set_ylabel('Repeats')
#     cbar_fr = fig.colorbar(im_fr, ax=ax_fr, fraction=0.025, pad=0.02)
#     cbar_fr.set_label("Rate (spike/s)")
    
#     # --- Row 2: Spatial encoding consistency vs temporal shift (full width)
#     ax_cons = fig.add_subplot(gs[1, 0])
#     ax_cons.plot(shifts, spatial_corr_per_shift, marker='o', color='black', label=str(neuron_name))
#     ax_cons.hlines(0, shifts[0], shifts[-1], colors='k', linestyles='-')
#     ax_cons.set_xlabel('Temporal shift (bins of 30)')
#     ax_cons.set_ylabel('Mean grid correlation')
#     ax_cons.set_title(f"Neuron {neuron_name} — Spatial encoding consistency vs. temporal shift")
#     ax_cons.legend(loc='best', fontsize=8)
#     if best_shift is not None:
#         ax_cons.axvline(best_shift, color='tab:red', linestyle='--', linewidth=1)
    
#     # --- Rows 3–5: 3 rows × (1 coverage + len(shifts)) columns
#     ncols_bottom = 1 + len(shifts)  # 1 coverage col + shifts
#     gs_bot = GridSpecFromSubplotSpec(
#         nrows=n_grids, ncols=ncols_bottom, subplot_spec=gs[2, 0], wspace=0.05, hspace=0.08
#     )
    
#     axs = np.empty((n_grids, ncols_bottom), dtype=object)
    
#     # Column headers
#     for c in range(ncols_bottom):
#         ax_title = fig.add_subplot(gs_bot[0, c])
#         fig.delaxes(ax_title)  # we'll place real axes shortly; title texts go on real axes
#     # we’ll set titles on real axes below (top row only)
    
#     # Fill axes
#     for r, g in enumerate(grid_order):
#         # Coverage column (col 0)
#         ax_cov = fig.add_subplot(gs_bot[r, 0])
#         axs[r, 0] = ax_cov
#         spatial_cov = grid_cvg[g].reshape(3, 3)
#         im_cov = ax_cov.imshow(spatial_cov, cmap='Greys', vmin=cov_min, vmax=cov_max)
#         ax_cov.set_xticks([]); ax_cov.set_yticks([])
#         if r == 0:
#             ax_cov.set_title("Coverage")
#         # annotate coverage numbers
#         for (i, j), v in np.ndenumerate(spatial_cov):
#             ax_cov.text(j, i, f"{int(v)}", ha='center', va='center', color='black', fontsize=8)
    
#         # Row label (left of coverage cell)
#         ax_cov.text(-0.45, 0.5, f"Grid combo {r+1}",
#                     transform=ax_cov.transAxes, va='center', ha='right', fontsize=10)
    
#         # Shift columns (1..len(shifts))
#         for si, sh in enumerate(shifts):
#             ax = fig.add_subplot(gs_bot[r, si+1])
#             axs[r, si+1] = ax
#             spatial = FR_maps_neuron[si][:, r].reshape(3, 3)
#             im = ax.imshow(spatial, cmap='coolwarm', vmin=rate_min, vmax=rate_max)
#             ax.set_xticks([]); ax.set_yticks([])
#             if r == 0:
#                 ax.set_title(f"{sh}°", fontsize=9)
    
#     # Outline the best-shift column across all three rows
#     if best_shift is not None and best_shift in shifts:
#         col_idx = 1 + int(np.where(shifts == best_shift)[0][0])  # +1 because col 0 = coverage
#         for r in range(n_grids):
#             axs[r, col_idx].add_patch(Rectangle((0, 0), 1, 1,
#                                transform=axs[r, col_idx].transAxes, fill=False,
#                                linewidth=2.5, edgecolor='black', zorder=10))
    
#     # Shared colorbar for rate maps (all shift columns)
#     # Use the last 'im' as mappable (safe because vmin/vmax consistent)
#     cbar_rate = fig.colorbar(im, ax=axs[:, 1:].ravel().tolist(), fraction=0.02, pad=0.01)
#     cbar_rate.set_label("Avg. rate per field")
    
#     # Optional colorbar for coverage column
#     # cbar_cov = fig.colorbar(im_cov, ax=axs[:, 0].ravel().tolist(), fraction=0.02, pad=0.04)
#     #cbar_cov.set_label("Coverage (samples)")
    
#     # Big y-label for bottom block
#     # fig.text(0.06, 0.21, "3×3 spatial maps", rotation=90, va='center', fontsize=11)
#     fig.suptitle(title_string, fontsize=18, fontweight="bold", y=0.995)

#     plt.show()
 
 

def compute_fut_spatial_tunings(sessions, trials = 'all_correct', combine_two_grids = False, sparsity_c = None, weighted = True, save_all = False):  
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
        data = mc.analyse.helpers_human_cells.filter_data(data_raw, sesh, trials)
        beh_df = data[f"sub-{sesh:02}"]['beh']
        # determine identical grids
        grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
            
        # for each cell, cross-validate the peak task-lag shift for spatial consistency.
        for neuron_idx, curr_neuron in enumerate(data[f"sub-{sesh:02}"]['normalised_neurons']):

            # if curr_neuron not in ['07-07-chan104-LOFC', '16-16-chan118-LHC', '01-01-mLF2aCa07-LACC']:
            #     continue  
            if curr_neuron not in ['11-11-mLP2Cb07-LPCC', '07-07-mLP2Cb02-LPCC', '10-10-mLP2Cb07-LPCC', '09-09-mLP2Cb05-LPCC']:
                continue
            # if curr_neuron not in ['04-04-mLT2dHa03-LHC', '04-04-mRT2cHb07-RHC','03-03-mLAHIP1-LHC']:
            #     continue
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
            fr_by_shift, corr_per_shift, dwell = comp_peak_spatial_tuning(neuron_data, locations, behaviour, curr_neuron, idx_same_grids, weighted = weighted)


            file_name = f"spatial_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons_weighted.csv"
            df = pd.read_csv(f"{group_dir_fut_spat}/{file_name}")
            cross_val = df['avg_consistency_at_peak'][df['neuron_id'] == curr_neuron][df['session_id']== sesh].to_list()

            title = f"overview for {curr_neuron} \n including {trials} repeats. \n cross-val value: {round(cross_val[0], 3)}"
            import pdb; pdb.set_trace()
            plot_neuron_overview(curr_neuron, beh_df, data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron], data[f"sub-{sesh:02}"]['locations'], fr_by_shift, corr_per_shift, idx_same_grids, title)
            
            
    
def loop_through_trial_combos(trial_list):
    for incl_trials in trial_list:
        compute_fut_spatial_tunings(sessions=[44], trials = incl_trials, combine_two_grids = True, sparsity_c = 'gridwise_qc', weighted = True, save_all=False)


if __name__ == "__main__":
    # trials can be 'all', 'all_correct', 'early', 'late'
    # loop through these different trials to compare a single cell.
    loop_through_trial_combos(trial_list = ['all_minus_explore'])
    #loop_through_trial_combos(trial_list = ['all', 'all_correct', 'early', 'late'])
    

# 44 11-11-mLP2Cb07-LPCC
# 44 07-07-mLP2Cb02-LPCC
# 44 10-10-mLP2Cb07-LPCC (but state)
# 43 09-09-mLP2Cb05-LPCC
    