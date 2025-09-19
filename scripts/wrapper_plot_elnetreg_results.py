#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:45:59 2025

Wrapper to plot human cell regression results

@author: xpsy1114
"""


import mc
import fire
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D



def load_result_dirs(file_name_list):
    several_results = {}
    # import pdb; pdb.set_trace()
    for file_name in file_name_list:
        result_dir = {"binned": {}, "raw":{}}
        results_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/corrs"
        
        # utah_and_UCLA_cells = [1,2,4,6,17,23,24,29,30,39,41,42,47,48, 52, 53, 54, 55, 3, 40, 50, 51, 56]
        # subjects = [f"sub-{i}" for i in utah_and_UCLA_cells]
        
        #only_baylor_cells = [5,7,8,9,10,11,12,13,14,15,16,18, 19,25,26,27,28,31,32,33,34,35, 36,37,38,43,44,45,46,49, 57,58,59]
        #subjects = [f"sub-{i}" for i in only_baylor_cells]
        
        subjects = [f"sub-{i}" for i in range(1, 65)]
        
        
        actual_subjects = []
        # check if on server or local
        if not os.path.isdir(results_folder):
            print("running on ceph")
            results_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group/elastic_net_reg/corrs"

        # loop through all subjects
        # first find how many subjects there are
        for sub in subjects:
            path_to_subfile = f"{results_folder}/{sub}_corrs_{file_name}"
            if os.path.isfile(path_to_subfile):
                actual_subjects.append(sub)
    
                with open(path_to_subfile, 'rb') as f:
                    sub_dir = pickle.load(f)
                    result_dir['raw'][sub] = {}
                    for neuron, model in sub_dir.items():
                        for model, inner in model.items():
                            if model not in result_dir['raw'][sub]:
                                result_dir['raw'][sub][model] = {}
                            for inner_neuron, values in inner.items():
                                result_dir['raw'][sub][model][inner_neuron] = values
                if os.path.isdir(f"{path_to_subfile}_fit_binned_by_state"):
                    with open(f"{path_to_subfile}_fit_binned_by_state", 'rb') as f:
                        sub_dir = pickle.load(f)
                        result_dir['binned'][sub] = {}
                        for neuron, model in sub_dir.items():
                            for model, inner in model.items():
                                if model not in result_dir['binned'][sub]:
                                    result_dir['binned'][sub][model] = {}
                                for inner_neuron, values in inner.items():
                                    result_dir['binned'][sub][model][inner_neuron] = values
        several_results[file_name] = result_dir
        #import pdb; pdb.set_trace()
    return several_results, actual_subjects




def load_data(subs):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    # check if on server or local
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    group_dir = f"{data_folder}/group/elastic_net_reg"
    
    if not os.path.isdir(group_dir):
        os.mkdir(group_dir)
    
    # first bring the subject list in the right format
    formatted_subjects = [f"{int(s.split('-')[1]):02}" for s in subs]
    data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, formatted_subjects)

    return data


def hist_early_late_per_roi(df_early, df_late, title_note="(per ROI)", bins=20, models=None, rois=None):
    early_color = "#00BFC4"  # turquoise-blue
    late_color  = "#E07B39"  # terracotta-orange

    if models is None:
        models = sorted(np.union1d(df_early["model"].unique(), df_late["model"].unique()))
    if rois is None:
        rois = sorted(np.union1d(df_early["roi"].unique(), df_late["roi"].unique()))

    # build the printed summary once
    summary = summarize_means_p(df_early, df_late)

    for model in models:
        e_m = df_early[df_early["model"] == model]
        l_m = df_late [df_late ["model"] == model]
        if e_m.empty and l_m.empty:
            continue

        n_roi = len(rois)
        fig, axes = plt.subplots(
            1, n_roi, figsize=(max(5.5, 4.8*n_roi), 4.8),
            sharey=True, constrained_layout=True
        )
        if n_roi == 1:
            axes = [axes]

        for ax, roi in zip(axes, rois):
            e_vals = e_m.loc[e_m["roi"] == roi, "avg_corr_val"].to_numpy(float)
            l_vals = l_m.loc[l_m["roi"] == roi, "avg_corr_val"].to_numpy(float)

            has_e = e_vals.size > 0
            has_l = l_vals.size > 0

            # consistent bin edges if both present
            if has_e and has_l:
                finite = np.concatenate([e_vals[np.isfinite(e_vals)], l_vals[np.isfinite(l_vals)]])
                if finite.size == 0:
                    edges = bins
                else:
                    vmin, vmax = np.nanmin(finite), np.nanmax(finite)
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                        vmin, vmax = -1.0, 1.0
                    edges = np.linspace(vmin, vmax, bins+1)
            else:
                edges = bins

            if has_e:
                ax.hist(e_vals, bins=edges, alpha=0.55, label="Early", edgecolor="none", color=early_color)
            if has_l:
                ax.hist(l_vals, bins=edges, alpha=0.55, label="Late",  edgecolor="none", color=late_color)

            # compute means & p (one-sided > 0)
            mean_e = float(np.nanmean(e_vals)) if has_e else np.nan
            mean_l = float(np.nanmean(l_vals)) if has_l else np.nan
            p_e    = one_sided_pos_p(e_vals) if has_e else np.nan
            p_l    = one_sided_pos_p(l_vals) if has_l else np.nan

            # draw vertical mean lines (no dots)
            ymin, ymax = ax.get_ylim()
            if has_e and np.isfinite(mean_e):
                ax.axvline(mean_e, ymin=0, ymax=1, color=early_color, lw=2, label=None)
                ax.text(mean_e, ymax*0.98, star(p_e), ha="center", va="top", fontsize=11, color=early_color)
            if has_l and np.isfinite(mean_l):
                ax.axvline(mean_l, ymin=0, ymax=1, color=late_color, lw=2, label=None)
                ax.text(mean_l, ymax*0.90, star(p_l), ha="center", va="top", fontsize=11, color=late_color)

            # small text box with mean & p (optional; remove if cluttered)
            lines = []
            if has_e: lines.append(f"E μ={mean_e:.3f}, p={p_e:.2g}, {star(p_e)}")
            if has_l: lines.append(f"L μ={mean_l:.3f}, p={p_l:.2g}, {star(p_l)}")
            if lines:
                ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                        ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

            ax.set_title(roi if (has_e or has_l) else f"{roi} (no data)")
            ax.set_xlabel("avg_corr_val")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.axvline(0, color="k", lw=1)
            if ax is axes[0]:
                ax.set_ylabel("Count")

        # legend with line proxies for means
        handles = [Line2D([0],[0], color=early_color, lw=2, label='Early mean'),
                   Line2D([0],[0], color=late_color,  lw=2, label='Late mean')]
        # add histogram patches only if present
        if not e_m.empty or not l_m.empty:
            axes[0].legend(handles=handles, frameon=False, loc="upper right")

        fig.suptitle(f"{model} — Early & Late: mean>0 tests {title_note}", fontsize=12)
        plt.show()

        # --- print the summary for this model ---
        to_print = (summary[summary["model"] == model]
                    .sort_values(["roi","cond"])
                    .reset_index(drop=True))
        print(f"\n=== Summary: {model} ===")
        # keep columns concise
        print(to_print[["roi","cond","n","mean","p","stars"]].to_string(index=False))


# --- helpers ---
def one_sided_pos_p(arr):
    """One-sample (mean>0) p-value; returns np.nan if fewer than 2 finite points."""
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    t, p_two = ttest_1samp(arr, 0.0, nan_policy="omit")
    return (p_two / 2.0) if t > 0 else 1.0

def star(p):
    if not np.isfinite(p): return "n/a"
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'

def summarize_means_p(df_early, df_late):
    """Build a tidy table of mean/p for Early and Late by model & ROI."""
    rows = []
    models = sorted(np.union1d(df_early["model"].unique(), df_late["model"].unique()))
    rois   = sorted(np.union1d(df_early["roi"].unique(),   df_late["roi"].unique()))

    for model in models:
        for roi in rois:
            e_vals = df_early.loc[(df_early["model"]==model) & (df_early["roi"]==roi), "avg_corr_val"].to_numpy(float)
            l_vals = df_late .loc[(df_late ["model"]==model) & (df_late ["roi"]==roi), "avg_corr_val"].to_numpy(float)

            if e_vals.size:
                mean_e = float(np.nanmean(e_vals))
                p_e    = one_sided_pos_p(e_vals)
                rows.append({"model":model, "roi":roi, "cond":"Early",
                             "n":int(np.isfinite(e_vals).sum()), "mean":mean_e, "p":p_e, "stars":star(p_e)})
            if l_vals.size:
                mean_l = float(np.nanmean(l_vals))
                p_l    = one_sided_pos_p(l_vals)
                rows.append({"model":model, "roi":roi, "cond":"Late",
                             "n":int(np.isfinite(l_vals).sum()), "mean":mean_l, "p":p_l, "stars":star(p_l)})
    return pd.DataFrame(rows, columns=["model","roi","cond","n","mean","p","stars"])


       


def plot_all(model_name_string_list, define_somehow_what_to_plot=None, remove_state=False):
    results, subjects = load_result_dirs(model_name_string_list)
    
    #result_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/elastic_net_reg/corrs/sig_cells_per_model_ceph_12-06-2025/"
    result_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/corrs/"
    
    # check if on server or local
    if not os.path.isdir(result_folder):
        print("running on ceph")
        result_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group/elastic_net_reg/"

    # import pdb; pdb.set_trace()
    
    
    # drop in 10th of sept 2025 for new 360 binned version.
    rows = []
    for res_ver, res_dict in results.items():
        for subj, subj_dict in res_dict["raw"].items():
            for model, model_dict in subj_dict.items():
                for cell, values in model_dict.items():
                    avg = np.nanmean(values)  # average per cell
                    rows.append((res_ver, subj, model, cell, avg))
    
    df = pd.DataFrame(rows, columns=["result_version", "subject", "model", "cell", "avg_corr_val"])
    # 1) ROI from the cell label
    # --- 1) Add ROI column with suffix-matching ---
    TARGET_ROIS = ["ACC","PCC","OFC","EC","HC","AMY"]
    
    def infer_roi(cell_label: str) -> str:
        sfx = str(cell_label).split("-")[-1].upper()
        for tag in TARGET_ROIS:
            if tag in sfx:
                return tag
        return "mixed"
    
    df = df.copy()
    df["roi"] = df["cell"].apply(infer_roi)
    
    # 2) split early/late
    EARLY, LATE = "360bins_early", "360bins_late"
    df_early = df[df["result_version"] == EARLY]
    df_late  = df[df["result_version"] == LATE]

    # run it
    hist_early_late_per_roi(df_early, df_late, title_note="(per ROI)", bins=20)

    import pdb; pdb.set_trace()
    
    #
    ##
    #
    
    
    # results_corr_by_roi = mc.analyse.plotting_cells.prep_result_for_plotting_by_rois(results['raw']) 
    # results_corr_by_roi_binned = mc.analyse.plotting_cells.prep_result_for_plotting_by_rois(results['binned'])
    
    # title_addition = "roi neurons binned after fit"
    # mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi_binned,title_addition)
    
    # title_addition = "raw correlation, only pos fit"
    # mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi,title_addition)
    
    all_results, all_results_binned, cleaned_results, cleaned_results_binned, cell_overview, cell_overview_binned = {}, {}, {}, {}, {}, {}
    cleaned_results_only_state, cleaned_results_binned_only_state, cell_overview_only_state, cell_overview_binned_only_state = {}, {}, {}, {}
    
    results_rem_sig_state, results_rem_sig_state_and_phase, cell_overview_rem_sig_state, cell_overview_rem_sig_state_and_phase = {}, {}, {}, {}
    
    all_results_big_rois, results_rem_sig_state_big_rois = {}, {}
    
    for results_for_version in results:
        # prepare data to plot.
        all_results[results_for_version] = mc.analyse.plotting_cells.prep_result_df_for_plotting_by_rois(results[results_for_version]['raw'])
        all_results_binned[results_for_version] = mc.analyse.plotting_cells.prep_result_df_for_plotting_by_rois(results[results_for_version]['binned'])
        
        
        # prepare data to plot in ROIs, lumping PFC together.
        all_results_big_rois[results_for_version] = mc.analyse.plotting_cells.prep_result_df_for_plotting_by_small_rois(results[results_for_version]['raw'])
        
        
        
        #
        x_percent_to_remove_state = 35
        x_percent_to_remove_phase = 20
        #mc.plotting.results.plot_overlap_in_cells(results_corr_by_roi_df, results_corr_by_roi_binned_df, x_percent_to_remove)
        
        # here you could consider doing 2 overlapping distributions to compare early-late better
        # also add where the lowest state-cell that was removed sat with a line.
        
        
        
        #
        # instead of removing the top performing cells, remove the significant state
        # and phase cells.
        # I currently can only do this for not-binned.
        # /Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/elastic_net_reg/curr_rings_split_clock_model_w_partial_musicboxes_only_reps_6-10_avg_in_20_bins_across_runs_sig_after_temp_perms.csv
        
        path_to_file_state = f"{result_folder}stat_model_{results_for_version}_sig_after_temp_perms.csv"
        
        
        if remove_state == True:
            results_rem_sig_state[results_for_version], cell_overview_rem_sig_state[results_for_version] = mc.analyse.helpers_human_cells.remove_certain_cells_for_x_model(all_results[results_for_version], path_to_file_state)
            results_rem_sig_state_big_rois[results_for_version], _ = mc.analyse.helpers_human_cells.remove_certain_cells_for_x_model(all_results_big_rois[results_for_version], path_to_file_state)
            
            
            
            
            path_to_file_phase = f"{result_folder}phas_model_{results_for_version}_sig_after_temp_perms.csv"
            results_rem_sig_state_and_phase[results_for_version], cell_overview_rem_sig_state_and_phase[results_for_version] = mc.analyse.helpers_human_cells.remove_certain_cells_for_x_model(results_rem_sig_state[results_for_version], path_to_file_phase)
            
            # next step: remove the top x_percent of state cells
           
            cleaned_results_only_state[results_for_version], cell_overview_only_state[results_for_version] = mc.analyse.helpers_human_cells.remove_top_x_percent_of_x_model(all_results[results_for_version], delete_from_model= 'stat_model', x_percent=x_percent_to_remove_state)
            # remove top x_percent % of state cells inn binned data.
            cleaned_results_binned_only_state[results_for_version], cell_overview_binned_only_state[results_for_version] = mc.analyse.helpers_human_cells.remove_top_x_percent_of_x_model(all_results_binned[results_for_version], delete_from_model= 'stat_model', x_percent=x_percent_to_remove_state)
            
        
            cleaned_results[results_for_version], cell_overview[results_for_version] = mc.analyse.helpers_human_cells.remove_top_x_percent_of_x_model(cleaned_results_only_state[results_for_version], delete_from_model= 'phas_model', x_percent=x_percent_to_remove_phase)
            # remove top x_percent % of state cells inn binned data.
            cleaned_results_binned[results_for_version], cell_overview_binned[results_for_version] = mc.analyse.helpers_human_cells.remove_top_x_percent_of_x_model(cleaned_results_binned_only_state[results_for_version], delete_from_model= 'phas_model', x_percent=x_percent_to_remove_phase)
            
        
        
        # title_addition = f"raw correlation, excl.{x_percent_to_remove}% best state"
        # mc.analyse.plotting_cells.plotting_df_based_corr_perm_histogram_by_ROIs(cleaned_results[results_for_version] ,title_addition)
        
        # title_addition = f"raw binned, excl.{x_percent_to_remove}% best state"
        # mc.analyse.plotting_cells.plotting_df_based_corr_perm_histogram_by_ROIs(cleaned_results_binned[results_for_version] ,title_addition)
        
        
    # mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= cleaned_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                                 df_late = cleaned_results['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
    #                                                                 title_string_add = f"excl. {x_percent_to_remove_state}% best state + phase ")
    
    
    # show overlap between early and late cells for top state and phase
    # mc.plotting.results.slope_plot_early_late_per_roi(df_early= all_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                                 df_late = all_results['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
    #                                                                 title_string_add = f"overlap early - late cells ")

    # # show overlap between early and late cells for top state and phase
    # mc.plotting.results.slope_plot_early_late_per_roi(df_early= all_results_binned['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                                 df_late = all_results_binned['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
    #                                                                 title_string_add = f"overlap early - late cells \n binned by state")
        
    
    
    # # show overlap between early and late cells, state + phase removed
    # mc.plotting.results.slope_plot_early_late_per_roi(df_early= cleaned_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                                 df_late = cleaned_results['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
    #                                                                 title_string_add = f"overlap early - late cells \n excl. {x_percent_to_remove_state}% best state + phase ")

    # only state removed
    # mc.plotting.results.slope_plot_early_late_per_roi(df_early= cleaned_results_only_state['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                                 df_late = cleaned_results_only_state['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
    #                                                                 title_string_add = f"overlap early - late cells \n excl. {x_percent_to_remove_state}% best state")

    
    mc.plotting.results.slope_plot_early_late_per_roi(df_early= all_results[results_for_version],
                                                                    df_late = all_results[results_for_version], 
                                                                    title_string_add = f"overlap early - late cells")
    
    
    
    # only state removed
    mc.plotting.results.slope_plot_early_late_per_roi(df_early= results_rem_sig_state['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = results_rem_sig_state['w_partial_musicboxes_only_reps_6-10_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"overlap early - late cells \n excl. sig. state cells")
    # show overlap between early and late cells, state + phase removed
    mc.plotting.results.slope_plot_early_late_per_roi(df_early= results_rem_sig_state_and_phase['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = results_rem_sig_state_and_phase['w_partial_musicboxes_only_reps_6-10_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"excl. sig. state and phase cells")

    mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= results_rem_sig_state['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                df_late = results_rem_sig_state['w_partial_musicboxes_only_reps_6-10_avg_in_20_bins_across_runs'], 
                                                                title_string_add = f"excl. sig. state cells")
    
    mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= results_rem_sig_state_big_rois['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                df_late = results_rem_sig_state_big_rois['w_partial_musicboxes_only_reps_6-10_avg_in_20_bins_across_runs'], 
                                                                title_string_add = f"excl. sig. state cells")
    
    # mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= results_rem_sig_state_and_phase['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                             df_late = results_rem_sig_state_and_phase['w_partial_musicboxes_only_reps_6-10_avg_in_20_bins_across_runs'], 
    #                                                             title_string_add = f"excl. sig. state and phase cells")
    
    
    
    mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= all_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = all_results['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"all cells")

    
    mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= cleaned_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = cleaned_results['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"excl. {x_percent_to_remove_state}% best state + {x_percent_to_remove_phase}% phase")
    
    
    
    mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= cleaned_results_only_state['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = cleaned_results_only_state['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"excl. {x_percent_to_remove_state}% best state")
    
    
    
    
    mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= cleaned_results_binned_only_state['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = cleaned_results_binned_only_state['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"excl. {x_percent_to_remove_state}% best state \n binned")
    
    
    
    mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= cleaned_results_binned['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = cleaned_results_binned['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"excl. {x_percent_to_remove_state}% best state + {x_percent_to_remove_phase}% phase \n binned")
    




    
        

    
    
    removed_state_cells = []
    for c in cell_overview['all_cells']:
        if c not in cell_overview['cells_to_keep']:
            removed_state_cells.append(c)
                     
    removed_state_cells_binned = []
    for c in cell_overview_binned['all_cells']:
        if c not in cell_overview_binned['cells_to_keep']:
            removed_state_cells_binned.append(c)
    
    
    
    title_addition = "raw correlation, only pos fit"
    mc.analyse.plotting_cells.plotting_df_based_corr_perm_histogram_by_ROIs(results_corr_by_roi_df,title_addition)
    
    
    title_addition_binned = "state-binned correlation, only pos fit"
    mc.analyse.plotting_cells.plotting_df_based_corr_perm_histogram_by_ROIs(results_corr_by_roi_binned_df,title_addition_binned)
    
    
    
    top_ten_cells_binned, predicted_cells_binned = mc.analyse.helpers_human_cells.identify_max_cells_for_model(cleaned_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs']['binned'])
    top_ten_cells_raw, predicted_cells_raw = mc.analyse.helpers_human_cells.identify_max_cells_for_model(results['raw'])
    
    
    
    top_twentyfive_cells_raw, predicted_cells_raw = mc.analyse.helpers_human_cells.identify_max_cells_for_model(results['raw'], x_percentage_of_cells=25)
    
    

    
    
    og_data = load_data(subjects)
    mc.analyse.helpers_human_cells.count_cells_per_roi(og_data)
    
    
    # for this I need the raw data. 
    #             for task in all_data[subject]['neurons']:
                # cells_to_store.append(task[cell_idx])
                # subset_dict = {}
                # subset_dict[cell_label] = cells_to_store.copy()
                # subset_dict['reward_configs'] = all_data[subject]['reward_configs'].copy()
                # subset_dict['locations'] = all_data[subject]['locations'].copy()
                # subset_dict['timings'] = all_data[subject]['timings'].copy()
    mc.analyse.helpers_human_cells.store_best_cells(top_ten_cells_binned, og_data, name_extension_string='binned')
    mc.analyse.helpers_human_cells.store_best_cells(top_ten_cells_raw, og_data, name_extension_string='raw')
    
    
    
    # to get an overview of how many cells there are.
    
    
    
    
    # import numpy as np
    # import matplotlib.pyplot as plt
    # for sub in all_data:
    #     for task in all_data[sub]['neurons']:
    #         for i, neuron in enumerate(task):
    #             curr_cell = f"{sub}_{all_data[sub]['cell_labels'][i]}_{i}"
    #             if curr_cell not in neuron_dict:
    #                 neuron_dict[curr_cell] = []   
                    
    #             avg_rate_hz = np.sum(neuron) / (len(neuron) * 0.025)
    #             neuron_dict[curr_cell].append(avg_rate_hz)
                
    # # Compute the average for each dictionary entry
    # averages = [np.mean(values) for values in neuron_dict.values()]

    # # Plot the resulting 600 data points as a histogram
    # plt.figure(figsize=(10, 6))
    # plt.hist(averages, bins=300, edgecolor='black')
    # plt.xlabel('Average Firing rate')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Average Values')
    # plt.show()


    
    
    
 
# if running from command line, use this one!   
# if __name__ == "__main__":
#     fire.Fire(plot_all)
#     # call this script like
#     # python wrapper_plot_elnetreg_results.py --model_name_string='w_partial_musicboxes_excl_rep1-2', --models_I_want='['withoutnow', 'only2and3future','onlynowandnext']' --exclude_x_repeats='[1,2]' --randomised_reward_locations=False --save_regs=True


if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    plot_all(
        model_name_string_list=['360bins_early','360bins_late', '360bins_all_minus_explore'], remove_state=False
        # model_name_string_list=['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs',
        #                    'w_partial_musicboxes_only_reps_6-10_avg_in_20_bins_across_runs']
        # model_name_string='w_partial_musicboxes_only_reps_0-1_avg_in_20_bins_across_runs'
        # model_name_string='w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'
        # model_name_string='w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'
        # model_name_string='w_partial_musicboxes_excl_rep1-1_avg_in_20_bins_across_runs'
        # sub-59_corrs_w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs_fit_binned_by_state
        # sub-1_corrs_w_partial_musicboxes_excl_rep1-2_avg_in_20_bins_across_runs_fit_binned_by_state
    )
    
# w_partial_musicboxes_excl_rep1-3_excl_rep1-3_pre_corr_binned-None_only_pos

