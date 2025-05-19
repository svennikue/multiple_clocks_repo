#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:41:15 2025

@author: Svenja Kuchenhoff

This script loads the original correlation values per cell and the permutation
and tests where in the distribution they lie.


"""


import mc
import fire
import os
import pickle
import matplotlib.pyplot as plt


def load_result_dirs(file_name, perm_string):
    result_dir = {"binned": {}, "raw":{}, "perm":{}}
    results_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/elastic_net_reg/corrs"
    
    # subjects = [f"sub-{i}" for i in range(1, 65)]
    subjects = [f"sub-{i}" for i in range(1, 7)]
    
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

            with open(f"{path_to_subfile}_fit_binned_by_state", 'rb') as f:
                sub_dir = pickle.load(f)
                result_dir['binned'][sub] = {}
                for neuron, model in sub_dir.items():
                    for model, inner in model.items():
                        if model not in result_dir['binned'][sub]:
                            result_dir['binned'][sub][model] = {}
                        for inner_neuron, values in inner.items():
                            result_dir['binned'][sub][model][inner_neuron] = values
        
            with open(f"{results_folder}/{perm_string}_{sub}_corrs_{file_name}", 'rb') as f:
                sub_dir = pickle.load(f)
                result_dir['perm'][sub] = {}
                for neuron, model in sub_dir.items():
                    for model, inner in model.items():
                        if model not in result_dir['perm'][sub]:
                            result_dir['perm'][sub][model] = {}
                        for inner_neuron, values in inner.items():
                            result_dir['perm'][sub][model][inner_neuron] = values
                            
                            
    
    return result_dir, actual_subjects




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





def plot_all(model_name_string, perms, define_somehow_what_to_plot=None):
    results, subjects = load_result_dirs(model_name_string, perms)
    
    # results_corr_by_roi = mc.analyse.plotting_cells.prep_result_for_plotting_by_rois(results['raw']) 
    # results_corr_by_roi_binned = mc.analyse.plotting_cells.prep_result_for_plotting_by_rois(results['binned'])
    
    # title_addition = "roi neurons binned after fit"
    # mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi_binned,title_addition)
    
    # title_addition = "raw correlation, only pos fit"
    # mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi,title_addition)
    results_corr_by_roi_df = mc.analyse.plotting_cells.prep_result_df_perms_for_plotting_by_rois(results['raw'], results['perm'])
    
    mc.plotting.results.plot_perms_per_cell_and_roi(results_corr_by_roi_df)
    
    
    import pdb; pdb.set_trace()
    
    
    
    
    title_addition = "raw correlation, only pos fit"
    mc.analyse.plotting_cells.plotting_df_based_corr_perm_histogram_by_ROIs(results_corr_by_roi_df,title_addition)
    
    results_corr_by_roi_binned_df = mc.analyse.plotting_cells.prep_result_df_for_plotting_by_rois(results['binned'])
    title_addition_binned = "state-binned correlation, only pos fit"
    mc.analyse.plotting_cells.plotting_df_based_corr_perm_histogram_by_ROIs(results_corr_by_roi_binned_df,title_addition_binned)
    
    
    
    top_ten_cells_binned, predicted_cells_binned = mc.analyse.helpers_human_cells.identify_max_cells_for_model(results['binned'])
    top_ten_cells_raw, predicted_cells_raw = mc.analyse.helpers_human_cells.identify_max_cells_for_model(results['raw'])
    
    
    
    og_data = load_data(subjects)
    
    
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
        model_name_string='w_partial_musicboxes_excl_rep1-1_avg_in_20_bins_across_runs',
        perms = '260perms_configs_shuffle'
        # sub-1_corrs_w_partial_musicboxes_excl_rep1-2_avg_in_20_bins_across_runs_fit_binned_by_state
    )
    
# w_partial_musicboxes_excl_rep1-3_excl_rep1-3_pre_corr_binned-None_only_pos

