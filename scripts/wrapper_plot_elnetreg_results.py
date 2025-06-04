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


def load_result_dirs(file_name_list):
    several_results = {}
    # import pdb; pdb.set_trace()
    for file_name in file_name_list:
        result_dir = {"binned": {}, "raw":{}}
        results_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/elastic_net_reg/corrs"
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





def plot_all(model_name_string_list, define_somehow_what_to_plot=None):
    results, subjects = load_result_dirs(model_name_string_list)
    
    # results_corr_by_roi = mc.analyse.plotting_cells.prep_result_for_plotting_by_rois(results['raw']) 
    # results_corr_by_roi_binned = mc.analyse.plotting_cells.prep_result_for_plotting_by_rois(results['binned'])
    
    # title_addition = "roi neurons binned after fit"
    # mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi_binned,title_addition)
    
    # title_addition = "raw correlation, only pos fit"
    # mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi,title_addition)
    
    all_results, all_results_binned, cleaned_results, cleaned_results_binned, cell_overview, cell_overview_binned = {}, {}, {}, {}, {}, {}
    cleaned_results_only_state, cleaned_results_binned_only_state, cell_overview_only_state, cell_overview_binned_only_state = {}, {}, {}, {}
    
    for results_for_version in results:
        # prepare data to plot.
        all_results[results_for_version] = mc.analyse.plotting_cells.prep_result_df_for_plotting_by_rois(results[results_for_version]['raw'])
        all_results_binned[results_for_version] = mc.analyse.plotting_cells.prep_result_df_for_plotting_by_rois(results[results_for_version]['binned'])
        
        #
        x_percent_to_remove_state = 35
        x_percent_to_remove_phase = 20
        #mc.plotting.results.plot_overlap_in_cells(results_corr_by_roi_df, results_corr_by_roi_binned_df, x_percent_to_remove)
        
        # here you could consider doing 2 overlapping distributions to compare early-late better
        # also add where the lowest state-cell that was removed sat with a line.
        
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
    mc.plotting.results.slope_plot_early_late_per_roi(df_early= all_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = all_results['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"overlap early - late cells ")

    # show overlap between early and late cells for top state and phase
    mc.plotting.results.slope_plot_early_late_per_roi(df_early= all_results_binned['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = all_results_binned['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"overlap early - late cells \n binned by state")
        
    
    
    # show overlap between early and late cells, state + phase removed
    mc.plotting.results.slope_plot_early_late_per_roi(df_early= cleaned_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = cleaned_results['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"overlap early - late cells \n excl. {x_percent_to_remove_state}% best state + phase ")

    # only state removed
    mc.plotting.results.slope_plot_early_late_per_roi(df_early= cleaned_results_only_state['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
                                                                    df_late = cleaned_results_only_state['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
                                                                    title_string_add = f"overlap early - late cells \n excl. {x_percent_to_remove_state}% best state")


    
    # mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= cleaned_results['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                                 df_late = cleaned_results['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
    #                                                                 title_string_add = f"excl. {x_percent_to_remove_state}% best state + {x_percent_to_remove_phase}% phase")
    
    
    # mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= cleaned_results_binned_only_state['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                                 df_late = cleaned_results_binned_only_state['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
    #                                                                 title_string_add = f"excl. {x_percent_to_remove_state}% best state \n binned")
    
    
    
    # mc.plotting.results.plotting_two_df_corr_perm_histogram_by_ROIs(df_early= cleaned_results_binned['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'],
    #                                                                 df_late = cleaned_results_binned['w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'], 
    #                                                                 title_string_add = f"excl. {x_percent_to_remove_state}% best state + {x_percent_to_remove_phase}% phase \n binned")
    




    import pdb; pdb.set_trace()
        

    
    
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
    
    
    
    top_ten_cells_binned, predicted_cells_binned = mc.analyse.helpers_human_cells.identify_max_cells_for_model(results['binned'])
    top_ten_cells_raw, predicted_cells_raw = mc.analyse.helpers_human_cells.identify_max_cells_for_model(results['raw'])
    
    
    
    top_twentyfive_cells_raw, predicted_cells_raw = mc.analyse.helpers_human_cells.identify_max_cells_for_model(results['raw'], x_percentage_of_cells=25)
    
    

    
    
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
        model_name_string_list=['w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs',
                           'w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs']
        # model_name_string='w_partial_musicboxes_only_reps_0-1_avg_in_20_bins_across_runs'
        # model_name_string='w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs'
        # model_name_string='w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs'
        # model_name_string='w_partial_musicboxes_excl_rep1-1_avg_in_20_bins_across_runs'
        # sub-59_corrs_w_partial_musicboxes_only_reps_5-max_avg_in_20_bins_across_runs_fit_binned_by_state
        # sub-1_corrs_w_partial_musicboxes_excl_rep1-2_avg_in_20_bins_across_runs_fit_binned_by_state
    )
    
# w_partial_musicboxes_excl_rep1-3_excl_rep1-3_pre_corr_binned-None_only_pos

