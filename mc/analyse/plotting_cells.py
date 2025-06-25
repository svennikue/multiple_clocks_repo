#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:41:55 2025

@author: Svenja Küchenhoff
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import seaborn as sns



def prep_result_df_for_plotting_by_small_rois(results):
    # Define your ROI labels
    ROI_labels = ['PFC','hippocampal', 'PFC', 'entorhinal', 'amyg and thal']
    
    # new strategy: use pandas dataframe.
    df = pd.DataFrame()
    i = 0
    for sub in results:
        for model in results[sub]:
            for cell_label in results[sub][model]:
                if 'ACC' in cell_label:
                    roi = 'PFC'
                elif 'vCC' in cell_label:
                    roi = 'PFC'
                elif 'AMC' in cell_label:
                    roi = 'PFC'
                elif 'PCC' in cell_label:
                    roi = 'PFC'
                elif 'OFC' in cell_label:
                    roi = 'PFC'
                elif 'MCC' in cell_label:
                    roi = 'hippocampal'
                elif 'HC' in cell_label:
                    roi = 'hippocampal'
                elif 'EC' in cell_label:
                    roi = 'entorhinal'
                elif 'AMYG' in cell_label:
                    roi = 'amyg and thal'
                else:
                    roi = 'amyg and thal'
                df.at[i, 'roi'] = roi
                df.at[i, 'cell'] = cell_label
                df.at[i, 'average_corr'] = np.mean(results[sub][model][cell_label])
                df.at[i, 'model'] = model
                i = i + 1
                           
    return df

    

def prep_result_df_for_plotting_by_rois(results):
    # Define your ROI labels
    ROI_labels = ['ACC', 'OFC', 'PCC','hippocampal', 'PFC', 'entorhinal', 'amygdala', 'mixed']
    
    # new strategy: use pandas dataframe.
    df = pd.DataFrame()
    i = 0
    for sub in results:
        for model in results[sub]:
            for cell_label in results[sub][model]:
                if 'ACC' in cell_label:
                    roi = 'ACC'
                elif 'vCC' in cell_label:
                    roi = 'ACC'
                elif 'AMC' in cell_label:
                    roi = 'ACC'
                elif 'PCC' in cell_label:
                    roi = 'PCC'
                elif 'OFC' in cell_label:
                    roi = 'OFC'
                elif 'MCC' in cell_label:
                    roi = 'hippocampal'
                elif 'HC' in cell_label:
                    roi = 'hippocampal'
                elif 'EC' in cell_label:
                    roi = 'entorhinal'
                elif 'AMYG' in cell_label:
                    roi = 'amygdala'
                else:
                    roi = 'mixed'
                df.at[i, 'roi'] = roi
                df.at[i, 'cell'] = cell_label
                df.at[i, 'average_corr'] = np.mean(results[sub][model][cell_label])
                df.at[i, 'model'] = model
                i = i + 1
                           
    return df


def prep_result_df_perms_for_plotting_by_rois(results, time_perm_results=None, task_perm_results=None):
    
    rows = []
    for sub in results:
        for model in results[sub]:
            for cell_label in results[sub][model]:
                row = {}
                if 'ACC' in cell_label:
                    roi = 'ACC'
                elif 'PCC' in cell_label:
                    roi = 'PCC'
                elif 'OFC' in cell_label:
                    roi = 'OFC'
                elif 'HC' in cell_label:
                    roi = 'hippocampal'
                elif 'EC' in cell_label:
                    roi = 'entorhinal'
                elif 'AMYG' in cell_label:
                    roi = 'amygdala'
                else:
                    roi = 'mixed'
                row['roi'] = roi
                row['cell'] = cell_label
                row['average_corr'] = np.mean(results[sub][model][cell_label])
                row['model'] = model
            
                # Add task_perm columns
                if task_perm_results:
                    matrix = task_perm_results[sub][model][cell_label]
                    for p_idx in range(matrix.shape[1]):
                        row[f"task_perm_{p_idx}"] = np.mean(matrix[:, p_idx])
                
                # Add time_perm columns
                if time_perm_results:
                    matrix = time_perm_results[sub][model][cell_label]
                    for p_idx in range(matrix.shape[1]):
                        row[f"time_perm_{p_idx}"] = np.mean(matrix[:, p_idx])
    
                # Append this row to the list
                rows.append(row)


    df = pd.DataFrame(rows)
                # if task_perm_results:
                #     for p_idx in range(0, task_perm_results[sub][model][cell_label].shape[1]):
                #         df.at[i, f"task_perm_{p_idx}"] = np.mean(task_perm_results[sub][model][cell_label][:,p_idx])
                # if time_perm_results:
                #     for p_idx in range(0, time_perm_results[sub][model][cell_label].shape[1]):
                #         df.at[i, f"time_perm_{p_idx}"] = np.mean(time_perm_results[sub][model][cell_label][:,p_idx])
                # i = i + 1


    n_perms_min = 0
    if time_perm_results and task_perm_results:
        n_perms_min = min(time_perm_results[sub][model][cell_label].shape[1], task_perm_results[sub][model][cell_label].shape[1])
    elif time_perm_results:
        n_perms_min = time_perm_results[sub][model][cell_label].shape[1]
    elif task_perm_results:
        n_perms_min = task_perm_results[sub][model][cell_label].shape[1]
    # import pdb; pdb.set_trace()           
    return df, n_perms_min


    
    
    
    
    
    

def plotting_df_based_corr_perm_histogram_by_ROIs(df_results, title_string_add):
    # import pdb; pdb.set_trace()
    # ROI_labels = ['hippocampal', 'ACC','PCC','OFC', 'entorhinal', 'amygdala', 'mixed']
    bins=50

    models = df_results['model'].unique().tolist()
    
    for model in models:
        filtered_df = df_results[df_results['model'] == model]
        rois = filtered_df['roi'].unique().tolist()
        n_roi = len(rois)
        # Create subplots: one row, n_roi columns
        fig, axes = plt.subplots(1, n_roi, figsize=(n_roi*5, 5), sharey=True)
        # In case there is only one ROI, wrap axes in a list for consistency.
        if n_roi == 1:
            axes = [axes]
        
        for ax, roi in zip(axes, rois):
            corrs_allneurons = filtered_df[filtered_df['roi'] == roi]['average_corr']
            
            # corrs_allneurons = roi_dict[roi]
            nan_filter = np.isnan(corrs_allneurons)
            # Remove any NaN values
            valid_corrs = corrs_allneurons[~np.isnan(corrs_allneurons)]
            mean_sample = np.mean(corrs_allneurons[~nan_filter])
            # Perform a two-tailed one-sample t-test
            ttest_result = st.ttest_1samp(corrs_allneurons[~nan_filter], 0)
            t_stat = ttest_result.statistic
            p_two = ttest_result.pvalue
            
            # Convert to a one-tailed p-value for H1: mean > 0.
            # If t_stat is positive, one-tailed p-value is half the two-tailed value.
            # Otherwise, if t_stat is negative, the one-tailed p-value is 1 - (p_two / 2).
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
            ax.hist(corrs_allneurons, bins=bins, color='skyblue', edgecolor='black')
            ax.axvline(0, color='black', linestyle='dashed', linewidth=2)
            
            # Set subplot title and labels
            ax.set_title(f"{model}\n {roi} \n for {len(corrs_allneurons[~nan_filter])} neurons \n {title_string_add}", fontsize=12)
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
        
        # # Optionally print additional information per model:
        # print(f"Model: {model}")
        # for roi in rois:
        #     print(f"  ROI {roi}: {len(roi_dict[roi])} neurons")










def prep_result_for_plotting_by_rois(results):
    import pdb; pdb.set_trace()
    # CAREFUL!!!!
    # THERE IS SOMETHIGN WRONG HERE!!!
    # I DONT GET THE CORRECT RESULTS
    # DONT USE
    
    # Define your ROI labels
    ROI_labels = ['ACC', 'OFC', 'PCC','hippocampal', 'PFC', 'entorhinal', 'amygdala', 'mixed']
    # Initialize a dictionary for the collapsed results.
    # This dictionary will be structured as:
    # collapsed_results[model][ROI] = list of correlation values (to be converted to an array later)
    collapsed_results = {}
    # import pdb; pdb.set_trace()
    # Assume 'results' is your nested dictionary and collapse_PFC is a Boolean flag.
    for subject, models in results.items():
        for model, cells in models.items():
            # Create dictionary for the model if it doesn't exist
            if model not in collapsed_results:
                collapsed_results[model] = {roi: [] for roi in ROI_labels}
            
            for cell_label, corr_values in cells.items():
                # Determine ROI based on the cell label.
                # Adjust the rules as needed.
                # if ('ACC' in cell_label) or ('PCC' in cell_label) or ('OFC' in cell_label):
                #     roi = 'PFC'
                if 'ACC' in cell_label:
                    roi = 'ACC'
                elif 'PCC' in cell_label:
                    roi = 'PCC'
                elif 'OFC' in cell_label:
                    roi = 'OFC'
                elif 'HC' in cell_label:
                    roi = 'hippocampal'
                elif 'EC' in cell_label:
                    roi = 'entorhinal'
                elif 'AMYG' in cell_label:
                    roi = 'amygdala'
                else:
                    roi = 'mixed'
                
                # Extend the list for the appropriate ROI with this cell's correlation values.
                # This assumes that corr_values is an iterable (e.g., a list or np.array).
                #collapsed_results[model][roi].extend(np.mean(corr_values))
                collapsed_results[model][roi].append(np.mean(corr_values))
   
    # import pdb; pdb.set_trace()
    # Optionally, convert each list to a 1D NumPy array
    for model in collapsed_results:
        for roi in collapsed_results[model]:
            collapsed_results[model][roi] = np.array(collapsed_results[model][roi])
    
    # Now collapsed_results is structured by model and ROI.
    # For example:
    # collapsed_results['model1']['hippocampal'] is a 1D array of all correlation values from cells whose label contained 'HC'
    #import pdb; pdb.set_trace()
    return collapsed_results



def prep_result_dir_for_plotting(result_dir):
    # first collapse across subjects.
    # Initialize empty lists to store values
    button_box_list, musicbox_list, state_list = [], [], []
    # import pdb; pdb.set_trace()
    # Loop through the nested dictionary
    for subject in result_dir.values():  # Iterate over subjects
        for reg_type, neurons in subject.items():  # Iterate over regression types
            for neuron_data in neurons.values():  # Iterate over neurons
                if reg_type == "buttonbox_reg":
                    button_box_list.append(np.mean(neuron_data))
                elif reg_type == "musicbox_reg":
                    musicbox_list.append(np.mean(neuron_data))
                elif reg_type == "state_reg":
                    state_list.append(np.mean(neuron_data))
    
    # Convert to NumPy arrays
    results_of_corr = {}
    results_of_corr['button_box'] = np.array(button_box_list)
    results_of_corr['musicbox'] = np.array(musicbox_list)
    results_of_corr['state'] = np.array(state_list)
    # results_of_corr['button_box'] = np.concatenate(button_box_list) if button_box_list else np.array([])
    # results_of_corr['musicbox'] = np.concatenate(musicbox_list) if musicbox_list else np.array([])
    # results_of_corr['state'] = np.concatenate(state_list) if state_list else np.array([])
    return results_of_corr


def plotting_corr_perm_histogram(results_of_corr, title_string_add):
    # finally, plot the distribution for each model                                      
    bins=50
    for model in results_of_corr:
        # something like this:
        corrs_allneurons=results_of_corr[model]
        nan_filter = np.isnan(corrs_allneurons)
        plt.figure()
        plt.title(f"correlation between {model} and all neurons")
        plt.hist(corrs_allneurons[~nan_filter],bins=bins,color='grey')
        #plt.xlim(-1,1)
        plt.axvline(0,color='black',ls='dashed')
        plt.tick_params(axis='both',  labelsize=20)
        plt.tick_params(width=2, length=6)
        # plt.savefig(reg_fig_dir+model+'GLM_analysis_all_neurons.svg',\
        #             bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        print(len(corrs_allneurons[~nan_filter]))
        print(st.ttest_1samp(corrs_allneurons[~nan_filter],0))
    
    
    
    for model in results_of_corr:
        # something like this:
        corrs_allneurons=results_of_corr[model]
        nan_filter = np.isnan(corrs_allneurons)
    
        # Compute the t-test for the correlations

        # Perform a two-tailed one-sample t-test
        ttest_result = st.ttest_1samp(corrs_allneurons[~nan_filter], 0)
        t_stat = ttest_result.statistic
        p_two = ttest_result.pvalue
        
        # Convert to a one-tailed p-value for H1: mean > 0.
        # If t_stat is positive, one-tailed p-value is half the two-tailed value.
        # Otherwise, if t_stat is negative, the one-tailed p-value is 1 - (p_two / 2).
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
        
        # Create a figure with a larger size for better aesthetics
        plt.figure(figsize=(10, 6))
        
        # Plot the histogram
        plt.hist(corrs_allneurons, bins=bins, color='skyblue', edgecolor='black')
        
        # Add a vertical dashed line at 0
        plt.axvline(0, color='black', linestyle='dashed', linewidth=2)
        
        # Add title and axis labels with increased font sizes
        plt.title(f"Correlation between {model} and {title_string_add} \n for {len(corrs_allneurons[~nan_filter])} neurons", fontsize=22)
        plt.xlabel("Correlation coefficient", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        
        # Adjust tick parameters for better readability
        plt.tick_params(axis='both', labelsize=16, width=2, length=6)
        
        # Annotate the figure with the significance level
        plt.text(0.95, 0.95, f"Significance: {significance}\n(p = {p_value:.3e})",
                 transform=plt.gca().transAxes, fontsize=16,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        
        plt.tight_layout()
        plt.show()
        
        # Print additional output if needed
        print("Number of neurons:", len(corrs_allneurons))
        print(ttest_result)
        
        
        
def plotting_corr_perm_histogram_by_ROIs(collapsed_results, title_string_add):
    ROI_labels = ['hippocampal', 'ACC','PCC','OFC', 'entorhinal', 'amygdala', 'mixed']
    bins=50
    for model in collapsed_results:
        # import pdb; pdb.set_trace()
        roi_dict = collapsed_results[model]
        # Determine which ROIs exist for this model (and preserve your ordering)
        rois = [roi for roi in ROI_labels if roi in roi_dict]
        n_roi = len(rois)
        
        # Create subplots: one row, n_roi columns
        fig, axes = plt.subplots(1, n_roi, figsize=(n_roi*5, 5), sharey=True)
        # In case there is only one ROI, wrap axes in a list for consistency.
        if n_roi == 1:
            axes = [axes]
        
        for ax, roi in zip(axes, rois):
            corrs_allneurons = roi_dict[roi]
            nan_filter = np.isnan(corrs_allneurons)
            # Remove any NaN values
            valid_corrs = corrs_allneurons[~np.isnan(corrs_allneurons)]
            mean_sample = np.mean(corrs_allneurons[~nan_filter])
            # Perform a two-tailed one-sample t-test
            ttest_result = st.ttest_1samp(corrs_allneurons[~nan_filter], 0)
            t_stat = ttest_result.statistic
            p_two = ttest_result.pvalue
            
            # Convert to a one-tailed p-value for H1: mean > 0.
            # If t_stat is positive, one-tailed p-value is half the two-tailed value.
            # Otherwise, if t_stat is negative, the one-tailed p-value is 1 - (p_two / 2).
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
            ax.hist(corrs_allneurons, bins=bins, color='skyblue', edgecolor='black')
            ax.axvline(0, color='black', linestyle='dashed', linewidth=2)
            
            # Set subplot title and labels
            ax.set_title(f"{model}\n {roi} \n for {len(corrs_allneurons[~nan_filter])} neurons \n {title_string_add}", fontsize=12)
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
        
        # Optionally print additional information per model:
        print(f"Model: {model}")
        for roi in rois:
            print(f"  ROI {roi}: {len(roi_dict[roi])} neurons")
           
            
            