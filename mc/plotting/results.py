#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:13:01 2024

this script offers several specific functions to plot my results.

@author: xpsy1114
"""

from matplotlib import pyplot as plt


import matplotlib.pyplot as plt
import numpy as np


def plot_perms_per_cell_and_roi(df_results, n_perms):
    # import pdb; pdb.set_trace()
    # ROI_labels = ['hippocampal', 'ACC','PCC','OFC', 'entorhinal', 'amygdala', 'mixed']
    bins=50

    models = df_results['model'].unique().tolist()
    cells = df_results['cell'].unique().tolist()
    
    
    # one goal: I want to know if the clocks model is still significant.
    # plot all clocks model cells that were higher than 0.05
    
    df_strong_cells = df_results[df_results['average_corr'] > 0.05]
    df_strong_clocks = df_strong_cells[df_strong_cells['model'] == 'clo_model'].reset_index(drop=True)
    #
    #
    # Plotting
    n_rows = int(np.ceil(np.sqrt(len(df_strong_clocks))))
    n_cols = int(np.ceil(len(df_strong_clocks) / n_rows))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    fig.suptitle("SMB model", fontsize=16, y=1.02)  # Title slightly above the top
    axs = axs.flatten()
    
    
    # Determine common x-axis range for centering
    all_values = df_strong_clocks[[f'perm_{i}' for i in range(n_perms)]].values.flatten()
    xlim = max(abs(np.min(all_values)), abs(np.max(all_values)))  # Symmetric about 0

    
    for idx, row in df_strong_clocks.iterrows():
        perm_values = row[[f'perm_{i}' for i in range(n_perms)]].values
        avg_corr = row['average_corr']
        
        ax = axs[idx]
        ax.hist(perm_values, bins=30, color='skyblue', edgecolor='black')
        ax.axvline(avg_corr, color='red', linestyle='--', linewidth=2)
        
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        
        # Center x-axis around 0
        ax.set_xlim(-xlim, xlim)
    
        # Calculate one-tailed p-value
        p_val = np.mean(perm_values >= avg_corr)
        ax.text(0.95, 0.95, f"p = {p_val:.3f}", ha='right', va='top', transform=ax.transAxes)
        
        ax.set_title(f"{row['roi']} | {row['cell']}", fontsize=10)
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Count")
    
    # Hide any unused subplots
    for ax in axs[len(df_strong_clocks):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the title
    plt.title("SMB model")
    plt.show()

    
    
    # df_strong_cells = df_results[df_results['average_corr'] > 0.05]
    # df_strong_state = df_strong_cells[df_strong_cells['model'] == 'state_reg'].reset_index(drop=True)
    # #
    # #
    # n_perms = 260
    # # Plotting
    # n_rows = int(np.ceil(np.sqrt(len(df_strong_state))))
    # n_cols = int(np.ceil(len(df_strong_state) / n_rows))
    
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    # fig.suptitle("State model", fontsize=16, y=1.02)  # Title slightly above the top
    # axs = axs.flatten()
    
    
    # # Determine common x-axis range for centering
    # all_values = df_strong_state[[f'perm_{i}' for i in range(n_perms)]].values.flatten()
    # xlim = max(abs(np.min(all_values)), abs(np.max(all_values)))  # Symmetric about 0

    
    # for idx, row in df_strong_state.iterrows():
    #     perm_values = row[[f'perm_{i}' for i in range(n_perms)]].values
    #     avg_corr = row['average_corr']
        
    #     ax = axs[idx]
    #     ax.hist(perm_values, bins=30, color='skyblue', edgecolor='black')
    #     ax.axvline(avg_corr, color='red', linestyle='--', linewidth=2)
        
    #     ax.axvline(0, color='black', linestyle='-', linewidth=1)
        
    #     # Center x-axis around 0
    #     ax.set_xlim(-xlim, xlim)
    
    #     # Calculate one-tailed p-value
    #     p_val = np.mean(perm_values >= avg_corr)
    #     ax.text(0.95, 0.95, f"p = {p_val:.3f}", ha='right', va='top', transform=ax.transAxes)
        
    #     ax.set_title(f"{row['roi']} | {row['cell']}", fontsize=10)
    #     ax.set_xlabel("Correlation")
    #     ax.set_ylabel("Count")
    
    # # Hide any unused subplots
    # for ax in axs[len(df_strong_state):]:
    #     ax.axis('off')
    
    # plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the title
    # plt.show()
    # #
    # #
    # #
    
    
    
    import pdb; pdb.set_trace()
    
    for model in models:
        filtered_df = df_results[df_results['model'] == model]
        rois = filtered_df['roi'].unique().tolist()
        n_roi = len(rois)
        # create a different plot per 
        
        
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
        
        
    
    



def overview_regression(results, rois, models, combo = False, only_reward_times = None, no_bins_per_state = None):
    # import pdb; pdb.set_trace()
    # Prepare data arrays for t-values and annotations for significance
    t_values = np.zeros((len(rois), len(models)))
    p_values = np.zeros_like(t_values)
    
    if combo == True:
        # first figure out the correct model order.
        for r, roi in enumerate(rois):
            models = results[roi]['label_regs']
        for r, roi in enumerate(rois):
            for m, model in enumerate(models):
                # import pdb; pdb.set_trace()
                t_values[r, m] = results[roi]['t_vals'][m]
                p_values[r, m] = results[roi]['p_vals'][m]
                
                
    else:                 
        for r, roi in enumerate(rois):
            for m, model in enumerate(models):
                if model in results[roi][model]['label_regs']:
                    t_values[r, m] = results[roi][model]['t_vals'][0]
                    p_values[r, m] = results[roi][model]['p_vals'][0]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 7))  # Adjust size as needed
    cax = ax.matshow(t_values, cmap='viridis')  # Choose a colormap that fits your preferences
    
    # Add a color bar
    fig.colorbar(cax, label='T-value magnitude')
    
    # Set up axes
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(rois)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(rois)
    
    # Rotate the tick labels for better readability
    plt.xticks(rotation=45)
    
    # Adding significance annotations
    for i in range(len(rois)):
        for j in range(len(models)):
            signif = '*' if p_values[i, j] < 0.05 else ''
            signif += '*' if p_values[i, j] < 0.01 else ''
            signif += '*' if p_values[i, j] < 0.005 else ''
            ax.text(j, i, signif, color='black', ha='center', va='center', fontsize=12)
    
    # Title and labels
    ax.set_title('T-values and Significance of Models Across ROIs')
    if only_reward_times == True:
        ax.set_title(f"T-values and Significance of State Across ROIs, only rew times, {no_bins_per_state} bins per state")
    if only_reward_times == False:
        ax.set_title(f"T-values and Significance of State Across ROIs, {no_bins_per_state} bins per state")
    if combo == True:
        ax.set_xlabel('Each model as regressor in combined GLM')
    else:     
        ax.set_xlabel('Models, in separate GLMs as single regressors')
    ax.set_ylabel('ROIs')
    
    plt.tight_layout()
    # Show the plot
    plt.show()
    
