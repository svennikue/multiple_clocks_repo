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
    for curr_model in models:
        df_strong_curr_model = df_strong_cells[df_strong_cells['model'] == curr_model].reset_index(drop=True)
        #
        #
        # Plotting
        n_rows = int(np.ceil(np.sqrt(len(df_strong_curr_model))))
        n_cols = int(np.ceil(len(df_strong_curr_model) / n_rows))
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12))
        fig.suptitle(f"{curr_model}", fontsize=15, y=0.99)  # Title slightly above the top
        axs = axs.flatten()
        
        
        # Determine common x-axis range for centering
        all_values = df_strong_curr_model[[f'perm_{i}' for i in range(n_perms)]].values.flatten()
        xlim = max(abs(np.min(all_values)), abs(np.max(all_values)))  # Symmetric about 0
    
        
        for idx, row in df_strong_curr_model.iterrows():
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
            ax.set_xlabel("Correlation", fontsize = 9)
            ax.set_ylabel("Count")
        
        # Hide any unused subplots
        for ax in axs[len(df_strong_curr_model):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 1.02])  # Adjust layout to make room for the title
        #plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the title
        # plt.title("SMB model")
        
        # then store these figures if on cluster. 
        plt.show()


    import pdb; pdb.set_trace()
    
    
    



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
    
