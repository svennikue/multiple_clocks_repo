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
import os


def plot_perms_per_cell_and_roi(df_results, n_perms, corr_thresh=0.05, save=False):
    # import pdb; pdb.set_trace()
    # ROI_labels = ['hippocampal', 'ACC','PCC','OFC', 'entorhinal', 'amygdala', 'mixed']
    bins=50
    models = df_results['model'].unique().tolist()
    cells = df_results['cell'].unique().tolist()
    
    
    # plot those cells that are strong for the respective model (corr higher than 0.05)
    df_strong_cells = df_results[df_results['average_corr'] > corr_thresh]
    for curr_model in models:
        # df_strong_curr_model = df_strong_cells[df_strong_cells['model'] == curr_model].reset_index(drop=True)
        df_strong_curr_model = df_strong_cells[df_strong_cells['model'] == curr_model].copy()

        # If 'time_perm_0' exists, compute p_val_time for each row
        if 'time_perm_0' in df_strong_curr_model.columns:
            p_val_times = []
            for _, row in df_strong_curr_model.iterrows():
                perm_values = row[[f'time_perm_{i}' for i in range(n_perms)]].values
                p_val_time = np.mean(perm_values >= row['average_corr'])
                p_val_times.append(p_val_time)
            df_strong_curr_model['p_val_time'] = p_val_times
        
            # Sort by p_val_time and take top 25
            if len(df_strong_curr_model) > 25:
                df_strong_curr_model = df_strong_curr_model.sort_values('p_val_time').head(25)
        else:
            df_strong_curr_model = df_strong_curr_model.sort_values('average_corr').head(25)
            
        df_strong_curr_model = df_strong_curr_model.reset_index(drop=True)

        # also store p vals for task perms
        if 'task_perm_0' in df_strong_curr_model.columns:
            p_val_task = []
            for _, row in df_strong_curr_model.iterrows():
                perm_values = row[[f'task_perm_{i}' for i in range(n_perms)]].values
                p_val_task = np.mean(perm_values >= row['average_corr'])
                p_val_task.append(p_val_task)
            df_strong_curr_model['p_val_task'] = p_val_task
            
        #
        #
        # Plotting
        n_rows = int(np.ceil(np.sqrt(len(df_strong_curr_model))))
        n_cols = int(np.ceil(len(df_strong_curr_model) / n_rows))
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12))
        fig.suptitle(f"{curr_model}", fontsize=15, y=0.99)  # Title slightly above the top
        axs = axs.flatten()
        
        # Custom colors
        color_task_perms = '#214066'   # dark turquoise blue
        color_time_perms = '#7A9DB1'   # blue-grey
        true_val_color = '#E2725B'   # terracotta/salmon


        
        # Determine common x-axis range for centering
        # all_values = df_strong_curr_model[[f'perm_{i}' for i in range(n_perms)]].values.flatten()
        
        if 'task_perm_0' in df_strong_curr_model.columns:
            all_values = df_strong_curr_model[[f'task_perm_{i}' for i in range(n_perms)]].values.flatten()
        if 'time_perm_0' in df_strong_curr_model.columns:
            all_values = df_strong_curr_model[[f'time_perm_{i}' for i in range(n_perms)]].values.flatten()
        if 'task_perm_0' in df_strong_curr_model.columns and 'time_perm_0' in df_strong_curr_model.columns:
            values_task = df_strong_curr_model[[f'task_perm_{i}' for i in range(n_perms)]].values.flatten()
            values_time = df_strong_curr_model[[f'time_perm_{i}' for i in range(n_perms)]].values.flatten()
            all_values = np.concatenate((values_task, values_time))

        xlim = max(abs(np.nanmin(all_values)), abs(np.nanmax(all_values)))  # Symmetric about 0
    
        
        for idx, row in df_strong_curr_model.iterrows():
            
            avg_corr = row['average_corr']
            
            ax = axs[idx]
            if 'task_perm_0' in df_strong_curr_model.columns:
                perm_values = row[[f'task_perm_{i}' for i in range(n_perms)]].values
                ax.hist(perm_values, bins=30, color=color_task_perms, alpha=0.5, label='Task perm.', edgecolor=None)
                # Calculate one-tailed p-value
                p_val_task = np.mean(perm_values >= avg_corr)
                ax.text(0.95, 0.70, f"p_task = {p_val_task:.3f}", ha='right', va='top', transform=ax.transAxes)
                
            if 'time_perm_0' in df_strong_curr_model.columns:
                perm_values = row[[f'time_perm_{i}' for i in range(n_perms)]].values
                ax.hist(perm_values, bins=30, color=color_time_perms, alpha=0.5, label='Time perm.', edgecolor=None)
                # Calculate one-tailed p-value
                p_val_time = np.mean(perm_values >= avg_corr)
                ax.text(0.95, 0.95, f"p_time = {p_val_time:.3f}", ha='right', va='top', transform=ax.transAxes)
                
                
            
            ax.axvline(avg_corr, color=true_val_color, linestyle='--', linewidth=2)
            
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            
            # Center x-axis around 0
            ax.set_xlim(-xlim, xlim)
        

            
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
        if save==True:
            fig_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/elastic_net_reg/corrs"
            if not os.path.isdir(fig_folder):
                fig_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group/elastic_net_reg/corrs"
            
            os.makedirs(f"{fig_folder}/figures", exist_ok=True)
            plt.savefig(f"{curr_model}_perms_best_cells.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
        
        
    
    



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
    
