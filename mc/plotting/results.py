#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:13:01 2024

this script offers several specific functions to plot my results.

@author: Svenja Küchenhoff
"""

from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sns
import scipy.stats as st 


def slope_plot_early_late_per_roi(df_early, df_late, title_string_add):
    
    # import pdb; pdb.set_trace()
    # Define your colors
    early_color = '#00BFC4'      # turquoise-blue
    late_color = '#E07B39'       # terracotta-orange
    
    # Merge the two DataFrames on cell and roi
    merged_df = pd.merge(
        df_early[['cell', 'roi', 'average_corr', 'model']],
        df_late[['cell', 'roi', 'average_corr', 'model']],
        on=['cell', 'roi', 'model'],
        suffixes=('_before', '_after')
    ).reset_index(drop=True)
    
    models = df_early['model'].unique().tolist()
    
    # only plot subset for now
    # import pdb; pdb.set_trace()
    models = ['complete_musicbox_reg', 'clo_model', 'curr_rings_split_clock_model', 'one_fut_rings_split_clock_model', 'two_fut_rings_split_clock_model', 'three_fut_rings_split_clock_model', 'phas_model', 'state_reg']
    
    
    # List of unique ROIs
    rois = merged_df['roi'].unique()
    n_roi = len(rois)
    
    for model in models:
        merged_df_model = merged_df[merged_df['model'] == model]
        
        # Plot
        fig, axes = plt.subplots(1, n_roi, figsize=(n_roi * 5, 5), sharey=True)
        if n_roi == 1:
            axes = [axes]
        
        for ax, roi in zip(axes, rois):
            df_roi = merged_df_model[merged_df_model['roi'] == roi]
        
            for _, row in df_roi.iterrows():
                # Grey line connecting before and after
                ax.plot([0, 1], [row['average_corr_before'], row['average_corr_after']], color='gray', linewidth=0.5)
        
            # Scatter points
            ax.scatter([0]*len(df_roi), df_roi['average_corr_before'], color=early_color, label='before', zorder=3)
            ax.scatter([1]*len(df_roi), df_roi['average_corr_after'], color=late_color, label='after', zorder=3)
        
            # Aesthetics
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Early', 'Late'])
            ax.set_title(roi)
            ax.set_ylabel('Average Correlation')
            ax.set_xlim(-0.5, 1.5)
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)
            ax.tick_params(axis='both', labelsize=12)
            ax.axhline(0, linestyle='solid', color='black', linewidth=1)
    
        # Only add legend to the first axis
        axes[0].legend()
        
        fig.suptitle(f"{model}\n — {title_string_add}", fontsize=12, y=0.97)
    
        
        plt.tight_layout()
        plt.show()

    
    

def plotting_two_df_corr_perm_histogram_by_ROIs(df_early, df_late, title_string_add):
    # import pdb; pdb.set_trace()
    
    # Define colors
    early_color = '#00BFC4'      # turquoise-blue
    late_color = '#E07B39'       # terracotta-orange
    
    line_thickness = 2
    
    # Function to get significance stars
    def get_significance(corrs):
        t_stat, p_two = st.ttest_1samp(corrs, 0)
        p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
        if p_one < 0.001:
            return '***'
        elif p_one < 0.01:
            return '**'
        elif p_one < 0.05:
            return '*'
        else:
            return ''
    
    models = df_early['model'].unique().tolist()
    
    # only plot subset for now
    # import pdb; pdb.set_trace()
    models = ['complete_musicbox_reg', 'location_reg', 'musicbox_onlynowand3future_complete_reg', 'musicbox_onlynextand2future_complete_reg', 'midn_model', 'phas_model', 'stat_model', 'phas_stat_model', 'clo_model', 'curr_rings_split_clock_model', 'one_fut_rings_split_clock_model', 'two_fut_rings_split_clock_model', 'three_fut_rings_split_clock_model']
    
    
    for model in models:
        df_early_model = df_early[df_early['model'] == model]
        df_late_model = df_late[df_late['model'] == model]
        
        rois = sorted(set(df_early_model['roi'].unique()).union(df_late_model['roi'].unique()))
        n_roi = len(rois)
        
        # fig, axes = plt.subplots(1, n_roi, figsize=(n_roi * 5, 5), sharey=True)
        
        fig, axes = plt.subplots(1, n_roi, figsize=(n_roi * 6, 3), sharey=True)

        if n_roi == 1:
            axes = [axes]
    
        for ax, roi in zip(axes, rois):
            # Get early and late data
            corrs_early = df_early_model[df_early_model['roi'] == roi]['average_corr'].dropna()
            corrs_late = df_late_model[df_late_model['roi'] == roi]['average_corr'].dropna()
    
            # Get stars for significance vs zero
            early_sig = get_significance(corrs_early)
            late_sig = get_significance(corrs_late)
    
            # KDE plots
            
            # # KDE plots with custom bandwidth and curve height
            # sns.kdeplot(
            #     corrs_early, ax=ax, color=early_color, fill=True, alpha=0.4,
            #     linewidth=line_thickness, bw_adjust=0.2, label=f"early {early_sig}"
            # )
            # sns.kdeplot(
            #     corrs_late, ax=ax, color=late_color, fill=True, alpha=0.4,
            #     linewidth=line_thickness, bw_adjust=0.2, label=f"late {late_sig}"
            # )
        

            # Vertical lines at means
            mean_early = corrs_early.mean()
            mean_late = corrs_late.mean()
            ax.axvline(mean_early, color=early_color, linestyle='solid', linewidth=line_thickness)
            ax.axvline(mean_late, color=late_color, linestyle='solid', linewidth=line_thickness)
    
            # Zero reference line
            ax.axvline(0, color='black', linestyle='dashed', linewidth=line_thickness)
    
            # Plot overlapping histograms with transparency (true frequency)
            ax.hist(
                corrs_early, bins=20, color=early_color, alpha=0.5,
                label=f"early {early_sig}", edgecolor='black'
            )
            ax.hist(
                corrs_late, bins=20, color=late_color, alpha=0.5,
                label=f"late {late_sig}", edgecolor='black'
            )
            
            # Add vertical lines for means
            ax.axvline(mean_early, color=early_color, linestyle='solid', linewidth=2)
            ax.axvline(mean_late, color=late_color, linestyle='solid', linewidth=2)
            
            # Add zero reference
            ax.axvline(0, color='black', linestyle='dashed', linewidth=2)
            
            # Y-axis now shows count (no need to change scale)
            ax.set_ylabel("Frequency", fontsize=10)


            # Labels and formatting
            ax.set_title(f"{roi}\n{len(corrs_early)} early / {len(corrs_late)} late neurons", fontsize=10)
            ax.set_xlabel("Correlation coefficient", fontsize=12)
            ax.tick_params(axis='both', labelsize=10, width=2, length=6)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend()
    
        # Move model name to top of entire figure
        fig.suptitle(f"{model}\n — {title_string_add}", fontsize=12, y=0.93)

        plt.tight_layout()
        plt.show()
        





def plot_overlap_in_cells(df1, df2, top_x_percent):
    # import pdb; pdb.set_trace()

    # Define your ROI order (top to bottom)
    # first get all rois
    rois = df1['roi'].unique().tolist()
    roi_rank = {roi: i for i, roi in enumerate(rois)}
    
    # --- Setup filtering ---
    def get_top_cells(df, model_name='stat_model', top_percent=top_x_percent):
        df_filtered = df[df['model'] == model_name]
        cutoff = df_filtered['average_corr'].quantile(1 - top_percent / 100)
        return df_filtered[df_filtered['average_corr'] >= cutoff]
    
    # --- Filter ---
    df1_top = get_top_cells(df1)
    df2_top = get_top_cells(df2)
    
    # --- Sets of cell IDs ---
    cells1 = set(df1_top['cell'])
    cells2 = set(df2_top['cell'])
    
    only1 = cells1 - cells2
    only2 = cells2 - cells1
    both = cells1 & cells2

    # --- Create plot data ---
    plot_data = []
    
    def add_points(df, cells, label, x_center):
        for cell in cells:
            row = df[df['cell'] == cell].iloc[0]
            roi = row['roi']
            corr = row['average_corr']
            if roi not in roi_rank:
                continue  # Skip unknown ROIs
            y_base = -roi_rank[roi]  # invert for top-to-bottom
            x = np.random.normal(loc=x_center, scale=0.2)
            y = np.random.normal(loc=y_base, scale=0.2)
            size = corr * 800  # adjust scaling more aggressively
            plot_data.append({'x': x, 'y': y, 'group': label, 'size': size, 'roi': roi})
    
    add_points(df1_top, only1, 'df1 only', -1)
    add_points(df2_top, only2, 'df2 only', 1)
    # Average the corr from both dfs for overlap
    for cell in both:
        row1 = df1_top[df1_top['cell'] == cell].iloc[0]
        row2 = df2_top[df2_top['cell'] == cell].iloc[0]
        roi = row1['roi']
        if roi not in roi_rank:
            continue
        avg_corr = (row1['average_corr'] + row2['average_corr']) / 2
        y_base = -roi_rank[roi]
        x = np.random.normal(loc=0, scale=0.2)
        y = np.random.normal(loc=y_base, scale=0.2)
        size = (avg_corr - 0.3) * 800
        plot_data.append({'x': x, 'y': y, 'group': 'overlap', 'size': size, 'roi': roi})
    
    plot_df = pd.DataFrame(plot_data)
    
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    for group, alpha in zip(['df1 only', 'df2 only', 'overlap'], [0.5, 0.5, 0.9]):
        subset = plot_df[plot_df['group'] == group]
        plt.scatter(subset['x'], subset['y'], s=subset['size'], alpha=alpha, label=group)
    
    # Add ROI labels on y-axis
    y_ticks = [-roi_rank[roi] for roi in rois]
    plt.yticks(y_ticks, rois)
    plt.xticks([])  # Remove x-axis ticks (groups are implicit)
    plt.xlabel('')
    plt.ylabel('ROI')
    plt.title(f'Overlapping Structured Representations by ROI\n(Top {top_x_percent}% average_corr, Model = state_model)')
    plt.legend()
    plt.tight_layout()
    plt.show()


    

def plot_perms_per_cell_and_roi(df_results, n_perms, corr_thresh=0.05, save=False, model_name_string=None):
    if save==True:
        res_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/elastic_net_reg/corrs"
        if not os.path.isdir(res_folder):
            res_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group/elastic_net_reg/corrs"
        

    models = df_results['model'].unique().tolist()
    cells = df_results['cell'].unique().tolist()
    rois = df_results['roi'].unique().tolist()
    # Custom colors
    color_task_perms = '#214066'   # dark turquoise blue
    color_time_perms = '#7A9DB1'   # blue-grey
    true_val_color = '#E2725B'   # terracotta/salmon
    # Always plotting the top 25 cells.
    # n_rows = int(np.ceil(np.sqrt(len(df_strong_curr_model))))
    # n_cols = int(np.ceil(len(df_strong_curr_model) / n_rows))
    n_rows = 5
    n_cols = 5

    # plot those cells that are strong for the respective model (corr higher than 0.05)
    # df_strong_cells = df_results[df_results['average_corr'] > corr_thresh]
    for curr_model in models:
        # df_strong_curr_model = df_strong_cells[df_strong_cells['model'] == curr_model].reset_index(drop=True)
        df_curr_model = df_results[df_results['model'] == curr_model].copy()
        
        # 1: COMPUTE SOME PERM STATS PER MODEL/CELL
        
        # If 'time_perm_0' exists, compute p_val_time for each row
        if 'time_perm_0' in df_curr_model.columns:
            p_val_times = []
            for _, row in df_curr_model.iterrows():
                perm_values = row[[f'time_perm_{i}' for i in range(n_perms)]].values
                p_val_time = np.mean(perm_values >= row['average_corr'])
                p_val_times.append(p_val_time)
            df_curr_model['p_val_time'] = p_val_times

        # also store p vals for task perms
        if 'task_perm_0' in df_curr_model.columns:
            p_val_tasks = []
            for _, row in df_curr_model.iterrows():
                perm_values = row[[f'task_perm_{i}' for i in range(n_perms)]].values
                p_val_task = np.mean(perm_values >= row['average_corr'])
                p_val_tasks.append(p_val_task)
            df_curr_model['p_val_task'] = p_val_tasks
        

        # and store the difference between both p val perms
        for idx, row in df_curr_model.iterrows():
            if 'task_perm_0' in df_curr_model.columns and 'time_perm_0' in df_curr_model.columns:
                perm_values_task = row[[f'task_perm_{i}' for i in range(n_perms)]].values
                perm_values_time = row[[f'time_perm_{i}' for i in range(n_perms)]].values
                _, p_value_diff_perms = ttest_ind(list(perm_values_task), list(perm_values_time))
                df_curr_model.loc[idx, 'p_val_perm_diff'] = p_value_diff_perms
         
        if save==True:
            # save the entire df for cells only for this model.
            os.makedirs(f"{res_folder}/cells_per_model", exist_ok=True)         
            df_curr_model.to_csv(f"{res_folder}/cells_per_model/{curr_model}_{model_name_string}.csv", index=False)
            
        
        # 2:  PRINTING STATS
        # then print some stats: percentage of cells, overall and per ROI for
        # each of the permutation ps

        results_file = []
        
        # first: overall
        n_cells = len(df_curr_model)
        # import pdb; pdb.set_trace()
        mean_avg_corr = np.mean(df_curr_model['average_corr'])
        
        print(f"for {curr_model}, for n = {n_cells} cells all over the brain, the mean corr is {mean_avg_corr:.3f}")
        results_file.append(f"for {curr_model}, for n = {n_cells} cells all over the brain, the mean corr is {mean_avg_corr:.3f}")
        
        
        if 'task_perm_0' in df_curr_model.columns:
            n_p_val_task_sig = len(df_curr_model[df_curr_model['p_val_task'] < 0.05])
            
            if n_p_val_task_sig > 0:
                mean_corr_sig_task = np.mean(df_curr_model[df_curr_model['p_val_task'] < 0.05])
            else:
                mean_corr_sig_task = 0
                   
            print(f"n = {n_p_val_task_sig} or {(n_p_val_task_sig/n_cells)*100:.3f} % cells are sig. for task config shuffles,")
            results_file.append(f"n = {n_p_val_task_sig} or {(n_p_val_task_sig/n_cells)*100:.3f} % cells are sig. for task config shuffles,")
        
        
        # this is the one I want to keep.
        if 'time_perm_0' in df_curr_model.columns:
            n_p_val_time_sig = len(df_curr_model[df_curr_model['p_val_time'] < 0.05])
            
            # and compute the mean for the significant cells.
            df_curr_model_sig = df_curr_model[df_curr_model['p_val_time'] < 0.05]
            n_sig_cells = len(df_curr_model_sig)
            if n_sig_cells > 0:
                mean_avg_corr_sig = np.mean(df_curr_model_sig['average_corr'])
            else:
                mean_avg_corr_sig = 0
            print(f"for {curr_model}, for n = {n_sig_cells} cells or or {(n_sig_cells/n_cells)*100:.3f} % of cells all over the brain are sig., their mean corr being r = {mean_avg_corr_sig:.3f}")
            results_file.append(f"for {curr_model}, for n = {n_sig_cells} cells or or {(n_sig_cells/n_cells)*100:.3f} % of cells all over the brain are sig., their mean corr being r = {mean_avg_corr_sig:.3f}")
            
            # also save the signficant cells as .csv
            if save == True:
                df_curr_model_sig.to_csv(f"{res_folder}/{model_name_string}_sig_after_temp_perms.csv", index=False)
            

        if 'task_perm_0' in df_curr_model.columns and 'time_perm_0' in df_curr_model.columns:
            n_p_val_perm_diff_sig = len(df_curr_model[df_curr_model['p_val_perm_diff'] < 0.05])
            
            if n_p_val_perm_diff_sig > 0:
                mean_corr_sig_diff = np.mean(df_curr_model[df_curr_model['p_val_perm_diff'] < 0.05])
            else:
                mean_corr_sig_diff = 0
            
            print(f"n = {n_p_val_perm_diff_sig} or {(n_p_val_perm_diff_sig/n_cells)*100} % have sig. different perm distributions.")
            results_file.append(f"n = {n_p_val_perm_diff_sig} or {(n_p_val_perm_diff_sig/n_cells)*100:.1f} % have sig. different perm distributions.")
        
        
        # second: per roi
        for roi in rois:
            df_curr_model_curr_roi = df_curr_model[df_curr_model['roi'] == roi].copy().reset_index(drop=True)
            n_cells_in_roi = len(df_curr_model_curr_roi)
            mean_avg_corr = np.mean(df_curr_model_curr_roi['average_corr'])
            print(f"for {curr_model}, for n = {n_cells_in_roi} cells in {roi}, mean corr is {mean_avg_corr:.3f}")
            results_file.append(f"for {curr_model}, for n = {n_cells_in_roi} cells in {roi}, mean corr is {mean_avg_corr:.3f}")
            
            if 'time_perm_0' in df_curr_model.columns:
                n_p_val_time_sig = len(df_curr_model_curr_roi[df_curr_model_curr_roi['p_val_time'] < 0.05])
                print(f"n = {n_p_val_time_sig} or {(n_p_val_time_sig/n_cells_in_roi)*100:.1f} % cells are sig. for time shuffles,")
                results_file.append(f"n = {n_p_val_time_sig} or {(n_p_val_time_sig/n_cells_in_roi)*100:.1f} % cells are sig. for task config shuffles,")
                
            
                # and compute the mean for the significant cells.
                df_curr_model_curr_roi_sig = df_curr_model_curr_roi[df_curr_model_curr_roi['p_val_time'] < 0.05]
                n_sig_cells_curr_model_curr_roi = len(df_curr_model_curr_roi_sig)
                if n_sig_cells_curr_model_curr_roi > 0:
                    mean_avg_corr_sig_curr_model_curr_roi = np.mean(df_curr_model_curr_roi_sig['average_corr'])
                else:
                    mean_avg_corr_sig_curr_model_curr_roi = 0
                print(f"for {curr_model}, for n = {n_sig_cells_curr_model_curr_roi} cells or {(n_sig_cells_curr_model_curr_roi/n_cells)*100:.1f} % of cells in {roi} are sig., their mean corr being r = {mean_avg_corr_sig_curr_model_curr_roi:.3f}")
                results_file.append(f"for {curr_model}, for n = {n_sig_cells_curr_model_curr_roi} cells or {(n_sig_cells_curr_model_curr_roi/n_cells)*100:.1f} % of cells in {roi} are sig., their mean corr being r = {mean_avg_corr_sig_curr_model_curr_roi:.3f}")
                
                
                
            if 'task_perm_0' in df_curr_model.columns:
                n_p_val_task_sig = len(df_curr_model_curr_roi[df_curr_model_curr_roi['p_val_task'] < 0.05])
                print(f"n = {n_p_val_task_sig} or {(n_p_val_task_sig/n_cells_in_roi)*100:.1f} % cells are sig. for task config shuffles,")
                results_file.append(f"n = {n_p_val_task_sig} or {(n_p_val_task_sig/n_cells_in_roi)*100:.1f} % cells are sig. for task shuffles,")
                
            if 'task_perm_0' in df_curr_model.columns and 'time_perm_0' in df_curr_model.columns:
                n_p_val_perm_diff_sig = len(df_curr_model_curr_roi[df_curr_model_curr_roi['p_val_perm_diff'] < 0.05])
                print(f"n = {n_p_val_perm_diff_sig} or {(n_p_val_perm_diff_sig/n_cells_in_roi)*100:.1f} % have sig. different perm distributions.")
                results_file.append(f"n = {n_p_val_perm_diff_sig} or {(n_p_val_perm_diff_sig/n_cells_in_roi)*100:.1f} % have sig. different perm distributions.")

            
        # Write everything to a .txt file at the end
        if save==True:
            with open(f"{res_folder}/{model_name_string}_stats.txt", 'w') as f:
                f.write('\n'.join(results_file))
    
    

        # 3. PLOTTING
        # plot the distributions for the nicest 25 cells per model, filtered for significant cells if possible
        
        if 'time_perm_0' in df_curr_model.columns and n_sig_cells > 0:
            if len(df_curr_model_sig) > 25:
                df_strong_curr_model = df_curr_model_sig.sort_values('average_corr', ascending=False).head(25).reset_index(drop=True)
            else:
                df_strong_curr_model = df_curr_model_sig.sort_values('average_corr', ascending=False).reset_index(drop=True) 
        else:
            # if there are no significant cells after temporal permutation testing
            # subset to only plot the top 25 cells
            if len(df_curr_model) > 25: 
                df_strong_curr_model = df_curr_model.sort_values('average_corr', ascending=False).head(25).reset_index(drop=True)
            else:
               df_strong_curr_model = df_curr_model.sort_values('average_corr', ascending=False).reset_index(drop=True) 
            
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12))
        fig.suptitle(f"{curr_model}", fontsize=15, y=0.99)  # Title slightly above the top
        axs = axs.flatten()
        
        # Determine common x-axis range for centering
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
                perm_values_task = row[[f'task_perm_{i}' for i in range(n_perms)]].values
                if pd.Series(perm_values_task).isna().all() == True:
                    perm_values_task = np.zeros(1)
                else:
                    ax.hist(perm_values_task, bins=30, color=color_task_perms, alpha=0.5, label='Task perm.', edgecolor=None)
                    # Calculate one-tailed p-value
                    p_val_task = np.mean(perm_values_task >= avg_corr)
                    ax.text(0.95, 0.70, f"p_task = {p_val_task:.3f}", ha='right', va='top', transform=ax.transAxes)
                
            if 'time_perm_0' in df_strong_curr_model.columns:
                perm_values_time = row[[f'time_perm_{i}' for i in range(n_perms)]].values
                ax.hist(perm_values_time, bins=30, color=color_time_perms, alpha=0.5, label='Time perm.', edgecolor=None)
                # Calculate one-tailed p-value
                p_val_time = np.mean(perm_values_time >= avg_corr)
                ax.text(0.95, 0.95, f"p_time = {p_val_time:.3f}", ha='right', va='top', transform=ax.transAxes)
                
            # true corr   
            ax.axvline(avg_corr, color=true_val_color, linestyle='--', linewidth=2)
            
            # 0 lin e
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

        # then store these figures if on cluster. 
        if save == True:
            os.makedirs(f"{res_folder}/figures", exist_ok=True)
            plt.savefig(f"{res_folder}/figures/{curr_model}_{model_name_string}_perms_best_cells.png", dpi=300, bbox_inches='tight')
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
    
