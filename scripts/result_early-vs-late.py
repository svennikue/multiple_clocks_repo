#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:15:49 2024

this script loads the early vs late repeats results and plots them for each
model.

@author: xpsy1114
"""
import sys
import mc
import os
import glob
from nilearn.image import load_img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress

regression_version = ['03-4-e' , '03-4-l'] 
RDM_version = '03-1' 


# import pdb; pdb.set_trace() 
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

# subjects = [f"sub-{subj_no}"]
subjects = ['sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18','sub-19', 'sub-20', 'sub-22', 'sub-23','sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28', 'sub-29', 'sub-30', 'sub-31', 'sub-32', 'sub-33', 'sub-34']
#subjects = ['sub-01']    

repeats = ['e', 'l']
ROIs = ['OFC_A11', 'hippocampus', 'ACC' ]


# first, per subject, go through the folders and load in a list of all files that end with *beta_std.nii.gz
# then, per file, load the file, and mask it into hippocampus, mpfc and OFC
# names of masks: OFC_a11_mask hippocampus_mask.nii.gz ACC_mask.nii.gz
# per ROI, select the peak value
# store value for early and late in a table per subject
# continue loop

# start a dictionary
peak_vals_dict = {}
all_models = []
for sub in subjects:
    # add an entry to dictionary per subject
    if sub not in peak_vals_dict:
        peak_vals_dict[sub] = {}
        
    for glm_by_repeat in regression_version:
        # add an entry to dicitionary for early and late respectively
        if glm_by_repeat not in peak_vals_dict[sub]:
            peak_vals_dict[sub][glm_by_repeat] = {}
        result_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/RSA_{RDM_version}_glmbase_{glm_by_repeat}/results-standard-space"
        mask_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks"
        if os.path.isdir(result_dir):
            print("Running on laptop.")
        else:
            result_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/func/RSA_{RDM_version}_glmbase_{glm_by_repeat}/results-standard-space"
            mask_dir = f"/home/fs0/xpsy1114/scratch/data/masks"
            print(f"Running on Cluster, now setting {result_dir} as result directory")
        
        beta_std_file_list = glob.glob(os.path.join(result_dir, '*beta_std.nii.gz'))
        
        for model_result in beta_std_file_list:
            brain_map = load_img(model_result)
            brain_map = brain_map.get_fdata()
            
            curr_model_basename = os.path.basename(model_result)
            curr_model = curr_model_basename[:-len('_beta_std.nii.gz')]
            all_models.append(curr_model)
            # add an entry to dictionary per curr_model
            if curr_model not in peak_vals_dict[sub][glm_by_repeat]:
                peak_vals_dict[sub][glm_by_repeat][curr_model] = {}
                
            for ROI in ROIs:
                curr_mask = load_img(f"{mask_dir}/{ROI}_mask_2mm.nii.gz")
                curr_mask = curr_mask.get_fdata()  
                masked_model = brain_map * curr_mask
                masked_model = masked_model.flatten()
                # save this into the dictionary per subject.
                peak_vals_dict[sub][glm_by_repeat][curr_model][ROI] = np.nanmax(masked_model)
                
                

# then, start to plot.


# Define models of interest and ROIs
models_of_interest = list(set(all_models))
# models_of_interest = ['CLOCKrewloc-combo-clrw-loc-ph-st', 'ONE-FUT-REW_combo_split-clock', 'TWO-FUT-REW_combo_split-clock']


# Generate a list of colors using a colormap
cmap = cm.get_cmap('viridis', len(subjects))
colors = [cmap(i) for i in range(len(subjects))]

# Find overall min and max values for axis limits
# all_values = []
# for subject in subjects:
#     for model in models_of_interest:
#         for roi in ROIs:
#             all_values.append(peak_vals_dict[subject]['03-4-e'][model][roi])
#             all_values.append(peak_vals_dict[subject]['03-4-l'][model][roi])
# min_val = min(all_values)
# max_val = max(all_values)

# Create plots
for model in models_of_interest:
    fig, axes = plt.subplots(1, len(ROIs), figsize=(15, 5), sharex=True, sharey=True)
    fig.suptitle(f'Early vs Late Repeats for {model}', fontsize=24)

    for idx, roi in enumerate(ROIs):
        ax = axes[idx]
        ax.set_title(f'{roi}', fontsize =22)
        ax.set_xlabel('Late Repeats', fontsize=20)
        ax.set_ylabel('Early Repeats', fontsize=20)
        
        # Collect all values for this subplot to determine axis limits
        early_values = []
        late_values = []
        for subject in subjects:
            early_values.append(peak_vals_dict[subject]['03-4-e'][model][roi])
            late_values.append(peak_vals_dict[subject]['03-4-l'][model][roi])
        
        # Determine min and max values for the current ROI
        min_val = min(early_values + late_values)
        max_val = max(early_values + late_values)
        
        
        # Fill background
        x_vals = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_vals, x_vals, max_val, color='#FFEBE5', zorder=0)  # Bright orange for y > x
        ax.fill_between(x_vals, min_val, x_vals, color='#E5F4FF', zorder=0)  # Bright blue for x > y

        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(early_values, late_values)
        line_x = np.linspace(min_val, max_val, 100)
        line_y = slope*line_x + intercept
        
        # Plot regression line
        ax.plot(line_x, line_y, color='darkgrey', linestyle='-', zorder=1)
        
        
        for subject_idx, (subject, regressions) in enumerate(peak_vals_dict.items()):
            early_value = regressions['03-4-e'][model][roi]
            late_value = regressions['03-4-l'][model][roi]
            ax.scatter(late_value, early_value, color=colors[subject_idx % len(colors)], label=subject)
        

        
        # Add text for correlation coefficient
        ax.text(0.05, 0.95, f'r={r_value:.2f}', transform=ax.transAxes, fontsize=22, verticalalignment='top')

            
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        
     # Create a single legend for the whole figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')


    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    plt.show()