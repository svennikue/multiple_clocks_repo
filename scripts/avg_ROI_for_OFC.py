#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:23:04 2024

create average in ROIs for the OFC conference poster

@author: xpsy1114
"""

import numpy as np
import nibabel as nib
import os
from nilearn.image import load_img
import matplotlib.pyplot as plt
import mc
import pickle
import sys
from pathlib import Path
import matplotlib

# import pdb; pdb.set_trace()

regression_version = '03-4' 
RSA_version = '03-1' 

out_dir = "/home/fs0/xpsy1114/scratch/data/derivatives/group/figures"


if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

# subjects = [f"sub-{subj_no}"]

# just for now
subjects = sub_list = ['sub-{0:02}'.format(i) for i in range(1, 34) if i not in [5, 21]]

ROIS = ['OFC', 'mpfc']
fmriplotting = False
fmri_save = True

results_dict = {}

for subj_i, sub in enumerate(subjects):
    subj_dict = {}
    result_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/RSA_{RSA_version}_glmbase_{regression_version}/results-standard-space/masked_beta_small"
    if os.path.isdir(result_dir):
        print("Running on laptop.")
    else:
        result_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/func/RSA_{RSA_version}_glmbase_{regression_version}/results-standard-space/masked_beta_small"
        print(f"Running on Cluster, setting {result_dir} as data directory")
    
    result_dir_path = Path(result_dir)
    # Get a list of files in the directory
    files = [file for file in result_dir_path.iterdir() if file.is_file()]
    
    for file_obj in files:
        file = file_obj.name
        if subj_i == 0:
            results_dict[file] = []
        subj_dict[file] = load_img(f"{result_dir}/{file}")
        subj_dict[file] = subj_dict[file].get_fdata() 
        results_dict[file].append(np.nanmean(subj_dict[file]))
        # debug
        #if file == 'ONE-FUT-REW_combo_split-clock_t_val_std.nii.gz_mpfc_mask.nii.gz':
        #    print(f"subj {sub} is in one mpfc {100 * np.nanmean(subj_dict[file])}")
        
order = ['CURR', 'ONE', 'TWO', 'THREE']
def custom_sort_key(filename):
    for index, prefix in enumerate(order):
        if filename.startswith(prefix):
            return index
    return len(order)  # Return a higher index if prefix not found (just in case)


# Set up the plot with 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 10))
fig.suptitle('Betas per subject, reward and ROI')

# Colors for scatter plots
#colors = ['#00BFFF', '#1E90FF', '#20B2AA']  # Deep Sky Blue, Dodger Blue, Light Sea Green
# colors = ['#008080', '#1E90FF', '#5F9EA0', '#00008B']  # Teal, Dark Cyan, Cadet Blue, dark night blue
# colors = ['#042E33','#04403D', '#035E4C', '#23877C' ] # shades of blue
colors = ['#FFC222','#D3A7B5', '#AF6280', '#921D56' ]# from the poster
matplotlib.rcParams['font.size'] = 30  # Increasing font size; adjust as needed


# Process each subplot
for index, key in enumerate(['mpfc_mask.nii.gz', 'ofc_mask.nii.gz']):
    ax = axes[index]
    groups = [k for k in results_dict if k.endswith(key)]
    sorted_groups = sorted(groups, key=custom_sort_key)
    for i, group in enumerate(sorted_groups):
        # Scatter plot
        y = results_dict[group]
        x = np.random.normal(i + 1, 0.1, size=len(y))  # Add jitter to x-axis for clarity
        ax.scatter(x, y, s=80, alpha=0.6, color=colors[i])
        
        # Boxplot
        ax.boxplot(y, positions=[i + 1], widths=0.7, showcaps=True, boxprops=dict(color=colors[i]),
                   whiskerprops=dict(color=colors[i]), medianprops=dict(color='black'),
                   capprops=dict(color=colors[i]), flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
                   meanprops=dict(style='.-', color='black', linewidth=3))
    if key == 'mpfc_mask.nii.gz':
        ax.set_title('MPFC')
    elif key == 'ofc_mask.nii.gz':
        ax.set_title('OFC')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([k.split('_')[0] for k in sorted_groups], rotation=45, ha='right')  # Rotate x-axis labels to 45 degrees
    ax.set_xlabel('Split Clock Elements')
    ax.set_ylabel('Betas')
    ax.axhline(0, color='grey', ls='dashed')
    

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to fit the overall title

# Save the figure as PNG and TIFF
plt.savefig(os.path.join(out_dir,'split-clock-OFC-and-MPFC-rsa03-1-GLM-03-4.png'), format='png', dpi=300)  # Save as PNG with high resolution
plt.savefig(os.path.join(out_dir,'split-clock-OFC-and-MPFC-rsa03-1-GLM-03-4.tiff'), format='tiff', dpi=300)  # Save as TIFF with high resolution


plt.show()
