#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:20:37 2026

@author: Svenja Küchenhoff

This is an attempt of identifying a gradient across subjects

"""

import os
import nilearn
import nilearn.image
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group"
if os.path.isdir(source_dir):
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch/data/derivatives/group"
    print(f"Running on Cluster, setting {source_dir} as data directory")
    
    
result_dir = f"{source_dir}/group_RSA_which-fut-isn-DSR_glmbase_all-paths-fixed_stickrews_split-buttons_cropped" 

files_temp_order = {"curr":'/CURRENT_QUARTER-split_quarters_DSR_beta.std.nii.gz',
                    "next":'NEXT_QUARTER-split_quarters_DSR_beta.std.nii.gz',
                    "next2": 'NEXT2_QUARTER-split_quarters_DSR_beta.std.nii.gz',
                    "next3": 'NEXT_QUARTER-split_quarters_DSR_beta.std.nii.gz'}

niftis = {}
for file_toggle in files_temp_order:
    niftis[file_toggle] = nilearn.image.load_img(f"{result_dir}/{files_temp_order[file_toggle]}")


peak_coordinates = {}
for condition, img in niftis.items():
    
    data = img.get_fdata()
    n_subjects = data.shape[3]
    
    peak_coordinates[condition] = {}
    
    for subj in range(n_subjects):
        
        subj_data = data[:, :, :, subj]
        peak_index = np.unravel_index(np.argmax(subj_data), subj_data.shape)
        
        mni_coord = nib.affines.apply_affine(img.affine, peak_index)
        
        peak_coordinates[condition][f"subj_{subj:02d}"] = {
            "voxel": peak_index,
            "mni": mni_coord
        }

print(peak_coordinates)


# choose axis
axis_name = "z"  
axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]

axis_projection = {}

for condition, subjects in peak_coordinates.items():
    axis_projection[condition] = []
    
    for subj in sorted(subjects.keys()):
        coord = subjects[subj]["mni"]  # using MNI coords
        axis_projection[condition].append(coord[axis_index])

# convert to arrays
for condition in axis_projection:
    axis_projection[condition] = np.array(axis_projection[condition])
    

conditions = ["curr", "next", "next2", "next3"]
n_subjects = len(axis_projection["curr"])

plt.figure()

for subj in range(n_subjects):
    values = [axis_projection[cond][subj] for cond in conditions]
    plt.plot(conditions, values)

plt.xlabel("Condition")
plt.ylabel(f"{axis_name}-coordinate (MNI)")
plt.title(f"Peak progression along {axis_name}-axis")
plt.show()


monotonic = 0

for subj in range(n_subjects):
    values = [axis_projection[cond][subj] for cond in conditions]
    if values == sorted(values):
        monotonic += 1

print("Subjects with perfect ordering:", monotonic, "/", n_subjects)



# ---------- NICE MEAN ORDERING PLOT ----------

conditions = ["curr", "next", "next2", "next3"]
x_positions = np.arange(len(conditions))
n_subjects = len(axis_projection["curr"])

# Stack data into matrix: subjects x conditions
data_matrix = np.vstack([
    axis_projection[cond] for cond in conditions
]).T   # shape: (subjects, conditions)

# Compute mean and SEM
mean_vals = data_matrix.mean(axis=0)
sem_vals = data_matrix.std(axis=0, ddof=1) / np.sqrt(n_subjects)

plt.figure(figsize=(8, 6))

# Plot individual subjects (light gray)
for subj in range(n_subjects):
    plt.plot(
        x_positions,
        data_matrix[subj, :],
        color='lightgray',
        linewidth=1,
        alpha=0.6
    )

# Plot mean trajectory (bold)
plt.errorbar(
    x_positions,
    mean_vals,
    yerr=sem_vals,
    marker='o',
    markersize=8,
    linewidth=3,
    capsize=5
)

plt.xticks(x_positions, conditions, fontsize=12)
plt.ylabel(f"{axis_name}-coordinate (MNI mm)", fontsize=13)
plt.xlabel("Condition", fontsize=13)
plt.title(f"Mean Peak Progression Along {axis_name}-Axis", fontsize=14)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



from scipy.stats import ttest_1samp

slopes = []
for subj in range(n_subjects):
    slope = np.polyfit(x_positions, data_matrix[subj, :], 1)[0]
    slopes.append(slope)

t_stat, p_val = ttest_1samp(slopes, 0)

print("Group-level slope test p-value:", p_val)
plt.title(f"Mean Peak Progression (p = {p_val:.3f})")