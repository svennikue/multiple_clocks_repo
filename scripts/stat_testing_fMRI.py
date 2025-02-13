#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:17:21 2025

@author: xpsy1114
"""

import numpy as np
from nilearn.image import load_img
import matplotlib.pyplot as plt
import mc
from scipy import stats
from statsmodels.stats.multitest import multipletests
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img

# Path to your ROI mask file (should be in the same space as your t-value image)
mask_path = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks/brainnetome_binMask30_A32sg_both.nii.gz'
# Load the mask image
roi_mask = load_img(mask_path)


data_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/RSA_03-1_glmbase_03-4_palm_03"
image_name = "masked_clocks_result.nii.gz"
pval_name = "masked_pval_clock.nii.gz"

res_img = load_img(f"{data_dir}/{image_name}")
p_val_img = load_img(f"{data_dir}/{pval_name}")


x, y, z = res_img.get_fdata().shape


masker = NiftiMasker(mask_img=roi_mask)
t_values = masker.fit_transform(res_img)
print(f"mean t-value in ROI is {np.mean(t_values)}")



p_values = masker.fit_transform(p_val_img)
print(f"mean p-value in ROI is {np.mean(p_values)}")
# Assume `p_values` is an array of p-values from your voxel-wise tests within the ROI
p_adjusted = multipletests(np.squeeze(p_values), alpha=0.05, method='fdr_bh')


# Get the corrected p-values and boolean array indicating which are significant
corrected_p_values = p_adjusted[1]
significant_voxels = p_adjusted[0]



# Perform a one-sample t-test against the hypothesis that the mean is 0
t_statistic, p_value_two_tailed = stats.ttest_1samp(np.squeeze(t_values), popmean=0)

# Check if the t-statistic supports the hypothesis that the mean is greater than 0
if t_statistic > 0:
    p_value_one_sided = p_value_two_tailed / 2
else:
    # If the t-statistic is negative, we retain the null hypothesis that the mean is <= 0
    p_value_one_sided = 1 - p_value_two_tailed / 2

# Print the results
print(f"T-statistic: {t_statistic}")
print(f"Two-tailed p-value: {p_value_two_tailed}")
print(f"One-sided p-value: {p_value_one_sided}")






from scipy.stats import t
import numpy as np

# Example configuration
num_permutations = 1000
extreme_count = 0

# Observed maximum absolute t-value
max_observed_t = np.max(np.abs(t_values))

# Generate permuted datasets and compute the maximum t-value for each
for _ in range(num_permutations):
    # Permute the signs of the t-values
    permuted_t_values = np.random.choice([-1, 1], size=len(t_values)) * t_values
    max_permuted_t = np.max(np.abs(permuted_t_values))
    
    # Count how often the permuted t-values are more extreme than the observed
    if max_permuted_t >= max_observed_t:
        extreme_count += 1

# Calculate empirical p-value
p_value = (extreme_count + 1) / (num_permutations + 1)  # +1 for the observed dataset

print(f"Empirical p-value from permutation test: {p_value:.4f}")



