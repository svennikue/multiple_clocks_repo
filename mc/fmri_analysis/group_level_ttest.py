#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:16:32 2024

@author: xpsy1114
"""

from nilearn.image import load_img
import nilearn
import nibabel as nib
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats

glm_version="02"
RSA_version="02"
palm_version="01"

group_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/group"
data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/group/group_RSA_{RSA_version}_glmbase_{glm_version}"
palm_dir = f"{group_dir}/RSA_{RSA_version}_glmbase_{glm_version}_palm_{palm_version}"


group_img = nib.load(f"{data_dir}/clocks_beta_std.nii.gz").get_fdata()

n_subj = len(group_img[0][0][0])

flat_group = group_img.reshape(-1, group_img.shape[-1]).T

p_vals = np.zeros(flat_group.shape[1])
t_vals = np.zeros(flat_group.shape[1])

#for vox in range(flat_group.shape[1]):
#    t_value, p_value = stats.ttest_1samp(flat_group[:,vox], 0)
#    p_vals[vox] = p_value
#    t_vals[vox] = t_value


# Calculate sample means, standard deviations, and size
means = np.mean(flat_group, axis=0)
stds = np.std(flat_group, axis=0, ddof=1)
n = flat_group.shape[0]

# Compute t-statistics (vectorized operation)
t_stats = means / (stds / np.sqrt(n))

# Compute p-values for two-tailed test
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-1))

# Plotting the distribution of t_values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(t_vals, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of t-values')
plt.xlabel('t-value')
plt.ylabel('Frequency')

# Plotting the distribution of p_values
plt.subplot(1, 2, 2)
plt.hist(p_vals, bins=50, color='green', alpha=0.7)
plt.title('Distribution of p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
