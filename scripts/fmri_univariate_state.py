#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:41:09 2025

@author: Svenja Küchenhoff

this script takes the estimated reward EVs and computes a univerariate version of state.

"""
import numpy as np
import os
from nilearn.image import load_img
import matplotlib.pyplot as plt
import mc
import pickle
import sys
from datetime import date
import json
import nibabel as nib
import scipy
import nilearn
import nilearn.image

# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
      
# --- Load configuration ---
# config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_simple.json"
config_file = sys.argv[2] if len(sys.argv) > 2 else "config_univ_state.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
regression_version = config.get("regression_version")
result_name = config.get("name")
fwhm = config.get("fwhm", 5)
today_str = date.today().strftime("%d-%m-%Y")


# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]


print(f"Now running univariate state estimation based on subj GLM {regression_version} for subj {subj_no}")

states = ["A", "B", "C", "D"]
for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        # DONT FORGET TO COMMENT THIS OUT!!!!
        regression_version = '03-4'
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")

    results_dir = f"{data_dir}/func/{result_name}_glmbase_{regression_version}/results" 
    os.makedirs(results_dir, exist_ok=True)

    # preparing the mask
    mask_file = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    mask_3d = mask_file.get_fdata()  
    n_vox_total = len(mask_3d.ravel())
    mask = (mask_3d > 0).ravel().astype(bool)
    # in smoothing, the 0s around the brain will 'bleed' into the brain.
    # if I, however, smooth the mask, then divide the smoothed imaged by
    # the smoothed value, I essentially correct the value at the edge back
    # to it's initial one, getting rid of the 'bleeding'.
    # so, first smooth the mask.
    smooth_mask = nilearn.image.smooth_img(mask_file, fwhm)
    
    # loading the data EVs and creating the data matrix
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(data_dir, regression_version=regression_version)
    data_EVs_stack = np.vstack([data_EVs[label] for label in all_EV_keys])
    # mask the data
    Y = data_EVs_stack[:, mask]
    
    # preparing regressors
    regressors = np.zeros((len(states), len(data_EVs_stack)))
    state_to_row = {s: i for i, s in enumerate(states)}
    
    for j, EV in enumerate(all_EV_keys):
        split_labels = EV.split("_")
        state_letter = split_labels[-2]
        if state_letter in state_to_row:
            i = state_to_row[state_letter]
            regressors[i,j] = 1
    X = regressors.T
        
        
    # then run the regression in parallel.
    # 1) mark voxels that have any NaN across conditions
    valid_vox = ~np.any(np.isnan(Y), axis=0)   # shape (n_mask_vox,)
    

    # 2) precompute pseudoinverse of X
    X_pinv = np.linalg.pinv(X)                 # shape (n_reg, n_cond)
    
    # 3) allocate betas (n_regressors x n_mask_voxels) and fill NaNs by default
    n_reg = X.shape[1]
    n_mask_vox = Y.shape[1]
    betas_masked = np.full((n_reg, n_mask_vox), np.nan, dtype=float)
    
    # 4) run GLM only on valid voxels
    Y_valid = Y[:, valid_vox]                  # (n_cond, n_valid_vox)
    
    # # z-score X and Y
    # import pdb; pdb.set_trace()
    # # Do not zscore this because they sum to a constant and therefore it would end up being singular
    # #X = scipy.stats.zscore(X, axis= nan_policy='raise')
    # Y = scipy.stats.zscore(Y_valid, axis=0, nan_policy='raise')
    
    betas_masked[:, valid_vox] = X_pinv @ Y_valid
 
    # then put betas back in the none-masked shape
    betas_full_flat = np.full((n_reg, n_vox_total), np.nan)
    betas_full_flat[:, mask] = betas_masked
    # and in the complete volume
    beta_vols = betas_full_flat.reshape(n_reg, *mask_3d.shape)

    # then save everything
    affine = mask_file.affine  # from your original data
    header = mask_file.header
    
    # also store the smoothed version.
    smooth_dir = os.path.join(f"{data_dir}/func/state_univ_glmbase_{regression_version}/smoothed")
    if not os.path.exists(smooth_dir):
        os.makedirs(smooth_dir, exist_ok=True)
    print(f"now smoothing the RDM and saving it here: {smooth_dir}")
    
    for i, state in enumerate(states):
        beta_img = nib.Nifti1Image(beta_vols[i], affine=affine, header=header)
        out_name = f"{sub}_state_{state}_univ_glmbase_{regression_version}.nii.gz"
        beta_img.to_filename(os.path.join(results_dir, out_name))
        
        print("Smoothing:", out_name)
        nifti_smooth = nilearn.image.smooth_img(beta_img, fwhm)
        # then divide by smoothed mask to get rid of bleeding
        np_nifti_smooth = nifti_smooth.get_fdata() / smooth_mask.get_fdata()
        np_nifti_smooth[mask_file.get_fdata() == 0.] = 0
        # save with a simple modified name
        out_file = os.path.join(smooth_dir, f"smooth_fwhm{fwhm}_{out_name}")
    
        nifti_smooth = nilearn.image.new_img_like(beta_img, np_nifti_smooth)
        nifti_smooth.to_filename(out_file)
        

    # --- SETTINGS SUMMARY (per subject) ---
    summary = {
        "subject": sub,
        "regression_version": regression_version,
        "data_dir": data_dir,
        "results_dir": results_dir
    }
    
    print("\n=== SETTINGS SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    
    # Save a copy alongside results for provenance
    with open(os.path.join(results_dir, f"{sub}_settings_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"(Saved summary → {os.path.join(results_dir, f'{sub}_settings_summary.json')})\n")
            





