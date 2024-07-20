#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:00:11 2024

@author: Svenja Kuchenhoff
"""

import os
import nibabel as nib
import numpy as np
import mc
import sys

regression_version = '03-4' 
RDM_version = '03-1-act'

# import pdb; pdb.set_trace() 
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

#subjects = [f"sub-{subj_no}"]
subjects = subs_list = [f'sub-{i:02}' for i in range(1, 36) if i not in (21, 29)]

print(f"Now masking results of RDM version {RDM_version} based on subj GLM {regression_version} for subj {subj_no} such that all std images overlap")


data_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives"
if os.path.isdir(data_dir):
    print("Running on laptop.")
else:
    data_dir = "/home/fs0/xpsy1114/scratch/data/derivatives"
    print(f"Running on Cluster, setting {data_dir} as data directory")


models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)
    
    

def load_niftis(subject_dirs, nifti_filename):
    niftis = []
    file_path_list = []
    for subject_dir in subject_dirs:
        file_path = os.path.join(subject_dir, nifti_filename)
        file_path_list.append(file_path)
        niftis.append(nib.load(file_path).get_fdata())
    return niftis, file_path_list

def create_common_mask(niftis):
    common_mask = np.ones_like(niftis[0], dtype=bool)
    for nii_data in niftis:
        common_mask &= np.isfinite(nii_data)  # Check for NaNs
        common_mask &= (nii_data != 0)        # Check for 0s
    return common_mask

def apply_mask(niftis, mask):
    masked_niftis = []
    for nii_data in niftis:
        masked_nii_data = nii_data * mask
        masked_niftis.append(masked_nii_data)
    return masked_niftis

def save_niftis(masked_niftis, subject_dirs, nifti_filename, output_root):
    for masked_nii_data, subject_dir in zip(masked_niftis, subject_dirs):
        nii = nib.load(os.path.join(subject_dir, nifti_filename))
        new_nii = nib.Nifti1Image(masked_nii_data, nii.affine, nii.header)
        output_dir = os.path.join(output_root, os.path.basename(subject_dir))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, nifti_filename)
        nib.save(new_nii, output_path)


# Get list of subject directories
subject_dirs = [os.path.join(data_dir, subject) for subject in subjects if os.path.isdir(os.path.join(data_dir, subject))]
if len(subject_dirs) > 35:
    raise ValueError(f"Expected 35 subject directories, but found {len(subject_dirs)}")



for model in models_I_want:
    # Load the NIfTI files from each subject directory
    niftis, nifti_path_list = load_niftis(subject_dirs, f"/func/RSA_{RDM_version}_glmbase_{regression_version}/results-standard-space/{model}_beta_std.nii.gz")
    
    print(f"currently loading niftis: {nifti_path_list}")
    # Create the common mask
    common_mask = create_common_mask(niftis)

    # Apply the mask to each NIfTI file
    masked_niftis = apply_mask(niftis, common_mask)

    # Save the masked NIfTI files to the corresponding output directories
    save_niftis(masked_niftis, f"{subject_dirs}/func/RSA_{RDM_version}_glmbase_{regression_version}/results-standard-space/{model}_beta_std_masked.nii.gz")

    print(f"Masked NIfTI files saved to {subject_dirs}")
