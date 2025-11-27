#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:27:33 2025

@author: Svenja Küchenhoff 

This script is taking as input 
"""


import sys
import os
from glob import glob
import json
import nilearn
import nilearn.image
import nibabel as nib

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
    
# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]


# --- Load configuration ---
# config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_simple.json"
config_file = sys.argv[2] if len(sys.argv) > 2 else "smooth5_config.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)
    
regression_version = config.get("regression_version")
fwhm = config.get("fwhm", 5)
name_RSA = config.get("name_of_RSA")

# NOTE: change the today string into pattern completion!
#RDM_version = f"{name_RSA}_{today_str}"
RSA_pattern = f"{name_RSA}_*" #ignore the date string


for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")
    
    # find all matching RDM folders (date varies)
    rsa_dirs = glob(f"{data_dir}/func/RSA_{RSA_pattern}_glmbase_{regression_version}")
    mask = nilearn.image.load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    
    print(f"these are the rsa_dirs we found: {rsa_dirs}")
    for RSA_dir in rsa_dirs:
        smooth_dir = os.path.join(RSA_dir, "smoothed")
        if not os.path.exists(smooth_dir):
            os.makedirs(smooth_dir, exist_ok=True)
        print(f"now smoothing the RDM and saving it here: {smooth_dir}")

        # get all files ending with *t_val.nii.gz
        nifti_files = glob(os.path.join(f"{RSA_dir}/results", "*t_val.nii.gz"))
        print(f"found the following files to smooth: {nifti_files}")
        # in smoothing, the 0s around the brain will 'bleed' into the brain.
        # if I, however, smooth the mask, then divide the smoothed imaged by
        # the smoothed value, I essentially correct the value at the edge back
        # to it's initial one, getting rid of the 'bleeding'.
        
        # so, first smooth the mask.
        smooth_mask = nilearn.image.smooth_img(mask, fwhm)
        
        for nifti_file in nifti_files:
            # import pdb; pdb.set_trace() 
            # base name if you need it
            base_name = os.path.basename(nifti_file)
            print("Smoothing:", base_name)
            
            nifti = nilearn.image.load_img(nifti_file)
            nifti_smooth = nilearn.image.smooth_img(nifti, fwhm)
            # then divide by smoothed mask to get rid of bleeding
            np_nifti_smooth = nifti_smooth.get_fdata() / smooth_mask.get_fdata()
            np_nifti_smooth[mask.get_fdata() == 0.] = 0
            # save with a simple modified name
            out_file = os.path.join(smooth_dir, f"smooth_fwhm{fwhm}_{base_name}")
            
            
            nifti_smooth = nilearn.image.new_img_like(nifti, np_nifti_smooth)
            nifti_smooth.to_filename(out_file)
                

