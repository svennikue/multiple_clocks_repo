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

regression_version = ['03-4-e' , '03-4-l'] 
RDM_version = '03-1' 


# import pdb; pdb.set_trace() 
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

subjects = [f"sub-{subj_no}"]

repeats = ['e', 'l']
ROIs = ['OFC_a11', 'hippocampus', 'ACC' ]


# first, per subject, go through the folders and load in a list of all files that end with *beta_std.nii.gz
# then, per file, load the file, and mask it into hippocampus, mpfc and OFC
# names of masks: OFC_a11_mask hippocampus_mask.nii.gz ACC_mask.nii.gz
# per ROI, select the peak value
# store value for early and late in a table per subject
# continue loop

# start a dictionary
for sub in subjects:
    # add an entry to dictionary per subject
    for glm_by_repeat in regression_version:
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
            curr_model = curr_model_basename[:-len('beta_std.nii.gz')]
            # add an entry to dictionary per curr_model
            
            for ROI in ROIs:
                curr_mask = load_img(f"{mask_dir}/{ROI}_mask.nii.gz")
                curr_mask = curr_mask.get_fdata()  
                masked_model = brain_map * curr_mask
                
                # save this into the dictionary per subject.
                peak_model_x_mask = np.max(masked_model)
                

# then, start to plot.
