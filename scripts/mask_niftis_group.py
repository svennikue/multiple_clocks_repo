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
# subjects = subs_list = [f'sub-{i:02}' for i in range(1, 36) if i not in (21, 29)]
subjects = subs_list = [f'sub-{i:02}' for i in range(2,4)]

print(f"Now masking results of RDM version {RDM_version} based on subj GLM {regression_version} for subj {subj_no} such that all std images overlap")
data_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives"



if os.path.isdir(data_dir):
    print("Running on laptop.")
else:
    data_dir = "/home/fs0/xpsy1114/scratch/data/derivatives"
    print(f"Running on Cluster, setting {data_dir} as data directory")

def get_file_basenames(example_subject_dir, pattern):
    # List all files matching the pattern in the example subject directory
    matching_files = [f for f in os.listdir(example_subject_dir) if f.endswith(pattern)]
    # Extract the basenames before the '*' part
    basenames = [f.split('*')[0] for f in matching_files]
    return basenames 


# Get list of subject directories
subject_dirs = [os.path.join(data_dir, subject) for subject in subjects if os.path.isdir(os.path.join(data_dir, subject))]
if len(subject_dirs) > 35:
    raise ValueError(f"Expected 35 subject directories, but found {len(subject_dirs)}")

result_path = f"/func/RSA_{RDM_version}_glmbase_{regression_version}/results-standard-space/"
subject_dirs = [(source_path + result_path) for source_path in subject_dirs]

# define which files you are looking for
pattern = 'beta_std.nii.gz'
# Get file basenames from example subject directory
basenames = get_file_basenames(subject_dirs[0], pattern)

# then, based on these basenames, load all niftis from all subjects that have these names
niftis = mc.analyse.handle_MRI_files.load_niftis(subject_dirs, basenames)

# Create the common mask
common_mask = mc.analyse.handle_MRI_files.create_common_mask(niftis)

# Apply the mask to each NIfTI file
masked_niftis = mc.analyse.handle_MRI_files.apply_mask(niftis, common_mask)

# Save the masked NIfTI files to the corresponding output directories
# these are the keys of the dictionary!
masked_extension = 'masked'
mc.analyse.handle_MRI_files.save_niftis(masked_niftis, next(iter(masked_niftis)), masked_extension)

print(f"Masked NIfTI files saved to with extension {masked_extension}")
