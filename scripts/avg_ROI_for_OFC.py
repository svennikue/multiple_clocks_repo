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


import pdb; pdb.set_trace()

regression_version = '03-4' 
RSA_version = '03' 

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

subjects = [f"sub-{subj_no}"]


ROIS = ['OFC', 'mpfc']
fmriplotting = False
fmri_save = True

results_dict = {}

for subj_i, sub in enumerate(subjects):
    subj_dict = {}
    result_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/RSA_{RSA_version}_glmbase_{regression_version}/results-standard-space/masked"
    if os.path.isdir(result_dir):
        print("Running on laptop.")
    else:
        result_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/func/RSA_{RSA_version}_glmbase_{regression_version}/results-standard-space/masked"
        print(f"Running on Cluster, setting {result_dir} as data directory")
    
    # Get a list of files in the directory
    files = [file for file in result_dir.iterdir() if file.is_file()]
    
    for file in files:
        if subj_i == 0:
            results_dict[file] = []
        subj_dict[file] = load_img(f"{result_dir}/{file}")
        subj_dict[file] = subj_dict[file].get_fdata() 
        results_dict[file].append(np.nanmean(subj_dict[file]))





