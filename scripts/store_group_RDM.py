#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:06:56 2025

@author: Svenja Küchenhoff


This script takes as an input the MNI standard space coordinates and the 
result files it is supposed to look at.

It then loops through all subject folders, transforms the MNI coordinates into
subjects space, loads the data RDM, and takes the respective coordinate's RDM array
and concatenates it across subjects.

It then stores a) an average version of that vector and b) the concatenated version.

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


# MNI coordinates: 
MNI_coords = [0,0,0]

# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    print("Running on laptop.")
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
    
# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]

for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        # DONT FORGET TO COMMENT THIS OUT!!!!
        regression_version = '03-4'
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")
    # data_rdm_dir = f"{data_dir}/func/data_RDMs_{RDM_version}_glmbase_{regression_version}"
    transform_mat = f"{data_dir}/func/preproc_clean_02.feat/reg/standard2example_func.mat"
    import pdb; pdb.set_trace(transform_mat) 
    # then open with np.loadtxt()
    
    # nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
    
    
    
    
    
    
    
    
    
    
    
    
    
