#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:25:52 2023

create fMRI data RDMs


@author: xpsy1114
"""

import numpy as np
import nibabel as nib
import os
from nilearn.image import resample_to_img
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from rsatoolbox.util.searchlight import get_volume_searchlight
from rsatoolbox.data.dataset import Dataset
from nilearn.image import load_img
from nilearn.masking import compute_brain_mask
from nilearn import plotting
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.rdm import RDMs
import pickle
# from rich.progress import track
from os import path, makedirs
#import fire
from nilearn.glm.contrasts import expression_to_contrast_vector, compute_contrast
from joblib import Parallel, delayed



import mc
import matplotlib.pyplot as plt

# import pdb; pdb.set_trace()

subjects = ['sub-01']
task_halves = ['1', '2']


for sub in subjects:
    for task_half in task_halves:
        data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/preproc_clean_0{task_half}.feat"
        file = "example_func.nii.gz"
        ref_img = load_img(f"{data_dir}/{file}")
        ref_img = ref_img.get_fdata()
        x, y, z = ref_img.get_fdata().shape
        
        # Prepare searchlight positions
        mask = load_img(f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/mask_0{task_half}_mask.nii.gz")
        mask = mask.get_fdata()
        
        # ok for some reason this doenst work
        # CONTINUE HERE!!!
        # resample the 
        # resampled_image = resample_to_img(mask, ref_img)
        
        
        # this doesnt work with spyder.
        # plotting.view_img(mask, cmap='gray', title='Brain Mask')
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5)
        
        
        pe_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/glm_04_pt01+.feat/stats"
        
        # Loop through files in the folder
        image_paths = []
        for filename in os.listdir(pe_path):
            if "pe" in filename:
                # Check if "pe" is in the file name
                image_paths.append(os.path.join(pe_path, filename))  # Get the full file path
                
                # with open(file_path, 'r') as file:
                #     # Read and process the file
                #     # For example, you can print the file content
                #     print(f"Reading file: {file_path}")
                #     content = file.read()
                #     print(content)
        
        
        # loop over all images
        data = np.zeros((80, x, y, z))
        for x, im in enumerate(image_paths):
            data[x] = nib.load(im).get_fdata()
            
        # STEP 2: get RDM for each voxel
        # reshape data so we have n_observastions x n_voxels
        data_2d = data.reshape([data.shape[0], -1])
        data_2d = np.nan_to_num(data_2d) # now this is 80timepoints x 153594 voxels
        
        # only one pattern per image
        image_value = np.arange(len(image_paths))
        
        # Get RDMs
        RDM = get_searchlight_RDMs(data_2d, centers, neighbors, image_value, method='correlation')

        
        import pdb; pdb.set_trace()
        
        