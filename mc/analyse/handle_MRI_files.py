#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:10:34 2024
any function that helps to deal with MRI files

@author: Svenja Küchenhoff
"""
import nibabel as nib
import numpy as np
import os


def apply_mask(niftis, mask):
    masked_niftis = {}
    # import pdb; pdb.set_trace() 
    for nii_data in niftis:
        #masked_niftis[nii_data] = niftis[nii_data] * mask
        masked_niftis[nii_data] = np.where(mask, niftis[nii_data], 0)
    return masked_niftis

def create_common_mask(niftis):
    # first create mask in the shape of the images
    common_mask = np.ones_like(niftis[next(iter(niftis))], dtype=bool)
    # then iteratively check if there are any nans or 0s and if so, add to mask
    for nii_data in niftis:
        common_mask &= np.isfinite(niftis[nii_data])  # Check for NaNs
        common_mask &= (niftis[nii_data] != 0)        # Check for 0s
    return common_mask

def save_niftis(nifti_dict, ref_nifti_path, output_file_name_extension, alternative_paths_to_save = None):
    ref_img = nib.load(ref_nifti_path)
    affine_matrix = ref_img.affine
    x,y,z = ref_img.shape    
    # make sure that dict keys are also new nifti paths!!!
    for nifti_path in nifti_dict:
        split_at = nifti_path.rfind('.nii.gz')
        if split_at != -1:
            out_path_file = nifti_path[:split_at] + '_' + output_file_name_extension + '.nii.gz'
        nifti_out = nib.Nifti1Image(nifti_dict[nifti_path], affine_matrix)
        nib.save(nifti_out, out_path_file)
        
        
def load_niftis(subject_dirs, nifti_filename_list):
    niftis = {}
    #niftis = []
    #file_path_list = []
    for subject in subject_dirs:
        for nifti_filename in nifti_filename_list:
            file_path = subject + nifti_filename
            if os.path.isfile(file_path):
                niftis[file_path] = nib.load(file_path).get_fdata()
                #file_path_list.append(file_path)
                #niftis.append(nib.load(file_path).get_fdata())
            else:
                print(f"Careful! {file_path} didn't exist! skipping this one")
    return niftis
