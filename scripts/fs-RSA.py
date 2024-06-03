#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:49:09 2024

surface based RSA

@author: xpsy1114
"""

from tqdm import tqdm
import numpy as np
import nibabel as nib
import os
import rsatoolbox.rdm as rsr
import rsatoolbox
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
from nilearn.image import load_img
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc
import pickle
import sys
import random

regression_version = '03' 
RDM_version = '02' 


# import pdb; pdb.set_trace() 
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '14'

subjects = [f"sub-{subj_no}"]
#subjects = ['sub-01']

load_old = False
visualise_RDMs = False


task_halves = ['1', '2']

print(f"Now running RSA for RDM version {RDM_version} based on subj GLM {regression_version} for subj {subj_no}")


models_I_want = mc.analyse.analyse_MRI_behav.models_I_want(RDM_version)


if regression_version in ['03', '04','03-99', '03-999', '03-9999', '03-l', '03-e']:
    no_RDM_conditions = 40 # only including rewards or only paths

for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data"
        print(f"Running on Cluster, setting {data_dir} as data directory")
    if RDM_version in ['03-999']:
        RDM_dir = f"{data_dir}/derivatives/{sub}/beh/RDMs_03_glmbase_{regression_version}"
    else:
        RDM_dir = f"{data_dir}/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
    if not os.path.exists(RDM_dir):
        os.makedirs(RDM_dir)  
    results_dir = f"{data_dir}/derivatives/{sub}/func/RSA_{RDM_version}_glmbase_{regression_version}"   
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.makedirs(f"{results_dir}/results")
    # results_dir = f"{data_dir}/derivatives/{sub}/func/RSA_{RDM_version}_glmbase_{regression_version}/results"  
    fs_dir = f"{data_dir}/freesurfer/{sub}"
    results_dir = f"{fs_dir}/RSA_{RDM_version}_glmbase_{regression_version}/results"
    
    reading_in_EVs_dict = {task_half: "" for task_half in ['01', '02']}
    for task_half in ['01', '02']:
        pe_path = f"{fs_dir}/glm_{regression_version}_pt{task_half}.feat"
        with open(f"{data_dir}/derivatives/{sub}/func/EVs_{regression_version}_pt{task_half}/task-to-EV.txt", 'r') as file:
            for line in file:
                index, name_ev = line.strip().split(' ', 1)
                name = name_ev.replace('ev_', '')
                reading_in_EVs_dict[task_half][f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
#                reading_in_EVs_dict_01[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_01, f"pe{int(index)+1}.nii.gz")
                
    

    surface = nib.freesurfer.io.read_geometry(f"{pe_path}/rh.pe40.mgh")
    # surface = nib.load(f"{fs_dir}/test-vol2surf.gii")
    # surface = nib.load(f"{pe_path}/rh.pe40.mgh")
    
    

    # example_func from half 1, as this is where the data is corrected to.
    # ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    # load the file which defines the order of the model RDMs, and hence the data RDMs
    # with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'rb') as file:
    #     sorted_keys = pickle.load(file)
    # with open(f"{RDM_dir}/sorted_regs.pkl", 'rb') as file:
    #     reg_keys = pickle.load(file)

  
    pe_path_01 = f"{data_dir}/func/glm_{regression_version}_pt01.feat/stats"
    reading_in_EVs_dict_01 = {}   
    with open(f"{data_dir}/func/EVs_{regression_version}_pt01/task-to-EV.txt", 'r') as file:
        for line in file:
            index, name_ev = line.strip().split(' ', 1)
            name = name_ev.replace('ev_', '')
            reading_in_EVs_dict_01[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_01, f"pe{int(index)+1}.nii.gz")
            
    pe_path_02 = f"{data_dir}/func/glm_{regression_version}_pt02.feat/stats"     
    reading_in_EVs_dict_02 = {}
    with open(f"{data_dir}/func/EVs_{regression_version}_pt02/task-to-EV.txt", 'r') as file:
        for line in file:
            index, name_ev = line.strip().split(' ', 1)
            name = name_ev.replace('ev_', '')
            reading_in_EVs_dict_02[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_02, f"pe{int(index)+1}.nii.gz")
    
    
    

    
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    # mask = load_img(f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz")
    # mask = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    # mask = mask.get_fdata()  