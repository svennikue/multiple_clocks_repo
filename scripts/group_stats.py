#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:42:28 2024

this is to get a closer look at the data at a group level, specifically the
data RDMs

@author: xpsy1114
"""
import mc
import os
from nilearn.image import load_img
import pickle
import numpy as np

plot_model_RDMs = True
regression_version = '03-4' 
RDM_version = '05'

subjects = subs_list = [f'sub-{i:02}' for i in range(1, 11) if i not in (21, 29)]


group_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group"
if not os.path.isdir(group_dir):
    os.makedirs(group_dir)
    
subj_data_RDM_dir = {}

if not os.path.isdir(f"{group_dir}/subj_data_RDM_dir.pkl"):
    for sub in subjects:
        data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
        if os.path.isdir(data_dir):
            print(f"Running on laptop, for {sub}")
        else:
            data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
            print(f"Running on Cluster, setting {data_dir} as data directory, for subject {sub}")
        
        mean_data_RDM_dir = f"{data_dir}/func/data_RDM_glmbase_{regression_version}"
        data_RDM_subj = load_img(f"{mean_data_RDM_dir}/data_RDM_std.nii.gz")
        subj_data_RDM_dir[sub] = data_RDM_subj.get_fdata()  
else:
    with open(f"{group_dir}/subj_data_RDM_dir.pkl", 'rb') as file:
        pickle.dump(subj_data_RDM_dir, file) 

with open(f"{group_dir}/subj_data_RDM_dir.pkl", 'wb') as file:
    pickle.dump(subj_data_RDM_dir, file)
        
# next, insert plotting functions -> find your effects of interest of course!
# maybe also load the data RDM as a comparison?
    # ACC [54, 63, 41]
    # visual cortex [72, 17, 9]
    # hippocampus [43, 50, 17]
    
# maybe use this
# for one example subject

RDM_dir = f"{data_dir}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
condition_names = mc.analyse.analyse_MRI_behav.get_conditions_list(RDM_dir)
# might also be useful to in the end actually plot the figures of each model RDM
# as well as each group avg data RDM!

standard_image = load_img('/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks/MNI152_T1_2mm_brain.nii.gz')
MNI_coords = [48, 87, 46] #ACC [54, 63, 41]
mc.plotting.deep_data_plt.plot_group_avg_RDM_by_coord(subj_data_RDM_dir, MNI_coords, condition_names)

MNI_coords = [59, 54, 26] #hippocampus
mc.plotting.deep_data_plt.plot_group_avg_RDM_by_coord(subj_data_RDM_dir, MNI_coords, condition_names)

MNI_coords = [53, 20, 47] #visual cortex
mc.plotting.deep_data_plt.plot_group_avg_RDM_by_coord(subj_data_RDM_dir, MNI_coords, condition_names)



avg_model_RDM = {}
for i, sub in enumerate(subjects):
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    RDM_dir = f"{data_dir}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
    models_I_want = mc.analyse.analyse_MRI_behav.models_I_want(RDM_version)
    models_I_want = mc.analyse.analyse_MRI_behav.models_I_want(RDM_version)
    if 'state_masked' in models_I_want:
        models_I_want.remove('state_masked')
    RSM_dict_betw_TH = {}
    for model in models_I_want:
        RSM_dict_betw_TH[model]= np.load(os.path.join(RDM_dir, f"RSM_{model}_{sub}_fmri_both_halves.npy"))
        if i == 0:
            avg_model_RDM[model] = RSM_dict_betw_TH[model]
        else:
            avg_model_RDM[model] = (avg_model_RDM[model] + RSM_dict_betw_TH[model])/2
 
        
if plot_model_RDMs:
    if not os.path.exists(RDM_dir):
        os.makedirs(RDM_dir)
    mc.simulation.RDMs.plot_RDMs(avg_model_RDM, len(avg_model_RDM[model]), string_for_ticks = condition_names['1'])

