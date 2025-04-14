#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 11th of April 2025

Running the RSA [mainly] wihtout RSA toolbox.

RDM settings (creating the representations):
    03-1 -> modelling only reward rings + split ‘clocks model’ = just rotating the reward location around. 

GLM ('regression') settings (creating the 'bins'):
    03-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)
    
@author: Svenja Küchenhoff, 2025
"""


from tqdm import tqdm
import numpy as np
import os
from nilearn.image import load_img
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc
import pickle
import sys


regression_version = '03-4' 
RDM_version = '03-1' #'02-act'

smoothing = True
fwhm = 5
load_old = False
visualise_RDMs = False


# import pdb; pdb.set_trace() 
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

subjects = [f"sub-{subj_no}"]
# subjects = subs_list = [f'sub-{i:02}' for i in range(1, 36) if i not in (21, 29)]


print(f"Now running RSA for RDM version {RDM_version} based on subj GLM {regression_version} for subj {subj_no}")

# based on RDM version, what will be modelled?
models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)
# based on GLM, how many regressors/RDM conditions will there be?
no_RDM_conditions = mc.analyse.analyse_MRI_behav.determine_number_of_conditions(regression_version, RDM_version)


for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")
        
    if RDM_version in ['03-999']:
        RDM_dir = f"{data_dir}/beh/RDMs_03_glmbase_{regression_version}"
    else:
        RDM_dir = f"{data_dir}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"

    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}" 
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}_smooth{fwhm}/results" 
    
    data_rdm_dir = f"{data_dir}/func/data_RDM_glmbase_{regression_version}"

    os.makedirs(RDM_dir, exist_ok=True) 
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_rdm_dir, exist_ok=True)  
    
    
    # if os.path.exists(results_dir):
    #     # move pre-existing files into a different folder.
    #     mc.analyse.analyse_MRI_behav.move_files_to_subfolder(results_dir)
    
    
    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    mask_file = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    mask = mask_file.get_fdata()  
    # save this file to save time
    if load_old:
        with open(f"{RDM_dir}/searchlight_centers.pkl", 'rb') as file:
            centers = pickle.load(file)
        with open(f"{RDM_dir}/searchlight_neighbors.pkl", 'rb') as file:
            neighbors = pickle.load(file)
    else:
        # creating the searchlights
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5) # Found 175.483 searchlights
        # if I use the grey matter mask, then I find 144.905 searchlights
        # save this structure
        with open(f"{RDM_dir}/searchlight_centers.pkl", 'wb') as file:
            pickle.dump(centers, file)
        with open(f"{RDM_dir}/searchlight_neighbors.pkl", 'wb') as file:
            pickle.dump(neighbors, file)   


    #
    # Step 2: loading and computing the data RDMs
    #
    #prepare the data RDM file. 
    data_RDM_file_2d= mc.analyse.analyse_MRI_behav.read_in_RDM_conds(regression_version, RDM_version, data_dir, RDM_dir, no_RDM_conditions, sort_as = 'dict-two-halves')
    condition_names= mc.analyse.analyse_MRI_behav.get_conditions_list(RDM_dir)

    if not os.path.exists(f"{data_rdm_dir}/data_RDM.npy"):
        if RDM_version in ['01', '01-1']:
            import pdb; pdb.set_trace()
            # this will need to be something like 
            # data_RDMs = mc.analyse.my_RSA.get_RDM_per_searchlight(data_RDM_file_2d, centers, neighbors, method = 'corr')
            # which is a correlation option that does not exist yet. Write it!!
            # mc.analyse.analyse_MRI_behav.save_data_RDM_as_nifti(data_RDMs, data_rdm_dir, "data_RDM.nii.gz", ref_img)
        else:
            data_RDMs = mc.analyse.my_RSA.get_RDM_per_searchlight(data_RDM_file_2d, centers, neighbors, method = 'crosscorr')
            mc.analyse.handle_MRI_files.save_data_RDM_as_nifti(data_RDMs, data_rdm_dir, "data_RDM.nii.gz", ref_img, centers)
    else:
        data_RDMs = np.load(f"{data_rdm_dir}/data_RDM.npy")

    if smoothing == True:
        if not os.path.exists(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}-pkl"):
            path_to_save_smooth = f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.nii.gz"
            print(f"now smoothing the RDM and saving it here: {path_to_save_smooth}")
            data_RDMs = mc.analyse.handle_MRI_files.smooth_RDMs(data_RDMs, ref_img, fwhm,use_rsa_toolbox = False, path_to_save=path_to_save_smooth,centers=centers)
        else:
            data_RDMs = np.load(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.npy")
           
    if visualise_RDMs == True:
        # ACC [54, 63, 41]
        mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [54, 63, 41], ref_img, condition_names)
        mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDMs, [54, 63, 41], ref_img, condition_names, centers = centers, no_rsa_toolbox=True)
        
        # visual cortex [72, 17, 9]
        mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [72, 17, 9], ref_img, condition_names)
        mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDMs, [72, 17, 9], ref_img, condition_names, centers = centers, no_rsa_toolbox=True)
        
        # hippocampus [43, 50, 17]
        mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [43, 50, 17], ref_img, condition_names)
        mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDMs, [43, 50, 17], ref_img, condition_names, centers = centers, no_rsa_toolbox=True)
        
        
    #
    # Step 3: load and compute the model RDMs.
    #
    # 3-1 load the data files I created.
    #
    data_dirs = {}
    for model in models_I_want:
        if model in ['state_masked']:
            data_dirs[model]= np.load(os.path.join(RDM_dir, f"datastate_{sub}_fmri_both_halves.npy")) 
        else:    
            data_dirs[model]= np.load(os.path.join(RDM_dir, f"data_{model}_{sub}_fmri_both_halves.npy")) 


    #
    # step 3-2: create model RDMs
    #
    # first, compute similarity esitmate for each model separately.
    #
    model_RDM_dir, RDM_my_model_dir = {},{}
    
    for model in data_dirs:
        print(model)
        if RDM_version in ['01', '01-1']:
            import pdb; pdb.set_trace()
            # I NEED TO WRITE A FUNCTION FOR THIS, one that isn't corss_corr.
            # model_RDM_dir[model] = rsr.calc_rdm(model_data, method='correlation', descriptor='conds')
        else:
            model_RDM_dir[model] = mc.analyse.my_RSA.compute_crosscorr([np.transpose(data_dirs[model])])
            
        #
        # STEP 4: evaluate the model fit between model and data RDMs.
        #
        # ACTUAL RSA - single regressors
        #
        RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(model_RDM_dir[model][0], d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {model}"))
        mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=RDM_my_model_dir[model], centers=centers, file_path = results_dir, file_name= f"{model}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)

    

  # SECOND RSA: combo models (put several models in one regression to control for one another)
  
  # I am interested in:
      # do SMBs predict mPFC beyond simple current location/state representations?
      # is there a separate representation of current and future reward location representations in mPFC?
      
    combo_model_rdms = {}
    # combo clocks and controls
    if RDM_version in ['03-1'] and regression_version in ['03', '03-4', '03-l', '03-e', '03-rep1', '03-rep2','03-rep3','03-rep4','03-rep5']:
        combo_model_name = 'split_clock'
        models_to_combine = ['curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew']
        stacked_model_RDMs = np.stack([model_RDM_dir[m][0] for m in models_to_combine], axis=1)
        estimates_combined_model_rdms = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(stacked_model_RDMs, d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {combo_model_name}"))
        for i, model in enumerate(models_to_combine):
            mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=estimates_combined_model_rdms, centers=centers, file_path = results_dir, file_name= f"{model.upper()}-{combo_model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
        
                
        combo_model_name = 'clocks-loc-state'
        models_to_combine = ['clocks', 'state', 'location']
        stacked_model_RDMs = np.stack([model_RDM_dir[m][0] for m in models_to_combine], axis=1)
        estimates_combined_model_rdms = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(stacked_model_RDMs, d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {combo_model_name}"))
        for i, model in enumerate(models_to_combine):
            mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=estimates_combined_model_rdms, centers=centers, file_path = results_dir, file_name= f"{model.upper()}-{combo_model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
        

