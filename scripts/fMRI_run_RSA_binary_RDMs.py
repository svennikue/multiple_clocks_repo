#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17th of October 2025.

Running the RSA [mainly] without RSA toolbox,
based on 
1. clean_fmri_behaviour
2. using estimated conditions, rather than timesteps, as an input.


# this is the new GLM I want to use:
    # glm_fut-steps_states_split-buttons
    # glm_fut-steps_split-buttons
    
@author: Svenja Küchenhoff, 2025
"""


from tqdm import tqdm
import numpy as np
import os
from nilearn.image import load_img
from rsatoolbox.util.searchlight import get_volume_searchlight
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc
import pickle
import sys
from datetime import date
import json
from fnmatch import fnmatch


def pair_correct_tasks(data_dict, keys_list):
    """
    Pairs th_1_* with th_2_* using the same base name, in the order of th_1_* keys in keys_list.
    Returns two matrices: rows align (th_1 row i matches th_2 row i).
    """
    th1_rows, th2_rows = [], []
    keys_th1, keys_th2 = [], []
    # keep order dictated by keys_list; consider only th_1_* keys
    for k1 in keys_list:
        if not k1.startswith('th_1_'):
            continue
        k2 = 'th_2_' + k1[len('th_1_'):]  # mirror name
        if k1 in data_dict and k2 in data_dict:
            th1_rows.append(np.asarray(data_dict[k1]))
            keys_th1.append(k1)
            th2_rows.append(np.asarray(data_dict[k2]))
            keys_th2.append(k2)
    if not th1_rows:
        raise ValueError("No valid th_1/th_2 pairs found in data_dict for the given keys_list.")
    th_1 = np.vstack(th1_rows)
    th_2 = np.vstack(th2_rows)
    
    
    # Build one-hot encodings: identical for TH1 and TH2
    idx = {b:i for i,b in enumerate(keys_th1)}
    oh = np.zeros((len(keys_th1), len(keys_th1)), dtype=int)
    for r, b in enumerate(keys_th1):
        oh[r, idx[b]] = 1
    oh1 = oh.copy()
    oh2 = oh.copy()

    return th_1, th_2, keys_th1, keys_th2, oh1, oh2


#
#
#  
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
      
# --- Load configuration ---
# config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_simple.json"
config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_state_bin.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
EV_string = config.get("load_EVs_from")
regression_version = config.get("regression_version")
today_str = date.today().strftime("%d-%m-%Y")
name_RSA = config.get("name_of_RSA")
RDM_version = f"{name_RSA}_{today_str}"

# conditions selection
conditions = config.get("EV_condition_selection")
parts_to_use = conditions["parts"]


# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]

# Flags
smoothing = config.get("smoothing", True)
fwhm = config.get("fwhm", 5)
load_searchlights = config.get("load_searchlights", False)

# conditions selection
rdms_to_run_masking = config.get("rdms_to_run_masking")


print(f"Now running RSA based on subj GLM {regression_version} for subj {subj_no}")

for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")
    
    data_rdm_dir = f"{data_dir}/func/data_RDMs_{RDM_version}_glmbase_{regression_version}"
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}" 
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}_smooth{fwhm}/results" 
    os.makedirs(results_dir, exist_ok=True)

    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    mask_file = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    mask = mask_file.get_fdata()  
    # save this file to save time
    if load_searchlights:
        with open(f"{data_dir}/func/searchlight_centers.pkl", 'rb') as file:
            centers = pickle.load(file)
        with open(f"{data_dir}/func/searchlight_neighbors.pkl", 'rb') as file:
            neighbors = pickle.load(file)
    else:
        # creating the searchlights
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5) # Found 175.483 searchlights
        # if I use the grey matter mask, then I find 144.905 searchlights
        # save this structure
        with open(f"{data_dir}/func/searchlight_centers.pkl", 'wb') as file:
            pickle.dump(centers, file)
            print("stored searchlight centres")
        with open(f"{data_dir}/func/searchlight_neighbors.pkl", 'wb') as file:
            pickle.dump(neighbors, file)   
            print("stored searchlight neighbors")

    #
    # Step 2: loading conditions for model and data RDMs
    # import pdb; pdb.set_trace()
    # loading the data EVs into dict
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs_th(data_dir, regression_version=regression_version)
    
    # exclude the conditions that you don't want to include in the model later.
    EV_keys = []        
    for ev in sorted(all_EV_keys):
        # simple include/exclude logic
        for p in parts_to_use:
            excludes = parts_to_use[p].get("exclude", [])
            # Exclude first
            if any((pat in ev) for pat in excludes):
                break
        else:
            # only append if none of the 4 parts triggered 'break'
            EV_keys.append(ev)
            
    
    # prepare labels for halved RDMs
    labels_half_RDM = [k.split('th_1_', 1)[1] for k in EV_keys if k.startswith('th_1_')]
    
    # make sure the EVs are paired in the correct order
    data_th1, data_th2, names_th1, names_th2, model_th1, model_th2 = pair_correct_tasks(data_EVs, EV_keys)
    
    # then concatenate the data
    data_concat = np.concatenate((data_th1, data_th2), axis = 0)
    # and concatenate the model.
    full_model_concat = np.concatenate((model_th1, model_th2), axis = 0)
    
    # 
    # Step 3: compute the model and data RDMs.
    # model RDMs
    models_concat, model_RDM_dict = {}, {}
    for model in rdms_to_run_masking: 
        model_RDM_dict[model] = mc.analyse.my_RSA.compute_crosscorr_and_filter(full_model_concat, plotting = False, labels = labels_half_RDM, mask = rdms_to_run_masking[model], binarise = True)

    # data RDMs
    data_RDM_dict = {}
    if not os.path.exists(f"{data_rdm_dir}/data_RDM_{model}.npy"): 
         # and searchlight-wise for data RDMs
         for model in rdms_to_run_masking:
             data_RDM_dict[model] = mc.analyse.my_RSA.get_RDM_per_searchlight(data_concat, centers, neighbors, method = 'crosscorr_and_filter', labels = labels_half_RDM, mask = rdms_to_run_masking[model])
             mc.analyse.handle_MRI_files.save_data_RDM_as_nifti(data_RDM_dict[model], data_rdm_dir, f"data_RDM_{model}", ref_img, centers)      
    else:
        for model in rdms_to_run_masking:
            data_RDM_dict[model] = np.load(f"{data_rdm_dir}/data_RDM_{model}.npy")

    if smoothing == True:
        if not os.path.exists(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.npy"):
            path_to_save_smooth = f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.nii.gz"
            print(f"now smoothing the RDM and saving it here: {path_to_save_smooth}")
            for model in rdms_to_run_masking:
                data_RDM_dict[model] = mc.analyse.handle_MRI_files.smooth_RDMs(data_RDM_dict[model], ref_img, fwhm,use_rsa_toolbox = False, path_to_save=path_to_save_smooth,centers=centers)
        
        else:
            for model in rdms_to_run_masking:
                data_RDM_dict[model] = np.load(f"{data_rdm_dir}/data_RDM_{model}_smooth_fwhm{fwhm}.npy")

    #
    # STEP 4: evaluate the model fit between model and data RDMs.
    #
    RSA_results = {}
    all_models = []
    for model in rdms_to_run_masking:
        print(model)
        all_models.append(model)
        # first, compute similarity esitmate for each model separately.
        # ACTUAL RSA - single regressor
        RSA_results[model] = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(model_RDM_dict[model][0], d) for d in tqdm(data_RDM_dict[model], desc=f"running GLM for all searchlights in {model}"))
        mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=RSA_results[model], centers=centers, file_path = results_dir, file_name= f"{model}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)

          
    # --- SETTINGS SUMMARY (per subject) ---
    summary = {
        "subject": sub,
        "EV_string": EV_string,
        "regression_version": regression_version,
        "RDM_version": RDM_version,
        "smoothing": smoothing,
        "fwhm": fwhm,
        "load_searchlights": load_searchlights,
        "n_all_EVs": len(EV_keys),
        "n_selected_EVs": len(EV_keys),
        "models_evaluated": all_models,
        "data_dir": data_dir,
        "results_dir": results_dir,
    }
    
    print("\n=== SETTINGS SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    
    # Save a copy alongside results for provenance
    with open(os.path.join(results_dir, f"{sub}_settings_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"(Saved summary → {os.path.join(results_dir, f'{sub}_settings_summary.json')})\n")
            

