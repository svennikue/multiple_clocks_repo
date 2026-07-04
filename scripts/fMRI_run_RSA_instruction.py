#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 07:36:34 2026

Running the instruction-phase RSA, per timepoint.

@author: Svenja Küchenhoff, 2026
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

def pair_correct_tasks(data_dict, keys_list):
    """
    data_dict: dict with keys like 'A1_forw_A_reward'
    keys_list: ordered list of keys you want to include and in what order
    Returns two matrices: one for the first element of each pair, one for its match.
    """
    # Define task pairing relationships
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    th_1, th_2, paired_list_control  = [], [], []
    # Loop through keys in the *specified order*
    for key in keys_list:
        assert key in data_dict, "Missmatch between model rdm keys and data RDM keys"
        task, direction, state, phase = key.split('_')  # e.g. ['A1', 'forw', 'A', 'reward']
        # Create the pairing suffix (e.g. from '1_forw' → '2_backw')
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        # Build the paired key (e.g. 'A2_backw_A_reward')
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        # Only add if both keys exist
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} with {pair_key}")

    # import pdb; pdb.set_trace()       
    th_1 = np.vstack(th_1)
    th_2 = np.vstack(th_2)
    # print(paired_list_control)
    return th_1, th_2, paired_list_control

#
#
# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")

# --- Load configuration ---
config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_instruction.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
EV_string = config.get("load_EVs_from")
regression_version = config.get("regression_version")

today_str = date.today().strftime("%d-%m-%Y")
name_RSA = config.get("name_of_RSA")
RDM_version = f"{name_RSA}"


# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]

# Flags
smoothing = config.get("smoothing", True)
fwhm = config.get("fwhm", 5)

# this should better be: what kind of searchlight_mask do you want?
# make sure to change this in the config files!
#load_searchlights = config.get("load_searchlights", False)
searchlight_mask = config.get("searchlight_mask", None)
include_diagonal = config.get("diagonal_included", True)

print(f"Now running RSA based on subj GLM {regression_version} for subj {subj_no}")


for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        # DONT FORGET TO COMMENT THIS OUT!!!!
        # regression_version = '03-4'
        only_load_labels = True 
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        only_load_labels = False
        print(f"Running on Cluster, setting {data_dir} as data directory")
      
    modelled_conditions_dir = f"{data_dir}/beh/modelled_EVs"
    data_rdm_dir = f"{data_dir}/func/data_RDMs_glmbase_{regression_version}_{searchlight_mask}"
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_{today_str}_glmbase_{regression_version}/results" 
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_{today_str}_glmbase_{regression_version}_smooth{fwhm}/results" 
    os.makedirs(results_dir, exist_ok=True)

    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    if searchlight_mask:
        if searchlight_mask == 'no_CSF':
            mask_file = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
            mask_name = '_no_CSF' # Found 166.240 searchlights with no CSF mask
        elif searchlight_mask == 'grey_matter':
            mask_file = load_img(f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz")
            mask_name = '_grey_matter'  # Found 126.404 searchlights with gm mask 
    else:
        mask_file = ref_img.copy() # full BOLD Found 175.483 searchlights
        mask_name = ''
    mask = mask_file.get_fdata()  
    path_to_searchlight_centers = f"{data_dir}/func/searchlight_centers{mask_name}.pkl"
    path_to_searchlight_neighbours = f"{data_dir}/func/searchlight_neighbors{mask_name}.pkl"
    if os.path.exists(path_to_searchlight_centers):
        with open(path_to_searchlight_centers, "rb") as f:
            centers = pickle.load(f)
        with open(path_to_searchlight_neighbours, "rb") as f:
            neighbors = pickle.load(f)
    else:
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5)
        with open(path_to_searchlight_centers, 'wb') as file:
            pickle.dump(centers, file)
            print("stored searchlight centres")
        with open(path_to_searchlight_neighbours, 'wb') as file:
            pickle.dump(neighbors, file)   
            print("stored searchlight neighbors")

    #
    # Step 2: loading conditions for model and data RDMs
    #
    # loading the model EVs into dict
    # TODO
    # this needs to be different. Only load the DSR and the reward-DSR per configuration.
    with open(f"{modelled_conditions_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'rb') as file:
        model_EVs = pickle.load(file)
        selected_models = ['DSR', 'DSR_reward', 'simple']
        
    # TODO
    # Then set up the model RDM: this will be either:
    # simple execution: same (1) if execution was the same, different (0) if execution was differnt.
    # DSR-like execution, rew only: assume a DSR-model starting from 'A' per configuration.
        # compare the similarity between DSR-traces (only rewards): i.e. same reward at same future from A: + similarity
    # DSR-like execution, paths and rewards: assume a DSR-model starting from 'A' per configuration.
        # compare the similarity between DSR-traces: i.e. same location same lag at future: + similarity


    # loading the data EVs into dict
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(data_dir, regression_version=regression_version, only_load_labels = only_load_labels)
    
    # TODO:
    # change because this is now only 1 EV per task - 10 per task half, 20 across.
    EV_keys = []        
    for ev in sorted(all_EV_keys):
        EV_keys.append(ev)
    
    print(f"including the following EVs in the RDMs: {EV_keys}")
    
    # TODO: also check if the pairing is still correct- it will have to be based, this time,
    # on the execution (same as before), but we need to consider the instruction as well (backw forw)
    data_th1, data_th2, paired_labels = pair_correct_tasks(data_EVs, EV_keys)
    data_concat = np.concatenate((data_th1, data_th2), axis = 0)
    
    # 
    # Step 3: compute the model and data RDMs.
    models_concat = {}
    model_RDM_dir = {}
    
    for model in model_EVs:
        model_th1, model_th2, model_paired_labels = pair_correct_tasks(model_EVs[model], EV_keys)
        # finally, concatenate th1 and th2 to do the cross-correlation after
        models_concat[model] = np.concatenate((model_th1, model_th2), axis = 0)
        model_RDM_dir[model] = mc.analyse.my_RSA.compute_hamming_distance(models_concat[model], plotting = False, include_diagonal=include_diagonal, model_name=model)


    # compute the data RDM
    if not os.path.exists(f"{data_rdm_dir}/data_RDM.npy"):
        # TODO
        # adjust this! this has to be a different similarity now (not across task halves)
        data_RDMs = mc.analyse.my_RSA.get_RDM_per_searchlight(data_concat, centers, neighbors, method = 'crosscorr', include_diagonal=include_diagonal) 
        mc.analyse.handle_MRI_files.save_data_RDM_as_nifti(data_RDMs, data_rdm_dir, "data_RDM", ref_img, centers) 
    else:
        data_RDMs = np.load(f"{data_rdm_dir}/data_RDM.npy")
        
    if smoothing == True:
        if not os.path.exists(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.npy"):
            path_to_save_smooth = f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}"
            print(f"now smoothing the RDM and saving it here: {path_to_save_smooth}")
            data_RDMs = mc.analyse.handle_MRI_files.smooth_RDMs(data_RDMs, ref_img, fwhm,use_rsa_toolbox = False, path_to_save=path_to_save_smooth,centers=centers)
        else:
            data_RDMs = np.load(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.npy")


    # import pdb; pdb.set_trace()
    # STEP 4: evaluate the model fit between model and data RDMs.
    #
    RSA_results = {}
    run_single_models = config.get("run_single_models", True)
    if run_single_models == True:
        for model in selected_models:
            RSA_results[model] = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(model_RDM_dir[model][0], d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {model}"))
            mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=RSA_results[model], centers=centers, file_path = results_dir, file_name= f"{model}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)

    # import pdb; pdb.set_trace()
    run_combo_models = config.get("run_combo_models", bool(config.get("combo_models")))
    if run_combo_models:
        combo_list = config["combo_models"]
        for combo in combo_list:
            combo_model_name = combo["name"]
            models_to_combine = combo["regressors"]
            print(f"running combo model {combo_model_name}")
            # check if these models have been computed in model_EVs
            missing = [m for m in models_to_combine if m not in model_RDM_dir]
            if missing:
                for m_int in missing:
                    if m_int.endswith('interaction'):
                        curr_m = m_int.split('_interaction')[0]
                        z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
                        model_RDM_dir[m_int] = [z(model_RDM_dir[curr_m][0]) * z(model_RDM_dir['path_rew'][0])]
                        # model_RDM_dir[m_int] = [model_RDM_dir[curr_m][0]*model_RDM_dir['path_rew'][0]]
                    else:
                        raise ValueError(f"Combo model {combo_model_name} not possible, as {missing} not computed")
  
            stacked_model_RDMs = np.stack([model_RDM_dir[m][0] for m in models_to_combine], axis=1)
            
            # check how correlated each model is whith each other.
            # corr = np.corrcoef(stacked_model_RDMs, rowvar=False)
            # for i in range(len(models_to_combine)):
            #     for j in range(i+1, len(models_to_combine)):
            #         print(f"{models_to_combine[i]} vs {models_to_combine[j]}: r={corr[i,j]:.3f}")
            # corr, fig, ax = mc.analyse.my_RSA.plot_model_correlations(stacked_model_RDMs, models_to_combine, conditions_masking=conditions_masking)
              
            estimates_combined_model_rdms = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(stacked_model_RDMs, d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {combo_model_name}"))
            for i, model in enumerate(models_to_combine):
                # TODO: Change the type of similarity to not throw away half of the matrix.
                mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=estimates_combined_model_rdms, centers=centers, file_path = results_dir, file_name= f"{model.upper()}-{combo_model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
            
    

    # --- SETTINGS SUMMARY (per subject) ---
    summary = {
        "subject": sub,
        "EV_string": EV_string,
        "EV_labels_in_RDM": model_paired_labels,
        "regression_version": regression_version,
        "RDM_version": RDM_version,
        "smoothing": smoothing,
        "fwhm": fwhm,
        "searchlight_mask": searchlight_mask,
        "n_all_EVs": len(all_EV_keys),
        "n_selected_EVs": len(EV_keys),
        "models_evaluated": selected_models,
        "diagonal is included": include_diagonal,
        "run_combo_models": run_combo_models,
        "combo_model_names": [c["name"] for c in config.get("combo_models", [])] if run_combo_models else [],
        "combo_model_regs_per_combo": [c["regressors"] for c in config.get("combo_models", [])] if run_combo_models else [],
        "data_dir": data_dir,
        "results_dir": results_dir
    }
    
    print("\n=== SETTINGS SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    
    # Save a copy alongside results for provenance
    with open(os.path.join(results_dir, f"{sub}_settings_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"(Saved summary → {os.path.join(results_dir, f'{sub}_settings_summary.json')})\n")
            

