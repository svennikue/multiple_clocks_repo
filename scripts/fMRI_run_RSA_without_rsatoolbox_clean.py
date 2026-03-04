#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17th of October 2025.

Running the RSA [mainly] without RSA toolbox,
based on 
1. clean_fmri_behaviour
2. create_fMRI_model_RDMs_on_clean_beh


# this is the old GLM I used:
GLM ('regression') settings (creating the 'bins'):
    03-4 -24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)
# this is the new GLM I want to use:
    # glm_all-rews-split_buttons
    
    
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
# config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_simple.json"
#config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_DSR_rew_vs_path_interaction_vis_combos.json"
config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_quarters_DSR_controls.json"
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
models_reverse = config.get("models_reverse", None)

masked_conditions = config.get("masked_conds", None)
conditions_masking = None
include_diagonal = config.get("diagonal_included", True)

weighted_models = config.get("weighted_models", None)
# conditions selection
conditions = config.get("EV_condition_selection")
parts_to_use = conditions["parts"]
parts_to_include_name = config.get("parts_to_include_name", None)
# model selection happens later in script

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
    if parts_to_include_name:
        data_rdm_dir = f"{data_dir}/func/data_RDMs_glmbase_{regression_version}_{searchlight_mask}_{parts_to_include_name}"
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_{today_str}_glmbase_{regression_version}/results" 
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_{today_str}_glmbase_{regression_version}_smooth{fwhm}/results" 
    os.makedirs(results_dir, exist_ok=True)

    # if masked_conditions:
    #     if masked_conditions[0] == 'load_same_loc_instate_mask':
    #         with open (f"{data_dir}/{masked_conditions[1]}", "r") as f:
    #             json_mask = json.load(f)
    #         masked_conditions = json_mask.get("masked_conditions")
        

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
    with open(f"{modelled_conditions_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'rb') as file:
        model_EVs = pickle.load(file)
        
    # these are the models that will be evaluated INDIVIDUALLY
    selected_models = config.get("models", list(model_EVs.keys()))
    if models_reverse:
        for m in ["state_action_DSR", "rew_stateactionDSR", "path_stateactionDSR"]:
            for rev in models_reverse:
                new_model = m + '-' + rev
                model_EVs[new_model] = model_EVs[m].copy()
    if weighted_models:
        for m in ["DSR", "rewDSR"]:
            for w in weighted_models:
                new_model = m + '-' + w + '-' + 'weighted'
                model_EVs[new_model] = model_EVs[m].copy()
                
    # loading the data EVs into dict
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(data_dir, regression_version=regression_version, only_load_labels = only_load_labels)
    # if you don't want all conditions created through FSL, exclude some here!
    # based on config file:
    # Ensure all four parts exist in config
    for _p in ("task", "direction", "state", "phase"):
        if _p not in parts_to_use:
            raise ValueError(f"Missing selection.parts['{_p}'] in config.")
            
    EV_keys = []        
    for ev in sorted(all_EV_keys):
        task, direction, state, phase = ev.split('_')
        # simple include/exclude logic
        for name, value in zip(["task", "direction", "state", "phase"], [task, direction, state, phase]):
            part = parts_to_use[name]
            includes = part.get("include", [])
            excludes = part.get("exclude", [])
            # Exclude first
            if any(fnmatch(value, pat) for pat in excludes):
                break  
            # If include list non-empty → must match at least one
            if includes and not any(fnmatch(value, pat) for pat in includes):
                break
        else:
            # only append if none of the 4 parts triggered 'break'
            EV_keys.append(ev)
    

    
    print(f"including the following EVs in the RDMs: {EV_keys}")
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
        if model == 'path_rew':
            model_RDM_dir[model] = mc.analyse.my_RSA.make_categorical_RDM(models_concat[model], plotting = False, include_diagonal=include_diagonal)
        elif model == 'duration':
            model_RDM_dir[model] = mc.analyse.my_RSA.make_distance_RDM(models_concat[model], plotting = False, include_diagonal=include_diagonal)
        elif model in ['location', 'DSR', 'prev_buttons', 'buttons_out', 'next_buttons', 'phys_abstr_space', 'action_DSR', 'state_action_DSR',
                       'state_action_glob', 'state_action_loc', "rewDSR", "pathDSR", "rew_stateactionDSR", "path_stateactionDSR",
                       'DSR_onefut', 'DSR_twofut','DSR_threefut','DSR_fourfut','DSR_fivefut','DSR_sixfut','DSR_sevenfut',
                       'curr_quarter','next_quarter','next2_quarter','next3_quarter']:
            model_RDM_dir[model] = mc.analyse.my_RSA.compute_hamming_distance(models_concat[model], plotting = False, include_diagonal=include_diagonal, model_name=model)
        elif model.endswith('diff'):
            cond = model.split('-')[1]
            model_RDM_dir[model] = mc.analyse.my_RSA.compute_hamming_difference(models_concat[model],combination = cond, plotting = False, include_diagonal=include_diagonal, model_name=model)
        elif model.endswith('weighted'):
            weight = model.split('-')[1]
            model_RDM_dir[model] = mc.analyse.my_RSA.compute_hamming_distance_weighted(models_concat[model], weight = weight, plotting = False, include_diagonal=include_diagonal, model_name=model)

        
        #elif model.startswith('button'):
        #    model_RDM_dir[model] = mc.analyse.my_RSA.make_distance_RDM_cosine_normratio(models_concat[model], plotting = False, include_diagonal=include_diagonal)
        else:
            model_RDM_dir[model] = mc.analyse.my_RSA.compute_crosscorr(models_concat[model], plotting= False, include_diagonal=include_diagonal)
            if model == 'A-state':
                A_state_mask = ~np.isnan(model_RDM_dir['A-state'][0])
                # import pdb; pdb.set_trace()
                # first make sure to not do this for the 'correct nans'
                nan_mask = np.isnan(model_RDM_dir['state'][0])
                # then turn all nans into 1s
                nan_mask_other_states = np.isnan(model_RDM_dir[model][0])
                model_RDM_dir[model][0][nan_mask_other_states] = 1
                #plt.figure(); plt.imshow(model_RDM_dir[model][0])
    
    if masked_conditions:
        conditions_masking = mc.analyse.my_RSA.make_category_masks(models_concat['path_rew'], plotting = False, include_diagonal=include_diagonal, mask_only_path_rew_combos=True)
    
    # n = 40
    # rdm = np.zeros((n,n))
    # rdm[np.triu_indices(n, 1)] = model_RDM_dir['state_action_glob-sa_ss_diff'][0]
    # plt.figure(); plt.imshow(rdm, cmap='coolwarm', vmin=0, vmax=1); plt.title('Same Action, Same State')
    # for i in range(0,n,4):
    #     plt.axvline(i-0.5, color='white', ls = 'dashed')
    # for i in range(0,n,4):
    #     plt.axhline(i-0.5, color='white', ls = 'dashed')
    # labels = ['A1_backw', 'A1_forw', 'B1_backw', 'B1_forw', 'C1_backw', 'C1_forw', 'D1_back', 'D1_forw', 'E1_backw', 'E1_forw']
    # plt.yticks(np.arange(2, rdm.shape[1], 4), labels)
    # plt.colorbar()
    
    # rew_EVs = []
    # for e in EV_keys:
    #     if e.endswith('reward'):
    #         rew_EVs.append(e)
    
    
    
    if 'A-state-ones' in selected_models:
        model_RDM_dir['state'][0][A_state_mask] = model_RDM_dir['state'][0][3]
        print(f"set state A in state regressor to {model_RDM_dir['state'][0][3]}")
    if 'A-state-mask' in selected_models:
        # A_state_mask = ~np.isnan(model_RDM_dir['A-state'][0])
        for model in model_RDM_dir:
            model_RDM_dir[model][0][A_state_mask] = np.nan
            print(f"excluding n = {np.sum(np.isnan(model_RDM_dir[model]))} datapoints from {len(model_RDM_dir[model][0])} because of state-A masking.")

    if not os.path.exists(f"{data_rdm_dir}/data_RDM.npy"):
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
            if model == 'A-state-mask' or model == 'A-state-ones':
                print("skipping computing A-state-mask or A-state-ones, already masked")
                continue
            print(model)
            # first, compute similarity esitmate for each model separately.
            # ACTUAL RSA - single regressors
            # for d in data_RDMs:
            #     RSA_results[model] = mc.analyse.my_RSA.evaluate_model(model_RDM_dir[model][0], d)
            #  RSA_results[model] = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(model_RDM_dir[model][0][0:171], d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {model}"))
            
            
            RSA_results[model] = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(model_RDM_dir[model][0], d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {model}"))
            mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=RSA_results[model], centers=centers, file_path = results_dir, file_name= f"{model}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)

    # import pdb; pdb.set_trace()
    run_combo_models = config.get("run_combo_models", bool(config.get("combo_models")))
    if run_combo_models:
        combo_list = config["combo_models"]
        for combo in combo_list:
            combo_model_name = combo["name"]
            models_to_combine = combo["regressors"].copy()
            if masked_conditions:
                for cond in conditions_masking:
                    print(f"now computing combo model, only considering conditions {cond}")
                    if cond == 'path-path':
                        if 'A-state' in models_to_combine:
                            models_to_combine.remove('A-state')
                    if cond == 'path-path' or cond == 'reward-reward':
                        if 'path_rew' in models_to_combine:
                            models_to_combine.remove('path_rew')
                    
                    print(f"running combo model {combo_model_name}, including {models_to_combine}")
                    
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
                    # #check how correlated each model is whith each other.
                    # corr = np.corrcoef(stacked_model_RDMs, rowvar=False)
                    # for i in range(len(models_to_combine)):
                    #     for j in range(i+1, len(models_to_combine)):
                    #         print(f"{models_to_combine[i]} vs {models_to_combine[j]}: r={corr[i,j]:.3f}")
                    # corr, fig, ax = mc.analyse.my_RSA.plot_model_correlations(stacked_model_RDMs, models_to_combine, conditions_masking=conditions_masking)
                    
                    estimates_combined_model_rdms = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(stacked_model_RDMs[conditions_masking[cond]], d[conditions_masking[cond]]) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {combo_model_name} only including {cond}"))
                    for i, model in enumerate(models_to_combine):
                        mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=estimates_combined_model_rdms, centers=centers, file_path = results_dir, file_name= f"{model.upper()}-{combo_model_name}-{cond}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)

            else:
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
            

