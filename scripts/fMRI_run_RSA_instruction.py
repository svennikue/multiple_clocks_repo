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
#import pdb; pdb.set_trace() 


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

    # sanity check: every task-half-1 key in keys_list must have found a partner
    n_th1_expected = sum(1 for k in keys_list if k.split('_')[0].endswith('1'))
    assert len(paired_list_control) == n_th1_expected, (
        f"Expected {n_th1_expected} pairs, got {len(paired_list_control)}. "
        "Some task-half-1 keys did not find their same-execution partner in the dict."
    )
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
TR = config.get("TR")
regression_version_full = f"{regression_version}-TR{TR}"


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
PLOTTING = True

# this should better be: what kind of searchlight_mask do you want?
# make sure to change this in the config files!
#load_searchlights = config.get("load_searchlights", False)
searchlight_mask = config.get("searchlight_mask", None)

print(f"Now running RSA based on subj GLM {regression_version} for subj {subj_no}")


for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        only_load_labels = False 
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        only_load_labels = False
        print(f"Running on Cluster, setting {data_dir} as data directory")
      
    modelled_conditions_dir = f"{data_dir}/beh/modelled_EVs"
    data_rdm_dir = f"{data_dir}/func/data_RDMs_glmbase_{regression_version_full}_{searchlight_mask}"
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version_full}/results"
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version_full}_smooth{fwhm}/results"
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
    # Model EVs — full dict (used to build DSR / rewDSR / simple below).
    with open(f"{modelled_conditions_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'rb') as file:
        model_EVs = pickle.load(file)
    # Which models to build + evaluate. Driven by the config so we can swap
    # between the original ['DSR', 'rewDSR', 'simple'] analysis and the new
    # ['curr_rew', 'next_rew', 'two_next_rew', 'three_next_rew'] split_rew_DSR
    # analysis without touching the script.
    selected_models = config.get("selected_models", ['DSR', 'rewDSR', 'simple'])
    # Data EVs — one PE per instruction-phase condition at this TR, per task half.
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs_instr_TRwise(
        data_dir, regression_version=regression_version, TR=TR,
        only_load_labels=only_load_labels,
    )
    EV_keys = sorted(all_EV_keys)
    print(f"including the following EVs in the RDMs: {EV_keys}")

    # Pair task halves by same-execution (A1_forw <-> A2_backw, etc.).
    data_th1, data_th2, paired_labels = pair_correct_tasks(data_EVs, EV_keys)
    data_concat = np.concatenate((data_th1, data_th2), axis=0)
    
    #
    # Step 3: compute the model RDMs.
    # Labels aligned with data pairing (row -> TH1 label, col -> TH2 label).
    th1_labels = [p.split(' with ')[0].replace('_instruction_onset', '') for p in paired_labels]
    th2_labels = [p.split(' with ')[1].replace('_instruction_onset', '') for p in paired_labels]

    model_RDM_dir = {}

    # Every non-'simple' model in `selected_models` is built the same way:
    # hamming dissim over the model's A_reward strings, TH1 x TH2, full off-block.
    for model in selected_models:
        if model == 'simple':
            continue
        a_rew_sub = {k: v for k, v in model_EVs[model].items() if k.endswith('_A_reward')}
        a_rew_keys = [k.replace('_instruction_onset', '_A_reward') for k in EV_keys]
        m_th1, m_th2, _ = pair_correct_tasks(a_rew_sub, a_rew_keys)
        model_RDM_dir[model] = mc.analyse.my_RSA.compute_hamming_instruction_RDM(m_th1, m_th2)

    # Simple — {-1, +1, NaN} based on same/different execution within the same task letter.
    if 'simple' in selected_models:
        model_RDM_dir['simple'] = mc.analyse.my_RSA.build_simple_instruction_RDM(th1_labels, th2_labels)
    
    if PLOTTING == True:
        # Plot each selected model RDM (plotted from the stored arrays — no recomputation).
        for model in selected_models:
            if model == 'simple':
                vmin, vmax, title = -1, 1, 'simple execution dissim'
            else:
                vmin, vmax, title = 0, 1, f'{model} A_reward hamming dissim'
            mc.analyse.my_RSA.plot_instruction_RDM(model_RDM_dir[model], th1_labels, th2_labels,
                                                   title=title, vmin=vmin, vmax=vmax,
                                                   save_path=f"{results_dir}_{model}")
    
        # Optional inspection plot: cosine dissim from one random searchlight.
        plot_example_data_RDM = config.get("plot_example_data_RDM", False)
        if plot_example_data_RDM and not only_load_labels:
            rng = np.random.default_rng(42)
            sl_idx = int(rng.integers(0, len(centers)))
            vox_ids = np.asarray(neighbors[sl_idx])
            sl_data = data_concat[:, vox_ids]
            n_conds = sl_data.shape[0] // 2
            example_data_RDM = mc.analyse.my_RSA.compute_cosine_instruction_RDM(sl_data[:n_conds], sl_data[n_conds:])
            mc.analyse.my_RSA.plot_instruction_RDM(
                example_data_RDM, th1_labels, th2_labels,
                title=f'example data RDM (searchlight #{sl_idx}, cosine dissim)', save_path=f"{results_dir}_data"
            )
    
        plt.show(block=False)
    
    #
    # Step 4: compute the data RDM per searchlight (cosine dissim, TH1 x TH2, full off-block).
    #
    os.makedirs(data_rdm_dir, exist_ok=True)
    if not os.path.exists(f"{data_rdm_dir}/data_RDM.npy"):
        data_RDMs = mc.analyse.my_RSA.get_instruction_RDM_per_searchlight(data_concat, centers, neighbors)
        mc.analyse.handle_MRI_files.save_data_RDM_as_nifti(data_RDMs, data_rdm_dir, "data_RDM", ref_img, centers)
    else:
        data_RDMs = np.load(f"{data_rdm_dir}/data_RDM.npy")

    if smoothing == True:
        if not os.path.exists(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.npy"):
            path_to_save_smooth = f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}"
            print(f"now smoothing the RDM and saving it here: {path_to_save_smooth}")
            data_RDMs = mc.analyse.handle_MRI_files.smooth_RDMs(data_RDMs, ref_img, fwhm, use_rsa_toolbox=False, path_to_save=path_to_save_smooth, centers=centers)
        else:
            data_RDMs = np.load(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.npy")

    #
    # Step 5: evaluate each single model against every searchlight data RDM.
    # NaN cells in the simple model automatically drop the corresponding data
    # cells from the OLS (see evaluate_model_vec). Both X and Y are z-scored.
    #
    RSA_results = {}
    run_single_models = config.get("run_single_models", True)
    if run_single_models == True:
        for model in selected_models:
            model_flat = np.asarray(model_RDM_dir[model]).ravel()
            RSA_results[model] = Parallel(n_jobs=3)(
                delayed(mc.analyse.my_RSA.evaluate_model)(model_flat, d)
                for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {model}")
            )
            mc.analyse.handle_MRI_files.save_my_RSA_results(
                result_file=RSA_results[model], centers=centers,
                file_path=results_dir, file_name=f"{model}",
                mask=mask, number_regr=0, ref_image_for_affine_path=ref_img,
            )

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
  
            # Each model_RDM_dir[m] is a (n_th1, n_th2) matrix; flatten to a
            # 1D regressor and stack so shape = (n_pairs, n_regressors).
            stacked_model_RDMs = np.stack(
                [np.asarray(model_RDM_dir[m]).ravel() for m in models_to_combine],
                axis=1,
            )
            
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
        "regression_version": regression_version,
        "TR": TR,
        "regression_version_full": regression_version_full,
        "RDM_version": RDM_version,
        "paired_labels": paired_labels,
        "smoothing": smoothing,
        "fwhm": fwhm,
        "searchlight_mask": searchlight_mask,
        "n_all_EVs": len(all_EV_keys),
        "n_selected_EVs": len(EV_keys),
        "models_evaluated": selected_models,
        "run_combo_models": run_combo_models,
        "combo_models": config.get("combo_models", []),
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
            

