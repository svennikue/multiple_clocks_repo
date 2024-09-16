#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:25:52 2023

create fMRI data RDMs

28.03.: I am changing something in the preprocessing. This is THE day to change the naming such that it all works well :)

RDM settings (creating the representations):
    01 -> instruction periods, similarity by order of execution, order of seeing, all backw presentations
    01-1 -> instruction periods, location similarity
    
    02 -> modelling paths + rewards, creating all possible models
    02-act -> modelling paths + rewards, also creating the action model.
    02-act-1phas -> only one phase per subpath! modelling paths + rewards, also creating the action model.
    
    03 -> modelling only reward anchors/rings + splitting clocks model in the same py function.
    03-1 -> modelling only reward rings + split ‘clocks model’ = just rotating the reward location around. 
    03-2 -> same as 03-1 but only considering task D and B (where 2 rew locs are the same)
    03-3 -> same as 03-1 but only considering B,C,D [excluding rew A] -> important to be paired with GLM 03-3!
    03-5 - STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. 
    03-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    03-99 ->  using 03-1 - reward locations and future rew model; but EVs are scrambled.
    03-999 ->  is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.    
    
    04 -> modelling only paths
    04-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    
    05 -> modelling only baths and only rewards to compare them later!
    xx-999 ->  is debugging 2.0: using whatever, but the voxels are scrambled.


GLM ('regression') settings (creating the 'bins'):
    01 - instruction EVs
    02 - 80 regressors; every task is divided into 4 rewards + 4 paths
    03 - 40 regressors; for every tasks, only the rewards are modelled [using a stick function]
    03-2 - 40 regressors; for every task, only the rewards are modelled (in their original time)
    03-3 - 30 regressors; for every task, only the rewards are modelled (in their original time), except for A (because of visual feedback)
    03-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)
        Careful! I computed one data (subject-level) GLM called 03-4. This is simply a 03 without button presses!
        Not the same as 03-4 in this sense, but ok to be used.
    03-99 - 40 regressors; no button press; I allocate the reward onsets randomly to different state/task combos  -> shuffled through whole task; [using a stick function]
    03-999 - 40 regressors; no button press; created a random but sorted sample of onsets that I am using -> still somewhat sorted by time, still [using a stick function]
    03-9999 - 40 regressors; no button press; shift all regressors 6 seconds earlier
    04 - 40 regressors; for every task, only the paths are modelled
    04-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E)
    05 - locations + button presses 
    06 - averaging across the entire task [for introduction analysis]
    06-rep 1 - averaging across the entire task, but only the first repeat.
    07 - entire path and reward period, collapsed (= 03 + 04)
    
@author: Svenja Küchenhoff, 2024
"""

# UNDER CONSTRUCTION
# NOT WORKING CURRENTL!!
# CONTINUE DEBUGGING FIRST!!!


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

regression_version = '03-4' 
RDM_version = '02-act'

binary = False
neuron_weighting = False
smoothing = True
fwhm = 5

# import pdb; pdb.set_trace() 
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

# subjects = [f"sub-{subj_no}"]
subjects = subs_list = [f'sub-{i:02}' for i in range(1, 36) if i not in (21, 29)]

load_old = False
visualise_RDMs = False


task_halves = ['1', '2']

print(f"Now running RSA for RDM version {RDM_version} based on subj GLM {regression_version} for subj {subj_no}")


models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)

# import pdb; pdb.set_trace() 

# based on GLM
if regression_version == '01':
    no_RDM_conditions = 20 # including all instruction periods
elif regression_version in ['02', '02-e', '02-l']:
    no_RDM_conditions = 80 # including all paths and rewards
elif regression_version in ['03', '04','03-99', '03-999', '03-9999', '03-l', '03-e']:
    no_RDM_conditions = 40 # only including rewards or only paths
elif regression_version == '03-3': #excluding reward A
    no_RDM_conditions = 30
elif regression_version in ['03-4', '04-4', '03-4-e', '03-4-l', '03-4-rep1', '03-4-rep2' , '03-4-rep3' , '03-4-rep4' ,'03-4-rep5' ]: # only including tasks without double reward locs: A,C,D  and only rewards
    no_RDM_conditions = 24
    
if regression_version in ['03-4', '04-4'] and RDM_version in ['03-5-A', '02-A', '03-A', '04-A', '04-5-A']: # only TASK A,C,D, only rewards B-C-D
    no_RDM_conditions = 18

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
    if not os.path.exists(RDM_dir):
        os.makedirs(RDM_dir)  
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}" 
    if smoothing == True:
        results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}_smooth{fwhm}" 
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.makedirs(f"{results_dir}/results")
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results" 
    if smoothing == True:
        results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}_smooth{fwhm}/results" 
    data_rdm_dir = f"{data_dir}/func/data_RDM_glmbase_{regression_version}"
    if not os.path.exists(data_rdm_dir):
        os.makedirs(data_rdm_dir)  
    # if os.path.exists(results_dir):
    #     # move pre-existing files into a different folder.
    #     mc.analyse.analyse_MRI_behav.move_files_to_subfolder(results_dir)
    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    # mask = load_img(f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz")
    mask_file = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    mask = mask_file.get_fdata()  
    # save this file to save time
    if load_old:
        with open(f"{RDM_dir}/searchlight_centers.pkl", 'rb') as file:
            centers = pickle.load(file)
        with open(f"{RDM_dir}/searchlight_neighbors.pkl", 'rb') as file:
            neighbors = pickle.load(file)
        #centers = np.load(f"{RDM_dir}/searchlight_centers.npy", allow_pickle=True)
        #neighbors = np.load(f"{RDM_dir}/searchlight_neighbors.npy", allow_pickle=True)
    else:
        # creating the searchlights
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5) # Found 175.483 searchlights
        # if I use the grey matter mask, then I find 144.905 searchlights
        # save this structure
        with open(f"{RDM_dir}/searchlight_centers.pkl", 'wb') as file:
            pickle.dump(centers, file)
        with open(f"{RDM_dir}/searchlight_neighbors.pkl", 'wb') as file:
            pickle.dump(neighbors, file)   
        #np.save(f"{RDM_dir}/searchlight_centers.npy", centers)
        #np.save(f"{RDM_dir}/searchlight_neighbors.npy", neighbors)
            
    # Step 2: loading and computing the data RDMs
    # if load_old:
    #     with open(f"{results_dir}/data_RDM.pkl", 'rb') as file:
    #         data_RDM = pickle.load(file)
    #     if visualise_RDMs == True:
    #         mc.analyse.analyse_MRI_behav.visualise_data_RDM(mni_x=53, mni_y = 30, mni_z= 2, data_RDM_file= data_RDM, mask=mask)
    # else:
    data_RDM_file_2d= mc.analyse.analyse_MRI_behav.read_in_RDM_conds(regression_version, RDM_version, data_dir, RDM_dir, no_RDM_conditions, sort_as = 'dict-two-halves')
    condition_names = mc.analyse.analyse_MRI_behav.get_conditions_list(RDM_dir)
    data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1,2)).transpose(),2*no_RDM_conditions)  
    # now prepare the data RDM file. 
    # final data RDM file; 
    if RDM_version in ['01', '01-1']:
        data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1)).transpose(),no_RDM_conditions)  
        data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, data_conds, method='correlation')
    else:
        # for all other cases, cross correlated between task-halves.
        # this is defining both task halves/ runs: 0 is first half, the second one is 1s
        sessions = np.concatenate((np.zeros(no_RDM_conditions), np.ones(no_RDM_conditions)))
        if not os.path.exists(f"{data_rdm_dir}/data_RDM-pkl"):
            data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, data_conds, method='crosscorr', cv_descr=sessions)
            # save  so that I don't need to recompute - or don't save bc it's massive
            data_RDM.save(f"{data_rdm_dir}/data_RDM-pkl", 'pkl')
            # potentially build in a progress function for this one! takes ages..
            mc.analyse.analyse_MRI_behav.save_data_RDM_as_nifti(data_RDM, data_rdm_dir, "data_RDM.nii.gz", ref_img)
        else:
            with open(f"{data_rdm_dir}/data_RDM-pkl", 'rb') as file:
                data_RDM_dir = pickle.load(file)
                data_RDM = rsatoolbox.rdm.rdms.rdms_from_dict(data_RDM_dir)
    
    # ACC [54, 63, 41]
    # mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [54, 63, 41], ref_img, condition_names)
    #mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDM, [54, 63, 41], ref_img, condition_names)
    
    # visual cortex [72, 17, 9]
    #mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [72, 17, 9], ref_img, condition_names)
    #mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDM, [72, 17, 9], ref_img, condition_names)
    
    # hippocampus [43, 50, 17]
    #mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [43, 50, 17], ref_img, condition_names)
    #mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDM, [43, 50, 17], ref_img, condition_names)
    
    # import pdb; pdb.set_trace()
    if smoothing == True:
        if not os.path.exists(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}-pkl"):
            path_to_save_smooth = f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.nii.gz"
            print(f"now smoothing the RDM and saving it here: {path_to_save_smooth}")
            data_RDM = mc.analyse.handle_MRI_files.smooth_RDMs(data_RDM, ref_img, path_to_save_smooth, fwhm)
            data_RDM.save(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}-pkl", 'pkl')
        else:
            with open(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}-pkl", 'rb') as file:
                print("now opening the smoothed RDM")
                data_RDM_dir = pickle.load(file)
                data_RDM = rsatoolbox.rdm.rdms.rdms_from_dict(data_RDM_dir)
                
    # ACC [54, 63, 41]
    #mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [54, 63, 41], ref_img, condition_names)
    #mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDM, [54, 63, 41], ref_img, condition_names)
    
    # visual cortex [72, 17, 9]
    #mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [72, 17, 9], ref_img, condition_names)
    #mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDM, [72, 17, 9], ref_img, condition_names)
    
    # hippocampus [43, 50, 17]
    #mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [43, 50, 17], ref_img, condition_names)
    #mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDM, [43, 50, 17], ref_img, condition_names)
    
    
    
    # Step 3: load and compute the model RDMs.
    # 3-1 load the data files I created.
    data_dirs = {}
    for model in models_I_want:
        if RDM_version in ['999', '9999']: # potentially delete?? this is now 03-99 nd 03-999
            RDM_dir = f"{data_dir}/beh/RDMs_09_glmbase_{regression_version}" # potentially delete??
        if model in ['state_masked']:
            data_dirs[model]= np.load(os.path.join(RDM_dir, f"datastate_{sub}_fmri_both_halves.npy")) 
        else:    
            data_dirs[model]= np.load(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves.npy")) 
    
        # add keys for the 2 weighted models
        if neuron_weighting == True and model in ['clocks_only-rew', 'clocks', 'clocks_no-rew']:
            data_dirs[f"{model}-sin"] = 0
            data_dirs[f"{model}-cos"] = 0
            
        
    # import pdb; pdb.set_trace()
    # step 3-2: create model RDMs
    # first, each model gets its own, separate estimation.
    model_RDM_dir = {}
    RDM_my_model_dir = {}
    for model in data_dirs:
        print(model)
        if model in ['clocks_only-rew-sin', 'clocks-sin', 'clocks_no-rew-sin', 'clocks_only-rew-cos', 'clocks-cos', 'clocks_no-rew-cos']:
            model_data = mc.analyse.analyse_MRI_behav.prepare_model_data(data_dirs[f"{model[:-4]}"], no_RDM_conditions, RDM_version)
        else:
            model_data = mc.analyse.analyse_MRI_behav.prepare_model_data(data_dirs[model], no_RDM_conditions, RDM_version)
        if RDM_version in ['01', '01-1']:
            model_RDM_dir[model] = rsr.calc_rdm(model_data, method='correlation', descriptor='conds')
        if model.endswith('-sin'):
            model_RDM_dir[model] = rsr.calc_rdm(model_data, method='weight_crosscorr', descriptor='conds', cv_descriptor='sessions', weighting = 'sin')
        elif model.endswith('-cos'):
            model_RDM_dir[model] = rsr.calc_rdm(model_data, method='weight_crosscorr', descriptor='conds', cv_descriptor='sessions', weighting = 'cos')
        else:
            model_RDM_dir[model] = rsr.calc_rdm(model_data, method='crosscorr', descriptor='conds', cv_descriptor='sessions')
        
        
        if model in ['state_masked']:
            state_mask = np.load(os.path.join(f"{data_dir}/beh/RDMs_03-5_glmbase_{regression_version}", f"RSM_state_masked_{sub}_fmri_both_halves.npy"))
            state_tril = state_mask.T[np.triu_indices(len(state_mask), 1)]
            nan_mask = np.isnan(state_tril)
            model_RDM_dir['state_masked'].dissimilarities[0][nan_mask] = np.nan
            
        fig, ax, ret_vla = rsatoolbox.vis.show_rdm(model_RDM_dir[model])
        # set up the model object
        model_model = rsatoolbox.model.ModelFixed(f"{model}_only", model_RDM_dir[model])
        
        
        # ACTUAL RSA - single models
        # STEP 4: evaluate the model fit between model and data RDMs.
        # for d in data_RDM:
        #     RDM_my_model_dir[model] = mc.analyse.analyse_MRI_behav.evaluate_model(model_model, d)
        
        
        for d in data_RDM:
            RDM_my_model_dir[model] = mc.analyse.analyse_MRI_behav.evaluate_model(model_model, d)
        
        
        
        if binary == True:           
            print("remember to set binary value depending on model RDM!!")
            # for d in data_RDM:
            #       RDM_my_model_dir[model] = mc.analyse.analyse_MRI_behav.evaluate_binary_model(model_model, d, binary_val=0.5)
            results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results-bin"
            if smoothing == True:
                results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}_smooth{fwhm}/results-bin"
            RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_binary_model)(model_model, d, binary_val=0.5) for d in tqdm(data_RDM, desc=f"running GLM for all searchlights in {model}"))
            
            # DELETE LATER, THIS IS FOR DEBUGGING PURPOSES!
            # mc.plotting.deep_data_plt.save_changed_voxel_val(20, RDM_my_model_dir[model], data_RDM, [34,37,34], results_dir, f"dummy_{model}", ref_img)
        
            mc.analyse.analyse_MRI_behav.save_RSA_result_binary(result_file=RDM_my_model_dir[model], data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model}", mask=mask, ref_image_for_affine_path=ref_img)
    
        else:
            RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(model_model, d) for d in tqdm(data_RDM, desc=f"running GLM for all searchlights in {model}"))
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_model_dir[model], data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    
  
       

    # import pdb; pdb.set_trace() 
  # SECOND RSA: combo models.
  # I am interested in:
      # combo clocks with midnight, phase, state and location included
      # combo split clocks with now, one future, two future, [three future]

    if RDM_version in ['01']:
        multiple_regressors_first = ['direction_presentation', 'execution_similarity', 'presentation_similarity']
        model_name = 'combo-instr'
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)
    
    # combo clocks and controls    
    elif RDM_version in ['02', '02-act', '02-act-1phas']: # modelling all
        # first: clocks with midnight, phase, state and location.
        multiple_regressors_first = ['clocks', 'midnight', 'state', 'location', 'phase']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)       
        model_name = 'combo-cl-mid-st-loc-ph'

    # combo clocks and controls
    elif RDM_version in ['02', '02-A'] and regression_version in ['03', '03-4', '04', '04-4']: # don't model location and midnight together if reduced to reward times as they are the same.
        # # first: clocks with midnight, phase, state and location.
    
        multiple_regressors_first = ['clocks', 'midnight', 'state', 'phase']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)    
        model_name = 'combo-cl-mid-st-ph'


    # combo clocks and controls
    elif RDM_version == '03' and regression_version in ['02', '04']: # modeling only reward rings
        # # first: clocks with midnight, phase, state and location.
        multiple_regressors_first = ['clocks_only-rew', 'midnight_only-rew', 'state', 'location', 'phase']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)        
        model_name = 'combo-cl-mid-st-loc-ph'


    # combo clocks and controls
    elif RDM_version == '03' and regression_version in ['03']: # don't model location and midnight together if reduced to reward times as they are the same.
        # # first: clocks with midnight, phase, state and location.
        multiple_regressors_first = ['clocks_only-rew', 'midnight_only-rew', 'state', 'phase']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)     
        model_name = 'combo-cl-mid-st-ph'


    elif RDM_version == '03-tasklag' and regression_version in ['03', '03-4']:
        # 'location', 'phase', 'state', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew', 'curr_rings_split_clock_sin', 'one_fut_rings_split_clock_sin', 'two_fut_rings_split_clock_sin', 'three_fut_rings_split_clock_sin', 'clocks_only-rew_sin']
    
        multiple_regressors_first_cos = ['clocks_only-rew', 'location', 'state', 'phase']
        results_combo_model_cos = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first_cos, model_RDM_dir, data_RDM)     
        model_name = 'combo-coscl-loc-st-ph'
    
          # already compute the cosine here bc only one will be taken after
        for i, model in enumerate(multiple_regressors_first_cos):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model_cos, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
    
    
        multiple_regressors_first = ['clocks_only-rew_sin', 'location', 'state', 'phase']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)     
        model_name = 'combo-sincl-loc-st-ph'
    

    
    # combo clocks and controls
    if RDM_version in ['03-1'] and regression_version in ['03', '03-4', '03-l', '03-e', '03-rep1', '03-rep2','03-rep3','03-rep4','03-rep5']:
        multiple_regressors_first = ['curr-and-future-rew-locs', 'location', 'phase', 'state']
        results_combo_model= mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)
        model_name = 'combo-clrw-loc-ph-st'

    if RDM_version in ['03-1-act']:
        multiple_regressors_first = ['curr-and-future-rew-locs', 'location', 'phase', 'state', 'action-box_only-rew', 'buttons']
        results_combo_model= mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)
        model_name = 'combo-clrw-loc-ph-st-act-but'

    # combo clocks and controls
    elif RDM_version == '04': #modelling only path rings
        multiple_regressors_first = ['clocks_no-rew', 'midnight_no-rew', 'state', 'location', 'phase']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)
        model_name = 'combo-cl-mid-st-loc-ph'
    
    # combo comparing no-reward rings with reward rings.
    elif RDM_version in ['05']:
          multiple_regressors_first = ['clocks_no-rew', 'clocks_only-rew']
          results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)
          model_name = 'combo_onlyrewnowrew-rings'
    
    if RDM_version not in ['03-5', '03-5-A', '04-5', '04-5-A']:  # these are the state models and I am not defining more than 1 model for it.
        # then, compute the first combo model.
        for i, model in enumerate(multiple_regressors_first):
              mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)


  # SECOND COMBO MODEL

    # combo split clocks
    if RDM_version in ['02', '03', '04', '02-A', '02-act', '02-act-1phas']:
          # second: split clock: now/ midnight; one future, two future, three future
          multiple_regressors = ['curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock']
          results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
          model_name = 'combo_split_clock'

    elif RDM_version == '03-tasklag' and regression_version in ['03', '03-4']:
          # 'location', 'phase', 'state', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew', 'curr_rings_split_clock_sin', 'one_fut_rings_split_clock_sin', 'two_fut_rings_split_clock_sin', 'three_fut_rings_split_clock_sin', 'clocks_only-rew_sin']
          # second: split clock: now/ midnight; one future, two future, three future
          multiple_regressors = ['curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock']
          results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
          model_name = 'combo_split_clock_cos'
          # do it once bc there need to be 2 combo models
          for i, model in enumerate(multiple_regressors):
              mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
        
          multiple_regressors = ['curr_rings_split_clock_sin', 'one_fut_rings_split_clock_sin', 'two_fut_rings_split_clock_sin', 'three_fut_rings_split_clock_sin',]
          results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
          model_name = 'combo_split_clock_sin'


    # combo split clocks    
    elif RDM_version in ['03-1', '03-1-act'] and regression_version in ['03', '03-4','03-l', '03-e', '03-rep1', '03-rep2', '03-rep3', '03-rep4', '03-rep5']:
      multiple_regressors = ['location', 'one_future_rew_loc', 'two_future_rew_loc', 'three_future_rew_loc']
      results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
      model_name = 'combo_split-clock'

    # combo split clocks with state to control
    elif RDM_version in ['03-1'] and regression_version in ['03', '03-4','03-l', '03-e', '03-rep1', '03-rep2', '03-rep3', '03-rep4', '03-rep5']:
      multiple_regressors = ['location', 'one_future_rew_loc', 'two_future_rew_loc', 'three_future_rew_loc', 'state']
      results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
      model_name = 'combo_split-clock-state'


    elif RDM_version in ['05']:
      # compare 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', and 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock'
      multiple_regressors = ['midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock']
      results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
      model_name = 'combo_onlyrewnowrew-split_clocks'

    if RDM_version not in ['03-5', '03-5-A', '04-5', '04-5-A']: # these are the state models and I am not defining more than 1 model for it
        # then, finally, compute the results for the second combo model.   
        for i, model in enumerate(multiple_regressors):
          mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)

    # THIRD COMBO MODEL
    if RDM_version in ['05'] and neuron_weighting == True:
        #  UNCOMMENT AGAIN LATER!!
        # multiple_regressors = ['clocks_no-rew-cos', 'clocks_only-rew-cos', 'clocks_no-rew-sin', 'clocks_only-rew-sin']
        # results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)
        # model_name = 'combo_onlyrew-nowrew-cos-sin'
    
        # for i, model in enumerate(multiple_regressors):
        #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
        
        # multiple_regressors = ['location', 'state', 'clocks_only-rew-cos', 'clocks_only-rew-sin']
        # results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_first, model_RDM_dir, data_RDM)
        # model_name = 'combo_onlyrew-cos-sin-loc-st'
    
        # for i, model in enumerate(multiple_regressors):
        #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
        
        
        multiple_regressors_one = ['clocks_only-rew', 'clocks_only-rew-sin']
        results_combo_model_one = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_one, model_RDM_dir, data_RDM)
        model_name_one = 'combo_onlyrew-clock-sin'
    
        for i, model in enumerate(multiple_regressors_one):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model_one, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name_one}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
        
        multiple_regressors = ['clocks_only-rew', 'clocks_only-rew-cos']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
        model_name = 'combo_onlyrew-clock-cos'
    
        for i, model in enumerate(multiple_regressors):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
        
        
        multiple_regressors_two = ['clocks_only-rew', 'clocks_only-rew-cos', 'clocks_only-rew-sin']
        results_combo_model_two = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors_two, model_RDM_dir, data_RDM)
        model_name_two = 'combo_onlyrew-clock-cos-sin'
    
        for i, model in enumerate(multiple_regressors_two):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model_two, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name_two}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
        
    if RDM_version in ['05', '03-5', '03-5-A', '04-5', '04-5-A']:
        multiple_regressors = ['state_masked', 'location']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
        model_name = 'combo_state_loc'
        for i, model in enumerate(multiple_regressors):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
    
    
    if RDM_version in ['03-1-act']:
        multiple_regressors = ['action-box_only-rew', 'buttonsXphase_only-rew', 'buttons', 'location', 'phase', 'state']
        results_combo_model= mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
        model_name = 'combo-actrw-buph-bu-loc-ph-st'
        for i, model in enumerate(multiple_regressors):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
    
    if RDM_version in ['03-1-act']:
        multiple_regressors = ['action-box_only-rew', 'clocks_only-rew', 'buttons', 'location']
        results_combo_model= mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
        model_name = 'combo-actrw-cl-bu-loc'
        for i, model in enumerate(multiple_regressors):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
    
        multiple_regressors = ['buttons', 'one_future_step2rew', 'two_future_step2rew', 'three_future_step2rew']
        results_combo_model = mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
        model_name = 'combo_split-actionbox'
        for i, model in enumerate(multiple_regressors):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
    
    if RDM_version in ['02-act', '02-act-1phas']:
        multiple_regressors = ['curr_subpath_buttons', 'one_future_subp_buttons', 'two_future_subp_buttons', 'three_future_subp_buttons']
        results_combo_model= mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
        model_name = 'combo_split-actionbox'
        for i, model in enumerate(multiple_regressors):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
    
        multiple_regressors = ['action-box','clocks', 'buttons', 'location']
        results_combo_model= mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
        model_name = 'combo-act-cl-bu-loc'
        for i, model in enumerate(multiple_regressors):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
    
        multiple_regressors = ['action-box','buttonsXphase', 'buttons', 'location', 'phase', 'state']
        results_combo_model= mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
        model_name = 'combo-act-buph-bu-loc-ph-st'
        for i, model in enumerate(multiple_regressors):
            mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
    