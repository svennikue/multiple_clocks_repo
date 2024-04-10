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
    
    03 -> modelling only rewards + splitting model in the same function.
    03-1 -> modelling only rewards + splitting the model after regression 
    03-2 -> same as 03-1 but only considering task D and B (where 2 rew locs are the same)
    03-3 -> same as 03-1 but only considering B,C,D [excluding rew A] -> important to be paired with GLM 03-3!
    03-5 - STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. 
    03-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    03-99 ->  using 03-1 - reward locations and future rew model; but EVs are scrambled.
    03-999 ->  is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.    
    
    04 -> modelling only paths
    04-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    
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
    
@author: Svenja Küchenhoff, 2024
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

RDM_version = '02-A' 
regression_version = '04-4' 


# import pdb; pdb.set_trace() 
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

subjects = [f"sub-{subj_no}"]
#subjects = ['sub-01']

load_old = False
visualise_RDMs = False


task_halves = ['1', '2']

print(f"Now running RSA for RDM version {RDM_version} based on subj GLM {regression_version} for subj {subj_no}")


if RDM_version in ['01', '01-1']: # 01 doesnt work yet! 
    models_I_want = ['direction_presentation', 'execution_similarity', 'presentation_similarity']

elif RDM_version in ['02', '02-A']: #modelling paths + rewards, creating all possible models 
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight', 'clocks']

elif RDM_version in ['03']: # modelling only rewards, splitting clocks within the same function
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew']
elif RDM_version in ['03-1', '03-2']:  # modelling only rewards, splitting clocks later in a different way - after the regression.
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
elif RDM_version in ['03-3']:  # modelling only rewards, splitting clocks later in a different way - after the regression; ignoring reward A
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc']
elif RDM_version in ['03-5', '03-5-A', '04-5', '04-5-A']:
    models_I_want = ['state']
elif RDM_version in ['03-99']:  # using 03-1 - reward locations and future rew model; but EVs are scrambled.
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
elif RDM_version in ['03-999']:  # is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew']
elif RDM_version in ['04', '04-A']: # only paths. to see if the human brain represents also only those rings anchored at no-reward locations
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_no-rew', 'clocks_no-rew']
      
# based on GLM
if regression_version == '01':
    no_RDM_conditions = 20 # including all instruction periods
elif regression_version == '02':
    no_RDM_conditions = 80 # including all paths and rewards
elif regression_version in ['03', '04','03-99', '03-999', '03-9999']:
    no_RDM_conditions = 40 # only including rewards or only paths
elif regression_version == '03-3': #excluding reward A
    no_RDM_conditions = 30
elif regression_version == '03-4': # only including tasks without double reward locs: A,C,D  and only rewards
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
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.makedirs(f"{results_dir}/results")
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results"  
    # if os.path.exists(results_dir):
    #     # move pre-existing files into a different folder.
    #     mc.analyse.analyse_MRI_behav.move_files_to_subfolder(results_dir)
    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    # load the file which defines the order of the model RDMs, and hence the data RDMs
    with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'rb') as file:
        sorted_keys = pickle.load(file)
    with open(f"{RDM_dir}/sorted_regs.pkl", 'rb') as file:
        reg_keys = pickle.load(file)
    # also store 2 dictionaries of the EVs
    if regression_version in ['03-3', '03-4']:
        regression_version = '03'
    if regression_version in ['04-4']:
        regression_version = '04'
            
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
    mask = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    mask = mask.get_fdata()  
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
    if load_old:
        with open(f"{results_dir}/data_RDM.pkl", 'rb') as file:
            data_RDM = pickle.load(file)
        if visualise_RDMs == True:
            mc.analyse.analyse_MRI_behav.visualise_data_RDM(mni_x=53, mni_y = 30, mni_z= 2, data_RDM_file= data_RDM, mask=mask)
            
    else:
        data_RDM_file_2d = {}
        data_RDM_file = {}
        data_RDM_file_1d = {}
        reading_in_EVs_dict = {}
        image_paths = {}
        
        
        # I need to do this slightly differently. I want to be super careful that I create 2 'identical' splits of data.
        # thus, check which folder has the respective task.
        for split in sorted_keys:
            if RDM_version == '01':
                # DOUBLE CHECK IF THIS IS EVEN STILL CORRECT!!!
                # for condition 1, I am ignoring task halves. to make sure everything goes fine, use the .txt file
                # and only load the conditions in after the task-half loop.
                pe_path = f"{data_dir}/func/glm_{regression_version}_pt0{split}.feat/stats"
                with open(f"{data_dir}/func/EVs_{RDM_version}_pt0{split}/task-to-EV.txt", 'r') as file:
                    for line in file:
                        index, name = line.strip().split(' ', 1)
                        reading_in_EVs_dict[f"{name}_EV_{index}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
            else:           
                i = -1
                image_paths[split] = [None] * no_RDM_conditions # Initialize a list for each half of the dictionary
                data_RDM_file[split] = [None] * no_RDM_conditions  # Initialize a list for each half of the dictionary
                for EV_no, task in enumerate(sorted_keys[split]):
                    for regressor_sets in reg_keys:
                        if regressor_sets[0].startswith(task):
                            curr_reg_keys = regressor_sets
                    for reg_key in curr_reg_keys:
                        # print(f"now looking for {task}")
                        for EV_01 in reading_in_EVs_dict_01:
                            if EV_01.startswith(reg_key):
                                i = i + 1
                                # print(f"looking for {task} and found it in 01 {EV_01}, index {i}")
                                image_paths[split][i] = reading_in_EVs_dict_01[EV_01]  # save path to check if everything went fine later
                                data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_01[EV_01]).get_fdata()
                        for EV_02 in reading_in_EVs_dict_02:
                            if EV_02.startswith(reg_key):
                                i = i + 1
                                # print(f"looking for {task} and found it in 01 {EV_02}, index {i}")
                                image_paths[split][i] = reading_in_EVs_dict_02[EV_02]
                                data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_02[EV_02]).get_fdata() 
                                # Convert the list to a NumPy array
                
                print(f"This is the order now: {image_paths[split]}")
                data_RDM_file[split] = np.array(data_RDM_file[split])
                # reshape data so we have n_observations x n_voxels
                data_RDM_file_2d[split] = data_RDM_file[split].reshape([data_RDM_file[split].shape[0], -1])
                data_RDM_file_2d[split] = np.nan_to_num(data_RDM_file_2d[split]) # now this is 80timepoints x 746.496 voxels
                
                if RDM_version == f"{RDM_version}_999": # scramble voxels randomly
                    data_RDM_file_1d[split] = data_RDM_file_2d[split].flatten()
                    np.random.shuffle(data_RDM_file_1d[split]) #shuffle all voxels randomly
                    data_RDM_file_2d[split] = data_RDM_file_1d[split].reshape(data_RDM_file_2d[split].shape) # and reshape

        
        if RDM_version in ['01']:
            data_RDM_file_2d = {}
            data_RDM_file = {}
            data_RDM_file[RDM_version] = [None] * no_RDM_conditions
            # sort across task_halves
            for i, task in enumerate(sorted(reading_in_EVs_dict.keys())):
                if task not in ['ev_press_EV_EV_index']:
                    image_paths[i] = reading_in_EVs_dict[task]
                    data_RDM_file[RDM_version][i] = nib.load(image_paths[i]).get_fdata()
            # Convert the list to a NumPy array
            data_RDM_file_np = np.array(data_RDM_file[RDM_version])
            # reshape data so we have n_observations x n_voxels
            data_RDM_file_2d = data_RDM_file_np.reshape([data_RDM_file_np.shape[0], -1])
            data_RDM_file_2d = np.nan_to_num(data_RDM_file_2d) # now this is 20timepoints x 746.496 voxels

            print(f"This is the order now: {image_paths}")  

        
        # define the conditions, combine both task halves
        data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1,2)).transpose(),2*no_RDM_conditions)  
        # now prepare the data RDM file. 
        # final data RDM file; 
        if RDM_version in ['01', '01-1']:
            data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1)).transpose(),no_RDM_conditions)  
            data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, data_conds, method='correlation')
        else:
            # this is defining both task halves/ runs: 0 is first half, the second one is 1s
            sessions = np.concatenate((np.zeros(int(data_RDM_file['1'].shape[0])), np.ones(int(data_RDM_file['2'].shape[0]))))  
            # for all other cases, cross correlated between task-halves.
            data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, data_conds, method='crosscorr', cv_descr=sessions)
            # save  so that I don't need to recompute - or don't save bc it's massive
            # with open(f"{results_dir}/data_RDM.pkl", 'wb') as file:
                # pickle.dump(data_RDM, file)

    
    # Step 3: load and compute the model RDMs.
    # load the data files I created.
    data_dirs = {}
    for model in models_I_want:
        if RDM_version in ['999', '9999']: # potentially delete?? this is now 03-99 nd 03-999
            RDM_dir = f"{data_dir}/beh/RDMs_09_glmbase_{regression_version}" # potentially delete??
        data_dirs[model]= np.load(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves.npy")) 
    
    
    # step 3: create model RDMs
    # first, each model gets its own, separate estimation.
    model_RDM_dir = {}
    RDM_my_model_dir = {}
    for model in data_dirs:
        model_data = mc.analyse.analyse_MRI_behav.prepare_model_data(data_dirs[model], no_RDM_conditions, RDM_version)
        if RDM_version in ['01', '01-1']:
            model_RDM_dir[model] = rsr.calc_rdm(model_data, method='correlation', descriptor='conds')
        else:
            model_RDM_dir[model] = rsr.calc_rdm(model_data, method='crosscorr', descriptor='conds', cv_descriptor='sessions')
        
        if RDM_version in ['03-5', '03-5-A', '04-5', '04-5-A']:
            state_mask = np.load(os.path.join(RDM_dir, f"RSM_state_masked_{sub}_fmri_both_halves.npy"))
            state_mask_flat = list(state_mask[np.triu_indices(len(state_mask), 1)])
            # state_mask_flat = [int(x) for x in state_mask_flat]
            boolean_mask = np.isnan(state_mask_flat)
            model_RDM_dir[model].dissimilarities[0][boolean_mask] = np.nan
            #model_RDM_dir[model].dissimilarities = np.where(state_mask_flat == 1, model_RDM_dir[model].dissimilarities, np.nan)
            #test = np.where(state_mask_flat == 1, model_RDM_dir[model].dissimilarities, np.nan)

        # import pdb; pdb.set_trace()
        fig, ax, ret_vla = rsatoolbox.vis.show_rdm(model_RDM_dir[model])
        # then compute the location model.
        model_model = rsatoolbox.model.ModelFixed(f"{model}_only", model_RDM_dir[model])
        # Step 4: evaluate the model fit between model and data RDMs.
        RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(model_model, d) for d in tqdm(data_RDM, desc=f"running GLM for all searchlights in {model}"))
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_model_dir[model], data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
      
    # second, combo models.
    # I am interested in:
        # combo clocks with midnight, phase, state and location included
        # combo split clocks with now, one future, two future, [three future]


    if RDM_version in ['01']:
        instruction_comp_RDM = rsatoolbox.rdm.concat(model_RDM_dir['direction_presentation'], model_RDM_dir['execution_similarity'], model_RDM_dir['presentation_similarity'])
        instruction_comp_model = rsatoolbox.model.ModelWeighted('instruction_comp_RDM', instruction_comp_RDM)
        results_instruction_comp_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(instruction_comp_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - different instruction models'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_instruction_comp_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "DIRECTION-PRESENTATION-combo-instr", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_instruction_comp_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "EXECUTION-SIM-combo-instr", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_instruction_comp_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "PRESENTATION-SIM-combo-instr", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)

        
        
    if RDM_version == '02': # modelling all
        # first: clocks with midnight, phase, state and location.
        clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks'], model_RDM_dir['midnight'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
        clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_loc_ph_RDM)
        # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
        results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state, loc'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CLOCK-combo-cl-mid-st-loc-ph", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "MIDN-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "STATE-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "LOC-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "PHASE-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 4, ref_image_for_affine_path=ref_img)

    if RDM_version in ['02', '02-A'] and regression_version in ['03', '03-4', '04', '04-4']: # don't model location and midnight together if reduced to reward times as they are the same.
        # first: clocks with midnight, phase, state and location.
        clocks_midn_states_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks'], model_RDM_dir['midnight'], model_RDM_dir['state'], model_RDM_dir['phase'])
        clocks_midn_states_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_ph_RDM)
        # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
        results_clocks_midn_states_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state')) 
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CLOCK-combo-cl-mid-st-ph", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "MIDN-combo_cl-mid-st-ph", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "STATE-combo_cl-mid-st-ph", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "PHASE-combo_cl-mid-st-ph", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)


    if RDM_version == '03' and regression_version in ['02', '04']: # modeling only reward rings
        # first: clocks with midnight, phase, state and location.
        clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks_only-rew'], model_RDM_dir['midnight_only-rew'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
        clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_loc_ph_RDM', clocks_midn_states_loc_ph_RDM)
        # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
        results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state, loc'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CLOCKrw-combo-cl-mid-st-loc-ph", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "MIDNrw-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "STATE-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "LOC-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "PHASE-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 4, ref_image_for_affine_path=ref_img)

    if RDM_version == '03' and regression_version in ['03']: # don't model location and midnight together if reduced to reward times as they are the same.
        # first: clocks with midnight, phase, state and location.
        clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks_only-rew'], model_RDM_dir['midnight_only-rew'], model_RDM_dir['state'], model_RDM_dir['phase'])
        clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_loc_ph_RDM', clocks_midn_states_loc_ph_RDM)
        # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
        results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state, loc'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CLOCKrw-combo-cl-mid-st-ph", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "MIDNrw-combo_cl-mid-st-ph", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "STATE-combo_cl-mid-st-ph", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "PHASE-combo_cl-mid-st-ph", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)


    if RDM_version == '04': #modelling only path rings
        # first: clocks with midnight, phase, state and location.
        clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks_no-rew'], model_RDM_dir['midnight_no-rew'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
        clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_loc_ph_RDM', clocks_midn_states_loc_ph_RDM)
        # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
        results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state, loc'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CLOCKnorw-combo-cl-mid-st-loc-ph", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "MIDNnorw-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "STATE-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "LOC-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "PHASE-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 4, ref_image_for_affine_path=ref_img)

    if RDM_version in ['02', '03', '04', '02-A']:
        # second: split clock: now/ midnight; one future, two future, three future
        split_clocks_RDM = rsatoolbox.rdm.concat(model_RDM_dir['curr_rings_split_clock'], model_RDM_dir['one_fut_rings_split_clock'], model_RDM_dir['two_fut_rings_split_clock'], model_RDM_dir['three_fut_rings_split_clock'])
        split_clocks_model = rsatoolbox.model.ModelWeighted('split_clocks_RDM', split_clocks_RDM)
        results_split_clocks_combo_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(split_clocks_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - split clocks'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_split_clocks_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CURR-RINGS_combo_split_clock", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_split_clocks_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "ONE-FUTR-RINGS_combo_split_clockh", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_split_clocks_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "TWO-FUTR-RINGS_combo_split_clock", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_split_clocks_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "THRE-FUTR-RINGS_combo_split_clock", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)

        
    if RDM_version in ['03-1'] and regression_version in ['03']:
        split_clocks_RDM = rsatoolbox.rdm.concat(model_RDM_dir['location'], model_RDM_dir['one_future_rew_loc'], model_RDM_dir['two_future_rew_loc'], model_RDM_dir['three_future_rew_loc'])
        split_clocks_model = rsatoolbox.model.ModelWeighted('split_clocks_RDM', split_clocks_RDM)
        
        results_current_and_all_future_rew_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(split_clocks_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - split clocks after regressiom'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CURR_REW-combo_split-clock", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "ONE-FUT-REW_combo_split-clock", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "TWO-FUT-REW_combo_split-clock", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "THREE-FUT-REW_combo_split-clock", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
        




    # if RDM_version == '05': 
        
    #     clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks'], model_RDM_dir['midnight'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
    #     #clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(clocks_RDM, midnight_RDM, state_RDM, loc_RDM, phase_RDM)
    #     clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_loc_ph_RDM)
    #     # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
    #     results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model 2'))
        
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CLOCK-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "MIDN-clock-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "STATE-clock-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "LOC-clock-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "PHASE-clock-combo_cl-mid-st-loc-ph", mask=mask, number_regr = 4, ref_image_for_affine_path=ref_img)

    # if RDM_version == '08': 
    #     #['reward_midnight_v2', 'reward_clocks_v2', 'state', 'task_prog']
        
    #     clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks'], model_RDM_dir['midnight'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
    #     #clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(clocks_RDM, midnight_RDM, state_RDM, loc_RDM, phase_RDM)
    #     clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_loc_ph_RDM)
    #     # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
    #     results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model 2'))
        
        
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_clock_betw_halves", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_midn_betw_halves", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_state_betw_halves", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_loc_betw_halves", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_phase_betw_halves", mask=mask, number_regr = 4, ref_image_for_affine_path=ref_img)

    # if RDM_version == '09' or RDM_version == '999':
    #     # I want to have a combined 
    #     current_and_all_future_rew_RDM = rsatoolbox.rdm.concat(model_RDM_dir['reward_location'], model_RDM_dir['one_future_rew_loc'], model_RDM_dir['two_future_rew_loc'], model_RDM_dir['three_future_rew_loc'])
    #     current_and_all_future_rew_model = rsatoolbox.model.ModelWeighted('current_and_all_future_rew_RDM', current_and_all_future_rew_RDM)
        
    #     results_current_and_all_future_rew_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(current_and_all_future_rew_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model'))
        
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CURR_REW-combo_split-clock", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "ONE-FUT-REW_combo_split-clock", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "TWO-FUT-REW_combo_split-clock", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "THREE-FUT-REW_combo_split-clock", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
        
    #     # additionally do clocks vs. current rew location 
    #     clocks_curr_rew_RDM = rsatoolbox.rdm.concat(model_RDM_dir['reward_location'], model_RDM_dir['reward_clocks_v2'])
    #     clocks_curr_rew_RDM_model = rsatoolbox.model.ModelWeighted('clocks_curr_rew_RDM', clocks_curr_rew_RDM)
        
    #     results_clocks_curr_rew_RDM_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_curr_rew_RDM_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model'))
        
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_curr_rew_RDM_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CURR-REW_combo_cl_mid", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_curr_rew_RDM_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "CLOCKS_combo_cl_mid", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        
    # if RDM_version == '10':
    #     # I want to have a combined 
    #     current_and_all_future_rew_RDM = rsatoolbox.rdm.concat(model_RDM_dir['reward_location'], model_RDM_dir['one_future_rew_loc'], model_RDM_dir['two_future_rew_loc'])
    #     current_and_all_future_rew_model = rsatoolbox.model.ModelWeighted('current_and_all_future_rew_RDM', current_and_all_future_rew_RDM)
        
    #     results_current_and_all_future_rew_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(current_and_all_future_rew_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model'))
        
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "curr_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "one_fut_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "two_fut_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)

    #     # additionally do clocks vs. current rew location 
    #     clocks_curr_rew_RDM = rsatoolbox.rdm.concat(model_RDM_dir['reward_location'], model_RDM_dir['reward_clocks_v2'])
    #     clocks_curr_rew_RDM_model = rsatoolbox.model.ModelWeighted('clocks_curr_rew_RDM', clocks_curr_rew_RDM)
        
    #     results_clocks_curr_rew_RDM_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(current_and_all_future_rew_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model'))
        
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_curr_rew_RDM_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "curr_reward_location_combo_cl_mid", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    #     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_curr_rew_RDM_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "clocks_location_combo_cl_mid", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)


####################### play around with RDM visualisation.





