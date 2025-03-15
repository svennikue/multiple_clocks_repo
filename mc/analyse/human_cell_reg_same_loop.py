
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:05:55 2025

elastic net regression
in same loop

@author: Svenja Küchenhoff

replicating and somewhat adjusting El-Gaby's Figure 5 regression



"""
import numpy as np
import mc
import matplotlib.pyplot as plt
import os
import pickle
import time
from operator import itemgetter
import scipy.stats as st
from sklearn.linear_model import ElasticNet
import copy

# DONE. firstly take the mean across held-out tasks! 
# then correct the randomised version: create musicbox but with randomised reward locations per subject
# then do that loads of times, create a permutation distribution of means and test vs actual true mean valiue
# DONE. then do the same but holding out current reward, then current +m next, and see how it changes entorhinal vs pfc
# mice firing rate round about 1 and 2 Hz
# humans higher or lower?? 
# try permuting rows of design matrix to test
# or use reward locations from wrong grids/ use regressors from different grids to test


### SETTINGS
data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
group_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
# check if on server or local
if not os.path.isdir(group_folder):
    print("running on ceph")
    group_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group/elastic_net_reg"
    data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"

file_name_all_subj_reg_prep = f"prep_data_for_regression"

subjects = [f"{i:02}" for i in range(1, 58) if i not in [9, 27, 43, 44]]
# 9 because the original timings from habiba are negative
# 27, 43 and 44 because they skipped a grid and I want to be careful to not mess up timings
# subjects = [f"{i:02}" for i in range(55,58) if i not in [9, 27, 43, 44]]

save_results = True
tt = time.time()
alpha=0.001 ##0.01 used in El-gaby paper
l1_ratio= 0.01

jumbled_regressors = False
randomised_reward_locations = False
models_I_want = ['withoutnow', 'only2and3future','onlynowandnext'] # ['withoutnow', 'only2and3future','onlynowandnext']# None or 'withoutnow, only2and3future','onlynowandnext'
exclude_repeats = [1,2] # None or values

# check following subjects cells:
    # s42
    # s 41



#### Setting up folder and files
if models_I_want:
    file_name_all_subj_reg_prep = f"prep_data_for_regression_w_partial_musicboxes"
if randomised_reward_locations == True:
    file_name_all_subj_reg_prep = f"prep_data_for_regression_random_rew_order"
if not os.path.isdir(group_folder):
    os.mkdir(group_folder)
    
if exclude_repeats:
    file_name_all_subj_reg_prep = f"{file_name_all_subj_reg_prep}_excl_rep{exclude_repeats[0]}-{exclude_repeats[-1]}"
    

reg_dir = f"{group_folder}/regression"
reg_fig_dir = f"{group_folder}/regression/figures"
if not os.path.isdir(reg_dir):
    os.mkdir(reg_dir)
if not os.path.isdir(reg_fig_dir):
    os.mkdir(reg_fig_dir)


#### Loading data
if os.path.isfile(os.path.join(group_folder,file_name_all_subj_reg_prep)):
    with open(os.path.join(group_folder,file_name_all_subj_reg_prep), 'rb') as f:
        data_and_regressors = pickle.load(f)
        print(f"opened stored dataset")
else:  
    print(f"loading and running preprocessing of human cells")
    data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)
    # PART 1 
    # prepare all regressors for the data I want, in the same array format as the cells.
    # any regressors I want (e.g. current location, next location, state etc)
    # for all grids, all subjects: 
        # simulate 4x 9 location neurons
        # fires when currently at location, when next reward, next next, and prev. reward location
    data_and_regressors = mc.analyse.helpers_human_cells.prep_regressors_for_neurons(data, models_I_want=models_I_want, exclude_x_repeats=exclude_repeats, randomised_reward_locations=randomised_reward_locations)
    
    # if save_results == True:
    #     # save the all_modelled_data dict such that I don't need to always run it again.
    #     with open(os.path.join(group_folder,file_name_all_subj_reg_prep), 'wb') as f:
    #         pickle.dump(data_and_regressors, f)
    #     print(f"saved the modelled data as {group_folder}/{file_name_all_subj_reg_prep}")
          

# del data # freeing up space

corr_dict = {}
corr_dict_binned = {}

# do this for only one subject to try with.
# for sub in data_and_regressors: 
#for sub in ['sub-01']: 
for sub in data_and_regressors:

    tt_start=time.time()
    corr_dict[sub] = {}
    corr_dict_binned[sub] = {}
    
    single_sub_dict = copy.deepcopy(data_and_regressors[sub])
    if sub == 'sub-15':
        single_sub_dict['reward_configs'] = np.delete(single_sub_dict['reward_configs'], 23, axis = 0)

    for model in single_sub_dict:
        if model.endswith('reg') or model.endswith('model'):
            corr_dict[sub][model], corr_dict_binned[sub][model] = {}, {}
            
    # # PART 2
    # # run a regression per neuron (ALL neurons and ALL subjects)
    # # load if already exists
    file_name = f"standard_regs_all_cells_all_models_{sub}"
    if models_I_want:
        file_name = f"all_regs_w_any_musicbox_all_cells_{sub}"
    if jumbled_regressors == True:
        file_name = f"jumbled_all_regs_all_cells_all_models_{sub}"
    if randomised_reward_locations == True:
        file_name = f"regs_all_cells_standard_models_pseudorandom_rew_locs_{sub}"
    
    if exclude_repeats:
        file_name = f"{file_name}_excl_rep{exclude_repeats[0]}-{exclude_repeats[-1]}"
    
    if os.path.isfile(os.path.join(group_folder,f"{file_name}")):
        with open(os.path.join(group_folder,file_name), 'rb') as f:
            result_dict = pickle.load(f)
            if models_I_want:
                print(f"opened stored dataset also of partial musicboxes for {sub}") 
            elif jumbled_regressors == True:
                print(f"opened jumbled stored dataset for {sub}") 
            else:
                print(f"opened stored dataset for {sub}") 
    
    # elif os.path.isfile(os.path.join(group_folder,f"all_regs_all_cells_all_models_{sub}")):
    #     if jumbled_regressors == False:
    #         with open(os.path.join(group_folder,f"all_regs_all_cells_all_models_{sub}"), 'rb') as f:
    #             result_dict = pickle.load(f)
    #             print(f"opened stored dataset for {sub}") 
            
    
    
    else:
        print(f"fitting and testing models for {sub}")
        result_dict = {}
        for cell_idx, cell in enumerate(single_sub_dict['neurons'][0]):
            # create one entry in result_dict per cell 
            curr_cell = f"{single_sub_dict['cell_labels'][cell_idx]}_{cell_idx}_{sub}"
            result_dict[curr_cell] = {}
            
            # check for unique grid combos 
            unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(single_sub_dict['reward_configs'], axis=0,
                                                                    return_index=True,
                                                                    return_inverse=True,
                                                                    return_counts=True)
        
            for model in single_sub_dict:
                if model.endswith('reg'):
                    corr_dict[sub][model][curr_cell] = np.zeros(len(unique_grids))
                    corr_dict_binned[sub][model][curr_cell] = np.zeros(len(unique_grids))
                    
                
            for task_idx, left_out_grid_idx in enumerate(idx_unique_grid): 
                test_grid_idx = np.where(idx_same_grids == idx_same_grids[left_out_grid_idx])[0]
                if not test_grid_idx.shape == (1,):
                    all_neurons_heldouttasks = itemgetter(*test_grid_idx)(single_sub_dict['neurons'])
                    curr_neuron_heldouttasks = [all_neurons[cell_idx] for all_neurons in all_neurons_heldouttasks]
                    curr_neuron_heldouttasks_flat = np.concatenate(curr_neuron_heldouttasks)
                else:
                    all_neurons_heldouttasks = single_sub_dict['neurons'][test_grid_idx[0]]
                    curr_neuron_heldouttasks_flat = all_neurons_heldouttasks[cell_idx]
                
                training_grid_idx=np.setdiff1d(np.arange(len(single_sub_dict['reward_configs'])),test_grid_idx)
                all_neurons_training_tasks = itemgetter(*training_grid_idx)(single_sub_dict['neurons'])
                curr_neuron_training_tasks = [all_neurons[cell_idx] for all_neurons in all_neurons_training_tasks]
                
                # depending on the permutations, change the train and test dataset.
                for entry in single_sub_dict:
                    # only the regressors I created.
                    if entry.endswith('reg') or entry.endswith('model'):
                        result_dict[curr_cell][entry] = []
                        
                        # then choose which tasks to take and go and select training regressors
                        simulated_neurons_training_tasks = itemgetter(*training_grid_idx)(single_sub_dict[entry])
                        # and test regressors
                        simulated_neurons_test_grids = itemgetter(*test_grid_idx)(single_sub_dict[entry])
                        if not test_grid_idx.shape == (1,):
                            simulated_neurons_test_grids_flat = np.transpose(np.concatenate(simulated_neurons_test_grids, axis = 1))
                        else:
                            simulated_neurons_test_grids_flat = np.transpose(simulated_neurons_test_grids)
                            
                        # for binning later on, store the state regressor
                        if entry == "state_reg":
                            state_model_curr_testgrid = simulated_neurons_test_grids_flat.copy()
                    
                        if jumbled_regressors == True:
                            # shuffle the order of regressors
                            indices = np.random.permutation(len(simulated_neurons_training_tasks))
                            shuffled_tasks = [simulated_neurons_training_tasks[i] for i in indices]
    
                            # Then concatenate along axis 1 and transpose as before
                            X = np.transpose(np.concatenate(shuffled_tasks, axis=1))
                        else:
                            X = np.transpose(np.concatenate(simulated_neurons_training_tasks, axis = 1))
                        y = np.concatenate(curr_neuron_training_tasks)
                        alpha=0.01
                        reg = ElasticNet(alpha=alpha, l1_ratio = l1_ratio, positive=True).fit(X, y)            
                        coeffs_flat=reg.coef_
                        
                        # save per neuron, for all models, across perms.
                        # rewrite!!
                        result_dict[curr_cell][entry].append(coeffs_flat)
    
                        # next, create the predicted activity neuron.
                        predicted_activity_curr_neuron = np.sum((result_dict[curr_cell][entry]*simulated_neurons_test_grids_flat), axis = 1)
                        #check with mohamady why this step - in particular why include the actual neural activity??
                        # predicted_activity_curr_neuron_scaled = predicted_activity_curr_neuron*(
                        #     np.mean(curr_neuron_heldouttasks_flat)/np.mean(predicted_activity_curr_neuron))
                        
                        # before doing this, first bin per state.
                        predicted_activity_curr_neuron_binned = mc.analyse.helpers_human_cells.neurons_to_state_bins(predicted_activity_curr_neuron, state_model_curr_testgrid)
                        curr_neuron_heldouttasks_flat_binned = mc.analyse.helpers_human_cells.neurons_to_state_bins(curr_neuron_heldouttasks_flat, state_model_curr_testgrid)
                        
                        Predicted_Actual_correlation_binned=st.pearsonr(curr_neuron_heldouttasks_flat_binned,predicted_activity_curr_neuron_binned)[0]
                        
                        Predicted_Actual_correlation=st.pearsonr(curr_neuron_heldouttasks_flat,predicted_activity_curr_neuron)[0]
                        # if np.isnan(Predicted_Actual_correlation):
                        #     import pdb; pdb.set_trace()
                        # most likely nan if the predicted activity is just 0 bc of low firing rate
                            
                        corr_dict[sub][entry][curr_cell][task_idx] = Predicted_Actual_correlation
                        corr_dict_binned[sub][entry][curr_cell][task_idx] = Predicted_Actual_correlation_binned
            
        if save_results == True:
            # save the all_modelled_data dict such that I don't need to always run it again.
            with open(os.path.join(group_folder,file_name), 'wb') as f:
                pickle.dump(result_dict, f)
            print(f"saved the fit GLM data as {group_folder}/{file_name}")

    tt_end=time.time()    
    print(f"{sub} took {tt_end- tt_start} time")
    
                
# # exclude subjects with super structured tasks.
# # doesnt really seem to make a difference.
# # # 39, 41, 42, 52, 54
# subset_subjects = [f"sub-{i:02}" for i in range(1, 58) if i not in [9, 27, 43, 44, 39, 41, 42, 52, 54]]
# subset_corr_dict = {}
# for sub in subset_subjects:
#     subset_corr_dict[sub] = corr_dict_binned[sub]
    
# results_corr_by_roi_subset = mc.analyse.plotting_cells.prep_result_for_plotting_by_rois(subset_corr_dict)
# title_addition = "subset roi neurons binned"
# mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi_subset,title_addition )



results_corr = mc.analyse.plotting_cells.prep_result_dir_for_plotting(corr_dict)
results_binned_corr = mc.analyse.plotting_cells.prep_result_dir_for_plotting(corr_dict_binned)
results_corr_by_roi = mc.analyse.plotting_cells.prep_result_for_plotting_by_rois(corr_dict_binned)
   
if jumbled_regressors == True:
    title_addition = "regressors in random grid order - all neurons binned"
    mc.analyse.plotting_cells.plotting_corr_perm_histogram(results_binned_corr, title_addition)
    title_addition = "regressors in random grid order - all neurons"
    mc.analyse.plotting_cells.plotting_corr_perm_histogram(results_corr, title_addition)
    title_addition = "regressors in random grid order - roi neurons binned"
    mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi,title_addition )

    
else:
    title_addition = "all neurons binned"
    mc.analyse.plotting_cells.plotting_corr_perm_histogram(results_binned_corr, title_addition)
    title_addition = "all neurons"
    mc.analyse.plotting_cells.plotting_corr_perm_histogram(results_corr, title_addition)
    title_addition = "roi neurons binned"
    mc.analyse.plotting_cells.plotting_corr_perm_histogram_by_ROIs(results_corr_by_roi,title_addition )



predicted_cells = mc.analyse.helpers_human_cells.identify_max_cells_for_model(corr_dict_binned)
mc.analyse.helpers_human_cells.store_best_cells(predicted_cells, data)


if save_results == True:
    # save the results so I don't always need to run this again!!
    file_name = f"standard_regs_corr_results"
    if models_I_want:
        file_name = f"all_regs_w_partial_musicboxes_corr_results"
    if jumbled_regressors == True:
        file_name = f"jumbled_regs_corr_results"
    
    with open(os.path.join(group_folder,file_name), 'wb') as f:
        pickle.dump(results_corr, f)
        
    with open(os.path.join(group_folder,f"{file_name}_binned"), 'wb') as f:
        pickle.dump(results_binned_corr, f)
        
    with open(os.path.join(group_folder,f"{file_name}_binned_by_roi"), 'wb') as f:
        pickle.dump(results_corr_by_roi, f)

    with open(os.path.join(group_folder,f"{file_name}_corrs_binned"), 'wb') as f:
        pickle.dump(corr_dict_binned, f)

        
    