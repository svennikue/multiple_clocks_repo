#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:47:17 2025

Wrapper that calls every subject file individually and runs the regression cell-wise
and in parallel

@author: Svenja Kuchenhoff
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
import fire
from joblib import Parallel, delayed



def get_data(sub, models_I_want=False, exclude_x_repeats=False, randomised_reward_locations=False):
    ### SETTINGS
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    # check if on server or local
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    group_dir = f"{data_folder}/group/elastic_net_reg"
    
    if not os.path.isdir(group_dir):
        os.mkdir(group_dir)
    
    subjects = [f"{sub:02}"]
    data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)
    
    file_name = f"sub-{subjects[0]}_prep_data_for_regression"
    if models_I_want:
        file_name = f"prep_data_for_regression_w_partial_musicboxes"
    if randomised_reward_locations == True:
        file_name = f"prep_data_for_regression_random_rew_order"
    if exclude_x_repeats:
        file_name = f"{file_name}_excl_rep{exclude_x_repeats[0]}-{exclude_x_repeats[-1]}"
    
    return data, group_dir, file_name
    
 
def run_elnetreg_cellwise(data, curr_cell):
    # parameters that seem to work, can be set flexibly
    alpha=0.001 ##0.01 used in El-gaby paper
    l1_ratio= 0.01
    
    
    cell_idx = int(curr_cell.split('_')[1])
    corr_dict, corr_dict_binned, coefs_per_model = {}, {}, {}
    for model in data:
        if model.endswith('reg') or model.endswith('model'):
            corr_dict[model], corr_dict_binned[model] = {}, {}
            
    unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(data['reward_configs'], axis=0,
                                                            return_index=True,
                                                            return_inverse=True,
                                                            return_counts=True)
            
    for model in data:
            if model.endswith('reg') or model.endswith('model'):
                corr_dict[model][curr_cell] = np.zeros(len(unique_grids))
                corr_dict_binned[model][curr_cell] = np.zeros(len(unique_grids))
                
    for task_idx, left_out_grid_idx in enumerate(idx_unique_grid): 
        test_grid_idx = np.where(idx_same_grids == idx_same_grids[left_out_grid_idx])[0]
        if not test_grid_idx.shape == (1,):
            all_neurons_heldouttasks = itemgetter(*test_grid_idx)(data['neurons'])
            curr_neuron_heldouttasks = [all_neurons[cell_idx] for all_neurons in all_neurons_heldouttasks]
            curr_neuron_heldouttasks_flat = np.concatenate(curr_neuron_heldouttasks)
        else:
            all_neurons_heldouttasks = data['neurons'][test_grid_idx[0]]
            curr_neuron_heldouttasks_flat = all_neurons_heldouttasks[cell_idx]
        
        training_grid_idx=np.setdiff1d(np.arange(len(data['reward_configs'])),test_grid_idx)
        all_neurons_training_tasks = itemgetter(*training_grid_idx)(data['neurons'])
        curr_neuron_training_tasks = [all_neurons[cell_idx] for all_neurons in all_neurons_training_tasks]
        
        # depending on the permutations, change the train and test dataset.
        for entry in data:
            # only the regressors I created.
            if entry.endswith('reg') or entry.endswith('model'):
                coefs_per_model[entry] = []
                
                # then choose which tasks to take and go and select training regressors
                simulated_neurons_training_tasks = itemgetter(*training_grid_idx)(data[entry])
                # and test regressors
                simulated_neurons_test_grids = itemgetter(*test_grid_idx)(data[entry])
                if not test_grid_idx.shape == (1,):
                    simulated_neurons_test_grids_flat = np.transpose(np.concatenate(simulated_neurons_test_grids, axis = 1))
                else:
                    simulated_neurons_test_grids_flat = np.transpose(simulated_neurons_test_grids)
                    
                # for binning later on, store the state regressor
                if entry == "state_reg":
                    state_model_curr_testgrid = simulated_neurons_test_grids_flat.copy()
            
            
                X = np.transpose(np.concatenate(simulated_neurons_training_tasks, axis = 1))
                y = np.concatenate(curr_neuron_training_tasks)
                alpha=0.01
                reg = ElasticNet(alpha=alpha, l1_ratio = l1_ratio, positive=True).fit(X, y)            
                coeffs_flat=reg.coef_
                
                # save per neuron, for all models, across perms.
                # rewrite!!
                coefs_per_model[entry].append(coeffs_flat)
                # next, create the predicted activity neuron.
                predicted_activity_curr_neuron = np.sum((coefs_per_model[entry]*simulated_neurons_test_grids_flat), axis = 1)
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
                corr_dict[entry][curr_cell][task_idx] = Predicted_Actual_correlation
                corr_dict_binned[entry][curr_cell][task_idx] = Predicted_Actual_correlation_binned
    # return corr_dict, corr_dict_binned            
    return corr_dict, corr_dict_binned, curr_cell

    
 
def compute_one_subject(sub, models_I_want, exclude_x_repeats, randomised_reward_locations, save_regs):
    data, group_dir, subj_reg_file = get_data(sub, models_I_want=models_I_want, exclude_x_repeats=exclude_x_repeats, randomised_reward_locations=randomised_reward_locations)
    simulated_regs = mc.analyse.helpers_human_cells.prep_regressors_for_neurons(data, models_I_want=models_I_want, exclude_x_repeats=exclude_x_repeats, randomised_reward_locations=randomised_reward_locations)
    
    group_dir_coefs = f"{group_dir}/coefs"
    group_dir_corrs = f"{group_dir}/corrs"
    if not os.path.isdir(group_dir_coefs):
        os.mkdir(group_dir_coefs)
    if not os.path.isdir(group_dir_corrs):
        os.mkdir(group_dir_corrs)
    
    if save_regs == True:
        # save the all_modelled_data dict such that I don't need to always run it again.
        with open(os.path.join(group_dir_coefs,subj_reg_file), 'wb') as f:
            pickle.dump(simulated_regs, f)
        print(f"saved the modelled data as {group_dir_coefs}/{subj_reg_file}")
    
    subjects = [f"{sub:02}"]
    single_sub_dict = copy.deepcopy(simulated_regs[f"sub-{subjects[0]}"])
    # then parallelise the computation I do on the cells.
    cells = []
    for cell_idx, cell in enumerate(single_sub_dict['neurons'][0]):
        # create one entry in result_dict per cell 
        cells.append(f"{single_sub_dict['cell_labels'][cell_idx]}_{cell_idx}_{sub}")

    # this won't work like this
    # I want to give a single cell each, looping through the cell names
    # currently that's stored differently than I need it (i.e. cell strings are a list, and then there is the dict)
    # the loop needs to replace this loop: 
    #    for cell_idx, cell in enumerate(single_sub_dict['neurons'][0]):
        
        # also figure out how to solve this in the internal loop, i.e. how to select the approriate cell
        # then once I have this, run for a single subject on the cluster
        # and install all the packages until that works
        # then try to debug this script for a single subject: why are the regressions off/different?
        # plot how they look like for state and location first
        # they should be pretty much the same, at least the state ones.
    print(f"starting parallel regression and correlation for all cells and models for subject {sub}")
    
    
    parallel_results = Parallel(n_jobs=-1)(delayed(run_elnetreg_cellwise)(single_sub_dict, c) for c in cells)
    
    results = {}
    results_binned = {}
    
    for corr_dict, corr_dict_binned, curr_cell in parallel_results:
        results[curr_cell] = corr_dict
        results_binned[curr_cell] = corr_dict_binned
    
    
    # parallel = Parallel(n_jobs=-1, return_as="generator")
    # parallel_results = parallel(delayed(run_elnetreg_cellwise)(single_sub_dict, c) for c in cells)
    
    # results = {}
    # results_binned = {}
    
    # for corr_dict, corr_dict_binned, curr_cell in parallel_results:
    #     results[curr_cell] = corr_dict
    #     results_binned[curr_cell] = corr_dict_binned
    
    result_file_name  = f"sub-{sub}_corrs"
    if models_I_want:
        result_file_name = f"sub-{sub}_corrs_w_partial_musicboxes"
    if randomised_reward_locations == True:
        result_file_name = f"sub-{sub}_corrs_random_rew_order"
    if exclude_x_repeats:
        result_file_name = f"{result_file_name}_excl_rep{exclude_x_repeats[0]}-{exclude_x_repeats[-1]}"
    
    with open(os.path.join(group_dir_corrs,result_file_name), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(group_dir_corrs,f"{result_file_name}_binned"), 'wb') as f:
        pickle.dump(results_binned, f)
        
    print(f"saved the modelled data as {group_dir_corrs}/{subj_reg_file}")

    
    
    
if __name__ == "__main__":
    fire.Fire(compute_one_subject)
    # call this script like
    # python wrapper_human_cells_elnetreg.py 5 --models_I_want='['withoutnow', 'only2and3future','onlynowandnext']' --exclude_x_repeats='[1,2]' --randomised_reward_locations=False --save_regs=True
