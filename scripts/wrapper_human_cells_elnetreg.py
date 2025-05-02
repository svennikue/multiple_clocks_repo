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
import sys



print("ARGS:", sys.argv)



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
    

def get_rid_of_low_firing_cells(data, hz_exclusion_threshold = 0.1, sd_exclusion_threshold = 1.5):
    
    neuron_dict = {}
    for sub in data:
        for task in data[sub]['neurons']:
            for i, neuron in enumerate(task):
                curr_cell = f"{sub}_{data[sub]['cell_labels'][i]}_{i}"
                if curr_cell not in neuron_dict:
                    neuron_dict[curr_cell] = []            
                avg_rate_hz = np.sum(neuron) / (len(neuron) * 0.025)
                neuron_dict[curr_cell].append(avg_rate_hz)
       
    excl_dict = {}
    neurons_to_exclude = []
    neurons_to_exclude_str = []
    for neuron in neuron_dict:
        excl_dict[f"avg_{neuron}"] = np.mean(neuron_dict[neuron])
        excl_dict[f"dev_{neuron}"] = np.std(neuron_dict[neuron])/np.mean(neuron_dict[neuron])
        neuron_index = neuron.split('_')[-1]
        if excl_dict[f"avg_{neuron}"] < hz_exclusion_threshold:
            neurons_to_exclude.append(int(neuron_index))
            neurons_to_exclude_str.append(neuron)
        elif excl_dict[f"dev_{neuron}"] > sd_exclusion_threshold:
            neurons_to_exclude.append(int(neuron_index))
            neurons_to_exclude_str.append(neuron)

    remaining_neurons = {}
    stuff_to_copy = ['reward_configs', 'timings', 'locations', 'buttons']
    to_clean = ['cell_labels', 'neurons']
    
    for sub in data:
        if sub not in remaining_neurons:
            remaining_neurons[sub] = {}
            for m in data[sub]:
                if m in stuff_to_copy:
                    remaining_neurons[sub][m] = data[sub][m]
                elif m in to_clean:
                    # # import pdb; pdb.set_trace() 
                    # if m not in remaining_neurons[sub]:
                    #     remaining_neurons[sub][m] = []
                    if m == 'neurons':
                        if m not in remaining_neurons[sub]:
                            remaining_neurons[sub][m] = []
                        for el in data[sub][m]:
                            cleaned = np.delete(el, neurons_to_exclude, axis=0)
                            remaining_neurons[sub][m].append(cleaned)
                    elif m == 'cell_labels':
                        remaining_neurons[sub][m] = [x for i, x in enumerate(data[sub][m]) if i not in neurons_to_exclude]


    print(f"exlcuding neurons {neurons_to_exclude_str} because average spike rate lower than {hz_exclusion_threshold} or variability bigger than std/mean = {sd_exclusion_threshold}")
    # import pdb; pdb.set_trace() 
    return remaining_neurons
    


def run_elnetreg_cellwise(data, curr_cell, fit_binned = None, bin_pre_corr = None):
    # parameters that seem to work, can be set flexibly
    alpha=0.0001 ##0.01 used in El-gaby paper
    # l1_ratio= 0.01
    
    
    cell_idx = int(curr_cell.split('_')[1])
    corr_dict, corr_dict_binned, coefs_per_model = {}, {}, {}
    model_string = []
    for model in data:
        if model.endswith('reg') or model.endswith('model'):
            model_string.append(model)
            corr_dict[model], corr_dict_binned[model] = {}, {}
            
    unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(data['reward_configs'], axis=0,
                                                            return_index=True,
                                                            return_inverse=True,
                                                            return_counts=True)
            
    for model in model_string:
        corr_dict[model][curr_cell] = np.zeros(len(unique_grids))
        corr_dict_binned[model][curr_cell] = np.zeros(len(unique_grids))
        coefs_per_model[model] = []
        
    for task_idx, left_out_grid_idx in enumerate(idx_unique_grid): 
        
        test_grid_idx = np.where(idx_same_grids == idx_same_grids[left_out_grid_idx])[0]
        if not test_grid_idx.shape == (1,):
            all_neurons_heldouttasks = itemgetter(*test_grid_idx)(data['neurons'])
            curr_neuron_heldouttasks = [all_neurons[cell_idx] for all_neurons in all_neurons_heldouttasks]
            curr_neuron_heldouttasks_flat = np.concatenate(curr_neuron_heldouttasks)
            # import pdb; pdb.set_trace() 
        else:
            all_neurons_heldouttasks = data['neurons'][test_grid_idx[0]]
            curr_neuron_heldouttasks_flat = all_neurons_heldouttasks[cell_idx]
        
        training_grid_idx=np.setdiff1d(np.arange(len(data['reward_configs'])),test_grid_idx)
        all_neurons_training_tasks = itemgetter(*training_grid_idx)(data['neurons'])
        curr_neuron_training_tasks = [all_neurons[cell_idx] for all_neurons in all_neurons_training_tasks]
        
  
        if fit_binned or bin_pre_corr:
            if fit_binned == 'by_state' or bin_pre_corr == 'by_state':
                binning_model = 'state_reg'
            elif fit_binned == 'by_loc_change' or bin_pre_corr == 'by_loc_change':
                binning_model = 'location_reg'
            # multimodel uses slightly different function and indexing works differently
            if fit_binned == 'by_state_loc_change' or bin_pre_corr == 'by_state_loc_change':
                binning_model = ['state_reg', 'location_reg']
                binning_curr_testgrid, binning_curr_training_grids = {}, {}
                for m in binning_model:
                    binning_training_grids = itemgetter(*training_grid_idx)(data[m])
                    binning_curr_training_grids[m] = np.transpose(np.concatenate(binning_training_grids, axis = 1))

                    binning_neurons_test_grids = itemgetter(*test_grid_idx)(data[m])
                    if not test_grid_idx.shape == (1,):
                        binning_curr_testgrid[m] = np.transpose(np.concatenate(binning_neurons_test_grids, axis = 1))
                    else:
                        binning_curr_testgrid[m] = np.transpose(binning_neurons_test_grids)
                if fit_binned:
                    curr_neuron_heldouttasks_flat = mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(curr_neuron_heldouttasks_flat, binning_curr_testgrid)
            # use single model to bin    
            else:
                binning_training_grids = itemgetter(*training_grid_idx)(data[binning_model])
                binning_curr_training_grids = np.transpose(np.concatenate(binning_training_grids, axis = 1))

                binning_neurons_test_grids = itemgetter(*test_grid_idx)(data[binning_model])
                if not test_grid_idx.shape == (1,):
                    binning_curr_testgrid = np.transpose(np.concatenate(binning_neurons_test_grids, axis = 1))
                else:
                    binning_curr_testgrid = np.transpose(binning_neurons_test_grids)
                if fit_binned:
                    curr_neuron_heldouttasks_flat = mc.analyse.helpers_human_cells.neurons_to_state_bins(curr_neuron_heldouttasks_flat, binning_curr_testgrid)
                
        # import pdb; pdb.set_trace()    
        # depending on the permutations, change the train and test dataset.
        for entry in model_string:
            # then choose which tasks to take and go and select training regressors
            simulated_neurons_training_tasks = itemgetter(*training_grid_idx)(data[entry])
            # and test regressors
            simulated_neurons_test_grids = itemgetter(*test_grid_idx)(data[entry])
            if not test_grid_idx.shape == (1,):
                simulated_neurons_test_grids_flat = np.transpose(np.concatenate(simulated_neurons_test_grids, axis = 1))
            else:
                simulated_neurons_test_grids_flat = np.transpose(simulated_neurons_test_grids)
               
            X = np.transpose(np.concatenate(simulated_neurons_training_tasks, axis = 1))
            
            if entry == 'stat_model':
                X = X[:, 0:3].copy()
                simulated_neurons_test_grids_flat = simulated_neurons_test_grids_flat[:, 0:3].copy()
            
            y = np.concatenate(curr_neuron_training_tasks)
            # import pdb; pdb.set_trace()

            if fit_binned:
                if fit_binned == 'by_state_loc_change':
                    simulated_neurons_test_grids_flat = mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(simulated_neurons_test_grids_flat, binning_curr_testgrid)
                    
                    X = mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(X, binning_curr_training_grids)
                    y = mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(y, binning_curr_training_grids)
                    
                else:
                    simulated_neurons_test_grids_flat = mc.analyse.helpers_human_cells.neurons_to_state_bins(simulated_neurons_test_grids_flat, binning_curr_testgrid)

                    X = mc.analyse.helpers_human_cells.neurons_to_state_bins(X, binning_curr_training_grids)
                    y = mc.analyse.helpers_human_cells.neurons_to_state_bins(y, binning_curr_training_grids)
                    
         
            reg = ElasticNet(alpha=alpha, positive=True, max_iter=100000).fit(X, y)  
            coeffs_flat=reg.coef_
            
            # save per neuron, for all models, across perms.
            # rewrite!!
            coefs_per_model[entry].append(coeffs_flat)
            # next, create the predicted activity neuron.
            # import pdb; pdb.set_trace()
            predicted_activity_curr_neuron = np.sum((coeffs_flat*simulated_neurons_test_grids_flat), axis = 1)
               
            # if not fit_binned == 'by_state':
            if bin_pre_corr:
                if bin_pre_corr == 'by_state_loc_change':
                   predicted_activity_curr_neuron_binned =  mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(predicted_activity_curr_neuron, binning_curr_testgrid)
                   curr_neuron_heldouttasks_flat_binned = mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(curr_neuron_heldouttasks_flat, binning_curr_testgrid)
                else:
                    # before doing this, first bin per state.
                    predicted_activity_curr_neuron_binned = mc.analyse.helpers_human_cells.neurons_to_state_bins(predicted_activity_curr_neuron, binning_curr_testgrid)
                    curr_neuron_heldouttasks_flat_binned = mc.analyse.helpers_human_cells.neurons_to_state_bins(curr_neuron_heldouttasks_flat, binning_curr_testgrid)
                    
                Predicted_Actual_correlation_binned=st.pearsonr(curr_neuron_heldouttasks_flat_binned,predicted_activity_curr_neuron_binned)[0]
                corr_dict_binned[entry][curr_cell][task_idx] = Predicted_Actual_correlation_binned
                    
            Predicted_Actual_correlation=st.pearsonr(curr_neuron_heldouttasks_flat,predicted_activity_curr_neuron)[0]
            # import pdb; pdb.set_trace() 
            # if np.isnan(Predicted_Actual_correlation):
            #     import pdb; pdb.set_trace()
            # most likely nan if the predicted activity is just 0 bc of low firing rate
            corr_dict[entry][curr_cell][task_idx] = Predicted_Actual_correlation
            
    # return corr_dict, corr_dict_binned 
    # for m in corr_dict:
    #     print(f"for model {m} mean is {np.mean(corr_dict[m][curr_cell])}")
        
    # import pdb; pdb.set_trace()       
    return corr_dict, corr_dict_binned, curr_cell

    
 
def compute_one_subject(sub, models_I_want, exclude_x_repeats, randomised_reward_locations, save_regs, fit_binned = None, bin_pre_corr = None, avg_across_runs = False):
    
    data, group_dir, subj_reg_file = get_data(sub, models_I_want=models_I_want, exclude_x_repeats=exclude_x_repeats, randomised_reward_locations=randomised_reward_locations)
    
    clean_data = get_rid_of_low_firing_cells(data)
    
    # if randomised_reward_locations == True:
    #     # first create the random grid configurations/shuffles.
    #     # use a set seed so this would always be the same shuffles
    #     # and then give it an index. So actually, randomised_reward_locations needs to be a list ['True', idx]
    #     # and then I feed the randomised reward_configs into the simulated-regs function. 
        
    #     # finally, also change what I store. I only need a single value per model/cell/subject.
    #       # it would be great if I could save all values as a final dictionary
    #       # this means, however, that if the flag 'randomised_reward_locations'[0] == True
    #       # that all print statements should be off, except for maybe 'perm 0,100,etc.
    #       # and then store in result_dict[subject][model][cell] = [perm_1_val, perm_2_val, ...,]
    #       # so i need to call it from inside this compute_one_subjet script
    #       # rng = np.random.default_rng(seed); seed = hash(f"{subject_id}_{perm_index}") % 2**32
    #       # seed could be based on permutation number and subject ID (although in my case, maybe just subject)

    
    
    simulated_regs = mc.analyse.helpers_human_cells.prep_regressors_for_neurons(clean_data, models_I_want=models_I_want, exclude_x_repeats=exclude_x_repeats, randomised_reward_locations=randomised_reward_locations, avg_across_runs=avg_across_runs)
    # import pdb; pdb.set_trace() 
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
    
    #corr_test, corr_test_binned, cell = run_elnetreg_cellwise(single_sub_dict, cells[0], fit_binned=fit_binned, bin_pre_corr=bin_pre_corr)   
    #import pdb; pdb.set_trace() 
    
    parallel_results = Parallel(n_jobs=-1)(delayed(run_elnetreg_cellwise)(single_sub_dict, c, fit_binned=fit_binned, bin_pre_corr=bin_pre_corr) for c in cells)
    
    
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
        if bin_pre_corr:
            result_file_name = f"{result_file_name}_excl_rep{exclude_x_repeats[0]}-{exclude_x_repeats[-1]}_pre_corr_binned"
        elif avg_across_runs == True:
            result_file_name = f"{result_file_name}_excl_rep{exclude_x_repeats[0]}-{exclude_x_repeats[-1]}_avg_in_12_bins_across_runs"
        elif fit_binned:
            result_file_name = f"{result_file_name}_excl_rep{exclude_x_repeats[0]}-{exclude_x_repeats[-1]}_fit-{fit_binned}_only_pos"
        else:
            result_file_name = f"{result_file_name}_excl_rep{exclude_x_repeats[0]}-{exclude_x_repeats[-1]}"


    # CHANGE TO ADDTION INSTEAD OF 
    
    
    with open(os.path.join(group_dir_corrs,result_file_name), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(group_dir_corrs,f"{result_file_name}_binned"), 'wb') as f:
        pickle.dump(results_binned, f)
    # import pdb; pdb.set_trace()    
    print(f"saved the modelled data as {group_dir_corrs}/{result_file_name}")

    
    
# if running from command line, use this one!   
if __name__ == "__main__":
    #print(f"starting regression for subject {sub}")
    fire.Fire(compute_one_subject)
#    call this script like
#    python wrapper_human_cells_elnetreg.py 5 --models_I_want='['withoutnow', 'onlynowand3future', 'onlynextand2future']' --exclude_x_repeats='[1,2,3]' --randomised_reward_locations=False --save_regs=True

# ['withoutnow', 'only2and3future','onlynowandnext', 'onlynowand3future', 'onlynextand2future']
# ['only','onlynowand3future', 'onlynextand2future']

# if __name__ == "__main__":
#     # For debugging, bypass Fire and call compute_one_subject directly.
#     compute_one_subject(
#         sub=59,
#         #models_I_want=['withoutnow', 'onlynowand3future', 'onlynextand2future'],
#         models_I_want=['withoutnow', 'only2and3future','onlynowandnext', 'onlynowand3future', 'onlynextand2future'],
#         exclude_x_repeats=[1,2],
#         randomised_reward_locations=False,
#         save_regs=True,
#         #fit_binned='by_state_loc_change' # 'by_loc_change', 'by_state', 'by_state_loc_change'
#         #bin_pre_corr='by_state_loc_change',
#         avg_across_runs=True
#     )


# these are hard-coded right now, so include them in the 'only' + models list 'state_reg', 'complete_musicbox_reg', 'reward_musicbox_reg', 'location_reg'




