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
from operator import itemgetter
from sklearn.linear_model import ElasticNet
import copy
import fire
from joblib import Parallel, delayed
import sys
from itertools import permutations, islice
import time


print("ARGS:", sys.argv)


def nan_safe_pearsonr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan  # Not enough data to compute correlation
    return np.corrcoef(x[mask], y[mask])[0, 1]




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
    

def get_rid_of_low_firing_cells(data, hz_exclusion_threshold = 0.2, sd_exclusion_threshold = 1.5):
    # 0.25 hz -> 1 spike per 4 secs
    # 0.2 hz -> 1 spike per 5 secs
    # 0.1 hz -> 1 spike per 10 secs
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
                remaining_neurons[sub]['excluded_cells'] = neurons_to_exclude_str
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


def generate_unique_grid_permutations(reward_configs, n_permutations=10, seed=42):
    """
    Generate unique deranged permutations of reward_configs:
    - No row stays in its original position.
    - All rows are unique, so no grouping needed.
    
    Returns:
        np.ndarray of shape (n_permutations, len(reward_configs), 4)
        np.ndarray of index vectors used for each permutation
    """
    np.random.seed(seed)
    n = len(reward_configs)
    orig_indices = np.arange(n)

    def random_derangement():
        while True:
            p = np.random.permutation(n)
            if not np.any(p == orig_indices):
                return p

    results = []
    index_vectors = set()  # track unique permutations

    attempts = 0
    max_attempts = n_permutations * 10  # prevent infinite loops

    while len(index_vectors) < n_permutations and attempts < max_attempts:
        perm = random_derangement()
        perm_tuple = tuple(perm)
        if perm_tuple not in index_vectors:
            index_vectors.add(perm_tuple)
            results.append(reward_configs[perm])
        attempts += 1

    if len(index_vectors) < n_permutations:
        print(f"Warning: only {len(index_vectors)} unique derangements found.")

    return np.array(results), np.array(list(index_vectors))



def run_elnetreg_cellwise(data, curr_cell, fit_binned = None, fit_residuals = None, comp_loc_perms= None):
    print(f"...fitting and testing cell {curr_cell}")
    # parameters that seem to work, can be set flexibly
    alpha=0.00001 ##0.01 used in El-gaby paper
    # l1_ratio= 0.01
    cell_idx = int(curr_cell.split('_')[1])
    corr_dict, coefs_per_model = {}, {}
    if fit_binned:
        coefs_per_model_binned = {}
        corr_dict_binned = {}
    if fit_residuals == True:
        corr_dict_residuals = {}
        coefs_per_model_residuals = {}
        
    model_string = []
    for model in data:
        if model.endswith('reg') or model.endswith('model'):
            model_string.append(model)
            corr_dict[model], corr_dict_binned[model] = {}, {}
     

    # prepare the held-out task order and the permutations 
    unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(data['reward_configs'], axis=0,
                                                            return_index=True,
                                                            return_inverse=True,
                                                            return_counts=True)
   
    perms  = 1
    if comp_loc_perms:
        start = time.time()
        unique_grid_permutations, perm_idx_vectors = generate_unique_grid_permutations(data['reward_configs'], n_permutations=comp_loc_perms)
        perms = len(unique_grid_permutations)
        end = time.time()
        print(f"Permutation generation took {end - start:.2f} seconds")
            
     # prepare the result dictionaries    
    for model in model_string:
        corr_dict[model][curr_cell] = np.zeros((len(unique_grids), perms))
        corr_dict_binned[model][curr_cell] = np.zeros((len(unique_grids), perms))
        # corr_dict_residuals[model][curr_cell] = np.zeros((len(unique_grids), perms))
        coefs_per_model[model] = []
        if fit_binned:
            coefs_per_model_binned[model] = []
        if fit_residuals == True:
            coefs_per_model_residuals[model] = []
                
                
    # import pdb; pdb.set_trace()   
    for p_idx in range(0,perms):
        start_perms = time.time()
        # create indices of shuffled task configs for simulations.
        if comp_loc_perms:
            unique_grids_sim, idx_unique_grid_sim, idx_same_grids_sim, counts_sim = np.unique(unique_grid_permutations[p_idx], axis=0,
                                                                    return_index=True,
                                                                    return_inverse=True,
                                                                    return_counts=True)
            curr_perm_idx_vector = perm_idx_vectors[p_idx]
            curr_perm_idx_vector = np.array([int(e) for e in curr_perm_idx_vector])
            
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
            
            if fit_binned:
                if fit_binned == 'by_state':
                    binning_model = 'state_reg'
                elif fit_binned == 'by_loc_change':
                    binning_model = 'location_reg'
                # multimodel uses slightly different function and indexing works differently
                if fit_binned == 'by_state_loc_change':
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
                    
                    curr_neuron_heldouttasks_flat_binned = mc.analyse.helpers_human_cells.neurons_to_state_bins(curr_neuron_heldouttasks_flat, binning_curr_testgrid)
                    
            # import pdb; pdb.set_trace()   
            # depending on the permutations, change the train and test dataset.
            for entry in model_string:
                if comp_loc_perms:
                    # start_perm = time.time()
                    # the models will have different indices than the neurons.
                    # take the shuffled task configs as indices for the simulations
                    left_out_grid_idx_sim = curr_perm_idx_vector[task_idx]
                    # there should never be several if I average repeated tasks, but just in case
                    test_grid_idx_sim = np.where(idx_same_grids_sim == idx_same_grids_sim[left_out_grid_idx_sim])[0]
                    
                    training_grid_idx_sim = curr_perm_idx_vector[~np.isin(curr_perm_idx_vector, test_grid_idx_sim)]
                    training_grid_idx_sim = np.array([int(e) for e in training_grid_idx_sim])
                    
                    # then use these perm vectors to refer to the OG grids
                    # then choose which tasks to take and go and select training regressors
                    simulated_neurons_training_tasks = itemgetter(*training_grid_idx_sim)(data[entry])
                    # and test regressors
                    simulated_neurons_test_grids = itemgetter(*test_grid_idx)(data[entry])

                else:
                    # if not permutated, take same order as for neurons
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
                reg = ElasticNet(alpha=alpha, positive=True, max_iter=100000).fit(X, y)  
                coeffs_flat=reg.coef_
                coefs_per_model[entry].append(coeffs_flat)
                # next, create the predicted activity neuron.
                predicted_activity_curr_neuron = np.sum((coeffs_flat*simulated_neurons_test_grids_flat), axis = 1)
                Predicted_Actual_correlation=nan_safe_pearsonr(curr_neuron_heldouttasks_flat,predicted_activity_curr_neuron)
                # if np.all(curr_neuron_heldouttasks_flat == curr_neuron_heldouttasks_flat[0]) or np.all(predicted_activity_curr_neuron == predicted_activity_curr_neuron[0]):
                #     import pdb; pdb.set_trace()
                
                corr_dict[entry][curr_cell][task_idx,p_idx] = Predicted_Actual_correlation
    
                if fit_binned:
                    if fit_binned == 'by_state_loc_change':
                        simulated_neurons_test_grids_flat_binned = mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(simulated_neurons_test_grids_flat, binning_curr_testgrid)
                        
                        X_binned = mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(X, binning_curr_training_grids)
                        y_binned = mc.analyse.helpers_human_cells.neurons_to_bins_multimodel(y, binning_curr_training_grids)
                        
                    else:
                        simulated_neurons_test_grids_flat_binned = mc.analyse.helpers_human_cells.neurons_to_state_bins(simulated_neurons_test_grids_flat, binning_curr_testgrid)
    
                        X_binned = mc.analyse.helpers_human_cells.neurons_to_state_bins(X, binning_curr_training_grids)
                        y_binned = mc.analyse.helpers_human_cells.neurons_to_state_bins(y, binning_curr_training_grids)
                    
                    reg_binned = ElasticNet(alpha=alpha, positive=True, max_iter=100000).fit(X_binned, y_binned)  
                    coeffs_flat_binned=reg_binned.coef_
                    coefs_per_model_binned[entry].append(coeffs_flat_binned)
                    # next, create the predicted activity neuron.
                    predicted_activity_curr_neuron_binned = np.sum((coeffs_flat*simulated_neurons_test_grids_flat_binned), axis = 1)
                    Predicted_Actual_correlation_binned=nan_safe_pearsonr(curr_neuron_heldouttasks_flat_binned,predicted_activity_curr_neuron_binned)
                    corr_dict_binned[entry][curr_cell][task_idx,p_idx] = Predicted_Actual_correlation_binned
                    # if np.all(curr_neuron_heldouttasks_flat_binned == curr_neuron_heldouttasks_flat_binned[0]) or np.all(predicted_activity_curr_neuron_binned == predicted_activity_curr_neuron_binned[0]):
                    #     import pdb; pdb.set_trace()
    
        # end_perm = time.time()
        # print(f"Fitting one permutation took {end_perm - start_perm:.2f} seconds")
        
    end_perms = time.time()
    print(f"Fitting all permutations took {end_perms - start_perms:.2f} seconds")
    
    all_results = {}
    all_results['curr_cell'] = curr_cell  
    # all_results['corr'] = np.empty((len(idx_unique_grid), perms))
    # if fit_binned:
    #     all_results['corr_dict_binned'] = corr_dict_binned
    # if fit_residuals == True:
    #     all_results['corr_dict_residuals'] = np.empty((len(idx_unique_grid), perms))
        
    all_results['corr'] = corr_dict
    if fit_binned:
        all_results['corr_dict_binned'] = corr_dict_binned
    if fit_residuals == True:
        all_results['corr_dict_residuals'] = corr_dict_residuals

        
    # import pdb; pdb.set_trace()
    return all_results

    
 
def compute_one_subject(sub, models_I_want, exclude_x_repeats, randomised_reward_locations, save_regs, fit_binned = None, fit_residuals= False, avg_across_runs = False, comp_loc_perms = False):
    
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


    print(f"starting parallel regression and correlation for all cells and models for subject {sub}")
    

    # for cell in cells:
    #     results = run_elnetreg_cellwise(single_sub_dict, cell, fit_binned=fit_binned, fit_residuals=fit_residuals, comp_loc_perms = comp_loc_perms)    
    
    
    parallel_results = Parallel(n_jobs=-1)(delayed(run_elnetreg_cellwise)(single_sub_dict, c, fit_binned=fit_binned, fit_residuals=fit_residuals, comp_loc_perms=comp_loc_perms) for c in cells)
    
    result_dir = {}
    result_dir['binned'], result_dir['raw'], result_dir['residuals'] = {}, {}, {}
    for cell_idx in parallel_results:
        curr_cell = cell_idx['curr_cell']
        result_dir['raw'][curr_cell] = cell_idx['corr']
        if fit_binned:
            result_dir['binned'][curr_cell] = cell_idx['corr_dict_binned']
        if fit_residuals == True:
            result_dir['residuals'][curr_cell] = cell_idx['corr_dict_binned']
    
    # import pdb; pdb.set_trace() 
    # define the basename
    result_file_name  = f"sub-{sub}_corrs"
    if models_I_want:
        result_file_name = f"sub-{sub}_corrs_w_partial_musicboxes"
    if randomised_reward_locations == True:
        result_file_name = f"sub-{sub}_corrs_random_rew_order"
    if exclude_x_repeats:
        if avg_across_runs == True:
            result_file_name = f"{result_file_name}_excl_rep{exclude_x_repeats[0]}-{exclude_x_repeats[-1]}_avg_in_20_bins_across_runs"
        else:
            result_file_name = f"{result_file_name}_excl_rep{exclude_x_repeats[0]}-{exclude_x_repeats[-1]}"
    
    if comp_loc_perms:
        result_file_name =  str(comp_loc_perms) + 'perms_configs_shuffle_' + result_file_name
            
    # I need to do some sort of extraction
    # but basically I want to do 
    # first, save the basic result
    with open(os.path.join(group_dir_corrs,result_file_name), 'wb') as f:
        pickle.dump(result_dir['raw'], f)
    
    # second, if fit_binned
    if fit_binned:
        result_file_name = f"{result_file_name}_fit_binned_{fit_binned}"
        with open(os.path.join(group_dir_corrs,result_file_name), 'wb') as f:
            pickle.dump(result_dir['binned'], f)
    
    # third, in case of fit_residuals
    if fit_residuals == True:
        result_file_name = f"{result_file_name}_fit_state_residuals"
        with open(os.path.join(group_dir_corrs,result_file_name), 'wb') as f:
            pickle.dump(result_dir['residuals'], f)

    print(f"saved the modelled data as {group_dir_corrs}/{result_file_name}")

    
    
# # if running from command line, use this one!   
if __name__ == "__main__":
    #print(f"starting regression for subject {sub}")
    fire.Fire(compute_one_subject)
#    call this script like
#    python wrapper_human_cells_elnetreg.py 5 --models_I_want='['withoutnow', 'onlynowand3future', 'onlynextand2future']' --exclude_x_repeats='[1,2,3]' --randomised_reward_locations=False --save_regs=True

# # ['withoutnow', 'only2and3future','onlynowandnext', 'onlynowand3future', 'onlynextand2future']
# # ['only','onlynowand3future', 'onlynextand2future']

# if __name__ == "__main__":
#     # For debugging, bypass Fire and call compute_one_subject directly.
#     compute_one_subject(
#         sub=7,
#         #models_I_want=['withoutnow', 'onlynowand3future', 'onlynextand2future'],
#         models_I_want=['onlynowand3future', 'onlynextand2future'],
#         exclude_x_repeats=[1],
#         randomised_reward_locations=False,
#         save_regs=True,
#         fit_binned='by_state', # 'by_loc_change', 'by_state', 'by_state_loc_change'
#         fit_residuals=False,
#         # fit_binned='by_state_loc_change' # 'by_loc_change', 'by_state', 'by_state_loc_change'
#         # introduce a fit residuals options!
#         # bin_pre_corr='by_state',
#         avg_across_runs=True,
#         comp_loc_perms=265 # since with 6 grids, I can only get !6 = 265 unique perms
#     )


# these are hard-coded right now, so include them in the 'only' + models list 'state_reg', 'complete_musicbox_reg', 'reward_musicbox_reg', 'location_reg'




