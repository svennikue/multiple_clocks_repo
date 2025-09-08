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
import pandas as pd


print("ARGS:", sys.argv)


def get_data(sub, models_I_want=False, only_repeats_included=False, randomised_reward_locations=False):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [f"{sub:02}"])
    file_name = f"{sub:02}_prep_data_for_regression"
    if models_I_want:
        file_name = "prep_data_for_regression_w_partial_musicboxes"
    if randomised_reward_locations == True:
        file_name = "prep_data_for_regression_random_rew_order"
    if only_repeats_included:
        file_name = f"{file_name}_reps_{only_repeats_included[0]}-{only_repeats_included[-1]}"
    
    return data_norm, data_folder, file_name


def nan_safe_pearsonr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan  # Not enough data to compute correlation
    return np.corrcoef(x[mask], y[mask])[0, 1]



    

def get_rid_of_low_firing_cells(data, hz_exclusion_threshold = 0.1, sd_exclusion_threshold = 1.5):
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
    to_clean = ['cell_labels', 'neurons', 'electrode_labels']
    # import pdb; pdb.set_trace() 
    # MAKE SURE TO COPY ELECTRODE LABELS!!!
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
                    elif m == 'electrode_labels':
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


def generate_unique_timepoint_permutations(data_dict, models, n_permutations=10, seed=42):
    """
    Generate unique deranged permutations of time points across tasks:
    - No timepoints remains in its original position
    """
    # shuffled_data = copy.deepcopy(data_dict)
    shuffled_data = {}
    np.random.seed(seed)
    def random_derangement(len_all_tasks):
        while True:
            p = np.random.permutation(len_all_tasks)
            if not np.any(p == np.arange(len_all_tasks)):
                return p
    
    for m in models:
        no_tasks = len(data_dict[m])
        concatenated_model = np.concatenate(data_dict[m], axis = 1)
        len_all_tasks = concatenated_model.shape[1]
        # then create random perms based on the length
        shuffled_data[m] = []
        for _ in range(n_permutations):
            idx = random_derangement(len_all_tasks)
            shuffled_model = concatenated_model[:, idx]
            shuffled_model_split = np.split(shuffled_model, no_tasks, axis = 1)
            shuffled_data[m].append(shuffled_model_split)
    
    return shuffled_data

    
    
def generate_unique_timepoint_permutations_neurons(data_dict, n_permutations=10, seed=42):
    """
    Generate unique deranged permutations of time points across tasks:
    - No timepoints remains in its original position
    """
    # shuffled_data = copy.deepcopy(data_dict)
    shuffled_data = {}
    np.random.seed(seed)
    def random_derangement(len_all_tasks):
        while True:
            p = np.random.permutation(len_all_tasks)
            if not np.any(p == np.arange(len_all_tasks)):
                return p

    no_tasks = len(data_dict['neurons'])
    concatenated_model = np.concatenate(data_dict['neurons'], axis = 1)
    len_all_tasks = concatenated_model.shape[1]
    # then create random perms based on the length
    shuffled_data['neurons'] = []
    for _ in range(n_permutations):
        idx = random_derangement(len_all_tasks)
        shuffled_model = concatenated_model[:, idx]
        shuffled_model_split = np.split(shuffled_model, no_tasks, axis = 1)
        shuffled_data['neurons'].append(shuffled_model_split)

    return shuffled_data



def run_elnetreg_cellwise(data, cleaned_beh, curr_cell, circular_perms=None, avg_across_runs=False):
    print(f"...fitting and testing cell {curr_cell}")
    # parameters that seem to work, can be set flexibly
    alpha=0.001 ##0.01 used in El-gaby paper
    # l1_ratio= 0.01
    # cell_idx = int(curr_cell.split('-')[0])-1
    corr_dict, coefs_per_model = {}, {}
        
    model_string = []
    for model in data:
        if model.endswith('model'):
            model_string.append(model)
            corr_dict[model] = {}

    # first, for the current cell, select the relevant behaviour.
    mask = cleaned_beh[f"consistent_FR_{curr_cell}"]
    # next, use this one to mask out what simulations I don't want/ what 
    
    grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
    # prepare the held-out task order and the permutations with the current mask.
    unique_grids, idx_unique_grid, cleaned_beh['curr_neurons_idx_same_grids'], counts = np.unique(cleaned_beh[grid_cols][mask].to_numpy(),
                                                     axis=0,
                                                     return_index=True,
                                                     return_inverse=True,
                                                     return_counts=True)
   
    perms  = 1
    if circular_perms:
        perms = circular_perms
        shuffled_data = {}
        shuffled_data['neurons'] = data['perm_neurons']

     # prepare the result dictionaries    
    for model in model_string:
        corr_dict[model][curr_cell] = np.zeros((len(unique_grids), perms))
        coefs_per_model[model] = []
        
      
    for p_idx in range(0,perms):
        start_perms = time.time()
        unique_grid_idx = np.unique(cleaned_beh['curr_neurons_idx_same_grids'].to_numpy())
        
        if avg_across_runs == True:
            curr_cell_df = data['normalised_neurons'][curr_cell]
            # 1) average all rows with the same grid id
            if circular_perms:
                rng = np.random.default_rng(123)
                R, T = curr_cell_df.shape
                arr = curr_cell_df.to_numpy().copy()
                for i in range(R):
                    k = int(rng.integers(0, T))
                    arr[i] = np.roll(arr[i], -k)      # left shift row i by k
                curr_cell_df = pd.DataFrame(arr, index=curr_cell_df.index, columns=curr_cell_df.columns)
            curr_cell_df_avg_by_grid = curr_cell_df.groupby(cleaned_beh['curr_neurons_idx_same_grids'], sort=False).mean()               # index = unique grid ids
            
        for left_out_grid_idx in unique_grid_idx:
            train_grid_mask = cleaned_beh['curr_neurons_idx_same_grids'] != left_out_grid_idx
            test_grid_mask = cleaned_beh['curr_neurons_idx_same_grids'] == left_out_grid_idx
            if avg_across_runs == True:
                curr_neuron_training_tasks = curr_cell_df_avg_by_grid.drop(left_out_grid_idx, errors='ignore').to_numpy()
                # pick the one that is already averaged by grids
                curr_neuron_training_tasks_flat = curr_neuron_training_tasks.reshape(-1).astype(float)
                curr_neuron_heldouttasks_flat = curr_cell_df_avg_by_grid.loc[left_out_grid_idx].to_numpy()   
            
            else:
                curr_neuron_training_tasks = data['normalised_neurons'][curr_cell].loc[train_grid_mask].to_numpy()
                curr_neuron_training_tasks_flat = curr_neuron_training_tasks.reshape(-1).astype(float)
                curr_neuron_heldouttask = data['normalised_neurons'][curr_cell].loc[test_grid_mask].to_numpy()
                
                # permuting the cells
                # only if not averaged, in this case that happens befre
                if circular_perms:
                    rng = np.random.default_rng(123)  
                    # circular shift of neurons time series by random bins along time_axis
                    T = curr_neuron_heldouttask.shape[1]
                    for i_r, repeat in enumerate(curr_neuron_heldouttask):
                        k = int(rng.integers(0, T))     # uniform in {0, 1, ..., T-1}
                        curr_neuron_heldouttask[i_r] = np.roll(repeat, -k)  # left shift by k
                
                curr_neuron_heldouttasks_flat = curr_neuron_heldouttask.reshape(-1).astype(float)


            for entry in model_string:
                # define test and train regressors
                modelled_neurons = data[entry]  # numpy array, shape: (N_trials, C, *S)
                curr_neurons_idx_same_grids = cleaned_beh['curr_neurons_idx_same_grids'].to_numpy()
                
                def flatten_keep_axis(x, keep_axis=1, dtype=float):
                    keep_axis %= x.ndim
                    return np.moveaxis(x, keep_axis, 0).reshape(x.shape[keep_axis], -1).astype(dtype)
                
                if avg_across_runs== True:
                    # --- TRAIN: average within each grid except the left-out one ---
                    # (uses sorted unique grids; if you need original order, use pandas.unique)
                    train_grids = np.unique(curr_neurons_idx_same_grids[curr_neurons_idx_same_grids != left_out_grid_idx])
                    # (G, C, *S): mean across trials within each grid
                    sel_train = np.stack([modelled_neurons[curr_neurons_idx_same_grids == g].mean(axis=0) for g in train_grids], axis=0)
                
                    # --- TEST: average all repetitions of the left-out grid ---
                    # (1, C, *S): keepdims=True so the next step can keep axis=1 consistently
                    sel_test = modelled_neurons[curr_neurons_idx_same_grids == left_out_grid_idx].mean(axis=0, keepdims=True)
                else:
                    # original selection without averaging
                    sel_train = modelled_neurons[curr_neurons_idx_same_grids != left_out_grid_idx]          # (N_train, C, *S)
                    sel_test  = modelled_neurons[curr_neurons_idx_same_grids == left_out_grid_idx]          # (N_test,  C, *S)
                
                # Flatten while keeping the channel axis (C) as rows
                sim_neurons_training_tasks_flat = flatten_keep_axis(sel_train, keep_axis=1)
                sim_neurons_test_tasks_flat     = flatten_keep_axis(sel_test,  keep_axis=1)

                reg = ElasticNet(alpha=alpha, positive=True).fit(sim_neurons_training_tasks_flat.T, curr_neuron_training_tasks_flat)


                coeffs_flat=reg.coef_
                coefs_per_model[entry].append(coeffs_flat)
                # next, create the predicted activity neuron.
                predicted_activity_curr_neuron = np.sum((coeffs_flat*sim_neurons_test_tasks_flat.T), axis = 1)
                Predicted_Actual_correlation=nan_safe_pearsonr(curr_neuron_heldouttasks_flat,predicted_activity_curr_neuron)

                corr_dict[entry][curr_cell][left_out_grid_idx,p_idx] = Predicted_Actual_correlation
                

    end_perms = time.time()
    print(f"Fitting all {p_idx} permutations took {end_perms - start_perms:.2f} seconds")
    
    all_results = {}
    all_results['curr_cell'] = curr_cell
    all_results['corr'] = corr_dict
    
    return all_results


def compute_one_subject(sub,trials, save_regs=False, avg_across_runs = False, comp_circular_perms=False):
    # load data
    data_raw, source_dir, subj_reg_file = get_data(sub)
    group_dir_corrs = f"{source_dir}/group/corrs"
    if not os.path.isdir(group_dir_corrs):
        os.mkdir(group_dir_corrs)
        
    # filter data for only those repeats that were 1) correct and 2) not the first one
    data = mc.analyse.helpers_human_cells.filter_data(data_raw, sub, trials)
    simulated_regs = mc.analyse.helpers_human_cells.prep_regressors_for_neurons_360(data)
    
    # prepare cleaning the cells by grids.
    beh_df = data[f"sub-{sub:02}"]['beh'].copy()
    grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
    unique_grids, _, beh_df['idx_same_grids'], _ = np.unique(
        beh_df[grid_cols].to_numpy(),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True
    )
    
    # depending on which cells this neuron is firing enough for, define a different mask.
    for neuron_idx, curr_neuron in enumerate(data[f"sub-{sub:02}"]['normalised_neurons']):
        # this has the filter for each neuron.
        beh_clean = mc.analyse.helpers_human_cells.extract_consistent_grids(data[f"sub-{sub:02}"]['normalised_neurons'][curr_neuron].to_numpy(), curr_neuron, beh_df)
        
        
    cells = list(data[f"sub-{sub:02}"]['normalised_neurons'])
    
    # run the analysis cell-wise.
    for cell_idx, cell in enumerate(cells):
         parallel_results = run_elnetreg_cellwise(simulated_regs[f"sub-{sub:02}"],beh_clean, cell, circular_perms=comp_circular_perms, avg_across_runs=avg_across_runs)    
    import pdb; pdb.set_trace()
         
    # or parallalised.
    #parallel_results = Parallel(n_jobs=-1)(delayed(run_elnetreg_cellwise)(simulated_regs[f"sub-{sub:02}"],beh_clean, cell, circular_perms=comp_circular_perms, avg_across_runs=avg_across_runs) for cell_idx, cell in enumerate(cells))
    
    result_dir = {}
    result_dir['raw']= {}
    for cell_idx in parallel_results:
        curr_cell = cell_idx['curr_cell']
        result_dir['raw'][curr_cell] = cell_idx['corr']

    # define the basename
    result_file_name  = f"sub-{sub}_corrs_360bins"
    
    # first, save the basic result
    with open(os.path.join(group_dir_corrs,result_file_name), 'wb') as f:
        pickle.dump(result_dir['raw'], f)

    print(f"saved the modelled data as {group_dir_corrs}/{result_file_name}")

 
    
 
# # # # if running from command line, use this one!   
# if __name__ == "__main__":
#     #print(f"starting regression for subject {sub}")
#     fire.Fire(compute_one_subject)
#     # call this script like
#     # python wrapper_human_cells_elnetreg.py 5 --models_I_want='['withoutnow', 'onlynowand3future', 'onlynextand2future']' --exclude_x_repeats='[1,2,3]' --randomised_reward_locations=False --save_regs=True

# ['withoutnow', 'only2and3future','onlynowandnext', 'onlynowand3future', 'onlynextand2future']
# ['only','onlynowand3future', 'onlynextand2future']

if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    compute_one_subject(
        sub=15,
        trials = 'early',
        save_regs=True,
        avg_across_runs=True,
        # comp_circular_perms=100
    )


# these are hard-coded right now, so include them in the 'only' + models list 'state_reg', 'complete_musicbox_reg', 'reward_musicbox_reg', 'location_reg'




