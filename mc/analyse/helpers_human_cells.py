#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:41:06 2025

@author: Svenja Küchenhoff
helpers functions to analyse human cells


"""
import pickle
import numpy as np
import os
import re
import scipy
import math
import mc
from matplotlib import pyplot as plt
import glob
import pandas as pd
import copy
# import pdb; pdb.set_trace()

def read_files_to_list_ordered(folder, pattern_string):
    # import pdb; pdb.set_trace()
    # This pattern matches "timings_rewards_grid<number>_sub50.csv" where <number> can be any integer
    pattern = re.compile(pattern_string)
    # List all files in the directory and filter them based on the pattern
    files = [f for f in os.listdir(folder) if pattern.match(f)]
    # Sort files by the numerical value following "grid"
    files.sort(key=lambda x: int(pattern.search(x).group(1)))
    # Read each file into a pandas DataFrame and store them in a list
    data_list = [np.genfromtxt(f"{folder}/{file}", delimiter=',') for file in files] 
    return data_list

def read_string_files_to_list_ordered(folder, pattern_string):
    # Create a list to store the file contents
    file_contents = []
    pattern = re.compile(pattern_string)
    files = [f for f in os.listdir(folder) if pattern.match(f)]
    # Sort files by the numerical value following "grid"
    files.sort(key=lambda x: int(pattern.search(x).group(1)))
    
    
    # # Construct the full pattern to search for files
    # full_pattern = f"{source_folder}/{pattern}"
    # # Use glob to find all files that match the pattern
    # file_list = sorted(glob.glob(full_pattern))
    # # Read each file and append the contents to the list
    for file in files:
        df = pd.read_csv(f"{folder}/{file}", header=None)
        file_contents.append(df.iloc[0].tolist())
    return file_contents



def buttons_to_ints(button_list):
    buttons_int = []
    for elem in button_list:
        if elem == 'LeftArrow':
            buttons_int.append(0)
        if elem == 'UpArrow':
            buttons_int.append(1)
        if elem == 'RightArrow':
            buttons_int.append(2)
        if elem == 'DownArrow':
            buttons_int.append(3)
        if elem == 'Return':
            buttons_int.append(99)
    return buttons_int
    


def button_change_indices(buttons):
    # Convert list to NumPy array
    arr = np.array(buttons)
    # Find the differences between consecutive elements
    changes = np.where(arr[:-1] != arr[1:])[0] + 1
    return changes
    

    
def load_cell_data(source_folder, subject_list):
    # load all files I prepared with Matlab into a subject dictionary
    data_dir = {}
    for sub in subject_list:
        print(f"loading files for subject {sub}")
        data_dir[f"sub-{sub}"] = {}
        data_dir[f"sub-{sub}"]["reward_configs"] = np.genfromtxt(f"{source_folder}/s{sub}/cells_and_beh/all_configs_sub{sub}.csv", delimiter=',')
        with open(f"{source_folder}/s{sub}/cells_and_beh/all_cells_region_labels_sub{sub}.txt", 'r') as file:
            data_dir[f"sub-{sub}"]["cell_labels"] = [line.strip() for line in file]
        timinges_pattern = r'timings_rewards_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        location_pattern = r'locations_per_25ms_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        cells_pattern = r'all_cells_firing_rate_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        button_pattern = r'buttons_per_25ms_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        data_dir[f"sub-{sub}"]["timings"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/cells_and_beh", timinges_pattern)
        data_dir[f"sub-{sub}"]["locations"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/cells_and_beh", location_pattern)
        data_dir[f"sub-{sub}"]["neurons"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/cells_and_beh", cells_pattern)
        
        data_dir[f"sub-{sub}"]["buttons"] = read_string_files_to_list_ordered(f"{source_folder}/s{sub}/cells_and_beh", button_pattern)
        # import pdb; pdb.set_trace() 
    return data_dir


def neurons_concat_per_ROI_acrosstasks(neuron_dict, order, unique_tasks = False, dont_average_tasks = False):
    # import pdb; pdb.set_trace()
    
    if unique_tasks == True:
        tasks_to_include = order[0:8]
    elif unique_tasks == False:
        tasks_to_include = order[0:6]
    if dont_average_tasks == True:
        tasks_to_include = ['task_A_0','task_A_1','task_A_2', 
                        'task_B_0','task_B_1','task_B_2',
                        'task_C_0', 'task_C_1','task_C_2',
                        'task_D_0', 'task_D_1', 'task_D_2',
                        'task_E_0', 'task_E_1', 'task_E_2',
                        'task_F_0', 'task_F_1', 'task_F_2']
        
    # next concatenate the neurons per ROI according to the order_of_task list
    neuron_temp_dict = {}
    for ROI in neuron_dict:
        if ROI not in neuron_temp_dict:
            neuron_temp_dict[ROI] = {}
        for task in tasks_to_include:
            task_data = {}
            for label in sorted(neuron_dict[ROI]):
                if label.startswith(task):
                    # Extract the cell label which follows the task label format "task_cellLabel"
                    _, cell_label = label.split(f"{task}_", 1)
            
                    # Initialize the cell label key if not present
                    if cell_label not in task_data:
                        task_data[cell_label] = neuron_dict[ROI][label]
                        
            # After processing all labels for the current task, add them to the main dictionary
            neuron_temp_dict[ROI][task] = task_data

    #import pdb; pdb.set_trace()
    neuron_concat_dict = {}
    for ROI in sorted(neuron_temp_dict):
        task_matrices = []
        for task in sorted(neuron_temp_dict[ROI]):
            cell_concat = []
            for cell in sorted(neuron_temp_dict[ROI][task]):
                #import pdb; pdb.set_trace()
                if len(neuron_temp_dict[ROI][task][cell]) < 2:
                    empty_cell = np.zeros((1,4))
                    cell_concat.append(empty_cell)
                else:
                    cell_concat.append(np.transpose(np.expand_dims(neuron_temp_dict[ROI][task][cell], axis = 1)))
            if cell_concat:
                #import pdb; pdb.set_trace()   
                task_concat = np.concatenate(cell_concat, axis=0)
                task_matrices.append(task_concat)

        # Concatenate all task matrices along columns (axis 1)
        if task_matrices:
            neuron_concat_dict[f"{ROI}_concat"] = np.concatenate(task_matrices, axis=1)
    # clean the 0 rows if there are any!!
    # i think some of the tasks dont have 3 repeats. I'm wondering how to solve this
    # including the same data twice would make some tasks a lot more similar to each other later
    # maybe just exclude those subjects???
    
    
    for ROI in neuron_concat_dict:
        # import pdb; pdb.set_trace()
        zero_rows = np.all(neuron_concat_dict[ROI] == 0, axis=1)
        if np.sum(zero_rows):
            print(f"careful - excluding cells for ROI {ROI}")
        not_all_zero_rows = ~zero_rows
        neuron_concat_dict[ROI] = neuron_concat_dict[ROI][not_all_zero_rows]
        
    
    return neuron_concat_dict



                    
def run_RSA(data, only_specific_model = False, per_ROI = True, plotting = True, simple_models = False, dont_avg_rep_tasks = False):
    # import pdb; pdb.set_trace()
    # normal version is going to be collapse all subjects, but split cells by ROIs
    # start by labelling the grids.
    # CONTINUE HERE!!
    if only_specific_model:
        # DO SOMETHING DIFFERENT THAN SORT THE GRIDS!!!
        prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data, unique = False)
        neurons, grid_labels = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data, specific_model = True)
        # now next step is concatenate all grids in the correct order
        # for all models.
        simulated_data_concat, order_of_tasks =  mc.analyse.helpers_human_cells.models_concat_and_avg_across_subj(prepared_data, specific_model = True)
    else:
        if dont_avg_rep_tasks == True:
            # CHANGE THIS HERE SUCH THAT ONE CAN ALSO CHOOSE TO NOT AVERAGE A TASK!
            # THIS WOULD MAKE THE RDMS BIGGER
            # WIHTOUT AVERAGING; JUST SORT BY TASK_A ETC
            prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data, unique = True, dont_average_tasks= True)
            neurons, grid_labels = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data, specific_model = False, collapse_PFC = True, dont_avg_grids=True)
            simulated_data_concat, order_of_tasks =  mc.analyse.helpers_human_cells.models_concat_and_avg_across_subj(prepared_data, specific_model = False, dont_average= True)

        else:
            # now next step is concatenate all grids in the correct order for all models.
            prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data, unique = True)
            neurons, grid_labels = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data, specific_model = False, collapse_PFC = True)
            simulated_data_concat, order_of_tasks =  mc.analyse.helpers_human_cells.models_concat_and_avg_across_subj(prepared_data, specific_model = False, dont_average= False)
      
    #import pdb; pdb.set_trace()
    
    if dont_avg_rep_tasks == False:
        # instead of pooling by subject, pool by ROI.
        # next concatenate the neurons per ROI according to the order_of_task list
        neurons_concat = mc.analyse.helpers_human_cells.neurons_concat_per_ROI_acrosstasks(neurons, order_of_tasks, unique_tasks = True)
        task_configs = order_of_tasks[0:8]
    
    elif dont_avg_rep_tasks == True:
        # instead of pooling by subject, pool by ROI.
        # next concatenate the neurons per ROI according to the order_of_task list
        neurons_concat = mc.analyse.helpers_human_cells.neurons_concat_per_ROI_acrosstasks(neurons, order_of_tasks, unique_tasks = True, dont_average_tasks=True)
        task_configs = order_of_tasks[0:8*2]
    
    RDM_dict = {}
    # plot the averaged simulated and cleaned data
    all_models_string = []
    for model in simulated_data_concat:
        all_models_string.append(model)
        # mc.simulation.predictions.plot_without_legends(simulated_data_concat[model], titlestring= f"{model} model, averaged across runs", intervalline= 4, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
        RDM_dict[model] = mc.simulation.RDMs.within_task_RDM(simulated_data_concat[model], plotting = True, titlestring = f"Between tasks {model} RSM, 4*8, averaged over runs", intervalline= 4)
    
    if only_specific_model:
        RDM_dict = {}
        all_models_string = only_specific_model.copy()
        for model in all_models_string:
            RDM_dict[model] = mc.simulation.RDMs.within_task_RDM(simulated_data_concat[model], plotting = True, titlestring = f"Between tasks {model} RSM, 4*8, averaged over runs", intervalline= 4)
    
    mc.simulation.RDMs.plot_RDMs(RDM_dict, len(task_configs))       

    if simple_models == True:
        regressors_to_include = ['location', 'curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew', 'state']
    else:
        # midn_model, phas_model, loc_model, stat_model, phas_stat, clo_model, curr_neurons
        regressors_to_include = ['clo_model', 'phas_model', 'loc_model', 'stat_model'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
    
    if only_specific_model: 
        regressors_to_include = all_models_string
        
    results= {}
    neuron_RDMs = {}
    all_rois_string = []
    for ROI in sorted(neurons_concat): 
        all_rois_string.append(ROI)
        plt.figure();
        plt.imshow(neurons_concat[ROI], aspect = 'auto')
        plt.title(ROI)
        neuron_RDMs[ROI] = mc.simulation.RDMs.within_task_RDM(neurons_concat[ROI], plotting = True, titlestring = f"Between tasks {ROI} RSM, 4*8, averaged over runs", intervalline= 4)
        regressors = {}
        for i, model in enumerate(sorted(regressors_to_include)):
            regressors[model] = RDM_dict[model].copy()
        results[ROI] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)

    mc.simulation.RDMs.plot_RDMs(neuron_RDMs, len(task_configs), flexyscale=False)       
    
    if only_specific_model == True:         
        mc.plotting.results.overview_regression(results, all_rois_string, regressors_to_include, combo = True)  
    
    if only_specific_model == False:  
        results_single_model = {}
        for ROI in neurons_concat:
            results_single_model[ROI] = {}
            for model in all_models_string:
                regressors = {}
                regressors[model] = RDM_dict[model].copy()
                results_single_model[ROI][model] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)
       
        mc.plotting.results.overview_regression(results_single_model, all_rois_string, all_models_string)
        
        if simple_models == True:
            regressors_to_include = ['curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew']
        else:
            # ['two_fut_rings_split_clock', 'phas_stat', 'phas_model', 'three_fut_rings_split_clock', 'loc_model', 'curr_rings_split_clock', 'stat_model', 'clo_model', 'midn_model', 'one_fut_rings_split_clock']
            regressors_to_include = ['clo_model', 'midn_model'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
        regressors = {}
        for i, model in enumerate(sorted(regressors_to_include)):
            regressors[model] = RDM_dict[model].copy()
        results= {}
        for ROI in neurons_concat: 
            results[ROI] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)
        mc.plotting.results.overview_regression(results, all_rois_string, regressors_to_include, combo = True)  
        
            
        if simple_models == True:
            regressors_to_include = ['curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew', 'state']
        else:
            # ['two_fut_rings_split_clock', 'phas_stat', 'phas_model', 'three_fut_rings_split_clock', 'loc_model', 'curr_rings_split_clock', 'stat_model', 'clo_model', 'midn_model', 'one_fut_rings_split_clock']
            regressors_to_include = ['clo_model', 'phas_model'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
        regressors = {}
        for i, model in enumerate(sorted(regressors_to_include)):
            regressors[model] = RDM_dict[model].copy()
        results= {}
        for ROI in neurons_concat: 
            results[ROI] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)
        mc.plotting.results.overview_regression(results, all_rois_string, regressors_to_include, combo = True)  
        
        if simple_models == False:
            # ['two_fut_rings_split_clock', 'phas_stat', 'phas_model', 'three_fut_rings_split_clock', 'loc_model', 'curr_rings_split_clock', 'stat_model', 'clo_model', 'midn_model', 'one_fut_rings_split_clock']
            regressors_to_include = ['curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
            regressors = {}
            for i, model in enumerate(sorted(regressors_to_include)):
                regressors[model] = RDM_dict[model].copy()
            results= {}
            for ROI in neurons_concat: 
                results[ROI] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)
            mc.plotting.results.overview_regression(results, all_rois_string, regressors_to_include, combo = True)  
            

    import pdb; pdb.set_trace()
    return results



def run_state_RSA(data,  per_ROI = True, plotting = True, only_reward_times = None, no_bins_per_state= None, sim_fake_data =False):
    
    # normal version is going to be collapse all subjects, but split cells by ROIs
    # start by labelling the grids.

    prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data, unique = False)
    neurons, grids = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data, specific_model = True)

        
    # now next step is concatenate all grids in the correct order
    # for all models.

    # instead, create state model in the correct order.
    states = ['A','B','C','D']
    for ROI in neurons:
        for task in neurons[ROI]:
            length_averaged_task = len(neurons[ROI][task])
            
    state_model = np.zeros((int(length_averaged_task),len(states)))
    length_state = int(len(neurons[ROI][task])/len(states))

    for s_i, state in enumerate(states):
        if s_i == 0:
            state_model[0:length_state,s_i] = 1
        else:
            state_model[length_state*s_i:length_state*(s_i+1),s_i] = 1
    state_model = np.tile(np.transpose(state_model), (1,len(grids)))
    # pool by ROI and concatenate the neurons per ROI according to the order_of_task list
    neurons_concat = mc.analyse.helpers_human_cells.neurons_concat_per_ROI_acrosstasks(neurons, grids)
    
    RDM_dict = {}
    RDM_dict['state'] = mc.simulation.RDMs.within_task_RDM(state_model, plotting = True, titlestring = f"Between tasks state RSM, {4*len(grids)}, averaged over runs", intervalline= length_averaged_task)
    
    
    if sim_fake_data:
        fake_neurons = {}
        fake_neurons['fake_ROI'], grid_labels = mc.analyse.helpers_human_cells.pool_by_grid(prepared_data)
        fake_neurons_concat = mc.analyse.helpers_human_cells.neurons_concat_per_ROI_acrosstasks(fake_neurons, grids)
        #RDM_dict['fake_neurons_RDM'] = mc.simulation.RDMs.within_task_RDM(fake_neurons_concat['fake_ROI_concat'], plotting = True, titlestring = f"Between tasks simualted neurons RSM, {4*len(grids)}, averaged over runs", intervalline= length_averaged_task)
        neurons_concat['fake_neurons'] = fake_neurons_concat['fake_ROI_concat'].copy()

    mc.simulation.RDMs.plot_RDMs(RDM_dict, len(grids))       
    results= {}
    neuron_RDMs = {}
    all_rois_string = []
    for ROI in sorted(neurons_concat): 
        results[ROI] = {}
           
    for ROI in sorted(neurons_concat): 
        all_rois_string.append(ROI)
        plt.figure();
        plt.imshow(neurons_concat[ROI], aspect = 'auto')
        plt.title(ROI)
        neuron_RDMs[ROI] = mc.simulation.RDMs.within_task_RDM(neurons_concat[ROI], plotting = True, titlestring = f"Between tasks {ROI} RSM, {4*len(grids)}, averaged over runs", intervalline= length_averaged_task)
        results[ROI]['state'] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], RDM_dict, mask_within = True, no_tasks = len(grids), t_val = True, plotting= False)
    mc.simulation.RDMs.plot_RDMs(neuron_RDMs, len(grids), flexyscale=False)       
    mc.plotting.results.overview_regression(results, all_rois_string, ['state'], combo = False, only_reward_times =only_reward_times, no_bins_per_state =no_bins_per_state)  

    import pdb; pdb.set_trace()
    return results    







def models_concat_and_avg_across_subj(data, specific_model = False, dont_average = False):
    # concatenate all grids in the correct order
    # first loop: collect the same model for the same task for all subjects with np.stack
    # second loop: average the same model same task across all subjects
    # and in the end average across task repeats
    # first figure out which grids to consider.
    temp_stacked = {}
    all_models_set = set()
    
    if specific_model == True:
        grid_labels = ['average_task_A', 'average_task_B', 'average_task_C', 
                       'average_task_D', 'average_task_E', 'average_task_F']
        grid_distribution = mc.analyse.helpers_human_cells.redistribute_grids(data, grid_labels)

    if dont_average == True:
        grid_labels = ['task_A', 'task_B', 'task_C', 
                           'task_D', 'task_E', 'task_F']
        

        for sub in data:
            for task in grid_labels:  
                count_to_three = 0
                for grid in data[sub]:
                    if count_to_three > 1:
                        continue
                    elif grid.startswith(task):
                        temp_stacked[f"{task}_{count_to_three}_{sub}"] = {}
                        count_to_three = count_to_three+1
        
        # if no average across tasks, only average across subjects.
        for sub in data:
            for task in grid_labels:  
                count_to_three = 0
                for grid in data[sub]:
                    if count_to_three > 1:
                        continue
                    elif grid.startswith(task):
                        curr_grid = f"{task}_{count_to_three}_{sub}"
                        count_to_three = count_to_three+1
                        
                        if grid == 'task_A_3_sub_sub-43':
                            grid = 'task_A_14_sub_sub-43'
                        if grid == 'task_A_19_sub_sub-51':
                            grid = 'task_A_11_sub_sub-51'
                        for model in data[sub][grid]:
                            if model not in ['neurons']:
                                all_models_set.add(model)
                                # here always append 3x 
                                if model not in temp_stacked[curr_grid]:
                                    temp_stacked[curr_grid][model] = [data[sub][grid][model]]
                                else:
                                    temp_stacked[curr_grid][model].append(data[sub][grid][model])
            
        temp_across_subs = {}
        for task in temp_stacked:
            task_across_subs = task.split("_sub")[0]
            temp_across_subs[task_across_subs] = {}
            for model in temp_stacked[task]:
                temp_across_subs[task_across_subs][model] = np.stack(temp_stacked[task][model], axis = 0)
        
        for task in temp_across_subs:
            for model in temp_across_subs[task]:
                temp_across_subs[task][model] = np.mean(temp_across_subs[task][model], axis = 0)
    
        # then, last step, concatenate all tasks. store which order you concatenated.
        models_task_concat = {}
        order_of_tasks = []
        for model in all_models_set:
            for task in sorted(temp_across_subs):
                order_of_tasks.append(task)
                if model not in models_task_concat:
                    models_task_concat[model] = temp_across_subs[task][model]
                else:
                    models_task_concat[model] = np.concatenate((models_task_concat[model], temp_across_subs[task][model]), axis = 1)
            
    
    

    elif dont_average == False:                  
        for sub in data:
            count_to_three = 0
            for grid in data[sub]: 
                if grid.startswith('average'):
                    if specific_model == True:
                        # first redistribute the tasks.
                        # Determine which predefined grid label this grid should be associated with
                        for label, grids in grid_distribution.items():
                            if grid in grids or grid == label:
                                current_grid = label  # Use the predefined grid label for processing
                                break
                    else: 
                        current_grid = grid 
                    temp_stacked[current_grid] = {}

        # reduce data for models and grids by subject dimension, make a list instead.
        for sub in data:
            for grid in data[sub]:
                if grid.startswith('average'):
                    if specific_model == True:
                        # first redistribute the tasks.
                        # Determine which predefined grid label this grid should be associated with
                        for label, grids in grid_distribution.items():
                            if grid in grids or grid == label:
                                current_grid = label  # Use the predefined grid label for processing
                                break
                    else: 
                        current_grid = grid 
                    for model in data[sub][current_grid]:
                        if model not in ['neurons']:
                            all_models_set.add(model)
                            if model not in temp_stacked[current_grid]:
                                temp_stacked[current_grid][model] = [data[sub][current_grid][model]]
                            else:
                                temp_stacked[current_grid][model].append(data[sub][current_grid][model])

        # stack all lists and average across subjects
        for task in temp_stacked:
            # probably do something like split after sub and then average across subs.
            for model in temp_stacked[task]:
                temp_stacked[task][model] = np.stack(temp_stacked[task][model], axis = 0)
            for model in temp_stacked[task]:
                temp_stacked[task][model] = np.mean(temp_stacked[task][model], axis = 0)
                      
        # then, last step, concatenate all tasks. store which order you concatenated.
        models_task_concat = {}
        order_of_tasks = []
        for model in all_models_set:
            for task in sorted(temp_stacked):
                order_of_tasks.append(task)
                if model not in models_task_concat:
                    models_task_concat[model] = temp_stacked[task][model]
                else:
                    models_task_concat[model] = np.concatenate((models_task_concat[model], temp_stacked[task][model]), axis = 1)
        
    return models_task_concat, order_of_tasks


def threshold_round(v, threshold=0.5):
    return np.round(v).astype(int) if threshold == 0.5 else np.floor(v + (1 - threshold)).astype(int)


def neurons_to_bins_multimodel(time_course, model_dict):
    for i, m in enumerate(model_dict):
        model_idx = np.transpose(copy.deepcopy(model_dict[m]))
        for idx, neuron in enumerate(model_idx):
            model_idx[idx] = neuron*(idx+1)
        if i == 0:
            model_bins = np.sum(model_idx, axis = 0)
        else:
            # import pdb; pdb.set_trace()
            model_bins = model_bins + np.sum(model_idx, axis = 0)
        
    # Identify change points (boundaries) in the index array
    # Apply custom rounding
    rounded_vec = threshold_round(model_bins, threshold=0.2)
    
    change_points = np.where(rounded_vec[1:] != rounded_vec[:-1])[0] + 1
    
    bins = np.split(time_course, change_points)  # Split the neural array at change points
        
    # Compute the average for each segment
    timecourse_binned = np.array([segment.mean(axis = 0) for segment in bins])
    return timecourse_binned

    


    

def neurons_to_state_bins(time_course, state_model):
    state_model_idx = np.transpose(copy.deepcopy(state_model))
        
    for idx, state in enumerate(state_model_idx):
        state_model_idx[idx] = state*(idx+1)
    
    state_model_bins = np.sum(state_model_idx, axis = 0)
    # Define threshold (e.g., values within 0.5 of an integer are rounded to that integer)

    # Apply custom rounding
    rounded_vec = threshold_round(state_model_bins, threshold=0.2)
    # Identify change points
    state_change_points = np.where(rounded_vec[1:] != rounded_vec[:-1])[0] + 1

    state_bins = np.split(time_course, state_change_points)  # Split the neural array at change points
    
    # Compute the average for each segment
    timecourse_binned = np.array([segment.mean(axis = 0) for segment in state_bins])
     
    # import pdb; pdb.set_trace()
    # ok i dont know what the fuck is happening
    # there is some weird shit going on about float to integer conversion
    # fix this!!
    # i need these to be exactly number/3 state change points!!
    # also make sure to include the last one...
    # test_idx = np.transpose(state_model)
    # for idx, state in enumerate(state_model_idx):
    #     test_idx[idx] = np.ceil(state*(idx+1))
            
    # state_model_bins = [int(e) for e in state_model_bins]
    # # Identify change points (boundaries) in the index array
    # state_change_points = np.where(np.diff(state_model_bins) != 0)[0] + 1  # Find where index changes
    # state_change_points = np.where(state_model_bins[:-1] != vec[1:])[0] + 1
    return timecourse_binned
    


def redistribute_grids(data_dict, grid_string):
    # Step 1: Find all unique grids
    all_grids = set()
    for sub in data_dict:
        for grid in data_dict[sub]:
            if grid.startswith('average'):
                all_grids.add(grid)
    
    # Step 2: Separate additional grids
    additional_grids = [grid for grid in all_grids if grid not in grid_string]
    
    # Initialize a counter for distributing grids
    from collections import defaultdict
    grid_distribution = defaultdict(list)
    
    # Step 3: Distribute additional grids equally among predefined grid_labels
    for i, grid in enumerate(additional_grids):
        assigned_label = grid_string[i % len(grid_string)]
        grid_distribution[assigned_label].append(grid)
        
    return grid_distribution
      

def pool_by_grid(data, specific_model = False):
    neurons = {}
    if specific_model == True:
        grid_labels = ['average_task_A', 'average_task_B', 'average_task_C', 
                       'average_task_D', 'average_task_E', 'average_task_F']
        grid_distribution = mc.analyse.helpers_human_cells.redistribute_grids(data, grid_labels)
    else:
        grid_labels = []
        
    for sub in data:
        for grid in data[sub]:
            if grid.startswith('average'):
                if specific_model == True:
                    # first redistribute the tasks.
                    # Determine which predefined grid label this grid should be associated with
                    for label, grids in grid_distribution.items():
                        if grid in grids or grid == label:
                            current_grid = label  # Use the predefined grid label for processing
                            break
                else:
                    current_grid = grid
                # Loop through cells in the subject's data    
                for i_c, cell in enumerate(data[sub][grid]['fake_neurons']):    
                    neurons[f"{current_grid}_{sub}_fake_ROI_{i_c}"] = cell
                                
    return neurons, grid_labels

    

                          
def pool_by_ROI_and_grid(data, specific_model = False, collapse_PFC = False, dont_avg_grids = False):
    # first create the ROI sets
    # import pdb; pdb.set_trace()
    ROIs = {}
        
    if collapse_PFC == True:
        ROI_labels = ['hippocampal', "PFC", "entorhinal", "amygdala", "mixed"]
    else:
        ROI_labels = ['hippocampal', "PCC", "ACC", "OFC", "entorhinal", "amygdala", "mixed"]
    for ROI in ROI_labels:
        ROIs[ROI] = set()
    for sub in data:
        for region in data[sub]['cell_labels']:
            if collapse_PFC == True:
                if 'ACC' in region or 'PCC' in region or 'OFC' in region:
                    ROIs['PFC'].add(region)
                elif 'HC' in region:
                    ROIs['hippocampal'].add(region)
                elif 'EC' in region:
                    ROIs['entorhinal'].add(region)
                elif 'AMYG' in region:
                    ROIs['amygdala'].add(region)
                    
                else:
                    ROIs['mixed'].add(region)
            else:
                if 'ACC' in region:
                    ROIs['ACC'].add(region)
                elif 'PCC' in region:
                    ROIs['PCC'].add(region)
                elif 'OFC' in region:
                    ROIs['OFC'].add(region)   
                elif 'HC' in region:
                    ROIs['hippocampal'].add(region)
                elif 'EC' in region:
                    ROIs['entorhinal'].add(region)
                elif 'AMYG' in region:
                    ROIs['amygdala'].add(region)
                    
                else:
                    ROIs['mixed'].add(region)

                    
                    
    print(f"mixed ROI includes: {ROIs['mixed']}")
    ROIs['entorhinal'] = {'REC', 'LEC'}
    # then create a neuron dictionary, split by ROIs, where I save how each cell 
    # of that ROI fires for each grid.
    
    if specific_model == True:
        grid_labels = ['average_task_A', 'average_task_B', 'average_task_C', 
                       'average_task_D', 'average_task_E', 'average_task_F']
        grid_distribution = mc.analyse.helpers_human_cells.redistribute_grids(data, grid_labels)
    else:
        grid_labels = []
    
        
    if dont_avg_grids == True:
        grid_labels = ['task_A', 'task_B', 'task_C', 
                       'task_D', 'task_E', 'task_F']
    neurons = {}
    for ROI in ROIs:
        neurons[ROI] = {}
    # import pdb; pdb.set_trace()  
    if dont_avg_grids == True:
        for sub in data:
            for task in grid_labels:  
                count_to_three = 0
                for grid in sorted(data[sub]):
                    if count_to_three > 1:
                        continue
                    elif grid.startswith(task):
                        if grid == 'task_A_3_sub_sub-43':
                            grid = 'task_A_14_sub_sub-43'
                        if grid == 'task_A_19_sub_sub-51':
                            grid = 'task_A_11_sub_sub-51'
                        
                        curr_grid = f"{task}_{count_to_three}_{sub}"
                        count_to_three = count_to_three+1
                        
                        # Loop through cells in the subject's data    
                        for i_c, cell in enumerate(data[sub]['cell_labels']):
                            # inner: loop through new dict
                            for ROI in ROIs:
                                if cell in ROIs[ROI]:
                                    if neurons[ROI].get(f"{curr_grid}_{i_c}_{cell}") is None:
                                        # if grid in ['task_A_3_sub_sub-43', 'task_A_19_sub_sub-51']:
                                        #     continue
                                        # else:
                                        neurons[ROI][f"{curr_grid}_{i_c}_{cell}"] = data[sub][grid]['neurons'][i_c]

    else:
        # outer: loop through old data dict
        for sub in data:
            for grid in data[sub]:
                if grid.startswith('average'):
                    if specific_model == True:
                        # first redistribute the tasks.
                        # Determine which predefined grid label this grid should be associated with
                        for label, grids in grid_distribution.items():
                            if grid in grids or grid == label:
                                current_grid = label  # Use the predefined grid label for processing
                                break
                    else:
                        current_grid = grid
                    # Loop through cells in the subject's data    
                    for i_c, cell in enumerate(data[sub]['cell_labels']):
                        # inner: loop through new dict
                        for ROI in ROIs:
                            if cell in ROIs[ROI]:
                                if neurons[ROI].get(f"{current_grid}_{sub}_{i_c}_{cell}") is None:
                                    neurons[ROI][f"{current_grid}_{sub}_{i_c}_{cell}"] = data[sub][grid]['neurons'][i_c]
                                    
                                
                                
    return neurons, grid_labels

        
        
    
  
def label_unique_grids(data_dict, unique = True, dont_average_tasks = False):
    # import pdb; pdb.set_trace()
    if unique == True:
        task_labels = {}
        for sub in data_dict:
            task_labels[sub] = {'task_A': ([1., 9., 5., 8.]), 'task_B': ([2., 5., 7., 6.]), 'task_C': ([3., 7., 9., 5.]), 'task_D': ([4., 8., 1., 3.]), 'task_E': ([6., 4., 2., 9.]), 'task_F': ([7., 3., 4., 2.]), 'task_G': ([8., 2., 6., 7.]), 'task_H': ([9., 1., 3., 4.])}
        #task_labels = {'task_A': ([1., 9., 5., 8.]), 'task_B': ([2., 5., 7., 6.]), 'task_C': ([3., 7., 9., 5.]), 'task_D': ([4., 8., 1., 3.]), 'task_E': ([6., 4., 2., 9.]), 'task_F': ([7., 3., 4., 2.]), 'task_G': ([8., 2., 6., 7.]), 'task_H': ([9., 1., 3., 4.])}
    else:
        # import pdb; pdb.set_trace()
        task_labels = {}
        for sub in data_dict:
            task_labels[sub] = {}
            unique_quadruplets = {tuple(row) for row in data_dict[sub]['reward_configs']}
            for i, quad in enumerate(unique_quadruplets):
                task_labels[sub][f'task_{chr(65 + i)}'] = [int(num) for num in quad]  # chr(65) is 'A'
                
    
        for sub in data_dict:
            keys_to_modify = []
            for unique_task in task_labels[sub]:
                for grid_idx, grid in enumerate(data_dict[sub]['reward_configs']):
                    if list(grid) == list(task_labels[sub][unique_task]):
                        if sub == 'sub-43' and grid_idx == 3:
                            continue
                        if sub == 'sub-15' and grid_idx == 23:
                            continue
                        new_key = f"{unique_task}_grid_{grid_idx}_sub_{sub}"
                        old_key = f"grid_{grid_idx}"
                        keys_to_modify.append((old_key, new_key))
            for old_grid_key, new_grid_key in keys_to_modify:
                # print(old_grid_key, new_grid_key)
                data_dict[sub][new_grid_key] = data_dict[sub].pop(old_grid_key)
    
    if dont_average_tasks == False:                      
        for sub in data_dict:
            # last, but not least, create grid averages.
            # if there aren't 3 averages, just make 2 averages instead.
            # import pdb; pdb.set_trace()
            for unique_task in task_labels[sub]:
                data_dict[sub][f"average_{unique_task}"] = {}
                for grid in data_dict[sub]:
                    if grid.startswith(unique_task):
                        for model in data_dict[sub][grid]:
                            if model not in data_dict[sub][f"average_{unique_task}"]:
                                data_dict[sub][f"average_{unique_task}"][model] = [data_dict[sub][grid][model]]
                            else:
                               data_dict[sub][f"average_{unique_task}"][model].append(data_dict[sub][grid][model]) 
        # and in the end average across task repeats
        for sub in data_dict:
            for grid in data_dict[sub]:
                temp_stacked = {}
                if grid.startswith('average'):
                    for model in data_dict[sub][grid]:
                        # first collect all grids that are the same
                        # Stack the arrays along a new axis (axis 0)
                        temp_stacked[model] = np.stack(data_dict[sub][grid][model], axis=0)
                        # Compute the mean across the stacked dimension
                    # then average
                    for model in temp_stacked:
                        data_dict[sub][grid][model] = np.mean(temp_stacked[model], axis=0)
                        
    # i think basically don't do anything here if no average is needed.
    
    # elif dont_average_tasks == True:
    #     import pdb; pdb.set_trace()
    #     for sub in data_dict:
    #         # dont create averages across tasks.
    #         for unique_task in task_labels[sub]:
    #             data_dict[sub][f"average_{unique_task}"] = {}
    #             for grid in data_dict[sub]:
    #                 if grid.startswith(unique_task):
    #                     for model in data_dict[sub][grid]:
    #                         if model not in data_dict[sub][f"average_{unique_task}"]:
    #                             data_dict[sub][f"average_{unique_task}"][model] = [data_dict[sub][grid][model]]
    #                         else:
    #                            data_dict[sub][f"average_{unique_task}"][model].append(data_dict[sub][grid][model]) 
        
        
    #     for sub in data_dict:
    #         for grid in data_dict[sub]:
    #             temp_stacked = {}
    #             for task in task_labels[sub]:
    #                 if grid.startswith(task):
    #                     for model in data_dict[sub][grid]:
    #                         # first collect all grids that are the same
    #                         # Stack the arrays along a new axis (axis 0)
    #                         temp_stacked[model] = np.stack(data_dict[sub][grid][model], axis=0)
    #                         # Compute the mean across the stacked dimension
    #                     # then average
    #                     for model in temp_stacked:
    #                         data_dict[sub][grid][model] = np.mean(temp_stacked[model], axis=0)
    return data_dict


def prep_regressors_for_neurons(data_dict, models_I_want = None, exclude_x_repeats = None, randomised_reward_locations = False, avg_across_runs = False):
    # settings
    #import pdb; pdb.set_trace()
    
    no_state = 4
    no_locations = 9
    no_buttons = 4
    all_models = ['state_reg','complete_musicbox_reg','reward_musicbox_reg','buttonbox_reg','location_reg','current_reward_reg', 'next_reward_reg', 'second_next_reward_reg', 'previous_reward_reg']
    models_same_length = ['location_reg','current_reward_reg', 'next_reward_reg', 'second_next_reward_reg', 'previous_reward_reg']
    
    # clean the attempt 
    for s in data_dict:
        if s == 'sub-59':
            data_prep_tmp = copy.deepcopy(data_dict)
            for entry in ['buttons', 'locations', 'neurons', 'timings']:
                data_dict[s][entry] = [x for i, x in enumerate(data_prep_tmp[s][entry]) if i != 21]
            # data_dict[s]['buttons'] = [x for i, x in enumerate(data_prep_tmp[s]['buttons']) if i != 21]
            # data_dict[s]['locations'] = [x for i, x in enumerate(data_prep_tmp[s]['locations']) if i != 21]
            # data_dict[s]['neurons'] = [x for i, x in enumerate(data_prep_tmp[s]['neurons']) if i != 21]
            # data_dict[s]['timings'] = [x for i, x in enumerate(data_prep_tmp[s]['timings']) if i != 21]
            data_dict[s]['reward_configs'] =  np.delete(data_prep_tmp[s]['reward_configs'], 21, axis = 0)
        
    data_prep = copy.deepcopy(data_dict)
    
    for sub in data_dict:
        print(f"now starting to process data from subject {sub}")
        for m in all_models:
            data_prep[sub][m]=[]
        # data_prep[sub]['state_reg'], data_prep[sub]['complete_musicbox_reg'], data_prep[sub]['reward_musicbox_reg'], data_prep[sub]['buttonbox_reg'], data_prep[sub]['location_reg'] = [],[],[],[],[]

        if models_I_want:
            if models_I_want[0] == 'only':
                # data_prep[sub] = {k: v for k, v in data_prep[sub].items() if not (k in ['state_reg', 'complete_musicbox_reg', 'reward_musicbox_reg', 'buttonbox_reg', 'location_reg'] and v == [])}
                for model in models_I_want[1:]:
                    data_prep[sub][f"musicbox_{model}_rew_reg"] = []
                    data_prep[sub][f"musicbox_{model}_complete_reg"] = []
            else:
                for model in models_I_want:
                    data_prep[sub][f"musicbox_{model}_rew_reg"] = []
                    data_prep[sub][f"musicbox_{model}_complete_reg"] = []
        
        if randomised_reward_locations == False:
            reward_configurations = data_dict[sub]['reward_configs'].copy()
            
        elif randomised_reward_locations == True:
            # randomize reward configurations 
            shuffled_tasks = np.zeros((data_dict[sub]['reward_configs'].shape))
            indices = np.random.permutation(len(data_dict[sub]['reward_configs']))
            for j, i in enumerate(indices):
                if i != j:
                    shuffled_tasks[j] = data_dict[sub]['reward_configs'][i]
                elif i == j:
                    if i+1 == len(indices):
                        shuffled_tasks[j] = data_dict[sub]['reward_configs'][i-1]
                    else:
                        shuffled_tasks[j] = data_dict[sub]['reward_configs'][i+1]

                    
            reward_configurations = shuffled_tasks.copy()
            data_prep[sub]['shuffled_tasks'] = shuffled_tasks.copy()
            
            
        for grid_idx, grid_config in enumerate(reward_configurations):
            
            if exclude_x_repeats:
                start_from_repeat = np.max(exclude_x_repeats)
                timings_task = data_dict[sub]['timings'][grid_idx][start_from_repeat:]
            else:  
                timings_task = data_dict[sub]['timings'][grid_idx]
                
            if (sub == 'sub-15' and grid_idx == 23) or (sub == 'sub-43' and grid_idx == 3):
                continue
            if (sub == 'sub-25' and grid_idx == 9) or (sub == 'sub-52' and grid_idx == 6) or (sub == 'sub-44' and grid_idx == 3) or (sub == 'sub-28' and grid_idx == 16) or (sub == 'sub-02' and grid_idx == 18):
                # cut the last row of the timings
                timings_task = timings_task[:-1, :]
            
            if randomised_reward_locations == False:
                # check the match between timings and reward configs
                mc.simulation.predictions.test_timings_rew(sub, data_dict[sub]['locations'][grid_idx],timings_task, grid_config, grid_idx)
            
            # first run the old way of modelling.
            # this needs to loop more
            # import pdb; pdb.set_trace()
            neurons_for_task = data_dict[sub]['neurons'][grid_idx].copy()
            # fields need to be between 0 and 8, and keep them as integers
            locations_curr_grid = [int((field_no-1)) for field_no in data_dict[sub]['locations'][grid_idx]]
            task_config = [int((field_no-1)) for field_no in data_dict[sub]['reward_configs'][grid_idx]]
             
            models_per_rep_dict, per_rep_prep = {}, {}
            regression_across_repeats = []
            for i, reps in enumerate(timings_task):
                timings_repeat = [int(r) for r in reps]
                # and for next repeat
                if i+1 == len(timings_task):
                    # if this is the last repeat, cut the file differently
                    timings_next_rep = None
                else:
                    timings_next_rep = [int(elem) for elem in timings_task[i+1]]
                    
                per_rep_prep[f"rep_{i}"] = mc.analyse.helpers_human_cells.prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_rep, task_config, neurons_for_task)
                regression_across_repeats.append(mc.simulation.predictions.create_x_regressors_per_state(per_rep_prep[f"rep_{i}"], only_for_rewards=False))
                
                
                if models_I_want[0] != 'only':
                    models_per_rep = mc.simulation.predictions.set_continous_models_ephys(per_rep_prep[f"rep_{i}"],  grid_size = 3, no_phase_neurons=3, fire_radius = 0.25, wrap_around = 1, plot = False, split_clock = True, use_orig_timings= True)
                    # then prepare concatenating all of them
                    for model in models_per_rep:
                        if model not in models_per_rep_dict:
                            models_per_rep_dict[model] = []
                        models_per_rep_dict[model].append(models_per_rep[model])
                        if model not in data_prep[sub]:
                            data_prep[sub][model] = []
                
                #
                #
                regression_one_rep = mc.simulation.predictions.create_x_regressors_per_state(per_rep_prep[f"rep_{i}"], only_for_rewards=False)
                # s = f"rep_{i}"
                
                # print(f"for rep {i} timing last repeat is {per_rep_prep[s]['timings_repeat'][-1]}")
                # print(f"for rep {i} trajectory is {len(per_rep_prep[s]['trajectory'])}")
                # print(f"for rep {i} regression is {len(regression_one_rep.transpose())}")
                # print(f"for rep {i} model is {len(models_per_rep['stat_model'].transpose())}")
                # #
                # #
                            
            for m in models_per_rep_dict:
                if m in data_prep[sub]:
                    data_prep[sub][m].append(np.concatenate(models_per_rep_dict[m], axis = 1))
            
            regression_across_repeats_concat = np.concatenate(regression_across_repeats, axis = 1)

            # this version cuts everything between starting the trial, and finding the very last reward.
            data_prep[sub]['neurons'][grid_idx]= data_prep[sub]['neurons'][grid_idx][:, int(timings_task[0,0]):int(timings_task[-1,-1])]
            data_prep[sub]['locations'][grid_idx] = data_prep[sub]['locations'][grid_idx][int(timings_task[0,0]):int(timings_task[-1,-1]+1)]
            data_prep[sub]['buttons'][grid_idx] = data_prep[sub]['buttons'][grid_idx][int(timings_task[0,0]):int(timings_task[-1,-1]+1)]
            length_curr_grid = len(data_prep[sub]['locations'][grid_idx])
            
            # what if I do this instead???
            length_curr_grid = int(timings_task[-1,-1]) - int(timings_task[0,0])

            # DEAL WITH THIS
            #
            #
            #
            # here, go and test if the dimensions are right. if not, dublicate last column.
            for m in data_prep[sub]:
                if m.endswith('model'):
                    # deal with this!!! 
                    # this isn't a valid way to deal with the dimensions
                    # the length_curr_grid is a useless variable
                    if data_prep[sub][m][grid_idx].shape[1] < length_curr_grid :
                        import pdb; pdb.set_trace()
                        x = data_prep[sub][m][grid_idx]
                        while x.shape[1] < length_curr_grid :
                            last_column = data_prep[sub][m][grid_idx][:, -1].reshape(-1, 1)  # Extract and reshape the last column
                            x = np.hstack((x, last_column))                            
                        data_prep[sub][m][grid_idx] = x 
            
            # if len(models_per_rep_dict) > 0:
            #     print(f"now dimensions match: length is {length_curr_grid} and dims are {data_prep[sub][m][grid_idx].shape}")
            

            #data_prep[sub]['state_reg'].append(np.zeros((no_state-1, length_curr_grid)))
            data_prep[sub]['state_reg'].append(np.zeros((no_state, length_curr_grid)))
            data_prep[sub]['complete_musicbox_reg'].append(np.zeros((no_state*no_locations, length_curr_grid)))
            # data_prep[sub]['complete_musicbox_reg'].append(np.zeros((3*no_state*no_locations, length_curr_grid)))
            data_prep[sub]['reward_musicbox_reg'].append(np.zeros((no_state*no_locations, length_curr_grid)))
            # data_prep[sub]['location_reg'].append(np.zeros((no_locations, length_curr_grid)))
            data_prep[sub]['buttonbox_reg'].append(np.zeros((no_state*no_buttons, length_curr_grid)))
            # split simple musicbox
            for model in models_same_length:
                data_prep[sub][model].append(np.zeros((no_locations, length_curr_grid)))
            
            if models_I_want:
                if models_I_want[0] == 'only':
                    for model in models_I_want[1:]:
                        if model == 'withoutnow':
                            less_rows = 1
                        if model in ['only2and3future', 'onlynowandnext', 'onlynowand3future', 'onlynextand2future']:
                            less_rows = 2
                        
                        data_prep[sub][f"musicbox_{model}_rew_reg"].append(np.zeros(((no_state-less_rows)*no_locations, length_curr_grid)))
                        data_prep[sub][f"musicbox_{model}_complete_reg"].append(np.zeros(((no_state-less_rows)*no_locations, length_curr_grid)))
                else:
                    for model in models_I_want:
                        if model == 'withoutnow':
                            less_rows = 1
                        if model in ['only2and3future', 'onlynowandnext', 'onlynowand3future', 'onlynextand2future']:
                            less_rows = 2
                        
                        data_prep[sub][f"musicbox_{model}_rew_reg"].append(np.zeros(((no_state-less_rows)*no_locations, length_curr_grid)))
                        data_prep[sub][f"musicbox_{model}_complete_reg"].append(np.zeros(((no_state-less_rows)*no_locations, length_curr_grid)))
                
            
            # the following ones should take the timings as input, and fill in the 
            # empty arrays of the following regressors, according to the timings
            # and locations/buttonpresses:
            if randomised_reward_locations == True:
                # if shuffled_tasks[grid_idx][0] != data_dict[sub]['reward_configs'][grid_idx][0]:
                #     grid_config = shuffled_tasks[grid_idx]
                # elif grid_idx == 0:
                #     grid_config = shuffled_tasks[grid_idx+1]
                # elif grid_idx == len(data_dict[sub]['reward_configs']):
                #     grid_config = shuffled_tasks[grid_idx-1]
                print(f"shuffled reward config is {grid_config} and actual was {data_dict[sub]['reward_configs'][grid_idx]}")
            
            # import pdb; pdb.set_trace()
            adjust_grid_idx = 0 
            if sub == 'sub-43' and grid_idx > 3:
                adjust_grid_idx = -1
                # account for the skipped grid. no added regressor for that grid, so data_prep dict is one shorter
            data_prep[sub]['state_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.state_cells(data_prep[sub]['state_reg'][grid_idx+adjust_grid_idx],timings_task, grid_config)
            data_prep[sub]['reward_musicbox_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.music_box_simple_cells(data_prep[sub]['locations'][grid_idx], data_prep[sub]['reward_musicbox_reg'][grid_idx+adjust_grid_idx], timings_task, grid_config)
            
            # split musicbox
            data_prep[sub]['current_reward_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.single_reward(data_prep[sub]['locations'][grid_idx], data_prep[sub]['current_reward_reg'][grid_idx+adjust_grid_idx], timings_task, grid_config, setting='current')
            data_prep[sub]['next_reward_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.single_reward(data_prep[sub]['locations'][grid_idx], data_prep[sub]['next_reward_reg'][grid_idx+adjust_grid_idx], timings_task, grid_config, setting='next')
            data_prep[sub]['second_next_reward_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.single_reward(data_prep[sub]['locations'][grid_idx], data_prep[sub]['second_next_reward_reg'][grid_idx+adjust_grid_idx], timings_task, grid_config, setting='second_next')
            data_prep[sub]['previous_reward_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.single_reward(data_prep[sub]['locations'][grid_idx], data_prep[sub]['previous_reward_reg'][grid_idx+adjust_grid_idx], timings_task, grid_config, setting='previous')
            
            
            data_prep[sub]['location_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.locations_cells(data_prep[sub]['locations'][grid_idx], data_prep[sub]['location_reg'][grid_idx+adjust_grid_idx])
            
            data_prep[sub]['complete_musicbox_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.musicbox_cells_complete(data_prep[sub]['locations'][grid_idx], data_prep[sub]['complete_musicbox_reg'][grid_idx+adjust_grid_idx], timings_task, grid_config)
            data_prep[sub]['buttonbox_reg'][grid_idx+adjust_grid_idx] = mc.simulation.predictions.button_box_simple_cells(data_prep[sub]['buttons'][grid_idx], data_prep[sub]['buttonbox_reg'][grid_idx+adjust_grid_idx], timings_task)
            if models_I_want:
                for model in models_I_want:
                    if models_I_want[0] == 'only':
                        for model in models_I_want[1:]:
                            data_prep[sub][f"musicbox_{model}_rew_reg"][grid_idx+adjust_grid_idx] = mc.simulation.predictions.music_box_simple_cells(data_prep[sub]['locations'][grid_idx], data_prep[sub][f"musicbox_{model}_rew_reg"][grid_idx+adjust_grid_idx], timings_task, grid_config, setting = model)
                            data_prep[sub][f"musicbox_{model}_complete_reg"][grid_idx+adjust_grid_idx] = mc.simulation.predictions.musicbox_cells_complete(data_prep[sub]['locations'][grid_idx], data_prep[sub]['complete_musicbox_reg'][grid_idx+adjust_grid_idx], timings_task, grid_config, setting = model)
                    else:
                        for model in models_I_want:
                            data_prep[sub][f"musicbox_{model}_rew_reg"][grid_idx+adjust_grid_idx] = mc.simulation.predictions.music_box_simple_cells(data_prep[sub]['locations'][grid_idx], data_prep[sub][f"musicbox_{model}_rew_reg"][grid_idx+adjust_grid_idx], timings_task, grid_config, setting = model)
                            data_prep[sub][f"musicbox_{model}_complete_reg"][grid_idx+adjust_grid_idx] = mc.simulation.predictions.musicbox_cells_complete(data_prep[sub]['locations'][grid_idx], data_prep[sub]['complete_musicbox_reg'][grid_idx+adjust_grid_idx], timings_task, grid_config, setting = model)
                       
            # the difference isn't always bigger in the first one. there are differences between subjects!
            # differences = np.zeros((timings_task.shape[0],timings_task.shape[1]-1))
            # for i,row in enumerate(timings_task):
            #     for j, elem in enumerate(row):
            #         if j < 4:
            #             differences[i,j] = row[j+1]-elem      
                        
            # on this level, there is the option to average across repeats
            if avg_across_runs == True:
                # import pdb; pdb.set_trace()
                # CAREFUL!! 
                # this for some reason is still not the same length???
                # check for the state regression.
                # i think sometime shte last state doesnt match.
                # how do the timings differ?
                # goal is that all states are exactly 1 after regression!!!
                data_prep_tmp = copy.deepcopy(data_prep)
                for m in data_prep[sub]:
                    if m.endswith('reg') or m.endswith('model') or m.endswith('neurons'):
                        # this needs to be concatenated and all
                        data_prep[sub][m][grid_idx+adjust_grid_idx] = mc.simulation.predictions.transform_data_to_betas(data_prep_tmp[sub][m][grid_idx+adjust_grid_idx], regression_across_repeats_concat)
    # import pdb; pdb.set_trace()
    return data_prep


def identify_max_cells_for_model(result_dir):
    # import pdb; pdb.set_trace()
    df = pd.DataFrame()
    all_cells = {}
    i = 0
    for sub in result_dir:
        for model in result_dir[sub]:
            for cell in result_dir[sub][model]:
                df.at[i, 'cell'] = cell
                df.at[i, 'average_corr'] = np.mean(result_dir[sub][model][cell])
                df.at[i, 'model'] = model
                if len(sub) == 5:
                    prefix, num_str = sub.split('-')
                    # Convert the number to an integer, then format it as a two-digit string.
                    formatted_num = f"{int(num_str):02}"
                    df.at[i, 'subject'] = f"{prefix}-{formatted_num}"
                else:
                    df.at[i, 'subject'] = sub
                i = i + 1
        
    for model in result_dir[sub]:
        all_cells[model] = df[df['model'] == model]
    
    top_ten = {}
    for model in all_cells:
        top_ten[model] = all_cells[model].sort_values(by=['average_corr'], ascending=False)[0:10]
    # import pdb; pdb.set_trace()
    return top_ten, all_cells
      

    

def store_best_cells(best_cells, all_data, name_extension_string = None):
    result_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/best_cells"
    # import pdb; pdb.set_trace()
    for model in best_cells:
        # import pdb; pdb.set_trace()
        for index, row in best_cells[model].iterrows():
            subject = row['subject']
            cell_label = row['cell']
            cell_idx_str = cell_label.split('_', 2)[1]
            cell_idx = int(cell_idx_str)
            cells_to_store = []
            for task in all_data[subject]['neurons']:
                cells_to_store.append(task[cell_idx])
            subset_dict = {}
            subset_dict[cell_label] = cells_to_store.copy()
            subset_dict['reward_configs'] = all_data[subject]['reward_configs'].copy()
            subset_dict['locations'] = all_data[subject]['locations'].copy()
            subset_dict['timings'] = all_data[subject]['timings'].copy()
            file_name = f"{cell_label}_best_for_{model}"
            if name_extension_string:
                file_name = f"{cell_label}_best_for_{model}_{name_extension_string}"
            with open(os.path.join(result_folder,file_name), 'wb') as f:
                pickle.dump(subset_dict, f)
    
    
            

           
def prep_neurons_and_state(data_dict, repeats, only_reward_times, no_bins_per_state, sim_fake_data = False):
    # import pdb; pdb.set_trace()
    # first, do some modifications (integers, start from 0, make timings to bins)
    # then, call the simulate functions that model the different neurons
    # do this for every repeat < for ever grid < for every subject.
    data_prep = {}
    
    # assume that every task has at least 10 repeats.
    for sub in data_dict:
        print(f"now starting to process data from subject {sub}")
        data_prep[sub] = {}
        data_prep[sub]['cell_labels'] = data_dict[sub]['cell_labels'].copy()
        data_prep[sub]['reward_configs'] = data_dict[sub]['reward_configs'].copy()
        #import pdb; pdb.set_trace()
        for grid_idx, grid_config in enumerate(data_dict[sub]['reward_configs']):
            if (sub == 'sub-43' and grid_idx == 3) or (sub == 'sub-15' and grid_idx == 23) or (sub == 'sub-02' and grid_idx == 18):
                # probably aborted the experiment here
                data_prep[sub][f"grid_{grid_idx}"] = {}
                continue
            data_prep[sub][f"grid_{grid_idx}"] = {}
            # timings task is start A - find A - find B - find C - find D
            timings_task = data_dict[sub]['timings'][grid_idx].copy()
            # fields need to be between 0 and 8, and keep them as integers
            locations_curr_grid = [int((field_no-1)) for field_no in data_dict[sub]['locations'][grid_idx]]
            # current neural recordings 
            neurons_for_task = data_dict[sub]['neurons'][grid_idx].copy()
            task_config = [int((field_no-1)) for field_no in data_dict[sub]['reward_configs'][grid_idx]]
            regression_across_repeats, simulated_state_data, prep_repeat_dict = {}, {}, {}
            repeats_list = []
            for repeat in range(repeats[0], repeats[1]):
                repeats_list.append(repeat)
                if repeat == 9 and sub == 'sub-52':
                    continue
                # then figure out the precise timings.
                if math.isnan(timings_task[repeat][-1]) == True:
                    # test if there are more nans than only the last, and if 
                    # this anyways the last repeat, skip this one
                    if math.isnan(timings_task[repeat][-2]) == True and repeat+1 == len(timings_task):
                        continue
                    if repeat+1 == len(timings_task):
                        # note this is a dumb fix. 
                        # cleaner if ignoring this trial as patient probably didn't finish
                        timings_task[repeat][-1] = timings_task[repeat][-2]+2
                timings_repeat = [int(elem) for elem in timings_task[repeat]]
                
                # I also need the next repeat
                if repeat+1 < len(timings_task):
                    if math.isnan(timings_task[repeat+1][-1]) == True:
                        # if this is the penultimum repeat and there are more nans, skip this one
                        if math.isnan(timings_task[repeat+1][-2]) == True and repeat+2 == len(timings_task):
                            continue
                        # note this is a dumb fix. 
                        # cleaner if ignoring this trial as patient probably didn't finish
                        timings_task[repeat+1][-1] = timings_task[repeat+1][-2]+2
                    
                    timings_next_repeat = [int(elem) for elem in timings_task[repeat+1]]
                else:
                    timings_next_repeat = None
                
                # then create the regressors 
                prep_repeat_dict[f"rep_{repeat}"] = mc.analyse.helpers_human_cells.prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_repeat, task_config, neurons_for_task)
                regression_across_repeats[f"rep_{repeat}"] = mc.simulation.predictions.create_x_regressors_per_state(prep_repeat_dict[f"rep_{repeat}"], no_regs_per_state=no_bins_per_state, only_for_rewards=only_reward_times)
                if sim_fake_data == True:
                    simulated_state_data[f"rep_{repeat}"] = mc.simulation.predictions.simulate_fake_data(prep_repeat_dict[f"rep_{repeat}"], model_to_simulate = 'state', repeat_idx = repeat)

            # lastly, run the same regression on the cells
            # the regressor sometimes is a few timebins shorter. make equal in size.
            # before doing so, normalise the neurons.
            # subtract mean and divide by standard deviation.
            # first, concatenate all repeats.
            for i, repeat in enumerate(sorted(regression_across_repeats)):
                if i == 0:
                    prep_repeat_dict["concat_neurons"] = prep_repeat_dict[repeat]['neuron_rep'].copy()
                    regression_across_repeats["concat"] = regression_across_repeats[repeat].copy()
                    if sim_fake_data == True:
                        simulated_state_data["concat"]  = simulated_state_data[repeat].copy()
                else:
                    prep_repeat_dict["concat_neurons"] = np.concatenate((prep_repeat_dict["concat_neurons"], prep_repeat_dict[repeat]['neuron_rep']), axis = 1)
                    regression_across_repeats["concat"] = np.concatenate((regression_across_repeats["concat"], regression_across_repeats[repeat]), axis = 1)
                    if sim_fake_data == True:
                        simulated_state_data["concat"]  = np.concatenate((simulated_state_data["concat"], simulated_state_data[repeat]), axis = 1)

            prep_repeat_dict["concat_neurons_normal"] = mc.simulation.predictions.normalise_neurons(prep_repeat_dict["concat_neurons"])
            data_prep[sub][f"grid_{grid_idx}"]['neurons'] = mc.simulation.predictions.transform_data_to_betas(prep_repeat_dict["concat_neurons_normal"], regression_across_repeats["concat"])     
            if sim_fake_data == True:
                data_prep[sub][f"grid_{grid_idx}"]['fake_neurons'] = mc.simulation.predictions.transform_data_to_betas(simulated_state_data["concat"], regression_across_repeats["concat"])  
                
    return data_prep           
 

def get_rid_of_low_firing_cells(data_one_sub, hz_exclusion_threshold = 0.1, sd_exclusion_threshold = 1.5):
     
    neuron_dict = {}
    for task in data_one_sub['neurons']:
        for i, neuron in enumerate(task):
            curr_cell = f"{data_one_sub['cell_labels'][i]}_{i}"
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
    
    
    for m in data_one_sub:
        if m in stuff_to_copy:
            remaining_neurons[m] = data_one_sub[m]
        elif m in to_clean:
            # # import pdb; pdb.set_trace() 
            # if m not in remaining_neurons[sub]:
            #     remaining_neurons[sub][m] = []
            if m == 'neurons':
                if m not in remaining_neurons:
                    remaining_neurons[m] = []
                for el in data_one_sub[m]:
                    cleaned = np.delete(el, neurons_to_exclude, axis=0)
                    remaining_neurons[m].append(cleaned)
            elif m == 'cell_labels':
                remaining_neurons[m] = [x for i, x in enumerate(data_one_sub[m]) if i not in neurons_to_exclude]

    
    print(f"exlcuding neurons {neurons_to_exclude_str} because average spike rate lower than {hz_exclusion_threshold} or variability bigger than std/mean = {sd_exclusion_threshold}")
    # import pdb; pdb.set_trace() 
    return remaining_neurons

   
def prep_and_model_human_cells(data_dict, repeats, model_simple = True):
    
    task_labels = {'task_A': ([1., 9., 5., 8.]), 'task_B': ([2., 5., 7., 6.]), 'task_C': ([3., 7., 9., 5.]), 'task_D': ([4., 8., 1., 3.]), 'task_E': ([6., 4., 2., 9.]), 'task_F': ([7., 3., 4., 2.]), 'task_G': ([8., 2., 6., 7.]), 'task_H': ([9., 1., 3., 4.])}
    
    # import pdb; pdb.set_trace()
    # first, do some modifications (integers, start from 0, make timings to bins)
    # then, call the simulate functions that model the different neurons
    # do this for every repeat < for ever grid < for every subject.
    
    modelled_data = {}
    for sub in data_dict:
        print(f"now starting to process data from subject {sub}")
        
        # clean up the data per subject
        # only consider those neurons that are firing enough
        modelled_data[sub] = mc.analyse.helpers_human_cells.get_rid_of_low_firing_cells(data_dict[sub])
        
        for grid_idx, grid_config in enumerate(data_dict[sub]['reward_configs']):
            # first identify which task you are currently modelling.
            for task_config_pre_defined in task_labels:
                if task_labels[task_config_pre_defined] == list(grid_config):
                    grid_label = f"{task_config_pre_defined}_{grid_idx}_sub_{sub}"
                    
            modelled_data[sub][grid_label] = {}
            # wrong timings for some reason in subject 51 in grid 17, 18 and probably more
            if (sub == 'sub-43' and grid_idx == 3) or (sub == 'sub-15' and grid_idx == 23) or (sub == 'sub-02' and grid_idx == 18) or (sub == 'sub-51' and grid_idx > 17):
                # probably aborted the experiment here
                continue
            # timings task is start A - find A - find B - find C - find D
            timings_task = data_dict[sub]['timings'][grid_idx]
            neurons_for_task = modelled_data[sub]['neurons'][grid_idx].copy()

            # fields need to be between 0 and 8, and keep them as integers
            locations_curr_grid = [int((field_no-1)) for field_no in data_dict[sub]['locations'][grid_idx]]
            task_config = [int((field_no-1)) for field_no in data_dict[sub]['reward_configs'][grid_idx]]
                
            regression_across_repeats, models_per_repeat, prep_repeat_dict  = {}, {}, {}
            # for repeat in range(1, len(timings_task)):  
            for repeat in range(repeats[0], repeats[1]):
                # first check if this repeat wasn't completed
                # for current repeat
                if math.isnan(timings_task[repeat][-1]) == True:
                    # test if there are more nans than only the last, and if 
                    # this anyways the last repeat, skip this one
                    if math.isnan(timings_task[repeat][-2]) == True and repeat+1 == len(timings_task):
                        continue
                    if repeat+1 == len(timings_task):
                        timings_task[repeat][-1] = timings_task[repeat][-2]+2
                timings_repeat = [int(elem) for elem in timings_task[repeat]]
                
                # and for next repeat
                if repeat+1 < len(timings_task):
                    if math.isnan(timings_task[repeat+1][-1]) == True:
                        # if this is the penultimum repeat and there are more nans, skip this one
                        if math.isnan(timings_task[repeat+1][-2]) == True and repeat+2 == len(timings_task):
                            continue
                        timings_task[repeat+1][-1] = timings_task[repeat+1][-2]+2
                    timings_next_repeat = [int(elem) for elem in timings_task[repeat+1]]
                else:
                    timings_next_repeat = None
                # this includes the trajectory, and the according neural recording
                prep_repeat_dict[f"rep_{repeat}"] = mc.analyse.helpers_human_cells.prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_repeat, task_config, neurons_for_task)
                
                # then model per repeat/grid/subject
                if model_simple == True:
                    models_per_repeat[f"rep_{repeat}"] = mc.simulation.predictions.set_simple_models_cells(prep_repeat_dict[f"rep_{repeat}"])
                else:
                    models_per_repeat[f"rep_{repeat}"] = mc.simulation.predictions.set_continous_models_ephys(prep_repeat_dict[f"rep_{repeat}"], split_clock = True)
                # then prepare regressors per repeat; will be same length as the walked path/trajectory/modelled data.
                regression_across_repeats[f"rep_{repeat}"] = mc.simulation.predictions.create_x_regressors_per_state(prep_repeat_dict[f"rep_{repeat}"], only_for_rewards=True)
            
            model_list = []
            for key in models_per_repeat[f"rep_{repeats[0]}"]:
                model_list.append(key)
            
            # then, per grid, average the runs to one. 
            # NOTE: later, HERE IS WHERE YOU COULD DO EARLY VS LATE !!
            # first, concatenate all repeats.
            # import pdb; pdb.set_trace()
            for i, rep in enumerate(sorted(models_per_repeat)):
                if i == 0:
                    prep_repeat_dict["concat_neurons"] = prep_repeat_dict[rep]['neuron_rep'].copy()
                    regression_across_repeats["concat"] = regression_across_repeats[rep].copy()
                    for model in model_list:
                        models_per_repeat[f"{model}_concat"] = models_per_repeat[rep][model].copy()
                else:
                    regression_across_repeats["concat"] = np.concatenate((regression_across_repeats["concat"], regression_across_repeats[rep]), axis = 1)
                    prep_repeat_dict["concat_neurons"] = np.concatenate((prep_repeat_dict["concat_neurons"], prep_repeat_dict[rep]['neuron_rep']), axis = 1)
                    for model in model_list:
                        models_per_repeat[f"{model}_concat"] = np.concatenate((models_per_repeat[f"{model}_concat"], models_per_repeat[rep][model]), axis = 1)
            # then average across all trials by running the regression and save each grid and model per subject.
            for model in model_list:
                modelled_data[sub][grid_label][model] = mc.simulation.predictions.transform_data_to_betas(models_per_repeat[f"{model}_concat"], regression_across_repeats["concat"])
            # lastly, run the same regression on the cells
            prep_repeat_dict["concat_neurons_normal"] = mc.simulation.predictions.normalise_neurons(prep_repeat_dict["concat_neurons"])
            modelled_data[sub][grid_label]['neurons'] = mc.simulation.predictions.transform_data_to_betas(prep_repeat_dict["concat_neurons_normal"], regression_across_repeats["concat"])     
            # import pdb; pdb.set_trace()
            # # the regressor sometimes is a few timebins shorter. make equal in size.
           
            # # before doing so, normalise the neurons.
            # # subtract mean and divide by standard deviation.
            # data_dict[sub]['neurons'][grid_idx] = mc.simulation.predictions.normalise_neurons(data_dict[sub]['neurons'][grid_idx])
            # # import pdb; pdb.set_trace()
            # # actually, 0 needs to be the first repeat index. How do I find that one??
            # start_modelling_from = int(timings_task[repeats[0]][0])
            # import pdb; pdb.set_trace()
            # # THIS ISNT RIGHT!!!!
            
            # modelled_data[sub][grid_label]['neurons'] = mc.simulation.predictions.transform_data_to_betas(prep_repeat_dict["concat_neurons"], regression_across_repeats["concat"])     
            # modelled_data[sub][grid_label]['neurons'] = mc.simulation.predictions.transform_data_to_betas(data_dict[sub]['neurons'][grid_idx][:, start_modelling_from:start_modelling_from + regression_across_repeats["concat"].shape[1]], regression_across_repeats["concat"])     
            # import pdb; pdb.set_trace()
    #fake_data = mc.analyse.helpers_human_cells(data_dict, modelled_data)
    
    print(f"the following models have been simulated and averaged for all repeats and all grids: {model_list}")
    return modelled_data
    

# THIS IS A LITTLE HELPER.
#def create_fake_data(og_data, model):
    
    # then store the 





def prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_repeat, reward_locs, neurons):
    # some pre-processing to create my models.
    # cut the neurons file such to include only the current repeat
    prep_dict = {}
    prep_dict['reward_locs'] = reward_locs
    prep_dict['timings_repeat'] = timings_repeat
    # because in the following, I will only regard the single repeats for themselves,
    # cut the timings such that it starts with the first bin of the repeat.
    prep_dict['timings_repeat'] = [elem - timings_repeat[0] for elem in timings_repeat]

    # locations for current repeat
    
    # 22/04/2025 update: don't do this, as it may use parts of the data twice and fucks up timings
    # 26.02.25 
    # curr trajectory should actually end only once you leave reward position. 
    # so instead of using timings_repeat[-1], find the relevant index.
    # this simply cuts longer chunks so that I can later select those longer chunks
    # in my regression and simulation.
    # start_at_rew, end_at_rew = mc.simulation.predictions.find_start_end_indices(locations_curr_grid, timings_repeat[-1])
    # prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:end_at_rew]
    # prep_dict['neuron_rep'] = neurons[:, timings_repeat[0]:end_at_rew] 

    # import pdb; pdb.set_trace()
    if timings_next_repeat:
        # note that timings_next_repeat[0] == timings_repeat[-1]
        # so collecting timings_next_repeat is not technically necessary
        prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:timings_next_repeat[0]]
        prep_dict['neuron_rep'] = neurons[:, timings_repeat[0]:timings_next_repeat[0]] 
        
        # this is if I want the actual time where they leave the rew location.
        # subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
        #                 locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
        #                 locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
        #                 locations_curr_grid[timings_repeat[3]:end_at_rew]]
        
        subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
                        locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
                        locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
                        locations_curr_grid[timings_repeat[3]:timings_next_repeat[0]]]
        
    elif not timings_next_repeat:
        # this means it's the last repeat and you have to include the last bin.
        # note: if you want to include the entire file after finding the reward,
        # do
        # prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:]
        # prep_dict['neuron_rep'] = neurons[:, timings_repeat[0]:] 
        # subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
        #                 locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
        #                 locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
        #                 locations_curr_grid[timings_repeat[3]:]]
        
        
        prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:timings_repeat[-1]+1]
        prep_dict['neuron_rep'] = neurons[:, timings_repeat[0]:timings_repeat[-1]+1] 
        subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
                        locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
                        locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
                        locations_curr_grid[timings_repeat[3]:timings_repeat[4]+1]]
    
    
    # actually, the fourth subpath has to last until the next one starts.
    # to find out the step number per subpath
    prep_dict['step_number'] = [0,0,0,0] 
    for path_no, subpath in enumerate(subpath_locs):
        for i, field in enumerate(subpath):
            if i == 0:
                count = 0
            elif field != subpath[i-1]:
                count+=1
        prep_dict['step_number'][path_no] = count     
    # mark where steps are made (vs. staying on same location)
    for field_no, field in enumerate(prep_dict['trajectory']):
        if field_no == 0:
            prep_dict['index_make_step'] = [0]
        elif field != prep_dict['trajectory'][field_no-1]:
            prep_dict['index_make_step'].append(field_no)
    return prep_dict


def prep_repeat_neurons(timings_repeat, t_first_bin, neurons):
    prep_dict = {}
    curr_neurons = neurons[:, t_first_bin:timings_repeat[-1]]
    # z-score neurons
    prep_dict['curr_neurons'] = scipy.stats.zscore(curr_neurons, axis=1)
    return prep_dict
