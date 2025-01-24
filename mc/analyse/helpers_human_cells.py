#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:41:06 2025

@author: Svenja Küchenhoff
helpers functions to analyse human cells


"""

import numpy as np
import os
import re
import scipy
import math
import mc
from matplotlib import pyplot as plt
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
    
    
def load_cell_data(source_folder, subject_list):
    # load all files I prepared with Matlab into a subject dictionary
    data_dir = {}
    for sub in subject_list:
        data_dir[f"sub-{sub}"] = {}
        data_dir[f"sub-{sub}"]["reward_configs"] = np.genfromtxt(f"{source_folder}/s{sub}/LFP/all_configs_sub{sub}.csv", delimiter=',')
        with open(f"{source_folder}/s{sub}/LFP/all_cells_region_labels_sub{sub}.txt", 'r') as file:
            data_dir[f"sub-{sub}"]["cell_labels"] = [line.strip() for line in file]
        timinges_pattern = r'timings_rewards_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        location_pattern = r'locations_per_50ms_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        cells_pattern = r'all_cells_firing_rate_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        data_dir[f"sub-{sub}"]["timings"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/LFP", timinges_pattern)
        data_dir[f"sub-{sub}"]["locations"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/LFP", location_pattern)
        data_dir[f"sub-{sub}"]["neurons"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/LFP", cells_pattern)
    return data_dir


def run_RSA(data, per_ROI = True, plotting = True):
    # normal version is going to be collapse all subjects, but split cells by ROIs
    # first collapse subjects.
    # also note that there are a couple of reward configs that are the same!!
    # similarity here should be very high.
    # eventually nan them out/average them?
    
    # then run one RSA per model on each ROI
        # important: sort the configurations!!
        # eventually go back and check which models I actually want.
        # I am assuming one thing we are interested in is actually the split clocks for only rewards?
    # the run a combined RSA on each ROI
    # then return results
    
    # next: reorder.
    
    # take every cell that belongs to a ROI.
    # second step: compute a within-task RSA
    # third step: average all cells and models that belong to a unique grid
    
    # and compute a between-task RSA
    
    # maybe start by labelling the grids.
    prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data)
    # instead of pooling by subject, pool by ROI.
    neurons = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data)

    
    # now next step is concatenate all grids in the correct order, for neurons 
    # per ROI and for all models. Maybe consider average the models across subjects first??
    # in theory, the data models should be the same across all subjects...

    simulated_data_concat, order_of_tasks =  mc.analyse.helpers_human_cells.models_concat_and_avg_across_subj(prepared_data)
    
    import pdb; pdb.set_trace()

    
    
    #
    #
    # # JUST COPIED
    # #
    # #
    # RDM_dict = {}
    # #import pdb; pdb.set_trace()
    # # plot the averaged simulated and cleaned data
    # for model in ave_models_between:
    #     mc.simulation.predictions.plot_without_legends(ave_models_between[model], titlestring= f"{model} model, averaged across runs for single mouse", intervalline= 4*no_bins_per_state, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
    #     RDM_dict[model] = mc.simulation.RDMs.within_task_RDM(ave_models_between[model], plotting = True, titlestring = f"Between tasks {model} RSM, 12*12, averaged over runs", intervalline= 4*no_bins_per_state)
    # mc.simulation.RDMs.plot_RDMs(RDM_dict, len(task_configs))       
    # # separately per phase
    # if split_by_phase:
    #     RDM_dict_phases = {}
    #     for model in ave_models_between:
    #         RDM_dict_phases[model] = {}
    #         # ceither all models or fine-tune this to only those I want by defining a string of models I loop through
    #         for phase in phase_string:
    #             mc.simulation.predictions.plot_without_legends(ave_models_between[model][:, phase_masks[phase]], titlestring=f"{phase} {model} across tasks", intervalline= 4*no_bins_per_state/3)
    #             RDM_dict_phases[model][phase] = mc.simulation.RDMs.within_task_RDM(ave_models_between[model][:, phase_masks[phase]], plotting=True, titlestring=f"RSM {phase} {model}, averaged over runs", intervalline= 4*no_bins_per_state/3)

    # # run regressions separetly for each phase
    # # import pdb; pdb.set_trace()
    # # try the new regression.
    
    
    # # midn_model, phas_model, loc_model, stat_model, phas_stat, clo_model, curr_neurons
    # regressors_to_include = ['clo_model', 'phas_model', 'loc_model', 'stat_model'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
    # regressors = {}
    # for i, model in enumerate(sorted(regressors_to_include)):
    #     print(f"the order of the regressors here is at {i} comes {model}")
    #     regressors[model] = RDM_dict[model].copy()

    # results_normal = mc.simulation.RDMs.GLM_RDMs(RDM_dict['curr_neurons'], regressors, mask_within, no_tasks = len(task_configs), plotting= False)
    
    
    
    
    #
    #
    #
    
    results= {}
    return results

    

def models_concat_and_avg_across_subj(data):
    # now next step is concatenate all grids in the correct order
    # first loop: collect the same model for the same task for all subjects with np.stack
    # second loop: average the same model same task across all subjects
    # and in the end average across task repeats
    
    temp_stacked = {}
    for sub in data:
        for grid in data[sub]:
            if grid.startswith('average'):
                temp_stacked[grid] = {}
    all_models_set = set()
    # reduce data for models and grids by subject dimension, make a list instead.
    for sub in data:
        for grid in data[sub]:
            if grid.startswith('average'):
                for model in data[sub][grid]:
                    if model not in ['neurons']:
                        all_models_set.add(model)
                        if model not in temp_stacked[grid]:
                            temp_stacked[grid][model] = [data[sub][grid][model]]
                        else:
                            temp_stacked[grid][model].append(data[sub][grid][model])
    # stack all lists and average across subjects
    for task in temp_stacked:
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


def pool_by_ROI_and_grid(data):
    # first create the ROI sets
    ROIs = {}
    ROI_labels = ['hippocampal', "PCC", "ACC", "OFC", "entorhinal", "amygdala", "mixed"]
    for ROI in ROI_labels:
        ROIs[ROI] = set()
    for sub in data:
        for region in data[sub]['cell_labels']:
            if 'HC' in region:
                ROIs['hippocampal'].add(region)
            elif 'ACC' in region:
                ROIs['ACC'].add(region)
            elif 'PCC' in region:
                ROIs['PCC'].add(region)
            elif 'EC' in region:
                ROIs['entorhinal'].add(region)
            elif 'AMYG' in region:
                ROIs['amygdala'].add(region)
            elif 'OFC' in region:
                ROIs['OFC'].add(region)
            else:
                ROIs['mixed'].add(region)
    ROIs['entorhinal'] = {'REC', 'LEC'}
    # then create a neuron dictionary, split by ROIs, where I save how each cell 
    # of that ROI fires for each grid.
    grid_labels = ['task_A', 'task_B', 'task_C', 'task_D', 'task_E', 'task_F', 'task_G', 'task_H']
    neurons = {}
    for ROI in ROIs:
        neurons[ROI] = {}
    # import pdb; pdb.set_trace()  
    # outer: loop through old data dict
    for sub in data:
        for grid in data[sub]:
            if grid.startswith('average'):
                for i_c, cell in enumerate(data[sub]['cell_labels']):
                    # inner: loop through new dict
                    for ROI in ROIs:
                        if cell in ROIs[ROI]:
                            if neurons[ROI].get(f"{grid}_{sub}_{i_c}_{cell}") is None:
                                if sub == 'sub-43':
                                    neurons[ROI][f"{grid}_{sub}_{i_c}_{cell}"] = []
                                else:
                                    neurons[ROI][f"{grid}_{sub}_{i_c}_{cell}"] = data[sub][grid]['neurons'][i_c]
    
    # import pdb; pdb.set_trace()
    return neurons
                                
        
        
    
  
def label_unique_grids(data_dict):
    task_labels = {'task_A': ([1., 9., 5., 8.]), 'task_B': ([2., 5., 7., 6.]), 'task_C': ([3., 7., 9., 5.]), 'task_D': ([4., 8., 1., 3.]), 'task_E': ([6., 4., 2., 9.]), 'task_F': ([7., 3., 4., 2.]), 'task_G': ([8., 2., 6., 7.]), 'task_H': ([9., 1., 3., 4.])}
    for sub in data_dict:
        keys_to_modify = []
        for unique_task in task_labels:
            for grid_idx, grid in enumerate(data_dict[sub]['reward_configs']):
                if list(grid) == list(task_labels[unique_task]):
                    new_key = f"{unique_task}_grid_{grid_idx}"
                    old_key = f"grid_{grid_idx}"
                    keys_to_modify.append((old_key, new_key))
        for old_grid_key, new_grid_key in keys_to_modify:
            # print(old_grid_key, new_grid_key)
            data_dict[sub][new_grid_key] = data_dict[sub].pop(old_grid_key)
        # import pdb; pdb.set_trace()
        # last, but not least, create grid averages.
        for unique_task in task_labels:
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
                    # make sure this works!!
                    print(f"{model} stacked has shape {temp_stacked[model].shape}")
                    # Compute the mean across the stacked dimension
                # then average
                for model in temp_stacked:
                    data_dict[sub][grid][model] = np.mean(temp_stacked[model], axis=0)
                #import pdb; pdb.set_trace()
    return data_dict
            
            
    
def prep_and_model_human_cells(data_dict):
    # first, do some modifications (integers, start from 0, make timings to bins)
    # then, call the simulate functions that model the different neurons
    # do this for every repeat < for ever grid < for every subject.
    modelled_data = {}
    for sub in data_dict:
        print(f"now starting to process data from subejct {sub}")
        modelled_data[sub] = {}
        modelled_data[sub]['cell_labels'] = data_dict[sub]['cell_labels'].copy()
        modelled_data[sub]['reward_configs'] = data_dict[sub]['reward_configs'].copy()
        #import pdb; pdb.set_trace()
        for grid_idx, grid_config in enumerate(data_dict[sub]['reward_configs']):
            if sub == 'sub-43' and grid_idx == 3:
                # probably aborted the experiment here
                modelled_data[sub][f"grid_{grid_idx}"] = {}
                continue
            modelled_data[sub][f"grid_{grid_idx}"] = {}
            # timings task is start A - find A - find B - find C - find D
            timings_task = data_dict[sub]['timings'][grid_idx].copy()
            # fields need to be between 0 and 8, and keep them as integers
            locations_curr_grid = [int((field_no-1)) for field_no in data_dict[sub]['locations'][grid_idx]]
            task_config = [int((field_no-1)) for field_no in data_dict[sub]['reward_configs'][grid_idx]]
            regression_across_repeats = {}
            models_per_repeat = {}
            for repeat in range(0, len(timings_task)):
                # then per repeat, simulate my neurons
                # if len(timings_task[repeat])
                if math.isnan(timings_task[repeat][-1]) == True:
                    if repeat+1 == len(timings_task):
                        # note this is a dumb fix. 
                        # cleaner if ignoring this trial as patient probably didn't finish
                        timings_task[repeat][-1] = timings_task[repeat][-2]+2
                timings_repeat = [int(elem) for elem in timings_task[repeat]]
                
                # I also need the next repeat
                if repeat+1 < len(timings_task):
                    if math.isnan(timings_task[repeat+1][-1]) == True:
                        # note this is a dumb fix. 
                        # cleaner if ignoring this trial as patient probably didn't finish
                        timings_task[repeat+1][-1] = timings_task[repeat+1][-2]+2
                    timings_next_repeat = [int(elem) for elem in timings_task[repeat+1]]
                else:
                    timings_next_repeat = None
                prep_repeat_dict = mc.analyse.helpers_human_cells.prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_repeat)
                # then model per repeat/grid/subject
                models_per_repeat[f"rep_{repeat}"] = mc.simulation.predictions.set_continous_models_ephys(prep_repeat_dict, split_clock = True)

                # then set everything back together
                #modelled_data[sub][grid][]
                regression_across_repeats[f"rep_{repeat}"] = mc.simulation.predictions.create_x_regressors_per_state(prep_repeat_dict, only_for_rewards=True)
            
            model_list = []
            for key in models_per_repeat[f"rep_{repeat}"]:
                model_list.append(key)
            
            # then, per grid, average the runs to one. 
            # NOTE: later, HERE IS WHERE YOU COULD DO EARLY VS LATE !!
            # first, concatenate all repeats.
            for i, repeat in enumerate(sorted(models_per_repeat)):
                if i == 0:
                    regression_across_repeats["concat"] = regression_across_repeats[repeat].copy()
                else:
                    regression_across_repeats["concat"] = np.concatenate((regression_across_repeats["concat"], regression_across_repeats[repeat]), axis = 1)
                for model in model_list:
                    if i == 0:
                        models_per_repeat[f"{model}_concat"] = models_per_repeat[repeat][model].copy()
                    else:
                        models_per_repeat[f"{model}_concat"] = np.concatenate((models_per_repeat[f"{model}_concat"], models_per_repeat[repeat][model]), axis = 1)
            
            for model in model_list:
                # then average across all trials by running the regression and save each grid and model per subject.
                # import pdb; pdb.set_trace()
                # something is wrong here for state.... not how I want it to look..
                # maybe because of the timings actually????
                # because I am kind of saying a new state starts once they reached the reward
                # maybe i need to recode this
                modelled_data[sub][f"grid_{grid_idx}"][model] = mc.simulation.predictions.transform_data_to_betas(models_per_repeat[f"{model}_concat"], regression_across_repeats["concat"])
            # lastly, run the same regression on the cells
            # the regressor sometimes is a few timebins shorter. make equal in size.
            modelled_data[sub][f"grid_{grid_idx}"]['neurons'] = mc.simulation.predictions.transform_data_to_betas(data_dict[sub]['neurons'][grid_idx][:, 0:regression_across_repeats["concat"].shape[1]], regression_across_repeats["concat"])     
    print(f"the following models have been simulated and averaged for all repeats and all grids: {model_list}")
    return modelled_data
    


def prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_repeat):
    # import pdb; pdb.set_trace()
    # some pre-processing to create my models.
    # cut the neurons file such to include only the current repeat
    prep_dict = {}
    # I think this needs to be adjusted to: 
    # adjust the timings such that they fit whatever you cut from the timings!    
    prep_dict['timings_repeat'] = [elem - timings_repeat[0] for elem in timings_repeat]
    # prep_dict['timings_repeat'] = timings_repeat, bins for ABCD
    # locations for current repeat
    prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:timings_repeat[-1]]
    # split trajectory into subpaths
    if timings_next_repeat:
        subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
                        locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
                        locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
                        locations_curr_grid[timings_repeat[3]:timings_next_repeat[0]]]
    elif not timings_next_repeat:
        # this means it's the last repeat, and the last timing is actually correct.
        subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
                        locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
                        locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
                        locations_curr_grid[timings_repeat[3]:timings_repeat[4]]]
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
