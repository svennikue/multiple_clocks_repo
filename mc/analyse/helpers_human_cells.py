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
    # for now, do per subject.
    # later, it might be interesting to do per grid or per cell.
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


def prep_and_model_human_cells(data_dict):
    modelled_data = {}
    for sub in data_dict:
        modelled_data[sub] = {}
        for grid_idx, grid_config in enumerate(data_dict[sub]['reward_configs']):
            modelled_data[sub][grid_idx] = {}
            # first convert trial times from ms to bin number to match neuron and location arrays 
            # (1 bin = 50ms)
            timings_task = data_dict[sub]['timings'][grid_idx].copy()
            for r, repeat in enumerate(timings_task):
                for s, timing in enumerate(repeat):
                    timings_task[r,s] = math.floor(timing/0.05)
                    
            # fields need to be between 0 and 8, and keep them as integers
            locations_curr_grid = [int((field_no-1)) for field_no in data_dict[sub]['locations'][grid_idx]]
            task_config = [int((field_no-1)) for field_no in data_dict[sub]['reward_configs'][grid_idx]]
            for repeat in range(0, len(timings_task)):
                # then per repeat, simulate my neurons
                timings_repeat = [int(elem) for elem in timings_task[repeat]]
                # first prep per repeat/grid/subject
                if repeat == 0:
                    first_bin = 0
                else:
                    timings_prev_repeat = [int(elem) for elem in timings_task[repeat-1]]
                    first_bin = timings_prev_repeat[-1] 
                prep_repeat_dict = mc.analyse.helpers_human_cells.prep_repeat(timings_repeat, first_bin, locations_curr_grid, data_dict[sub]['neurons'][grid_idx])
                
                # CONTINUE HERE!!!
                # SEE HOW MUCH I CAN TAKE FROM THE EPHYS MODELS!
                import pdb; pdb.set_trace()
                # then model per repeat/grid/subject
                modelled_data[sub][grid_idx][repeat] = mc.simulation.predictions.set_continous_models_ephys(prep_repeat_dict,  plot = False)
                # then set everything back together
                #modelled_data[sub][grid][]
    return modelled_data
    


def prep_repeat(timings_repeat, t_first_bin, locations_curr_grid, neurons):
    # some pre-processing to create my models.
    # include curr_neurons
    prep_dict = {}
    prep_dict['timings_repeat'] = timings_repeat
    prep_dict['trajectory'] = locations_curr_grid[t_first_bin:timings_repeat[-1]]
    curr_neurons = neurons[:, t_first_bin:timings_repeat[-1]]
    subpath_locs = [locations_curr_grid[t_first_bin:timings_repeat[1]], locations_curr_grid[timings_repeat[0]:timings_repeat[1]], locations_curr_grid[timings_repeat[1]:timings_repeat[2]], locations_curr_grid[timings_repeat[2]:timings_repeat[3]]]

    # z-score neurons
    prep_dict['curr_neurons'] = scipy.stats.zscore(curr_neurons, axis=1)
    
    # to count subpaths
    # subpath_file = [locations_task[row[0]:row[1]], locations_task[row[1]:row[2]], locations_task[row[2]:row[3]], locations_task[row[3]:row[4]]]
    # timings_curr_run = [(elem - row[0]) for elem in row]

    # to find out the step number per subpath
    prep_dict['step_number'] = [0,0,0,0] 
    for path_no, subpath in enumerate(subpath_locs):
        for i, field in enumerate(subpath):
            if i == 0:
                count = 0
            elif field != subpath[i-1]:
                count+=1
        prep_dict['step_number'][path_no] = count
       
    # mark where steps are made
    for field_no, field in enumerate(prep_dict['trajectory']):
        if field_no == 0:
            prep_dict['index_make_step'] = [0]
        elif field != prep_dict['trajectory'][field_no-1]:
            prep_dict['index_make_step'].append(field_no)

    return prep_dict