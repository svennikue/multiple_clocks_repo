#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon April 29 2024

This is a debug script which plots for a specific subject and task across task halves:
    a) the paths that the participants took
    b) the simulations for the models of these paths
    c) the RDM


@author: Svenja Küchenhoff, 2024
"""

import pandas as pd
import os
import numpy as np
import mc
import matplotlib.pyplot as plt
import pickle
import sys

# import pdb; pdb.set_trace()

regression_version = '03' 
RDM_version = '03-1' 
task_of_interest = 'B'
fmriplotting = False
subj_no = '07'
temporal_resolution = 10


sub = f"sub-{subj_no}"
task_halves = ['1', '2']
runs = ['0', '1', '2', '3', '4']
models_I_want = mc.analyse.analyse_MRI_behav.models_I_want(RDM_version)

keep_models = ['location', 'clocks_only-rew', 'midnight_only-rew']

# what I will want to do is create a dicitonary in which I can always have the respective 2 things:
    # per config, for half 1 and 2 the path
    # per config, for half 1 and 2 the model
    # per config, for half 1 and 2 the RDM

# plot: what is the config? Where are the rewards?
# plot: where did the subject walk per run and task half       

# initialize some dictionaries
models_between_task_halves = {}
raw_models_betw_task_halves = {}
sorted_models_split = {}
configs_dict = {}
reg_list = []
trajectories = {}
for task_half in task_halves:
    data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
    RDM_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
    if os.path.isdir(data_dir_beh):
        print("Running on laptop.")
    else:
        data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
        RDM_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
        print(f"Running on Cluster, setting {data_dir_beh} as data directory")
        
    file = data_dir_beh + f"{sub}_fmri_pt{task_half}.csv"

    configs, rew_list, rew_index, walked_path, steps_subpath_alltasks_empty, subpath_after_steps, timings, regressors = mc.analyse.analyse_MRI_behav.extract_behaviour(file)

    # so now, account for the temporal resolution that you want:
    for reg in regressors:
        regressors[reg] = np.repeat(regressors[reg], repeats = temporal_resolution)

    
    # only do the one you are actually interested in.
    configs = np.array([config for config in configs if config.startswith(task_of_interest)])
    # overview of the reward fields per task.
    steps_subpath_alltasks = mc.analyse.analyse_MRI_behav.subpath_files(configs, subpath_after_steps, rew_list, rew_index, steps_subpath_alltasks_empty)
    
    # prep result dictionaries.
    #all_models_dict = {f"{model}": {key: "" for key in configs} for model in models_I_want}
    all_models_dict = {f"{model}": {key: "" for key in configs} for model in keep_models}
    trajectories_dict = {f"{config}": {run: [] for run in runs} for config in configs}
    # finally, create simulations and time-bin per run.
    for config in configs:
        print(f"the config is {rew_list[config]} for {config}")
        # select complete trajectory of current task.
        trajectory = walked_path[config]
        trajectory = [[int(value) for value in sub_list] for sub_list in trajectory]
        # select the timings of this task
        timings_curr_run = timings[config]
  
        # select file that shows step no per subpath
        step_number = [[int(value) for value in sub_list] for sub_list in steps_subpath_alltasks[config]]
        index_run_no = np.array(range(len(step_number)))
        # only consider some of the repeats for the only later or only early trials!
        if regression_version in ['03-e', '03-4-e']:
            # step_number = step_number[0:2].copy()
            index_run_no = index_run_no[0:2].copy()
            runs = runs[0:2].copy()
        elif regression_version in ['03-l', '03-4-l']:
            # step_number = step_number[2:].copy()
            index_run_no = index_run_no[2:].copy()
            runs = runs[2:].copy()
                
        # make file that shows cumulative steps per subpath
        cumsteps_task = np.cumsum([np.cumsum(task)[-1] for task in step_number])

        # then start looping through each subpath within one task
        repeats_model_dict = {}
        for no_run in index_run_no:
            # first check if the run is not completed. if so, skip the uncomplete part.
            if len(subpath_after_steps[config]) < 20: # 5 runs a 4 subpaths
                stop_after_x_runs = len(subpath_after_steps[config]) // 4 # 4 subpaths
                if no_run >= stop_after_x_runs:
                    continue
         
            if no_run == 0:
                # careful: fields is always one more than the step number
                curr_trajectory = trajectory[0:cumsteps_task[no_run]+1]
                curr_timings = timings_curr_run[0:cumsteps_task[no_run]+1]
                curr_stepnumber = step_number[no_run]
            elif no_run > 0:
                # careful: fields is always one more than the step number
                curr_trajectory = trajectory[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
                curr_timings = timings_curr_run[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
                curr_stepnumber = step_number[no_run]
                curr_cumsumsteps = cumsteps_task[no_run]
            
            for field in curr_trajectory:
                trajectories_dict[config][str(no_run)].append(mc.simulation.predictions.field_to_number(field, 3))
            
            
            # KEY STEP
            # create all models.
            if RDM_version == '01-1': # creating location instruction stuff
                result_model_dict = mc.simulation.predictions.create_instruction_model(rew_list[config], trial_type=config)
            elif RDM_version in ['02', '02-A']: # default, modelling all and splitting clocks.
                result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = False, only_path= False, split_clock = True)
            elif RDM_version in ['03', '03-5', '03-5-A', '03-A']: # modelling only rewards + splitting clocks [new]
                result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path = False, split_clock=True)
            elif RDM_version in ['03-1', '03-2', '03-3']:# modelling only clocks + splitting clocks later in different way.
                result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path= False, split_clock = False)    
            elif RDM_version in ['04', '04-5-A', '04-A']: # modelling only paths + splitting clocks [new]
                result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = False, only_path = True, split_clock=True)
            
            
            # now for all models that are creating or not creating the splits models with my default function, this checking should work.
            if RDM_version not in ['03-1', '03-2', '03-3', '03-5', '03-5-A','04-5', '04-5-A']:
                # test if this function gives the same as the models you want, otherwise break!
                model_list = list(result_model_dict.keys())
                if model_list != models_I_want:
                    print('careful! the model dictionary did not output your defined models!')
                    print(f"These are the models you wanted: {models_I_want}. And these are the ones you got: {model_list}")
                    import pdb; pdb.set_trace()
            
            result_model_dict = {key: value for key, value in result_model_dict.items() if key in keep_models}
            # models need to be concatenated for each run and task
            if no_run == 0 or (regression_version in ['03-l', '03-4-l'] and no_run == 2):
                for model in result_model_dict:
                    repeats_model_dict[model] = result_model_dict[model].copy()
            else:
                for model in result_model_dict:
                    repeats_model_dict[model] = np.concatenate((repeats_model_dict[model], result_model_dict[model]), 1)
        
        
        # NEXT STEP: prepare the regression- select the correct regressors, filter keys starting with 'A1_backw'
        regressors_curr_task = {key: value for key, value in regressors.items() if key.startswith(config)}
        
        if regression_version in ['03-e', '03-4-e']:
            regressors_curr_task = {regressor: regressors_curr_task[regressor][0:len(repeats_model_dict[model][2])] for regressor in regressors_curr_task}

        if regression_version in ['03-l', '03-4-l']:
            regressors_curr_task = {regressor: regressors_curr_task[regressor][(-1*len(repeats_model_dict[model][2])):] for regressor in regressors_curr_task}

        print(f"now looking at regressor for task {config}")
        
        # check that all regressors have the same length in case the task wasn't completed.
        if len(subpath_after_steps[config]) < 20:
            # if I cut the task short, then also cut the regressors short.
            for reg_type, regressor_list in regressors_curr_task.items():
            # Truncate the list if its length is greater than the maximum length
                regressors_curr_task[reg_type] = regressor_list[:(np.shape(repeats_model_dict[list(repeats_model_dict)[0]])[1])]
        
        # Ensure all lists have the same length
        list_lengths = set(len(value) for value in regressors_curr_task.values())
        if len(list_lengths) != 1:
            raise ValueError("All lists must have the same length.")
        
        # if not all regressors shall be included, filter them according to the regression setting
        if regression_version in ['02']:
            if RDM_version == '02-A':
                regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key}
            else:
                regressors_curr_task = {key: value for key, value in regressors_curr_task.items()}
        
        if regression_version in ['03','03-1','03-2', '03-4', '03-99', '03-999', '03-l', '03-e', '03-4-e', '03-4-l']:
            if RDM_version in ['02-A', '03-A', '03-5-A']: # additionally get rid of the A-state.
                regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key and key.endswith('reward')}
            else:
                regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if key.endswith('reward')}
        
        if regression_version in ['04', '04-4']:
            if RDM_version in ['04-5-A', '02-A', '04-A']:
                regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key and key.endswith('path')}    
            else:
                regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if key.endswith('path')}
            
        # sort alphabetically.
        sorted_regnames_curr_task = sorted(regressors_curr_task.keys())
        # Create a list of lists sorted by keys
        sorted_regs = [regressors_curr_task[key] for key in sorted_regnames_curr_task]
        regressors_matrix = np.array(sorted_regs)
        reg_list.append(sorted_regnames_curr_task)
        
        # then do the ORDERED time-binning for each model - across the 5 repeats.
        for model in all_models_dict:
            if RDM_version == '01-1':
                all_models_dict[model][config] = result_model_dict[model]
            # run the regression on all simulated data, except for those as I have a different way of creating them:
            elif model not in ['one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr-and-future-rew-locs']:
                all_models_dict[model][config] = mc.simulation.predictions.transform_data_to_betas(repeats_model_dict[model], regressors_matrix)

        # once the regression took place, the location model is the same as the midnight model.
        # thus, it will also be the same as predicting future rewards, if we rotate it accordingly!
        # temporally not do this
        
        # DON"T FORGET TO UNCOMMENT!
        # if RDM_version in ['03-1', '03-2', '03-3']:
        #     # now do the rotating thing. 
        #     all_models_dict['one_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -1, axis = 1) 
        #     all_models_dict['two_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -2, axis = 1) 
        #     if RDM_version in ['03-1', '03-2']:
        #         all_models_dict['three_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -3, axis = 1) 
            
        #     # try something.
        #     all_models_dict['curr-and-future-rew-locs'][config] = np.concatenate((all_models_dict['one_future_rew_loc'][config], all_models_dict['two_future_rew_loc'][config], all_models_dict['three_future_rew_loc'][config]), 0)
            
    # then, lastly, safe the all_models_dict in the respective task_half.
    raw_models_betw_task_halves[task_half] = repeats_model_dict
    models_between_task_halves[task_half] = all_models_dict
    configs_dict[task_half] = {config: values for config, values in rew_list.items() if config in configs}
    trajectories[task_half] = trajectories_dict


# sort the models into two equivalent halves, just in case this went wrong before.
sorted_keys_dict = mc.analyse.extract_and_clean.order_task_according_to_rewards(configs_dict)
#models_sorted_into_splits = {task_half: {model: {config: "" for config in sorted_keys_dict[task_half]} for model in models_I_want} for task_half in task_halves}

models_sorted_into_splits = {task_half: {model: {config: "" for config in sorted_keys_dict[task_half]} for model in keep_models} for task_half in task_halves}

for half in models_between_task_halves:
    for model in models_between_task_halves[half]:
        for task in models_between_task_halves[half][model]:
            if task in sorted_keys_dict['1']:
                models_sorted_into_splits['1'][model][task] = models_between_task_halves[half][model][task]
            elif task in sorted_keys_dict['2']:
                models_sorted_into_splits['2'][model][task] = models_between_task_halves[half][model][task]                
# then, do the concatenation across the ordered tasks.

for split in models_sorted_into_splits:
    for model in models_sorted_into_splits[split]:
        models_sorted_into_splits[split][model] = np.concatenate([models_sorted_into_splits[split][model][task] for task in sorted_keys_dict[split]], 1)
        
    

# then, in a last step, create the RDMs
# concatenate the conditions from the two task halves (giving you 2*nCond X nVoxels matrix), 
# and calculate the correlations between all rows of this matrix. This gives you a symmetric matrix 
# (of size 2*nCond X 2*nCond), where the (non-symmetric) nCond X nCond bottom left square (or top right, 
# doesn't matter because it's symmetric) (i.e. a quarter of the original matrix) has all the correlations 
# across THs. 


RSM_dict_betw_TH = {}
for model in models_sorted_into_splits[split]:
    RSM_dict_betw_TH[model] = mc.simulation.RDMs.within_task_RDM(np.concatenate((models_sorted_into_splits['1'][model], models_sorted_into_splits['2'][model]),1), plotting = True, titlestring= model)
    # mc.simulation.predictions.plot_without_legends(RSM_dict_betw_TH[model])
    if RDM_version in ['03-5', '03-5-A', '04-5', '04-5-A']:
        if RDM_version in ['03-5','04-5']:
            exclude = 0
        elif RDM_version in ['03-5-A', '04-5-A']:
            exclude = 1
        # this is really just playing around. the proper work will be in fmri_do_RSA!!!
        RSM_dict_betw_TH_mask = mc.simulation.predictions.create_mask_same_tasks(RSM_dict_betw_TH[model], configs_dict, exclude)
        # import pdb; pdb.set_trace()
        RSM_dict_betw_TH['state_masked'] = np.where(RSM_dict_betw_TH_mask == 1, RSM_dict_betw_TH[model], np.nan)
        # import pdb; pdb.set_trace()
        #RSM_dict_betw_TH['state_masked'] = RSM_dict_betw_TH[model]* RSM_dict_betw_TH_mask
        

# then average the lower triangle and the top triangle of this nCond x nCond matrix, 
# by adding it to its transpose, dividing by 2, and taking only the lower or 
# upper triangle of the result.    
corrected_RSM_dict = {}
for model in RSM_dict_betw_TH:
    # import pdb; pdb.set_trace()
    corrected_model = (RSM_dict_betw_TH[model] + np.transpose(RSM_dict_betw_TH[model]))/2
    corrected_RSM_dict[model] = corrected_model[0:int(len(corrected_model)/2), int(len(corrected_model)/2):]
   

# just for me. what happens if I add the ['reward_location', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']?
# addition_model = corrected_RSM_dict['reward_location'] + corrected_RSM_dict['one_future_rew_loc'] + corrected_RSM_dict['two_future_rew_loc'] + corrected_RSM_dict['three_future_rew_loc'] 

if not os.path.exists(RDM_dir):
    os.makedirs(RDM_dir)
mc.simulation.RDMs.plot_RDMs(corrected_RSM_dict, len(configs), RDM_dir, sorted_keys_dict['1'])

# make my own correlation matrix.
# Schema - Partial Schema - Subgoal Progress - Location - State
intercorr_RDM_dict = {}
corr_RDMs = np.empty((len(corrected_RSM_dict),len(corrected_RSM_dict)))
for x, RDM_one in enumerate(corrected_RSM_dict):
    for y, RDM_two in enumerate(corrected_RSM_dict):
        if y == 0:
            tick_string = [RDM_two]
        else:
            tick_string.append(RDM_two)
        corr_RDMs[x,y] = mc.simulation.RDMs.corr_matrices_pearson(corrected_RSM_dict[RDM_one], corrected_RSM_dict[RDM_two])[0][1]               
intercorr_RDM_dict['correlation_try_two'] = corr_RDMs
mc.simulation.RDMs.plot_RDMs(intercorr_RDM_dict, len(corr_RDMs), RDM_dir, string_for_ticks = tick_string)       




# plotting the trajectories.
mc.analyse.analyse_MRI_behav.plot_trajectories(trajectories)
        
# plotting the models.
for model in result_model_dict:
    mc.simulation.predictions.plot_without_legends(repeats_model_dict[model], titlestring=model)
    for config in all_models_dict[model]:
        mc.simulation.predictions.plot_without_legends(all_models_dict[model][config], titlestring=model)
    


