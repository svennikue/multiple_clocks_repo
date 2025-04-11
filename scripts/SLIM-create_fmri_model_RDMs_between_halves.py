#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 7th of April 2025

from the behavioural files I collected in the experiment, extract behaviour
and use behavioural data to model simulations for the model RDMs. Finally, bin the
simulations according to the GLM I am using for the fMRI data.

RDM settings (creating the representations):
    03-1 -> modelling only reward rings + split ‘clocks model’ = just rotating the reward location around.  

GLM ('regression') settings (creating the 'bins'):
    03-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)

@author: Svenja Küchenhoff, 2025
"""

import os
import numpy as np
import mc
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd
import copy

# import pdb; pdb.set_trace()

regression_version = '03-4' 
RDM_version = '03-1'

no_phase_neurons = 3

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

subjects = [f"sub-{subj_no}"]

temporal_resolution = 10
task_halves = ['1', '2']
fmriplotting = True 
fmri_save = True

models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)
   
for sub in subjects:
    # initialize some dictionaries
    models_between_task_halves, sorted_models_split, configs_dict = {}, {}, {}
    reg_list = []
    for task_half in task_halves:
        data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        RDM_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
        file = data_dir_beh + f"{sub}_fmri_pt{task_half}.csv"
        
        # crucial step 1: get the behavioural data I need from the subject files.
        configs, rew_list, rew_index, walked_path, steps_subpath_alltasks_empty, subpath_after_steps, timings, regressors, keys_executed, keys_not_executed, timings_keys_not_executed = mc.analyse.analyse_MRI_behav.extract_behaviour(file)
        
        for reg in regressors:
            regressors[reg] = np.repeat(regressors[reg], repeats = temporal_resolution)
        
        # overview of how many steps to each reward per subpath in each task
        steps_subpath_alltasks = mc.analyse.analyse_MRI_behav.subpath_files(configs, subpath_after_steps, rew_list, rew_index, steps_subpath_alltasks_empty)

        if regression_version in ['03-4']:
            # in these regressions, tasks B and D aren't included.
            for config in configs:
                if config.startswith('B') or config.startswith('D'):
                    del rew_list[config]
            configs = np.array([config for config in configs if config.startswith('A') or config.startswith('C') or config.startswith('E')])

        # finally, create simulations and time-bin per run.
        # prepare the between-tasks dictionary.
        all_models_dict = {f"{model}": {key: "" for key in configs} for model in models_I_want}

        if not RDM_version == '01':
            for config in configs:
                print(f"the config is {rew_list[config]} for {config}")
                # select complete trajectory of current task.
                trajectory = walked_path[config]
                trajectory = [[int(value) for value in sub_list] for sub_list in trajectory]
                
                buttons_pressed = []
                for press in keys_executed[config]:
                    if pd.isna(press[0]):
                        buttons_pressed.append(None)
                    else:
                        buttons_pressed.append(int(press[0]))   
  
                # select file that shows step no per subpath
                step_number = [[int(value) for value in sub_list] for sub_list in steps_subpath_alltasks[config]]
                
                # but only consider some of the repeats for the only later or only early trials!
                index_run_no = mc.analyse.analyse_MRI_behav.determine_index_by_reg_version(regression_version, step_number)
                
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
                        curr_timings = timings[config][0:cumsteps_task[no_run]+1]
                        curr_stepnumber = step_number[no_run]
                        # but keys isn't
                        curr_keys = buttons_pressed[0:cumsteps_task[no_run]]
                    elif no_run > 0:
                        # careful: fields is always one more than the step number
                        curr_trajectory = trajectory[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
                        curr_timings = timings[config][cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
                        # but keys isn't
                        curr_keys = buttons_pressed[cumsteps_task[no_run-1]:cumsteps_task[no_run]]
                        curr_stepnumber = step_number[no_run]
                        curr_cumsumsteps = cumsteps_task[no_run]
                    
                    # check if locations map back to the reward configs
                    errors = 0
                    for i, step in enumerate(np.cumsum(curr_stepnumber)):
                        if curr_trajectory[step] != rew_list[config][i]:
                            errors = errors + 1
                            print("careful! reward doesn't match location!")
                    
                    # KEY STEP
                    # create all models.
                    if RDM_version in ['03-1', '03-2', '03-3']:# modelling only clocks + splitting clocks later in different way.
                        # USING A SIMPLER MODEL NOW
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri_simple(curr_trajectory, curr_timings, curr_stepnumber, rew_list[config])
                        # This used to be the function. 
                        # Note: change to result_model_dict if you want to use it again
                        # also change select_models_I_want function back!
                        # result_model_dict_old = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path= False, split_clock = False)    
                    
                    # models need to be concatenated for each run and task
                    if no_run == 0:
                        for model in result_model_dict:
                            repeats_model_dict[model] = copy.deepcopy(result_model_dict[model])
                    else:
                        for model in result_model_dict:
                            repeats_model_dict[model] = np.concatenate((repeats_model_dict[model], result_model_dict[model]), 1)
                
  
                # NEXT STEP: prepare the regression- select the correct regressors, filter keys starting with 'A1_backw'
                regressors_curr_task = {key: value for key, value in regressors.items() if key.startswith(config)}

                # identify at which index the next task starts.
                subpath_to_find_indices = next(key for key in regressors_curr_task if key.startswith(f"{config}_A_path"))
                new_task_starts = np.diff(regressors_curr_task[subpath_to_find_indices], prepend=0)
                index_next_repeat = list(np.where(new_task_starts == 1)[0])

                # cut the regressors depending on which repeats to include
                version_spec_start, version_spec_end = mc.simulation.predictions.define_start_and_end_of_repeat_regressors(regression_version, repeats_model_dict[model], index_next_repeat)
                regressors_curr_task = {regressor: regressors_curr_task[regressor][version_spec_start:version_spec_end] for regressor in regressors_curr_task}
                print(f"now looking at regressor for task {config}")
                
                
                # check that all regressors have the same length in case the task wasn't completed.
                if len(subpath_after_steps[config]) < 20:
                    # if I cut the task short, then also cut the regressors short.
                    for reg_type, regressor_list in regressors_curr_task.items():
                    # Truncate the list where the task stopped
                        regressors_curr_task[reg_type] = regressor_list[:(np.shape(repeats_model_dict[list(repeats_model_dict)[0]])[1])]
                
                # Ensure all regressors of one task have the same length
                list_lengths = set(len(value) for value in regressors_curr_task.values())
                if len(list_lengths) != 1:
                    raise ValueError("All lists must have the same length.")
                
                # if not all regressors shall be included, filter them according to the regression setting
                if regression_version in ['03-4']:
                    regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if key.endswith('reward')}
                
                # sort alphabetically to be sure.
                sorted_regnames_curr_task = sorted(regressors_curr_task.keys())
                # Create a list of lists sorted by keys
                sorted_regs = [regressors_curr_task[key] for key in sorted_regnames_curr_task]
                regressors_matrix = np.array(sorted_regs)
                reg_list.append(sorted_regnames_curr_task)
                
                # then do the ORDERED time-binning for each model - across the 5 repeats.
                for model in all_models_dict:
                    if model not in ['one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr-and-future-rew-locs', 'state_masked', 'one_future_step2rew', 'two_future_step2rew', 'three_future_step2rew', 'curr-and-future-steps2rew']:
                        all_models_dict[model][config] = mc.simulation.predictions.transform_data_to_betas(repeats_model_dict[model], regressors_matrix)
    
        # END OF LOOP GOING THROUGH ALL CONFIGURATIONS.
        # then, lastly, safe the all_models_dict in the respective task_half.
        models_between_task_halves[task_half] = copy.deepcopy(all_models_dict)
        print(f"task half {task_half}")
        configs_dict[task_half] = rew_list
    # END OF LOOP GOING THROUGH BOTH TASK HALVES.


    sorted_keys_dict = mc.analyse.extract_and_clean.order_task_according_to_rewards(configs_dict)   
    # first, sort the models into two equivalent halves, just in case this went wrong before.
    models_sorted_into_splits = {task_half: {model: {config: "" for config in sorted_keys_dict[task_half]} for model in models_I_want} for task_half in task_halves}
    test = {task_half: {model: "" for model in models_I_want} for task_half in task_halves}
    
    for half in models_between_task_halves:
        for model in models_between_task_halves[half]:
            for task in models_between_task_halves[half][model]:
                if task in sorted_keys_dict['1']:
                    models_sorted_into_splits['1'][model][task] = models_between_task_halves[half][model][task]
                elif task in sorted_keys_dict['2']:
                    models_sorted_into_splits['2'][model][task] = models_between_task_halves[half][model][task]                
    # then, do the concatenation across the ordered tasks.
    # import pdb; pdb.set_trace()
    for split in models_sorted_into_splits:
        for model in models_sorted_into_splits[split]:
            test[split][model] = np.concatenate([models_sorted_into_splits[split][model][task] for task in sorted_keys_dict[split]], 1)
            models_sorted_into_splits[split][model] = np.concatenate([models_sorted_into_splits[split][model][task] for task in sorted_keys_dict[split]], 1)
            
        

    # then, in a last step, create the RDMs
    # concatenate the conditions from the two task halves (giving you 2*nCond X nVoxels matrix), 
    # and calculate the correlations between all rows of this matrix. This gives you a symmetric matrix 
    # (of size 2*nCond X 2*nCond), where the (non-symmetric) nCond X nCond bottom left square (or top right, 
    # doesn't matter because it's symmetric) (i.e. a quarter of the original matrix) has all the correlations 
    # across THs. 

    # note: this is not what I actually use for the RSA. 
    # I am using the concatenated models_sorted_into_splits TH1, then TH2 for the model RDMs.
    
    RSM_dict_betw_TH = {}
    for model in models_sorted_into_splits[split]:
        RSM_dict_betw_TH[model] = mc.simulation.RDMs.within_task_RDM(np.concatenate((models_sorted_into_splits['1'][model], models_sorted_into_splits['2'][model]),1), plotting = False, titlestring= model)
        # mc.simulation.predictions.plot_without_legends(RSM_dict_betw_TH[model])
            
    corrected_RSM_dict = {}
    for model in RSM_dict_betw_TH:
        # import pdb; pdb.set_trace()
        corrected_model = RSM_dict_betw_TH[model][int(len(RSM_dict_betw_TH[model])/2):, 0:int(len(RSM_dict_betw_TH[model])/2):]
        corrected_model = (corrected_model + np.transpose(corrected_model))/2
        corrected_RSM_dict[model] = corrected_model


    if fmriplotting:
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        mc.simulation.RDMs.plot_RDMs(corrected_RSM_dict, len(configs), RDM_dir, sorted_keys_dict['1'],  one_minus = True)
    
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
                corr_RDMs[x,y] = mc.simulation.RDMs.corr_matrices(corrected_RSM_dict[RDM_one], corrected_RSM_dict[RDM_two])[0]              
        intercorr_RDM_dict['correlation_try_two'] = corr_RDMs
        mc.simulation.RDMs.plot_RDMs(intercorr_RDM_dict, len(corr_RDMs), RDM_dir, string_for_ticks = tick_string, one_minus = False)       

    
    if fmri_save: 
        # then save these matrices.
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir) 
            
        for RDM in corrected_RSM_dict:
            np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_both_halves"), corrected_RSM_dict[RDM])
        # also save the regression files
        for model in models_sorted_into_splits['1']:
            np.save(os.path.join(RDM_dir, f"data_{model}_{sub}_fmri_both_halves"), np.concatenate((models_sorted_into_splits['1'][model], models_sorted_into_splits['2'][model]),1))
        # and lastly, save the order in which I put the RDMs.
        
        with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'wb') as file:
            pickle.dump(sorted_keys_dict, file)
            
        with open(f"{RDM_dir}/sorted_regs.pkl", 'wb') as file:
            pickle.dump(reg_list, file)
                
    