#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:34:17 2023

from the behavioural files I collected in the experiment, extract behaviour
and use behavioural data to model simulations for the model RDMs. Finally, bin the
simulations according to the GLM I am using for the fMRI data.

28.03.: I am changing something in the preprocessing. This is THE day to change the naming such that it all works well :)

RDM settings (creating the representations):
    01 -> instruction periods, similarity by order of execution, order of seeing, all backw presentations
    01-1 -> instruction periods, location similarity
    02 -> modelling paths + rewards, creating all possible models
    02-A -> modelling everything but excluding state A
    02-act -> modelling paths + rewards, also creating the action model.
    02-act-1phas -> only one phase per subpath! modelling paths + rewards, also creating the action model.
    03 -> modelling only reward anchors/rings + splitting clocks model in the same py function.
    03-im -> imaginary number model of the clock to check for differences in task-lag, otherwise like 03
    03-tasklag -> weight the musicbox by tasklags: sine and cosine of task-lag angle.
    03-A -> same as 03 but only considering B,C,D [excluding rew A]

    03-1 -> modelling only reward rings + split ‘clocks model’ = just rotating the reward location around.  
    '03-1-act' -> same as 03-1, and additionally add the action model.
    
    
    03-2 -> same as 03-1 but only considering task D and B (where 2 rew locs are the same)
    03-5 - STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. 
    03-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    03-99 ->  using 03-1 - reward locations and future rew model; but EVs are scrambled.
    03-999 ->  is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.
    
    04 -> modelling only paths
    04-5 -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw.
    04-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING state A
    
    05 -> modelling only paths and only rewards to compare them later!
    

GLM ('regression') settings (creating the 'bins'):
    01 - instruction EVs
    02 - 80 regressors; every task is divided into 4 rewards + 4 paths
    02-4 - 48 regressors; every task is divided into 4 rewards + 4 paths; but only for the tasks where every reward is at a different location (A,C,E)
    03 - 40 regressors; for every tasks, only the rewards are modelled [using a stick function]
    03-e 40 regressors; for evert task, only take the first 2 repeats.
    03-l 40 regressors; for every task, only take the last 3 repeats.
        careful! sometimes, some trials are not finished and thus don't have any last runs. these are then empty regressors.
    03-rep1 40 regressors; for every task, only take the first repeat
    03-rep2 40 regressors; for every task, only take the second repeat
    03-rep3 40 regressors; for every task, only take the third repeat
    03-rep4 40 regressors; for every task, only take the fourth repeat
    03-rep5 40 regressors; for every task, only take the fifth repeat
    03-2 - 40 regressors; for every task, only the rewards are modelled (in their original time)
    03-3 - 30 regressors; for every task, only the rewards are modelled (in their original time), except for A (because of visual feedback)
    03-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)
    03-4-e - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function); only early repeats
    03-4-l - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function); only late repeats
    03-4-rep1, rep2,.... 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function); only one repeat
    03-99 - 40 regressors; no button press; I allocate the reward onsets randomly to different state/task combos  -> shuffled through whole task; [using a stick function]
    03-999 - 40 regressors; no button press; created a random but sorted sample of onsets that I am using -> still somewhat sorted by time, still [using a stick function]
    03-9999 - 40 regressors; no button press; shift all regressors 6 seconds earlier
    04 - 40 regressors; for every task, only the paths are modelled
    04-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E)
    05 - locations + button presses 
    06 - averaging across the entire task [for introduction analysis]
    06-rep 1 - averaging across the entire task, but only the first repeat.
    07 - entire path and reward period, collapsed (= 03 + 04)
    07-4 - entire path and reward period, collapsed (= 03 + 04); only for the tasks where every reward is at a different location (A,C,E)

@author: Svenja Küchenhoff, 2024
"""

import os
import numpy as np
import mc
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd

# import pdb; pdb.set_trace()

regression_version = '03-4' 
RDM_version = '02-act'

if RDM_version in ['02-act-1phas']:
    no_phase_neurons = 1
else:
    no_phase_neurons = 3

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

subjects = [f"sub-{subj_no}"]
# subjects = subs_list = [f'sub-{i:02}' for i in range(1, 35) if i not in (21, 29)]

temporal_resolution = 10

task_halves = ['1', '2']
fmriplotting = True # incorrect for 01 false for 03-im!
fmri_save = True

add_run_counts_model = False # this doesn't work with the current analysis

  
models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)
if 'state_masked' in models_I_want:
    models_I_want.remove('state_masked')

# import pdb; pdb.set_trace()
        
for sub in subjects:
    # initialize some dictionaries
    models_between_task_halves = {}
    sorted_models_split = {}
    configs_dict = {}
    reg_list = []
    for task_half in task_halves:
        data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        RDM_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
        if os.path.isdir(data_dir_beh):
            print(f"Running on laptop, now subject {sub}")
        else:
            data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
            RDM_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
            print(f"Running on Cluster, setting {data_dir_beh} as data directory")
            
        file = data_dir_beh + f"{sub}_fmri_pt{task_half}.csv"
        
        # crucial step 1: get the behavioural data I need from the subject files.
        configs, rew_list, rew_index, walked_path, steps_subpath_alltasks_empty, subpath_after_steps, timings, regressors, keys_executed, keys_not_executed, timings_keys_not_executed = mc.analyse.analyse_MRI_behav.extract_behaviour(file)

        # recombine the regressors for GLM 07
        if regression_version in ['07', '07-4']:
            # Iterate over the keys of the dictionary
            for key in regressors.keys():
                if '_path' in key:
                    # Extract the common prefix
                    prefix = key.replace('_path', '')
                    # Check if the corresponding 'reward' key exists
                    reward_key = prefix + '_reward'
                    if reward_key in regressors:
                        # Sum the arrays and save them in the new dictionary
                        for i, elem in enumerate(regressors[key]):
                            regressors[key][i] = elem + regressors[reward_key][i]
                            
        # lastly, remove all '_reward' regressors as they are integrated in the first bit.
        regressors = {key: value for key, value in regressors.items() if not key.endswith('_reward')}

        # so now, account for the temporal resolution that you want:
        for reg in regressors:
            regressors[reg] = np.repeat(regressors[reg], repeats = temporal_resolution)
        
        # overview of the reward fields per task.
        steps_subpath_alltasks = mc.analyse.analyse_MRI_behav.subpath_files(configs, subpath_after_steps, rew_list, rew_index, steps_subpath_alltasks_empty)

        if regression_version in ['02-4', '03-4', '03-4-e', '03-4-l','04-4', '03-4-rep1', '03-4-rep2', '03-4-rep3', '03-4-rep4', '03-4-rep5', '07-4']:
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
                # select the timings of this task
                timings_curr_run = timings[config]
  
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
                    #import pdb; pdb.set_trace()
                    if no_run == 0:
                        # careful: fields is always one more than the step number
                        curr_trajectory = trajectory[0:cumsteps_task[no_run]+1]
                        curr_timings = timings_curr_run[0:cumsteps_task[no_run]+1]
                        curr_stepnumber = step_number[no_run]
                        # but keys isn't
                        curr_keys = buttons_pressed[0:cumsteps_task[no_run]]
                    elif no_run > 0:
                        # careful: fields is always one more than the step number
                        curr_trajectory = trajectory[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
                        curr_timings = timings_curr_run[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
                        # but keys isn't
                        curr_keys = buttons_pressed[cumsteps_task[no_run-1]:cumsteps_task[no_run]]
                        curr_stepnumber = step_number[no_run]
                        curr_cumsumsteps = cumsteps_task[no_run]
                    
                    # KEY STEP
                    # create all models.
                    if RDM_version == '01-1': # creating location instruction stuff
                        result_model_dict = mc.simulation.predictions.create_instruction_model(rew_list[config], trial_type=config)
                    elif RDM_version in ['02', '02-A']: # default, modelling all and splitting clocks.
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = False, only_path= False, split_clock = True)
                    elif RDM_version in ['02-act', '02-act-1phas']:
                        models_from_02 = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = False, only_path= False, split_clock = True, no_phase_neurons=no_phase_neurons)
                        model_from_action = mc.simulation.predictions.create_action_model_RDMs_fmri(curr_keys, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, only_rew = False, only_path= False, split_future_actions = True, no_phase_neurons=no_phase_neurons)    
                        result_model_dict = {**models_from_02, **model_from_action}
                        
                    elif RDM_version in ['03', '03-5', '03-5-A', '03-A']: # modelling only rewards + splitting clocks [new]
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path = False, split_clock=True)
                    
                    elif RDM_version in ['03-im']: # modelling the clocks with imaginary numbers -> NOT SURE IF THIS LED ANYWHERE
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path= False, split_clock = True, imaginary = True)    
                    elif RDM_version in ['03-tasklag']: # modelling the clocks with imaginary numbers
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path= False, split_clock = True, imaginary = False, lag_weighting = True)    
                    elif RDM_version in ['03-1-act']:
                        models_from_03_1 = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path= False, split_clock = False)    
                        model_from_action = mc.simulation.predictions.create_action_model_RDMs_fmri(curr_keys, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, only_rew = True, only_path= False, split_future_actions = False)    
                        result_model_dict = {**models_from_03_1, **model_from_action}
                    elif RDM_version in ['03-1', '03-2', '03-3']:# modelling only clocks + splitting clocks later in different way.
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path= False, split_clock = False)    
                    elif RDM_version in ['04', '04-5-A', '04-A']: # modelling only paths + splitting clocks [new]
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = False, only_path = True, split_clock=True)
                    elif RDM_version in ['05']:
                        models_from_03_1 = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path= False, split_clock = False)    
                        model_from_04 = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = False, only_path = True, split_clock=True)
                        # then put both together:
                        result_model_dict = {**models_from_03_1, **model_from_04}
                    

                    
                    # import pdb; pdb.set_trace()
                    # now for all models that are creating or not creating the splits models with my default function, this checking should work.
                    if RDM_version not in ['03-1','03-1-act', '03-2', '03-3', '03-5', '03-5-A','04-5', '04-5-A', '05', '03-tasklag']:
                        # test if this function gives the same as the models you want, otherwise break!
                        model_list = list(result_model_dict.keys())
                        if model_list != models_I_want:
                            print('careful! the model dictionary did not output your defined models!')
                            print(f"These are the models you wanted: {models_I_want}. And these are the ones you got: {model_list}")
                            import pdb; pdb.set_trace()
                    
                    # models need to be concatenated for each run and task
                    if no_run == 0 or (regression_version in ['03-l', '03-4-l'] and no_run == 2) or (regression_version in ['03-rep1', '03-rep2', '03-rep3', '03-rep4', '03-rep5', '03-4-rep1', '03-4-rep2', '03-4-rep3', '03-4-rep4', '03-4-rep5']):
                        for model in result_model_dict:
                            repeats_model_dict[model] = result_model_dict[model].copy()
                    else:
                        for model in result_model_dict:
                            repeats_model_dict[model] = np.concatenate((repeats_model_dict[model], result_model_dict[model]), 1)
                
                   
                # NEXT STEP: prepare the regression- select the correct regressors, filter keys starting with 'A1_backw'
                regressors_curr_task = {key: value for key, value in regressors.items() if key.startswith(config)}

                # identify at which index the next task starts.
                index_next_repeat = []
                subpath_to_find_indices = next(key for key in regressors_curr_task if key.startswith(f"{config}_A_path"))
                for i in range(len(regressors_curr_task[subpath_to_find_indices])):
                    if regressors_curr_task[subpath_to_find_indices][i] == 1 and (i == 0 or regressors_curr_task[subpath_to_find_indices][i-1] == 0):
                        index_next_repeat.append(i)
                
                if regression_version in ['03-e', '03-4-e', '03-rep1', '03-4-rep1']:
                    regressors_curr_task = {regressor: regressors_curr_task[regressor][0:len(repeats_model_dict[model][2])] for regressor in regressors_curr_task}

                if regression_version in ['03-l', '03-4-l', '03-rep5', '03-4-rep5']:
                    regressors_curr_task = {regressor: regressors_curr_task[regressor][(-1*len(repeats_model_dict[model][2])):] for regressor in regressors_curr_task}
                
                if regression_version in ['03-rep2', '03-4-rep2']:
                    regressors_curr_task = {regressor: regressors_curr_task[regressor][index_next_repeat[1]:index_next_repeat[2]] for regressor in regressors_curr_task}

                if regression_version in ['03-rep3', '03-4-rep3']:
                    regressors_curr_task = {regressor: regressors_curr_task[regressor][index_next_repeat[2]:index_next_repeat[3]] for regressor in regressors_curr_task}

                if regression_version in ['03-rep4', '03-4-rep4']:
                    regressors_curr_task = {regressor: regressors_curr_task[regressor][index_next_repeat[3]:index_next_repeat[4]] for regressor in regressors_curr_task}
                



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
                if regression_version in ['02', '02-4']:
                    if RDM_version == '02-A':
                        regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key}
                    else:
                        regressors_curr_task = {key: value for key, value in regressors_curr_task.items()}
                
                if regression_version in ['03','03-1', '03-1-act', '03-2', '03-4', '03-99', '03-999', '03-l', '03-e', '03-4-e', '03-4-l', '03-4-rep1', '03-4-rep2', '03-4-rep3', '03-4-rep4', '03-4-rep5']:
                    if RDM_version in ['02-A', '03-A', '03-5-A']: # additionally get rid of the A-state.
                        regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key and key.endswith('reward')}
                    else:
                        regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if key.endswith('reward')}
                
                if regression_version in ['04', '04-4']:
                    if RDM_version in ['04-5-A', '02-A', '04-A']:
                        regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key and key.endswith('path')}    
                    else:
                        regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if key.endswith('path')}
     
                if regression_version in ['06', '06-rep1']:
                    regressors_curr_task ={}
                    regressors_curr_task[config] = np.ones(len(repeats_model_dict[model][2]))
                    if regression_version == '06-rep1':
                        regressors_curr_task[config][:] = 0
                        regressors_curr_task[config][0:index_next_repeat[1]] = 1
                 
                # import pdb; pdb.set_trace() 
                if regression_version in ['07','07-4']:
                    regressors_curr_task = {key: value for key, value in regressors_curr_task.items()}
                    
                    
                    
                # import pdb; pdb.set_trace() 
                    # regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if key.endswith('reward')}
                    # x = {key: value for key, value in regressors_curr_task.items() if key.endswith('reward')}
                    
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
                    elif model not in ['one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr-and-future-rew-locs', 'state_masked', 'one_future_step2rew', 'two_future_step2rew', 'three_future_step2rew', 'curr-and-future-steps2rew']:
                        all_models_dict[model][config] = mc.simulation.predictions.transform_data_to_betas(repeats_model_dict[model], regressors_matrix)
    
                # import pdb; pdb.set_trace()
                # once the regression took place, the location model is the same as the midnight model.
                # thus, it will also be the same as predicting future rewards, if we rotate it accordingly!
                # temporally not do this
                
                if RDM_version in ['03-1','03-1-act', '03-2', '03-3', '05']:
                    # now do the rotating thing. 
                    all_models_dict['one_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -1, axis = 1) 
                    all_models_dict['two_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -2, axis = 1) 
                    if RDM_version in ['03-1', '03-1-act', '03-2', '05']:
                        all_models_dict['three_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -3, axis = 1) 
                    # try if curr-and-future-rew-locs is the same as rew clocks
                    all_models_dict['curr-and-future-rew-locs'][config] = np.concatenate((all_models_dict['location'][config], all_models_dict['one_future_rew_loc'][config], all_models_dict['two_future_rew_loc'][config], all_models_dict['three_future_rew_loc'][config]), 0)
                    
                    # do the same for the buttons!
                    if RDM_version in ['03-1-act']:
                        all_models_dict['one_future_step2rew'][config] = np.roll(all_models_dict['buttons'][config], -1, axis = 1) 
                        all_models_dict['two_future_step2rew'][config] = np.roll(all_models_dict['buttons'][config], -2, axis = 1) 
                        all_models_dict['three_future_step2rew'][config] = np.roll(all_models_dict['buttons'][config], -3, axis = 1) 
                        # try if curr-and-future-steps-to-reward is the same as action box
                        all_models_dict['curr-and-future-steps2rew'][config] = np.concatenate((all_models_dict['buttons'][config], all_models_dict['one_future_step2rew'][config], all_models_dict['two_future_step2rew'][config], all_models_dict['three_future_step2rew'][config]), 0)
                  
                # import pdb; pdb.set_trace()
                    
                    
        # then, lastly, safe the all_models_dict in the respective task_half.
        models_between_task_halves[task_half] = all_models_dict
        print(f"task half {task_half}")
        configs_dict[task_half] = rew_list
  



    # out of the between-halves loop.
    sorted_keys_dict = mc.analyse.extract_and_clean.order_task_according_to_rewards(configs_dict)   
    
    if RDM_version == '01': # I have to work on this one further for the replay analysis (temporal + spatial)
        models_between_tasks = mc.analyse.analyse_MRI_behav.similarity_of_tasks(configs_dict) 
        models_sorted_into_splits = models_between_tasks.copy()
        split = list(models_sorted_into_splits.keys())[0]
        # this doens't work anymore now after the changes!! continue working on it
   
    elif not RDM_version == '01':
        # first, sort the models into two equivalent halves, just in case this went wrong before.
        # DOUBLE CHECK IF THIS SORTING ACTUALLY WORKS!!!!
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
    
    
    RSM_dict_betw_TH = {}
    for model in models_sorted_into_splits[split]:
        RSM_dict_betw_TH[model] = mc.simulation.RDMs.within_task_RDM(np.concatenate((models_sorted_into_splits['1'][model], models_sorted_into_splits['2'][model]),1), plotting = False, titlestring= model)
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
            


    # import pdb; pdb.set_trace()
    corrected_RSM_dict = {}
    for model in RSM_dict_betw_TH:
        # import pdb; pdb.set_trace()
        corrected_model = RSM_dict_betw_TH[model][int(len(RSM_dict_betw_TH[model])/2):, 0:int(len(RSM_dict_betw_TH[model])/2):]
        corrected_model = (corrected_model + np.transpose(corrected_model))/2
        corrected_RSM_dict[model] = corrected_model

    
    
    # just for me. what happens if I add the ['reward_location', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']?
    # addition_model = corrected_RSM_dict['reward_location'] + corrected_RSM_dict['one_future_rew_loc'] + corrected_RSM_dict['two_future_rew_loc'] + corrected_RSM_dict['three_future_rew_loc'] 
    
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
        if RDM_version in ['03-5', '03-5-A', '04-5', '04-5-A']:
            np.save(os.path.join(RDM_dir, "RSM_state_mask_across_halves"), RSM_dict_betw_TH_mask)
        if RDM_version not in ['01']:    
            for RDM in corrected_RSM_dict:
                np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_both_halves"), corrected_RSM_dict[RDM])
        else:
            for RDM in RSM_dict_betw_TH:
                np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_across_halves"), RSM_dict_betw_TH[RDM])
        # also save the regression files
        for model in models_sorted_into_splits['1']:
            np.save(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves"), np.concatenate((models_sorted_into_splits['1'][model], models_sorted_into_splits['2'][model]),1))
        
        # and lastly, save the order in which I put the RDMs.
        
        with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'wb') as file:
            pickle.dump(sorted_keys_dict, file)
            
        with open(f"{RDM_dir}/sorted_regs.pkl", 'wb') as file:
            pickle.dump(reg_list, file)
                
    