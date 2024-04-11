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
    
    03 -> modelling only rewards + splitting model in the same function.
    03-A -> same as 03 but only considering B,C,D [excluding rew A]

    03-1 -> modelling only rewards + splitting the model after regression 
    03-2 -> same as 03-1 but only considering task D and B (where 2 rew locs are the same)
    03-5 - STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. 
    03-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    03-99 ->  using 03-1 - reward locations and future rew model; but EVs are scrambled.
    03-999 ->  is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.
    
    04 -> modelling only paths
    04-5 -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw.
    04-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING state A


GLM ('regression') settings (creating the 'bins'):
    01 - instruction EVs
    02 - 80 regressors; every task is divided into 4 rewards + 4 paths
    03 - 40 regressors; for every tasks, only the rewards are modelled [using a stick function]
    03-2 - 40 regressors; for every task, only the rewards are modelled (in their original time)
    03-3 - 30 regressors; for every task, only the rewards are modelled (in their original time), except for A (because of visual feedback)
    03-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)
    03-99 - 40 regressors; no button press; I allocate the reward onsets randomly to different state/task combos  -> shuffled through whole task; [using a stick function]
    03-999 - 40 regressors; no button press; created a random but sorted sample of onsets that I am using -> still somewhat sorted by time, still [using a stick function]
    03-9999 - 40 regressors; no button press; shift all regressors 6 seconds earlier
    04 - 40 regressors; for every task, only the paths are modelled
    04-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E)
    05 - locations + button presses 
    

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
RDM_version = '03' 

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

subjects = [f"sub-{subj_no}"]
temporal_resolution = 10

task_halves = ['1', '2']
fmriplotting = False
fmri_save = True

add_run_counts_model = False # this doesn't work with the current analysis

if RDM_version in ['01', '01-1']: # 01 doesnt work yet! 
    models_I_want = ['direction_presentation', 'execution_similarity', 'presentation_similarity']

elif RDM_version in ['02', '02-A']: #modelling paths + rewards, creating all possible models 
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight', 'clocks']

elif RDM_version in ['03', '03-A']: # modelling only rewards, splitting clocks within the same function
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew']
elif RDM_version in ['03-1', '03-2']:  # modelling only rewards, splitting clocks later in a different way - after the regression.
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
elif RDM_version in ['03-5', '03-5-A', '04-5', '04-5-A']:
    models_I_want = ['state']
elif RDM_version in ['03-3']:  # modelling only rewards, splitting clocks later in a different way - after the regression; ignoring reward A
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc']
elif RDM_version in ['03-99']:  # using 03-1 - reward locations and future rew model; but EVs are scrambled.
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
elif RDM_version in ['03-999']:  # is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']

elif RDM_version in ['04', '04-A']: # only paths. to see if the human brain represents also only those rings anchored at no-reward locations
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_no-rew', 'clocks_no-rew']

    
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
            print("Running on laptop.")
        else:
            data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
            RDM_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
            print(f"Running on Cluster, setting {data_dir_beh} as data directory")
            
        file = f"{sub}_fmri_pt{task_half}"
        file_all = f"{sub}_fmri_pt{task_half}_all.csv"
        
        # load the two required excel sheets
        df = pd.read_csv(data_dir_beh + f"{file}.csv")
        # the first row is empty so delete to get indices right
        df = df.iloc[1:].reset_index(drop=True)
        # fill gapss
        df['round_no'] = df['round_no'].fillna(method='ffill')
        df['task_config'] = df['task_config'].fillna(method='ffill')
        df['repeat'] = df['repeat'].fillna(method='ffill')
        # so that I cann differenatiate task config and direction
        df['config_type'] = df['task_config'] + '_' + df['type']
        
        # add columns whith field numbers 
        for index, row in df.iterrows():
            # current locations
            df.at[index, 'curr_loc_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_y'], is_y=True, is_x = False)
            df.at[index, 'curr_loc_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_x'], is_x=True, is_y = False)
            df.at[index, 'curr_rew_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_y'], is_y=True, is_x = False)
            df.at[index, 'curr_rew_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_x'], is_x=True, is_y = False)
            # and prepare the regressors: config type, state and reward/walking specific.
            if not pd.isna(row['state']):
                if not np.isnan(row['rew_loc_x']):
                    df.at[index, 'time_bin_type'] =  df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_reward'
                elif np.isnan(row['rew_loc_x']):
                    df.at[index, 'time_bin_type'] = df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_path'
        

        # create a dictionnary with all future regressors, to make sure the names are not messed up.
        time_bin_types = df['time_bin_type'].dropna().unique()
        regressors = {}
        for time_bin_type in time_bin_types:
            regressors[time_bin_type] = []
           

        configs = df['config_type'].dropna().unique()
        
        
        walked_path = {}
        timings = {}
        rew_list = {}
        rew_timing = {}
        rew_index = {}
        subpath_after_steps = {}
        steps_subpath_alltasks = {}
        for config in configs:
            walked_path[config] = []
            timings[config] = []
            rew_list[config] = []
            rew_timing[config] = []
            rew_index[config] = []
            subpath_after_steps[config] = []
            steps_subpath_alltasks[config] = []
        
        
        for index, row in df.iterrows():
            task_config = row['config_type']
            time_bin_type = row['time_bin_type']
            
            #iterate through the regression dictionary first
            for key in regressors.keys():
                # check if the key starts with the task_config value
                if key.startswith(task_config):
                    if time_bin_type == key:
                        regressors[key].append(1)
                    elif pd.notna(time_bin_type):
                        regressors[key].append(0) 

            # in case a new task has just started
            if not np.isnan(row['next_task']): 
                # first check if this is the first task of several repeats.
                if (index == 0) or (row['config_type'] != df.at[index -1, 'config_type']):
                    timings[task_config].append(row['next_task'])
                else: # if it isnt, then take the reward start time from last rew D as start field.
                    timings[task_config].append(df.at[index -1, 't_step_press_global'])
                walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
            
            # if this is just a normal walking field
            elif not np.isnan(row['t_step_press_global']): # always except if this is reward D 
                # if its reward D, then it will be covered by the first if: if not np.isnan(row['next_task']): 
                timings[task_config].append(df.at[index - 1, 't_step_press_global'])  # Extract value from index-1
                walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
           
            # next check if its a reward field
            if not np.isnan(row['rew_loc_x']): # if this is a reward field.
                # check if this is either at reward D(thus complete) or ignore interrupted trials
                # ignore these as they are not complete.
                if (index+2 < len(df)) or (row['state'] == 'D'):
                    rew_timing[task_config].append(row['t_reward_start'])
                    rew_list[task_config].append([row['curr_rew_x_coord'], row['curr_rew_y_coord']])
                    subpath_after_steps[task_config].append(int(index-row['repeat']))  
                    if row['state'] == 'D':
                        rew_index[task_config].append(len(walked_path[task_config])) #bc step has not been added yet
                        # if this is the last run of a task
                        if (index+2 < len(df)):
                            # first check if there are more tasks coming after, otherwise error
                            if (row['config_type'] != df.at[index +1, 'config_type']):
                                walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                                timings[task_config].append(df.at[index -1, 't_reward_start'])
                        else:
                            # however also add these fields if this is the very last reward!
                            if row['repeat'] == 4:
                                walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                                timings[task_config].append(df.at[index -1, 't_step_press_global'])
                                
                    else:
                        rew_index[task_config].append(len(walked_path[task_config])-1) 
                else:
                    continue

        
        # so now, account for the temporal resolution that you want:
        for reg in regressors:
            regressors[reg] = np.repeat(regressors[reg], repeats = temporal_resolution)
        
        # this is the end of the loop in which I go through the table, extract rows and 
        # create these behavioural files.
        # put the results from this loop in dictionaries for better usability.
        rew_index_per_config = dict(zip(configs, rew_index))
        subpath_per_config = dict(zip(configs, subpath_after_steps))
        
        # overview of the reward fields per task.
        for config in rew_list:
            rew_list[config] = [[int(value) for value in sub_list] for sub_list in rew_list[config][0:4]]
          
        
        # next step: create subpath files with rew_index and how many steps there are per subpath.
        for config in subpath_after_steps:
            # if task is completed
            if (len(subpath_after_steps[config])%4) == 0:
                for r in range(0, len(subpath_after_steps[config]), 4):
                    subpath = subpath_after_steps[config][r:r+4]
                    steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                    if r == 0:
                        steps.insert(0, rew_index[config][r])
                    if r > 0:
                        steps.insert(0, (subpath[0]- subpath_after_steps[config][r-1]))
                    steps_subpath_alltasks[config].append(steps)
            # if task not completed
            elif (len(subpath_after_steps[config])%4) > 0:
                completed_tasks = len(subpath_after_steps[config])-(len(subpath_after_steps[config])%4)
                for r in range(0, completed_tasks, 4):
                    subpath = subpath_after_steps[config][r:r+4]
                    steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                    if r == 0:
                        steps.insert(0, rew_index[config][r])
                    if r > 0:
                        steps.insert(0, (subpath[0]- subpath_after_steps[config][r-1]))
                    steps_subpath_alltasks[config].append(steps)    

        if regression_version in ['03-4', '04-4']:
            for config in configs:
                if config.startswith('B') or config.startswith('D'):
                    del rew_list[config]
                
            configs = np.array([config for config in configs if config.startswith('A') or config.startswith('C') or config.startswith('E')])

    
        # finally, create simulations and time-bin per run.
        # first, prep result dictionaries.
        all_models_dict = {f"{model}": {key: "" for key in configs} for model in models_I_want}
        # and prepare the between-tasks dictionary.
        
        if not RDM_version == '01':
            for config in configs:
                # import pdb; pdb.set_trace()
                print(f"the config is {rew_list[config]} for {config}")
                # select complete trajectory of current task.
                trajectory = walked_path[config]
                trajectory = [[int(value) for value in sub_list] for sub_list in trajectory]
                # select the timings of this task
                timings_curr_run = timings[config]
                
                # select file that shows step no per subpath
                step_number = steps_subpath_alltasks[config]
                step_number = [[int(value) for value in sub_list] for sub_list in step_number]
                # make file that shows cumulative steps per subpath
                cumsteps_per_run = [np.cumsum(task)[-1] for task in step_number]
                cumsteps_task = np.cumsum(cumsteps_per_run)
                
                # then start looping through each subpath within one task
                repeats_model_dict = {}
                for no_run in range(len(step_number)):
                    # first check if the run is not completed. if so, skip the uncomplete part.
                    if len(subpath_after_steps[config]) < 20:
                        stop_after_x_runs = len(subpath_after_steps[config]) // 4
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
                    
                    # models  need to be concatenated for each run and task
                    if no_run == 0:
                        for model in result_model_dict:
                            repeats_model_dict[model] = result_model_dict[model].copy()
                    else:
                        for model in result_model_dict:
                            repeats_model_dict[model] = np.concatenate((repeats_model_dict[model], result_model_dict[model]), 1)
                
                
                # NEXT STEP: prepare the regression- select the correct regressors, filter keys starting with 'A1_backw'
                regressors_curr_task = {key: value for key, value in regressors.items() if key.startswith(config)}
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
                
                if regression_version in ['03','03-1','03-2', '03-4', '03-99', '03-999']:
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
                    elif model not in ['one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']:
                        all_models_dict[model][config] = mc.simulation.predictions.transform_data_to_betas(repeats_model_dict[model], regressors_matrix)
    
    
                # once the regression took place, the location model is the same as the midnight model.
                # thus, it will also be the same as predicting future rewards, if we rotate it accordingly!
                if RDM_version in ['03-1', '03-2', '03-3']:
                    # now do the rotating thing. 
                    all_models_dict['one_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -1, axis = 1) 
                    all_models_dict['two_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -2, axis = 1) 
                    if RDM_version in ['03-1', '03-2']:
                        all_models_dict['three_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -3, axis = 1) 
                
        # then, lastly, safe the all_models_dict in the respective task_half.
        models_between_task_halves[task_half] = all_models_dict
        print(f"task half {task_half}")
        configs_dict[task_half] = rew_list

    
        

    
    # out of the between-halves loop.
    if RDM_version == '01': # I have to work on this one further for the replay analysis (temporal + spatial)
        models_between_tasks = mc.analyse.analyse_MRI_behav.similarity_of_tasks(configs_dict) 
        models_sorted_into_splits = models_between_tasks.copy()
        # this doens't work anymore now after the changes!! continue working on it
        
    elif not RDM_version == '01':
        # first, sort the models into two equivalent halves, just in case this went wrong before.
        # import pdb; pdb.set_trace()
        sorted_keys_dict = mc.analyse.extract_and_clean.order_task_according_to_rewards(configs_dict)
        models_sorted_into_splits = {task_half: {model: {config: "" for config in sorted_keys_dict[task_half]} for model in models_I_want} for task_half in task_halves}
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
    # import pdb; pdb.set_trace()
    
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

    if fmriplotting:
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

    
    if fmri_save: 
        # then save these matrices.
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        if RDM_version in ['03-5', '03-5-A', '04-5', '04-5-A']:
            np.save(os.path.join(RDM_dir, f"RSM_state_mask_across_halves"), RSM_dict_betw_TH_mask)
        for RDM in corrected_RSM_dict:
            np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_both_halves"), corrected_RSM_dict[RDM])
            
        # also save the regression files
        for model in models_sorted_into_splits['1']:
            np.save(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves"), np.concatenate((models_sorted_into_splits['1'][model], models_sorted_into_splits['2'][model]),1))
        
        # and lastly, save the order in which I put the RDMs.
        
        with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'wb') as file:
            pickle.dump(sorted_keys_dict, file)
            
        with open(f"{RDM_dir}/sorted_regs.pkl", 'wb') as file:
            pickle.dump(reg_list, file)
                