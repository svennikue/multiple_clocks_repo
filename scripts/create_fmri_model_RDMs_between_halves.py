#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:34:17 2023

from the behavioural files I collected in the experiment, extract behaviour
and use behavioural data to model simulations for the model RDMs. Finally, bin the
simulations according to the GLM I am using for the fMRI data.

RDM settings (creating the representations):
    
    right. so actually, all RDM versions that are taken rn are:
        02 -> modelling only reward rings, new way of model splitting.
        03 -> modelling only no-reward rings, new way of model splitting.
        05 -> default RDM, modelling everything
        09 -> RDM only modelling rewards, different way of model splitting (after regression), and some debugging modes
        10 -> like 09, but only including the tasks with 2 same reward locations.
        11 -> instruction periods.
    
    02 [new] is model only the rewards, and split the clocks in the same function; this result should be the same as RDM 09
    03 [new] is to see if the human brain represents also only those rings anchored at no-reward locations; splitting the clocks in the same function
    
    02 was for the transfer of status report. It evolved from here.
    03 is teporal resolution = 1
    04 is another try of bringing the results back...
    05 is both task halves combined, with clocks, midnight, phase, state, loc model. default, modelling all and splitting clocks
    
    # DELELTE 06 -> replace with something new
    06 is both task halves combined, with the reduced midnight and clocks: only coding for rewards, and only considering reward times
        this might have been different in the past, but I didn't pull through with it anyways.
        now, 06 is now kind of redudant because it is the same as 05, but just the regression changes (representing both rew and paths in clock)

    DELTE 07 -> replace with something new. Is redundant with 09!!   
    07 is the second version of having midnight/clocks but only at reward locations: by 0-ing all non-reward ones
        [modelling only clocks + splitting clocks later in different way]
        
        
    DELETE 08 -> replace with something new.     
    08 is using combination models (GLMs with several models in the do RSA script)
    09 is looking at the split musicbox, when only reward locations are coded for.
        [modelling only clocks + splitting clocks later in different way] -> are 7 and 9 redundant???
    09-9 is kind-of debugging: try RDM 09 with glm 07 but only include those tasks, where the reward location is the same twice (B and D)
    999 is debugging: using 09 - reward locations and future rew model; but EVs are scrambled.
    9999 is debugging 2.0: using 09 - reward locations and future rew model; but the voxels are scrambled.
    10 is like 09, so rewards only and the split musicbox models as well, but excluding the state A (because of the visual feedback) 
        [modelling only clocks + splitting clocks later in different way]
        # ok this is actually dependent on the glm. it's glm 08. delete this one!
    11 is only the instruction period, simply 0 and 1 distances
    11-1 is the instruciton period but in a location model: are the location-sequences similar
    12 is modelling everything, including a split musicbox model of path + reward representations.
    
GLM ('regression') settings (creating the 'bins'):
    03 was location_EVs. (very long back)
    06 is reward + path phase per task [80 EVs] new, better script is now 06 #'04_pt01+_that_worked' 
    07 is only button press and rewards.
    08 is rewards only and without A (because of the visual feedback)
    09 is the instruction period only.
    10 is only paths
    11 is only rewards as a stick function
    

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

regression_version = '09' 
RDM_version = '11-1' 

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

subjects = [f"sub-{subj_no}"]
temporal_resolution = 10

task_halves = ['1', '2']
fmriplotting = True
fmriplotting_debug = False
fmri_save = True

add_run_counts_model = False # this doesn't work with the current analysis

if RDM_version in ['02']: # 2 was still free. so use this to model only the rewards, and split the clocks in the same function.
    # this result should be the same as RDM 09
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew']
elif RDM_version in ['03']: # 3 was still free. to see if the human brain represents also only those rings anchored at no-reward locations
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_no-rew', 'clocks_no-rew']
    
elif RDM_version in ['05']:
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight', 'clocks']
elif RDM_version in ['09','09-9', '999', '9999', '10']:  # delete 10 at some point
    models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
    #models_I_want = ['reward_location', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'reward_midnight_v2', 'reward_clocks_v2']
elif RDM_version in ['11', '11-1']: # 11 doesnt work yet! 
    models_I_want = ['instruction']


#elif RDM_version == '07':
#    models_I_want = ['reward_midnight_v2', 'reward_clocks_v2', 'state', 'task_prog', 'reward_location']
# RDM 08
# ??? (the model is using combination models (GLMs with several models in the do RSA script))

    
for sub in subjects:
    if add_run_counts_model == True:
        temp_models_I_want = models_I_want.copy()
        temp_models_I_want.append('run_count_model')
        models_between_tasks = {f"{model}": {key: "" for key in ['1', '2']} for model in temp_models_I_want}
    else:
        models_between_tasks = {f"{model}": {key: "" for key in ['1', '2']} for model in models_I_want}
    configs_dict = {}
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
      
        
        # [delete later - this is a debug step]
        # check if the regressors and extracted runs are aligned: the length of each regressor should be equal to len(elem) for elem in walked_path
        if fmriplotting_debug:
            # first check if regressors are all euqal, only plot if so
            # first check if C1_backw exists.
            if 'C1_backw' in configs:
                if len(regressors['C1_backw_A_subpath']) == len(regressors['C1_backw_A_reward']) == len(regressors['C1_backw_B_subpath']) == len (regressors['C1_backw_B_reward']) == len(regressors['C1_backw_C_subpath']) ==  len(regressors['C1_backw_C_reward']) ==  len(regressors['C1_backw_D_subpath']) == len(regressors['C1_backw_D_reward']):
                    plot_dict = np.concatenate((regressors['C1_backw_A_subpath'], regressors['C1_backw_A_reward'], regressors['C1_backw_B_subpath'], regressors['C1_backw_B_reward'], regressors['C1_backw_C_subpath'], regressors['C1_backw_C_reward'], regressors['C1_backw_D_subpath'], regressors['C1_backw_D_reward'])).reshape(8, -1)
                    plt.figure(); plt.imshow(plot_dict, aspect = 'auto')           
                     
                    # write a test figure for the regressors   
                    for config in configs:
                        #config = 'C1'
                        count = 0
                        to_plot = np.zeros((8,100)) + 3
                        for key in regressors.keys():
                            if key.startswith(config):
                                to_plot[count, 0 : len(regressors[key])] = regressors[key]
                                count += 1
                        plt.figure(); plt.imshow(to_plot, aspect = 'auto')   
                        for index in rew_index_per_config[config]:
                            plt.plot([index, index,index,index], [1,3,5,7], color = 'black', marker = 'x', markersize =8)
               
        
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

        if RDM_version == '09-9':
            configs = np.array([config for config in configs if config.startswith('D') or config.startswith('B')])
            
        # finally, create simulations and time-bin per run.
        # first, prep result dictionaries.
        if add_run_counts_model == True:
            all_models_dict = {f"{model}": {key: "" for key in configs} for model in temp_models_I_want}
        else:
            all_models_dict = {f"{model}": {key: "" for key in configs} for model in models_I_want}
        # and prepare the between-tasks dictionary.
        
        if not RDM_version == '11':
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
                    if RDM_version in ['02']: # modelling only rewards + splitting clocks [new]
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path = False, split_clock=True)
                    elif RDM_version in ['03']: # modelling only paths + splitting clocks [new]
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = False, only_path = True, split_clock=True)
                    elif RDM_version in ['05', '06']: # default, modelling all and splitting clocks. probs delete 06 at some point
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = False, only_path= False, split_clock = True)
                    elif RDM_version in ['07', '09', '09-9','10']:# modelling only clocks + splitting clocks later in different way. prons delete 07 at some point.
                        result_model_dict = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False, only_rew = True, only_path= False, split_clock = False)
                    elif RDM_version == '11':
                         result_model_dict = mc.analyse.analyse_MRI_behav.similarity_of_tasks(rew_list)
                    elif RDM_version == '11-1': # creating location instruction stuff
                        result_model_dict = mc.simulation.predictions.create_instruction_model(rew_list[config], trial_type=config)
                    
                    # now for all models that are creating or not creating the splits models with my default function, this checking should work.
                    if RDM_version not in ['07', '09', '09-9','10']:
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
                
                # and create a run-counter-model.
                if add_run_counts_model == True:
                    run_count_repeats = mc.simulation.predictions.create_run_count_model_fmri(step_number, len(step_number), norm_number_of_runs=5, wrap_around = 1, temporal_resolution = 10, plot = False)
                    models_I_want.append('run_count_model')
                    repeats_model_dict['run_count_model'] = run_count_repeats
                # INCLUDE THIS COUNT-RUNS MODEL AT SOME POINT - but in a different analysis way. requires a very different glm
                
                
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
                
                # if not all regressors shall be included, filter them according to the regression setting.
                if regression_version in ['07', '11']:
                    regressors_curr_task = {k: v for k, v in regressors_curr_task.items() if k.endswith('reward')}
                elif regression_version == '08': # additionally get rid of the A-state.
                    regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key}
                elif regression_version == '10':
                    regressors_curr_task = {k: v for k, v in regressors_curr_task.items() if k.endswith('path')}
                    
                # sort alphabetically.
                sorted_regnames_curr_task = sorted(regressors_curr_task.keys())
                # Create a list of lists sorted by keys
                sorted_regs = [regressors_curr_task[key] for key in sorted_regnames_curr_task]
                regressors_matrix = np.array(sorted_regs)
                if fmriplotting_debug:
                    plt.figure(); plt.imshow(regressors_matrix, aspect = 'auto')
                
                # then do the ORDERED time-binning for each model - across the 5 repeats.
                for model in all_models_dict:
                    if RDM_version == '11-1':
                        all_models_dict[model][config] = result_model_dict[model]
                    # run the regression on all simulated data, except for those as I have a different way of creating them:
                    elif model not in ['one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']:
                        all_models_dict[model][config] = mc.simulation.predictions.transform_data_to_betas(repeats_model_dict[model], regressors_matrix)


                # once the regression took place, the location model is the same as the midnight model.
                # thus, it will also be the same as predicting future rewards, if we rotate it accordingly!
                if RDM_version in ['09', '10', '09-9']:
                    # now do the rotating thing. 
                    all_models_dict['one_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -1, axis = 1) 
                    all_models_dict['two_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -2, axis = 1) 
                    if RDM_version in ['09', '09-9']:
                        all_models_dict['three_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -3, axis = 1) 

          
        # once through all task configuration - create the location_between etc. matrix by concatenating the task configurations in alphabetical order
        model_dict_sorted_keys = sorted(configs)
        print(f"I am sorting in this order: {model_dict_sorted_keys}")
        
        configs_dict[task_half] = rew_list
        
        # concatenate the model simulations of all tasks in the same order.
        if not RDM_version == '11':
            for model in all_models_dict:
                if model == 'reward_midnight_count': # this can probably go at some point
                    models_between_tasks[model][task_half] = np.concatenate([all_models_dict[model][key] for key in model_dict_sorted_keys])
                else:
                    models_between_tasks[model][task_half] = np.concatenate([all_models_dict[model][key] for key in model_dict_sorted_keys], 1)
                # THEN DO IT ALL AGAIN FOR THE NEXT TASK HALF
            
    # this one has to be done between task halves, so it's outside of this loop.
    if RDM_version == '11': # I have to work on this one further for the replay analysis (temporal + spatial)
        models_between_tasks = mc.analyse.analyse_MRI_behav.similarity_of_tasks(configs_dict)
    
        
    # then, in a last step, create the RDMs
    # concatenate the conditions from the two task halves (giving you 2*nCond X nVoxels matrix), 
    # and calculate the correlations between all rows of this matrix. This gives you a symmetric matrix 
    # (of size 2*nCond X 2*nCond), where the (non-symmetric) nCond X nCond bottom left square (or top right, 
    # doesn't matter because it's symmetric) (i.e. a quarter of the original matrix) has all the correlations 
    # across THs. 
    
    RSM_dict_betw_TH = {}
    for model in models_between_tasks:
        if model == 'reward_midnight_count':
            RSM_dict_betw_TH[model] = mc.simulation.RDMs.within_task_RDM(np.concatenate((models_between_tasks[model]['1'], models_between_tasks[model]['2'])), plotting = False, titlestring= model)
        else:
            RSM_dict_betw_TH[model] = mc.simulation.RDMs.within_task_RDM(np.concatenate((models_between_tasks[model]['1'], models_between_tasks[model]['2']),1), plotting = False, titlestring= model)
        # mc.simulation.predictions.plot_without_legends(RSM_dict_betw_TH[model])
    

    # then average the lower triangle and the top triangle of this nCond x nCond matrix, 
    # by adding it to its transpose, dividing by 2, and taking only the lower or 
    # upper triangle of the result.    
    corrected_RSM_dict = {}
    for model in RSM_dict_betw_TH:
        corrected_model = (RSM_dict_betw_TH[model] + np.transpose(RSM_dict_betw_TH[model]))/2
        corrected_RSM_dict[model] = corrected_model[0:int(len(corrected_model)/2), int(len(corrected_model)/2):]
   

    # just for me. what happens if I add the ['reward_location', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']?
    # addition_model = corrected_RSM_dict['reward_location'] + corrected_RSM_dict['one_future_rew_loc'] + corrected_RSM_dict['two_future_rew_loc'] + corrected_RSM_dict['three_future_rew_loc'] 

    if fmriplotting:
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        mc.simulation.RDMs.plot_RDMs(corrected_RSM_dict, len(configs), RDM_dir, model_dict_sorted_keys)
    
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
        for RDM in corrected_RSM_dict:
            np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_both_halves"), corrected_RSM_dict[RDM])
            
        # also save the regression files
        for model in models_between_tasks:
            np.save(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves"), np.concatenate((models_between_tasks[model]['1'],models_between_tasks[model]['2']),1))
