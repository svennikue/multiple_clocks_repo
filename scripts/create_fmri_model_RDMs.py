#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:34:17 2023

extract behaviour such that I can create my model RDMs for the fMRI data.


@author: xpsy1114
"""

import pandas as pd
import os
import numpy as np
import mc
import matplotlib.pyplot as plt
import pickle


#import pdb; pdb.set_trace()
# subjects = ['sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18','sub-19', 'sub-20',  'sub-22', 'sub-23','sub-24']

subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
task_halves = ['1', '2']
#subjects = ['sub-01']
# task_halves = ['1']
RDM_version = '04' # 04 res 10, more ordered stuff in code with dicts 11.12. # 03 is with resolution 1, no continuational model '02' # version 02 = with temp_res = 10
regression_version = '06'

fmriplotting = False
fmriplotting_debug = False
fmri_save = True

temporal_resolution = 10



for sub in subjects:
    for task_half in task_halves:
        data_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
        data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        file = f"{sub}_fmri_pt{task_half}"
        file_all = f"{sub}_fmri_pt{task_half}_all.csv"
        
        # load regressors as those used in fMRI.
        # 11.12.: I don't think this is actually useful as this is a different
        # temporal resolution. In theory, the regression should harmonise this issue.
        # EV_folder = f'/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/EVs_{regression_version}_pt0{task_half}/'
        # with open(f"{EV_folder}my_EV_dict", 'rb') as f:
        #     regressor_dict = pickle.load(f)
        
        #import pdb; pdb.set_trace()
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
        

        # create a dictionnary with all future regressors, to make sure 
        # the names are not messed up.
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
        
        # import pdb; pdb.set_trace()
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


        # finally, create simulations and time-bin per run.
        # first, prep result dictionaries.
        clocks_dict = {key: "" for key in configs}
        midnight_dict = {key: "" for key in configs}
        location_dict = {key: "" for key in configs}
        phase_dict = {key: "" for key in configs}
        state_dict = {key: "" for key in configs}
        # loop through all tasks, and within the tasks, through all runs.
        # import pdb; pdb.set_trace()
        for config in configs:
            # import pdb; pdb.set_trace()
            print(f"the config is {rew_list[config]} for {config}")
            # select complete trajectory of current task.
            trajectory = walked_path[config]
            trajectory = [[int(value) for value in sub_list] for sub_list in trajectory]
            
            # double check how this is used!!
            timings_curr_run = timings[config]
            
            # select file that shows step no per subpath
            step_number = steps_subpath_alltasks[config]
            step_number = [[int(value) for value in sub_list] for sub_list in step_number]
            # make file that shows cumulative steps per subpath
            cumsteps_per_run = [np.cumsum(task)[-1] for task in step_number]
            cumsteps_task = np.cumsum(cumsteps_per_run)
            
            # then start looping through each subpath within one task
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
                # DOUBLE CHECK HOW THE 'TIMINGS' ONE IS USED!!!
                location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, temporal_resolution = temporal_resolution, plot=False)
                
                # these need to be concatenated for each run and task
                if no_run == 0:
                    clocks_repeats = clocks_model.copy()
                    midnight_repeats = midnight_model.copy()
                    location_repeats = location_model.copy()
                    phase_repeats = phase_model.copy()
                    state_repeats = state_model.copy()
                else:
                    clocks_repeats = np.concatenate((clocks_repeats, clocks_model), 1)
                    midnight_repeats = np.concatenate((midnight_repeats, midnight_model), 1)
                    location_repeats = np.concatenate((location_repeats, location_model), 1)
                    phase_repeats = np.concatenate((phase_repeats, phase_model), 1)
                    state_repeats = np.concatenate((state_repeats, state_model), 1)
            
            # NEXT STEP: prepare the regression.
            # then select the correct regressors, filter keys starting with 'A1_backw'
            regressors_curr_task = {key: value for key, value in regressors.items() if key.startswith(config)}
            
            # check that all regressors have the same length in case the task wasn't completed.
            if len(subpath_after_steps[config]) < 20:
                # if I cut the task short, then also cut the regressors short.
                for reg_type, regressor_list in regressors_curr_task.items():
                # Truncate the list if its length is greater than the maximum length
                    regressors_curr_task[reg_type] = regressor_list[:(np.shape(clocks_repeats)[1])]
            # Ensure all lists have the same length

            
            list_lengths = set(len(value) for value in regressors_curr_task.values())
            if len(list_lengths) != 1:
                raise ValueError("All lists must have the same length.")
            # sort it alphabetically.
            sorted_regnames_curr_task = sorted(regressors_curr_task.keys())
            # Create a list of lists sorted by keys
            sorted_regs = [regressors_curr_task[key] for key in sorted_regnames_curr_task]
            regressors_matrix = np.array(sorted_regs)
            if fmriplotting_debug:
                plt.figure(); plt.imshow(regressors_matrix, aspect = 'auto')
            
            
            # instead of concatenating these, I will save them in a new dictionary.
            # this will then later allow me to create the big matrices, but in alphabetical order!
            clocks_dict[config] = mc.simulation.predictions.transform_data_to_betas(clocks_repeats, regressors_matrix)
            midnight_dict[config] = mc.simulation.predictions.transform_data_to_betas(midnight_repeats, regressors_matrix)
            location_dict[config] = mc.simulation.predictions.transform_data_to_betas(location_repeats, regressors_matrix)
            phase_dict[config] = mc.simulation.predictions.transform_data_to_betas(phase_repeats, regressors_matrix)
            state_dict[config] = mc.simulation.predictions.transform_data_to_betas(state_repeats, regressors_matrix)
     
        
        # then, create the location_between etc. matrix by concatenating the keys of each dict by alphabeticla order
        model_dict_sorted_keys = sorted(configs)
        print(f"I am sorting in this order: {model_dict_sorted_keys}")
        clocks_between  = np.concatenate([clocks_dict[key] for key in model_dict_sorted_keys], 1)
        midnight_between  = np.concatenate([midnight_dict[key] for key in model_dict_sorted_keys], 1)
        location_between  = np.concatenate([location_dict[key] for key in model_dict_sorted_keys], 1)
        phase_between  = np.concatenate([phase_dict[key] for key in model_dict_sorted_keys], 1)
        state_between  = np.concatenate([state_dict[key] for key in model_dict_sorted_keys], 1)


        # then, in a last step, create the RDMs
        RSM_dict = {}
        RSM_dict['Location'] = mc.simulation.RDMs.within_task_RDM(location_between, plotting = False, titlestring = 'Location RDM')
        RSM_dict['Schema'] = mc.simulation.RDMs.within_task_RDM(clocks_between, plotting = False, titlestring = 'Clock RDM')
        RSM_dict['Partial Schema'] = mc.simulation.RDMs.within_task_RDM(midnight_between, plotting = False, titlestring = 'Midnight RDM')
        RSM_dict['Subgoal Progress']= mc.simulation.RDMs.within_task_RDM(phase_between, plotting = False, titlestring = 'Phase RDM')
        RSM_dict['State']= mc.simulation.RDMs.within_task_RDM(state_between, plotting = False, titlestring= 'State RDM')
        
        
        if fmriplotting:
            save_in_dir = f"{data_dir}derivatives/{sub}/beh/RDMs_{RDM_version}"
            mc.simulation.RDMs.plot_RDMs(RSM_dict, len(configs), save_in_dir)
        
            # make my own correlation matrix.
            # Schema - Partial Schema - Subgoal Progress - Location - State
            intercorr_RDM_dict = {}
            corr_RDMs = np.empty((len(RSM_dict),len(RSM_dict)))
            for x, RDM_one in enumerate(RSM_dict):
                for y, RDM_two in enumerate(RSM_dict):
                    if y == 0:
                        tick_string = [RDM_two]
                    else:
                        tick_string.append(RDM_two)
                    corr_RDMs[x,y] = mc.simulation.RDMs.corr_matrices_pearson(RSM_dict[RDM_one], RSM_dict[RDM_two])[0][1]               
            intercorr_RDM_dict['correlation_try_two'] = corr_RDMs
            mc.simulation.RDMs.plot_RDMs(intercorr_RDM_dict, 5, save_in_dir, string_for_ticks = tick_string)       
            
            
        # import pdb; pdb.set_trace()    
        if fmri_save: 
            # then save these matrices.
            RDM_dir = f"{data_dir}derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
            if not os.path.exists(RDM_dir):
                os.makedirs(RDM_dir)
            for RDM in RSM_dict:
                np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_pt{task_half}"), RSM_dict[RDM])
                
            # also save the regression files
            np.save(os.path.join(RDM_dir, f"data_location_{sub}_fmri_pt{task_half}"), location_between)
            np.save(os.path.join(RDM_dir, f"data_clock_{sub}_fmri_pt{task_half}"), clocks_between)
            np.save(os.path.join(RDM_dir, f"data_midnight_{sub}_fmri_pt{task_half}"), midnight_between)
            np.save(os.path.join(RDM_dir, f"data_phase_{sub}_fmri_pt{task_half}"), phase_between)
            np.save(os.path.join(RDM_dir, f"data_state_{sub}_fmri_pt{task_half}"), state_between)
        
