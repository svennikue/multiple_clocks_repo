#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:17:11 2025

@author: Svenja Küchenhoff

use behavioural data to create conditions for data RDMs 
i.e. simulate models based on behaviour, then use regressors to bring 
in correct shape to create model RDMs


the logic is to per task half go through each grid-configurations, and per
repeat simulate how neurons of the respective model would fire, given the 
trajectory of this specific run, and the reward locations of this current grid.

it's a nested loop that goes: task half > grid > repeat > model



there are a bunch of settings I will add here, but start with
RDM settings (creating the representations):
    03-1 -> modelling only reward rings + split ‘clocks model’ = just rotating the reward location around.  

GLM ('regression') settings (creating the 'bins'):
    03-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)


"""
import os
import numpy as np
import pandas as pd
import mc


def load_behaviour(subject_idx, task_half):
    sub=f"sub-{subject_idx:02}"
    # check if on server or local
    data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
    if os.path.isdir(data_dir_beh):
        print(f"Running on laptop, now loading subject {sub}, task half {task_half}")
    else:
        data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
        print(f"Running on Cluster, setting {data_dir_beh} as data directory")

    behaviour = pd.read_csv(data_dir_beh + f"{sub}_beh_clean_fmri_pt{task_half}.csv")
    
    return behaviour





# 
#
# these are all the things I extracted last time. Do I need them all??
# create a dictionnary with all future regressors, to make sure the names are not messed up.
def extract_behavioural_variables(df):
    import pdb; pdb.set_trace()
    # these are the combinations of the task (1 of 10) 
    # plus subpath or reward
    # plus state A,B,C, or D.
    # -> 10*2*4 unique conditions.
    time_bin_types = df['time_bin_type'].dropna().unique()
    
    regressors = {}
    for time_bin_type in time_bin_types:
        regressors[time_bin_type] = []
       
    configs = df['config_type'].dropna().unique()
    
    # initialise all dictionaries
    walked_path, timings, rew_list, rew_timing, rew_index, subpath_after_steps = {}, {}, {}, {}, {}, {}
    steps_subpath_alltasks, keys_executed, keys_not_exe, timings_not_exe = {}, {}, {}, {}
     # and all lists per dictionary
    for config in configs:
        walked_path[config], keys_executed[config], timings[config], rew_list[config] = [], [], [], []
        rew_timing[config], rew_index[config], subpath_after_steps[config], steps_subpath_alltasks[config] = [], [], [], []
        keys_not_exe[config], timings_not_exe[config] = [], []
    
    
    for index, row in df.iterrows():
        # iterate through rows of the dataframe.
        # each row reflects a change in position on the screen: either because
        # a new task started, or because the participant pressed a button and moved.
        task_config = row['config_type']
        time_bin_type = row['time_bin_type']
        
        #iterate through the regression dictionary first
        for key in regressors.keys():
            # depending on which 'time bin type' it is, 
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
            keys_executed[task_config].append([row['curr_key']])
            
            # check in case a key had been pressed that wasn't executed
            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                keys_not_exe[task_config].append([row['non-exe_key']])
                timings_not_exe[task_config].append([row['non-exe_key_time']])
        
        # if this is just a normal walking field
        elif not np.isnan(row['t_step_press_global']): # always except if this is reward D 
            # if its reward D, then it will be covered by the first if: if not np.isnan(row['next_task']): 
            timings[task_config].append(df.at[index - 1, 't_step_press_global'])  # Extract value from index-1
            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
            keys_executed[task_config].append([row['curr_key']])
            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                keys_not_exe[task_config].append([row['non-exe_key']])
                timings_not_exe[task_config].append([row['non-exe_key_time']])
       
        
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
                            keys_executed[task_config].append([row['curr_key']])
                            timings[task_config].append(df.at[index -1, 't_reward_start'])
                            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                                keys_not_exe[task_config].append([row['non-exe_key']])
                                timings_not_exe[task_config].append([row['non-exe_key_time']])
                            
                    else:
                        # however also add these fields if this is the very last reward!
                        if row['repeat'] == 4:
                            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            keys_executed[task_config].append([row['curr_key']])
                            timings[task_config].append(df.at[index -1, 't_step_press_global'])
                            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                                keys_not_exe[task_config].append([row['non-exe_key']])
                                timings_not_exe[task_config].append([row['non-exe_key_time']])
                            
                else:
                    rew_index[task_config].append(len(walked_path[task_config])-1) 
            else:
                continue
    
    return configs, rew_list, rew_index, walked_path, steps_subpath_alltasks, subpath_after_steps, timings, regressors, keys_executed, keys_not_exe, timings_not_exe
    

#
#
# ok I am going to try something: make all of this a dictionary
# 
#
# these are all the things I extracted last time. Do I need them all??
# create a dictionnary with all future regressors, to make sure the names are not messed up.
def extract_behavioural_variables_dict(df):
    import pdb; pdb.set_trace()
    # these are the combinations of the task (1 of 10) 
    # plus subpath or reward
    # plus state A,B,C, or D.
    # -> 10*2*4 unique conditions.
    time_bin_types = df['time_bin_type'].dropna().unique()
    regressors = {}
    for time_bin_type in time_bin_types:
        regressors[time_bin_type] = []
      
    behav = {}
    configs = df['config_type'].dropna().unique()
    
    # for each grid, I want to extract behavioural data about:
    list_of_vars = ['walked_path', 'timings', 'rew_list', 'rew_timing', 'rew_index', 'subpath_after_steps', 
                    'steps_subpath_alltasks', 'keys_executed', 'keys_not_exe', 'timings_not_exe']
    
    for config in configs:
        behav[config] = {}
        for var in list_of_vars:
            behav[config][var] = []
  
    # iterate through rows of the dataframe.
    # each row reflects a change in position on the screen: either because
    # a new task started, or because the participant pressed a button and moved.
    for index, row in df.iterrows():
        # To later average across repeats within the same task condition, 
        # create regressors that are the same length within one configuration,
        # and set it to 1 only for the condition the current row belongs to
        for key in regressors.keys():
            # depending on which 'time bin type' it is, 
            # check if the key starts with the task_config value
            if key.startswith(row['config_type']):
                if row['time_bin_type'] == key:
                    regressors[key].append(1)
                elif pd.notna(row['time_bin_type']):
                    regressors[key].append(0) 
           
                    
        # in case a new task has just started
        if not np.isnan(row['next_task']): 
            # first check if this is the first task of several repeats.
            if (index == 0) or (row['config_type'] != df.at[index -1, 'config_type']):
                behav[row['config_type']]['timings'].append(row['next_task'])
            else: # if it isnt, then take the reward start time from last rew D as start field.
                behav[row['config_type']]['timings'].append(df.at[index -1, 't_step_press_global'])
            walked_path[row['config_type']].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
            keys_executed[row['config_type']].append([row['curr_key']])
            
            # check in case a key had been pressed that wasn't executed
            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                keys_not_exe[row['config_type']].append([row['non-exe_key']])
                timings_not_exe[row['config_type']].append([row['non-exe_key_time']])
        
        # if this is just a normal walking field
        elif not np.isnan(row['t_step_press_global']): # always except if this is reward D 
            # if its reward D, then it will be covered by the first if: if not np.isnan(row['next_task']): 
            behav[row['config_type']]['timings'].append(df.at[index - 1, 't_step_press_global'])  # Extract value from index-1
            walked_path[row['config_type']].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
            keys_executed[row['config_type']].append([row['curr_key']])
            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                keys_not_exe[row['config_type']].append([row['non-exe_key']])
                timings_not_exe[row['config_type']].append([row['non-exe_key_time']])
       
        
        # next check if its a reward field
        if not np.isnan(row['rew_loc_x']): # if this is a reward field.
            # check if this is either at reward D(thus complete) or ignore interrupted trials
            # ignore these as they are not complete.
            if (index+2 < len(df)) or (row['state'] == 'D'):
                rew_timing[row['config_type']].append(row['t_reward_start'])
                rew_list[row['config_type']].append([row['curr_rew_x_coord'], row['curr_rew_y_coord']])
                subpath_after_steps[row['config_type']].append(int(index-row['repeat']))  
                if row['state'] == 'D':
                    rew_index[task_config].append(len(walked_path[task_config])) #bc step has not been added yet
                    # if this is the last run of a task
                    if (index+2 < len(df)):
                        # first check if there are more tasks coming after, otherwise error
                        if (row['config_type'] != df.at[index +1, 'config_type']):
                            walked_path[row['config_type']].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            keys_executed[row['config_type']].append([row['curr_key']])
                            behav[row['config_type']]['timings'].append(df.at[index -1, 't_reward_start'])
                            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                                keys_not_exe[row['config_type']].append([row['non-exe_key']])
                                timings_not_exe[row['config_type']].append([row['non-exe_key_time']])
                            
                    else:
                        # however also add these fields if this is the very last reward!
                        if row['repeat'] == 4:
                            walked_path[row['config_type']].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            keys_executed[row['config_type']].append([row['curr_key']])
                            timings[row['config_type']].append(df.at[index -1, 't_step_press_global'])
                            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                                keys_not_exe[row['config_type']].append([row['non-exe_key']])
                                timings_not_exe[row['config_type']].append([row['non-exe_key_time']])
                            
                else:
                    rew_index[row['config_type']].append(len(walked_path[row['config_type']])-1) 
            else:
                continue
    
    return behav


#
#

    

def simulate_models_for_one_sub(sub, RDM_version, glm_version):
    for th in [1,2]:
        beh_df = load_behaviour(sub, th) # first load behavioural table
        # configs, rew_list, rew_index, walked_path, steps_subpath_alltasks, subpath_after_steps, timings, regressors, keys_executed, keys_not_exe, timings_not_exe = extract_behavioural_variables(beh_df)
        beh_dict = extract_behavioural_variables_dict(beh_df)
        
        
    
    
    
    
    
# # if running from command line, use this one!   
# if __name__ == "__main__":
#     #print(f"starting regression for subject {sub}")
#     fire.Fire(clean_behaviour_for_sub)
#     # call this script like, replacing 
#     # python clean_behaviour_csv.py 5


if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    simulate_models_for_one_sub(
        sub=2,
        RDM_version='03-1',
        # which model RDMs to generate
        glm_version='03-4'
        # GLM ('regression') settings (creating the conditions):
    )