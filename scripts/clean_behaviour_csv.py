#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 16:56:05 2025

@author: Svenja Küchenhoff

take the relevant entries of the behavioural table and only store those that 
are actually required in the remaining analysis.

"""

import mc
import pandas as pd
import numpy as np
import os
import ast


def load_behaviour_for_sub(subject_idx, task_half):
    sub=f"sub-{subject_idx:02}"
    # check if on server or local
    data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
    if os.path.isdir(data_dir_beh):
        print(f"Running on laptop, now subject {sub}, task half {task_half}")
    else:
        data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
        print(f"Running on Cluster, setting {data_dir_beh} as data directory")

    behaviour = pd.read_csv(data_dir_beh + f"{sub}_fmri_pt{task_half}.csv")
    
    return behaviour



def delete_unnecessary_fields(og_df):
    # the first row is empty so delete to get indices right
    og_df = og_df.iloc[1:].reset_index(drop=True)
    df_clean = og_df.copy()

    # drop all unused columns for better readability.
    df_clean = og_df.drop(columns=['rep_runs.thisRepN', 'rep_runs.thisTrialN', 'rep_runs.thisN', 'rep_runs.thisIndex',
                           't_step_end_global', 'sand_box.started', 'sand_box.stopped', 'foot.started', 'foot.stopped',
                           'reward.started', 'reward.stopped', 'TR_key.keys', 'TR_key.rt', 'TR_key.started',
                           'nav_key_task.stopped', 'break_key.keys', 'break_key.started', 'break_key.stopped', 
                           'progressbar_background.started', 'progressbar_background.stopped', 'progress_bar.started', 
                           'progress_bar.stopped', 'reward_progress.started', 'reward_progress.stopped', 
                           'plus_coin_txt.started', 'plus_coin_txt.stopped', 'reward_A_feedback.started', 
                           'reward_A_feedback.stopped', 'TR_key.stopped', 'participant', 'date'])
    
    return df_clean
 
    
 
    

def add_fields_I_want(df):
    # fill gaps
    df['round_no'] = df['round_no'].fillna(method='ffill')
    df['task_config'] = df['task_config'].fillna(method='ffill')
    df['repeat'] = df['repeat'].fillna(method='ffill')
    # so that I cann differenatiate task config and direction
    df['config_type'] = df['task_config'] + '_' + df['type']

    
    # per grid, the navigation keys are stored in the first column
    # thus, these can be used as indices of where a new grid starts.
    # there will be 10 different tasks per TH, with 5 repeats.
    indices_with_nav_keys = df[df['nav_key_task.started'].notna()].index.to_list()
    
    
    # loop to add a colum with nav_key presses that counted based on nav_key_task.rt and t_step_press_curr_run
    # and one with the actual keys they pressed based on nav_key_task.keys and t_step_press_curr_run
    # and one with keys presssed, but never executed.
    for grid_no, row_index in enumerate(indices_with_nav_keys):
        curr_list_of_keys = ast.literal_eval(df.at[row_index, 'nav_key_task.keys'])
        curr_key_times = ast.literal_eval(df.at[row_index, 'nav_key_task.rt'])
        count_error_keys = 0
        overall_error_counter = 0
        
        if grid_no == 0:
            # import pdb; pdb.set_trace()
            for i in range(0, indices_with_nav_keys[grid_no]):
                # if the data stored a value smaller than t = 0, correct that
                if round(df.at[i, 't_step_press_curr_run'],3) < 0:
                    curr_key_times = np.insert(curr_key_times, 0, 0)
                    curr_list_of_keys = np.insert(curr_list_of_keys, 0, 0)
                    df.at[i, 't_step_press_curr_run'] = 0  
                # next, track which button was pressed. It is possible to press more buttons than
                # actually are executed (only the button that was pressed last is executed)
                # thus, check if the button press is aligned with the time the subject moved
                if round(df.at[i, 't_step_press_curr_run'],3) == round(curr_key_times[i + overall_error_counter],3):
                    count_error_keys = 0
                    df.at[i, 'curr_key'] = curr_list_of_keys[i]
                    df.at[i, 'curr_key_time'] = curr_key_times[i]
                else:
                    wrong_keys = [str(curr_list_of_keys[i + overall_error_counter])]
                    wrong_times = [str(round(curr_key_times[i + overall_error_counter],4))]
                    count_error_keys += 1
                    overall_error_counter += 1
                    
                    while round(df.at[i, 't_step_press_curr_run'],3) != round(curr_key_times[i + overall_error_counter],3) :
                        wrong_keys.append(str(curr_list_of_keys[i + overall_error_counter]))
                        wrong_times.append(str(round(curr_key_times[i + overall_error_counter], 4)))
                        count_error_keys += 1
                        overall_error_counter +=1
                    
                    # if these columns don't exist yet, there will be an error if I try to fill with
                    # several items. instead, first create with 0, then fill.
                    df.at[i, 'non-exe_key_time'] = 0
                    df.at[i, 'non-exe_key'] = 0
                    
                    # once back to a correct key, fill in the one that you missed previously
                    df.at[i, 'non-exe_key'] = wrong_keys
                    df.at[i, 'non-exe_key_time'] = wrong_times
                    
                    df.at[i, 'curr_key'] = curr_list_of_keys[i + overall_error_counter]
                    df.at[i, 'curr_key_time'] = curr_key_times[i + overall_error_counter]
                    df.at[i, 'non-exe_key_counter'] = count_error_keys
                    count_error_keys = 0        
    
        elif grid_no > 0:               
            for i_list,i in enumerate(range(indices_with_nav_keys[grid_no-1]+1, indices_with_nav_keys[grid_no])): 
                # for some sad reason, there are some (rare) glitches in the behavioural tables.
                # one glitch is that the first time of t_step_press_curr_run is shorter than 0
                if round(df.at[indices_with_nav_keys[grid_no-1]+1, 't_step_press_curr_run'],3) <= 0:
                    curr_key_times = np.insert(curr_key_times, 0, 0)
                    curr_list_of_keys = np.insert(curr_list_of_keys, 0, 0)
                    df.at[indices_with_nav_keys[grid_no-1]+1, 't_step_press_curr_run'] = 0
                # another glitch is that the first time of t_step_press_curr_run is even later than the last recorded press of this task
                if round(df.at[indices_with_nav_keys[grid_no-1]+1, 't_step_press_curr_run'],3) > round(df.at[indices_with_nav_keys[grid_no]-1, 't_step_press_curr_run'],3):
                    df.at[indices_with_nav_keys[grid_no-1]+1, 't_step_press_curr_run'] = curr_key_times[i_list]
                # another glitch is that there is a negative time somewhere in the middle of the task
                if round(df.at[i, 't_step_press_curr_run'],3) < 0:
                    df.at[i, 't_step_press_curr_run'] = curr_key_times[i_list]
    
                # then, test for what I am actually interested in:
                    # which of the key presses was the recorded one?
                if round(df.at[i, 't_step_press_curr_run'],3) == round(curr_key_times[i_list + overall_error_counter],3):
                    df.at[i, 'curr_key'] = curr_list_of_keys[i_list + overall_error_counter]
                    df.at[i, 'curr_key_time'] = curr_key_times[i_list + overall_error_counter]
                else:
                    wrong_keys = [str(curr_list_of_keys[i_list + overall_error_counter])]
                    wrong_times = [str(round(curr_key_times[i_list + overall_error_counter],4))]
                    count_error_keys += 1
                    overall_error_counter += 1
                    
                    while round(df.at[i, 't_step_press_curr_run'],3) != round(curr_key_times[i_list + overall_error_counter],3) :
                        wrong_keys.append(str(curr_list_of_keys[i_list + overall_error_counter]))
                        wrong_times.append(str(round(curr_key_times[i_list + overall_error_counter], 4)))
                        count_error_keys += 1
                        overall_error_counter +=1
                    
                    # if these columns don't exist yet, there will be an error if I try to fill with
                    # several items. instead, first create with 0, then fill.
                    df.at[i, 'non-exe_key_time'] = 0
                    df.at[i, 'non-exe_key'] = 0
                    
                    # once back to a correct key, fill in the one that you missed previously
                    df.at[i, 'non-exe_key'] = wrong_keys
                    df.at[i, 'non-exe_key_time'] = wrong_times
                    
                    df.at[i, 'curr_key'] = curr_list_of_keys[i_list + overall_error_counter]
                    df.at[i, 'curr_key_time'] = curr_key_times[i_list + overall_error_counter]
                    df.at[i, 'non-exe_key_counter'] = count_error_keys
                    count_error_keys = 0
           
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
    
    return df
    


def save_processed_behavioural_csv(df, subject_idx, th):
    sub=f"sub-{subject_idx:02}"
    # check if on server or local
    data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
    if not os.path.isdir(data_dir_beh):
        data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
    
    print(f"Now storing processed data in {data_dir_beh}/{sub}_beh_clean_fmri_pt{th}.csv")

    df.to_csv(data_dir_beh + f"{sub}_beh_clean_fmri_pt{th}.csv")

    
    


def clean_behaviour_for_sub(sub):
    for th in [1,2]:
        # First load
        raw_df = load_behaviour_for_sub(sub, task_half=th)
        # second, delete all that isn't necessary
        df_cleaned = delete_unnecessary_fields(raw_df)
        # third, add fields I am going to make use of later
        df_completed = add_fields_I_want(df_cleaned)
        # fourth, store the new csv file for later use.
        save_processed_behavioural_csv(df_completed, sub, th)
        
        

    
# # if running from command line, use this one!   
# if __name__ == "__main__":
#     #print(f"starting regression for subject {sub}")
#     fire.Fire(clean_behaviour_for_sub)
#     # call this script like, replacing 
#     # python clean_behaviour_csv.py 5


if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    clean_behaviour_for_sub(
        sub=2
    )