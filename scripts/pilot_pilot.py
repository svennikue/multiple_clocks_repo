#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:33:36 2023

@author: xpsy1114
here, I am having a look a the pilot data.
# my goals are: 
# 1. plot how long every task takes
# 2. plot how long every subpath takes
# 3. look at the interaction between fields and pathlength -> which paths are easiest?
# 4. check if they tend to walk the same routes

"""

import pandas as pd
import numpy as np
import mc
import matplotlib.pyplot as plt

data_dir = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/behaviour/piloting_data/'


# filenames = ['00_pilot.csv', '01_pilot.csv', '02_pilot.csv', '03_pilot.csv', '04_pilot.csv', '05_pilot.csv', '06_pilot.csv', '07_pilot.csv', ]
#filenames = ['05_pilot.csv', '06_pilot.csv', '07_pilot.csv', '08_pilot.csv', '09_pilot.csv']
filenames = ['01_practice.csv', '01_MRI_pt1.csv', '01_MRI_pt2.csv']

# no 1. How long does every task take?
for file in filenames:
    df = pd.read_csv(data_dir+file)
    
    #drop some unnecessary columns
    to_drop = ['rep_runs.thisRepN', 'rep_runs.thisTrialN', 'rep_runs.thisIndex', 'sand_box.started', 'sand_box.stopped', 'foot.started', 'foot.stopped']
    df.drop(to_drop, inplace=True, axis = 1)

    
    # identify where the next task begins by iterating through the DataFrame 
    # and collecting the indices where the column is not empty
    index_next_task = []
    for index, row in df.iterrows():
        if not pd.isna(row['start_ABCD_screen']):
            index_next_task.append(index)
    
    # compute the task length for each task
    for i, index in enumerate(index_next_task):
        if i+1 < len(index_next_task):
            df.at[index, 'task_length'] = df.at[index_next_task[i+1] - 1 , 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen']   
            if 'type' in df.columns:
                df.at[index, 'type'] = df.at[index+ 1, 'type']
        
    df_clean = df.dropna(subset = ['round_no'])


    # plot path lenghts per task configuration
    max_repeats = df['rep_runs.thisN'].max(skipna=True)
    x_axis = np.arange(0, max_repeats, 1)
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    typestr = 'forw'
    for repeat, data in df_clean.groupby('round_no'):
        curr_task = np.empty(int(max_repeats))
        curr_task.fill(np.NaN)
        curr_task[0:len(data['task_length'])] = data['task_length']
        if 'type' in df_clean.columns:
            first_entry = data['type'].first_valid_index()
            typestr = data['type'][first_entry]
        plt.plot(x_axis, curr_task, label=f'Task no. {repeat}, {typestr}')
    
    plt.xticks(x_axis)
    plt.xlabel('Repeats')
    plt.ylabel('Time in seconds')
    plt.legend()
    plt.title(f'Task Length by Task Config for {file}')
    plt.show()

import pdb; pdb.set_trace()

# no 2: plot how long every subpath takes, across participants and separetly per task configuration.
# filenames = ['00_pilot.csv', '01_pilot.csv', '02_pilot.csv', '03_pilot.csv', '04_pilot.csv', '05_pilot.csv', '06_pilot.csv', '07_pilot.csv', ]
# to do that, just look at column 't_step_press_curr_run', filtered for those rows that have a reward.
# plot reward times per subpath by looking at column 'reward_delay'. (not filled for every row, unfortunately...)
# filenames = ['07_pilot.csv']
for file in filenames:
    df = pd.read_csv(data_dir + file)
    
    # identify where the next task begins by iterating through the DataFrame 
    # and collecting the indices where the column is not empty
    index_next_task = []
    for index, row in df.iterrows():
        if not pd.isna(row['start_ABCD_screen']):
            index_next_task.append(index)
            
    index_next_task = index_next_task[1:]
                    
    # identify where the next reward starts by iterating through the DataFrame 
    # and collecting the indices where the column is not empty
    index_next_reward = []
    for index, row in df.iterrows():
        if not pd.isna(row['t_reward_start']):
            index_next_reward.append(index)

    # Update 06.10.23: I don't think I need this anymore, I fixed it in the exp code
    # fill the missing last reward_delay columns.
    # they should be t_reward_afterwait-t_reward_start
    # take every 4th reward index to do so.
    #for i in range(3, len(index_next_reward), 4):
    #   df.at[index_next_reward[i], 'reward_delay'] = df.at[index_next_reward[i], 't_reward_afterwait'] - df.at[index_next_reward[i], 't_reward_start'] 
    
    # fill gaps in the round_no column
    df['round_no'] = df['round_no'].fillna(method='ffill')
    
    # import pdb; pdb.set_trace()
    # create a new column in which you plot how long ever subpath takes (with rew)
    j = 0
    for i, task_index in enumerate(index_next_task):
        while (len(index_next_reward) < j) and (index_next_reward[j] < task_index):
            df.at[index_next_reward[j], 'cum_subpath_length_without_rew'] = df.at[index_next_reward[j], 't_step_press_curr_run'] + df.at[index_next_reward[j]-1, 'length_step'] 
            df.at[index_next_reward[j], 'cum_subpath_length_with_rew'] = df.at[index_next_reward[j], 't_step_press_curr_run'] + df.at[index_next_reward[j]-1, 'length_step'] + df.at[index_next_reward[j], 'reward_delay'] 
            j += 1
        df.at[task_index-1, 'cum_subpath_length_without_rew'] = df.at[task_index-1, 't_step_press_curr_run'] + df.at[task_index-2, 'length_step'] 
        df.at[task_index-1, 'cum_subpath_length_with_rew'] = df.at[task_index-1, 't_step_press_curr_run'] + df.at[task_index-2, 'length_step'] + df.at[task_index-1, 'reward_delay'] 
    
    states = ['A', 'B', 'C', 'D']*len(index_next_task)
    
    # then, write the not- cumulative columns.
    for i, reward_index in enumerate(index_next_reward):
        if i < len(states):
            df.at[reward_index, 'state'] = states[i]
        if i > 0:
            df.at[reward_index, 'subpath_length_without_rew'] = df.at[reward_index, 'cum_subpath_length_without_rew'] - df.at[index_next_reward[i-1], 'cum_subpath_length_without_rew'] - df.at[reward_index, 'reward_delay']
            df.at[reward_index, 'subpath_length_with_rew'] = df.at[reward_index, 'cum_subpath_length_with_rew'] - df.at[index_next_reward[i-1], 'cum_subpath_length_with_rew']

    
    for i in range(0, len(index_next_reward), 4):
        df.at[index_next_reward[i], 'subpath_length_without_rew'] = df.at[index_next_reward[i], 'cum_subpath_length_without_rew'] 
        df.at[index_next_reward[i], 'subpath_length_with_rew'] = df.at[index_next_reward[i], 'cum_subpath_length_with_rew']

    
    #first reduce to only including those rows that have values for rewards.
    df_clean = df.dropna(subset = ['subpath_length_with_rew'])
    
    # group by task config and subpath
    grouped = df_clean.groupby(['round_no', 'state'])['subpath_length_with_rew'].mean().unstack()
    # Plot the bar chart
    ax = grouped.plot(kind='bar', figsize=(10, 6))
    # Customize the plot
    plt.xlabel('Task configuration')
    plt.ylabel('Average subpath lenght (reward included)')
    plt.title('Average subpath length (reward included) per subpath in each task')
    plt.legend(title='States', loc='upper right')
    plt.axhline(5, color='black', linestyle='--', label='Baseline at 5.5 secs')
    
    # Show the plot
    plt.show()
    
    # group by task config and subpath
    grouped_without = df_clean.groupby(['round_no', 'state'])['subpath_length_without_rew'].mean().unstack()
    # Plot the bar chart
    ax = grouped_without.plot(kind='bar', figsize=(10, 6))
    # Customize the plot
    plt.xlabel('Task configuration')
    plt.ylabel('Average subpath lenght (reward excluded)')
    plt.title('Average subpath length (reward excluded) per subpath in each task')
    plt.legend(title='States', loc='upper right')
    plt.axhline(5, color='black', linestyle='--', label='Baseline at 4 secs')


    # Show the plot
    plt.show()
    
    

    
    
