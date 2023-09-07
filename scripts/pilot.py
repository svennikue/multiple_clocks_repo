#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:33:36 2023

@author: xpsy1114
here, I am having a look a the pilot data.

"""

import pandas as pd
import numpy as np
import mc
import matplotlib.pyplot as plt

dataDir = '/Users/xpsy1114/Documents/projects/multiple_clocks/experiment/piloting_data/'


# filenames = ['00_pilot.csv', '01_pilot.csv', '02_pilot.csv', '03_pilot.csv', '04_pilot.csv', '05_pilot.csv', '06_pilot.csv', '07_pilot.csv', ]
filenames = ['05_pilot.csv', '06_pilot.csv', '07_pilot.csv', ]


for file in filenames:
    df = pd.read_csv(dataDir+file)
    
    #drop some unnecessary columns
    to_drop = ['rep_runs.thisRepN', 'rep_runs.thisTrialN', 'rep_runs.thisIndex', 'sand_box.started', 'sand_box.stopped', 'foot.started', 'foot.stopped']
    df.drop(to_drop, inplace=True, axis = 1)
    
    
    # my goals are: 
        # 1. plot how long every task takes
        # 2. plot how long every subpath takes
        # 3. look at the interaction between fields and pathlength -> which paths are easiest?
        # 4. check if they tend to walk the same routes
        
        
    # no 1. How long does every task take?
    # look at 'round_no' for the same task 
    # look at 'start_ABCD_screen' to compute how long each one took
    
    # Initialize an empty list to store the indices
    index_next_task = []
    
    # Iterate through the DataFrame and collect the indices where the column is not empty
    for index, row in df.iterrows():
        if not pd.isna(row['start_ABCD_screen']):
            index_next_task.append(index)
    

    for i, index in enumerate(index_next_task):
        if i+1 < len(index_next_task):
            df.at[index, 'task_length'] = df.at[index_next_task[i+1] - 1 , 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen']   
            if 'type' in df.columns:
                df.at[index, 'type'] = df.at[index+ 1, 'type']
        
    df_clean = df.dropna(subset = ['round_no'])
    
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
    
    plt.xlabel('Repeats')
    plt.ylabel('Time in seconds')
    plt.legend()
    plt.title(f'Task Length by Task Config for {file}')
    plt.show()


