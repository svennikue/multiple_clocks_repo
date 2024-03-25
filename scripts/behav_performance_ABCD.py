#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:25:31 2024

@author: Svenja Kuchenhoff

This script analyses the behaviour in the multiple clocks task, based on the
table that is output from the taks during the fMRI.

- compares performance (time) between task configurations
- compares performance (time) between runs within task configurations
- compares performance (time) between participants
"""

import pandas as pd
import os
import numpy as np
import mc
import matplotlib.pyplot as plt
import pickle
import re
import sys
import seaborn as sns


#import pdb; pdb.set_trace()

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'
    
    
#subjects = [f"sub-{subj_no}"]
# subj 21 is an outlier.
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18','sub-19', 'sub-20', 'sub-22', 'sub-23','sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28', 'sub-29', 'sub-30', 'sub-31', 'sub-32', 'sub-33', 'sub-34']
#subjects = ['sub-01']


# to debug task_halves = ['1']
task_halves = ['1', '2']

plotting = False


average_task_length = {}
sorted_df = {}


# Custom sort function
def custom_sort_key(config_type):
    # Sort by suffix ('forw' or 'backw'), then by the rest of the string
    if config_type.endswith('forw'):
        return (0, config_type)
    elif config_type.endswith('backw'):
        return (1, config_type)
    else:
        return (2, config_type)
            
            

for i, sub in enumerate(subjects):
    for task_half in task_halves:
        data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        funcDir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func"
        analysisDir = "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
        if os.path.isdir(data_dir_beh):
            print("Running on laptop.")
        else:
            data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
            funcDir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/func"
            analysisDir = "/home/fs0/xpsy1114/scratch/analysis"
            print(f"Running on Cluster, setting {data_dir_beh} as data directory")

        file = f"{sub}_fmri_pt{task_half}"
        file_all = f"{sub}_fmri_pt{task_half}_all.csv"
        

        # load behavioural file
        df = pd.read_csv(data_dir_beh + f"{file}.csv")
        df_analysed, df_clean = mc.analyse.analyse_MRI_behav.analyse_pathlength_beh(df)
        
        df['config_type'] = df['task_config'] + '_' + df['type']
        df['config_type'] = df['config_type'].fillna(method='ffill')
        
        # identify all task configurations and states there are
        task_names = df['config_type'].dropna().unique().tolist()
        state_names = df['state'].dropna().unique().tolist()
        
        # for i, task in enumerate(task_names):
        #     for s, state in enumerate(state_names):
        #         partial_df = df[((df['config_type'] == task) & (df['state'] == state))]
        #         import pdb; pdb.set_trace()
        

        # - between task: were there tasks that were longer 
        # Filter out rows where 'task_length' is NaN or non-numeric
        df_taskwise = df[pd.to_numeric(df['task_length'], errors='coerce').notna()]
        # Calculate the average task_length for each config_type
        if task_half == '1':
            task_half_one = df_taskwise.groupby('config_type')['task_length'].mean().reset_index()
        elif task_half == '2':
            task_half_two = df_taskwise.groupby('config_type')['task_length'].mean().reset_index()
            average_task_length[sub] = pd.concat([task_half_one, task_half_two])
        
        # for the within-task-across-runs analysis, create one mega df
        if task_half == '1' and i == 0:
            mega_df_one = df_analysed[['task_length', 'rep_runs.thisN', 'config_type']]
            mega_df = df_analysed[['task_length', 'rep_runs.thisN', 'config_type']]
        elif task_half == '2' and i == 0:
            mega_df_two = df_analysed[['task_length', 'rep_runs.thisN', 'config_type']]
        elif i > 0:
            if task_half == '1':
                new_partial_df_one = df_analysed[['task_length', 'rep_runs.thisN', 'config_type']]
                mega_df_one = pd.concat([mega_df_one, new_partial_df_one])
            if task_half == '2':
                new_partial_df_two = df_analysed[['task_length', 'rep_runs.thisN', 'config_type']]
                mega_df_two = pd.concat([mega_df_two, new_partial_df_two])
            new_partial_df = df_analysed[['task_length', 'rep_runs.thisN', 'config_type']]
            mega_df = pd.concat([mega_df, new_partial_df])
    # Sort the DataFrame
    sorted_df[sub] = average_task_length[sub].sort_values(by='config_type', key=lambda x: x.map(custom_sort_key))
      


# do the within-task-across-runs analysis

# for both task halves together.
# Calculate the average task_length for each combination of rep_runs.thisN and config_type
average_df = mega_df.groupby(['rep_runs.thisN', 'config_type'])['task_length'].mean().reset_index()
# Plot settings

# Prepare color palettes
task_names = mega_df['config_type'].dropna().unique().tolist()
colors_backw = sns.color_palette('Blues', n_colors=20)  # Shades of blue
colors_forw = sns.color_palette('Reds', n_colors=20)   # Shades of red
# Define the order of config_types and assign colors
color_map = {config: colors_backw[i] if 'backw' in config else colors_forw[i] for i, config in enumerate(task_names)}


sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Plotting
for config in task_names:
    subset = average_df[average_df['config_type'] == config]
    sns.lineplot(x='rep_runs.thisN', y='task_length', data=subset, label=config, color=color_map[config])

# Adjust plot settings
plt.xticks([0, 1, 2, 3, 4])
plt.title('Average Task Length Across Subjects by Config Type and Repeat')
plt.xlabel('Repeat (rep_runs.thisN)')
plt.ylabel('Average Task Length')
plt.ylim(23,50)
plt.legend(title='Config Type', bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()

plt.show()

# split by task half. task half 1.
# Calculate the average task_length for each combination of rep_runs.thisN and config_type
average_df_one = mega_df_one.groupby(['rep_runs.thisN', 'config_type'])['task_length'].mean().reset_index()
# Plot settings

# Prepare color palettes
task_names = mega_df_one['config_type'].dropna().unique().tolist()
colors_backw = sns.color_palette('Blues', n_colors=20)  # Shades of blue
colors_forw = sns.color_palette('Reds', n_colors=20)   # Shades of red
# Define the order of config_types and assign colors
color_map = {config: colors_backw[i] if 'backw' in config else colors_forw[i] for i, config in enumerate(task_names)}


sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Plotting
for config in task_names:
    subset = average_df_one[average_df['config_type'] == config]
    sns.lineplot(x='rep_runs.thisN', y='task_length', data=subset, label=config, color=color_map[config])

# Adjust plot settings
plt.xticks([0, 1, 2, 3, 4])
plt.title('1st half: Average Task Length Across Subjects by Task and Repeat')
plt.xlabel('Repeat no.')
plt.ylabel('Average Task Length')
plt.ylim(23,50)
plt.legend(title='Task', bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()

plt.show()

# split by task half. task half 2.
# Calculate the average task_length for each combination of rep_runs.thisN and config_type
average_df_two = mega_df_two.groupby(['rep_runs.thisN', 'config_type'])['task_length'].mean().reset_index()
# Plot settings

# Prepare color palettes
task_names = mega_df_two['config_type'].dropna().unique().tolist()
colors_backw = sns.color_palette('Blues', n_colors=20)  # Shades of blue
colors_forw = sns.color_palette('Reds', n_colors=20)   # Shades of red
# Define the order of config_types and assign colors
color_map = {config: colors_backw[i] if 'backw' in config else colors_forw[i] for i, config in enumerate(task_names)}


sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Plotting
for config in task_names:
    subset = average_df_two[average_df['config_type'] == config]
    sns.lineplot(x='rep_runs.thisN', y='task_length', data=subset, label=config, color=color_map[config])

# Adjust plot settings
plt.xticks([0, 1, 2, 3, 4])
plt.title('2nd half: Average Task Length Across Subjects by Task and Repeat')
plt.xlabel('Repeat no.')
plt.ylabel('Average Task Length')
plt.ylim(23,50)
plt.legend(title='Task', bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()

plt.show()   


# Calculate mean and standard deviation for each config_type across all subjects
# Concatenate all DataFrames for aggregate calculations
all_data = pd.concat(sorted_df.values())

# Calculate mean and standard deviation for each config_type across all subjects
mean_data = all_data.groupby('config_type')['task_length'].mean().reset_index()
std_data = all_data.groupby('config_type')['task_length'].std().reset_index()

# Sort the data frames
mean_data['sort_key'] = mean_data['config_type'].apply(custom_sort_key)
std_data['sort_key'] = std_data['config_type'].apply(custom_sort_key)
mean_data.sort_values(by='sort_key', inplace=True)
std_data.sort_values(by='sort_key', inplace=True)

# then analyse if any of the subjects were an outlier.
content_outlier_dict_outer = subjects.copy()
content_outlier_dict_outer.append('mean_plus_std')
content_outlier_dict_inner = task_names.copy()
content_outlier_dict_inner.append('sum_outlier')
outlier_subj = {subject: {task_name: None for task_name in content_outlier_dict_inner} for subject in content_outlier_dict_outer}

# import pdb; pdb.set_trace()
for sub in subjects:
    subj_df = sorted_df[sub]
    count_is_outlier = 0
    for task_name in task_names:
        mean = mean_data[mean_data['config_type'] == task_name]['task_length'].values[0]
        std = std_data[std_data['config_type'] == task_name]['task_length'].values[0] 
        outlier_subj['mean_plus_std']['config_type'] = mean + std
        if subj_df[subj_df['config_type'] == task_name]['task_length'].values[0] < outlier_subj['mean_plus_std']['config_type']:
            outlier_subj[sub][task_name] = 'no outlier'
        elif subj_df[subj_df['config_type'] == task_name]['task_length'].values[0] > outlier_subj['mean_plus_std']['config_type']:
            outlier_subj[sub][task_name] = 'is outlier'
            count_is_outlier = count_is_outlier + 1
    outlier_subj[sub]['sum_outlier'] = count_is_outlier



sns.set(style="whitegrid") 
plt.figure(figsize=(12, 8))
for subj in average_task_length:
    sns.lineplot(x='config_type', y='task_length', data=sorted_df[subj], color='grey', linestyle='-', linewidth =1, alpha =0.5, marker='o', label=subj)

# Plot the average line
sns.lineplot(x='config_type', y='task_length', data=mean_data, color='blue', linestyle='-', marker='o', linewidth=2, label='Average')
# plot the standard deviation
plt.fill_between(x=mean_data['config_type'], y1=mean_data['task_length'] - std_data['task_length'], y2=mean_data['task_length'] + std_data['task_length'], color='blue', alpha=0.2)

plt.title('Average performance by task across subjects')
plt.xlabel('Task')
plt.ylabel('Time in secs')
plt.xticks(rotation=45)
# Move the legend outside of the figure
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
        
# - within task, across runs
    
    
    # - between participants
        
        

       