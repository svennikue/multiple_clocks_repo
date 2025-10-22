#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:06:25 2025
Based on clean behavioural tables,
create regressors that I want to use for the fMRI.

I will store a standard set of models 
(currently "location", "curr_rew", "next_rew", "two_next_rew", "three_next_rew", "state")
in all possible regressors: both task halves, path x rewards x unique_tasks
You can choose later which regressors you want to use.


logic is as follows:
create the models based on the behaviour in time = 'steps'.
create regressors based on 'path' or 'reward' also in time = 'steps'
regress each model into the same binned dimension the fMRI is in.
I want to end with regressors that go like: '{model}_A1_backw_A_reward.txt'

note: needs clean_fmri_behaviour.py to have run first.

@author: Svenja Küchenhoff
"""

import pandas as pd
import numpy as np
import os
import pickle
import mc
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

subjects = [f"sub-{subj_no}"]
# subjects = subs_list = [f'sub-{i:02}' for i in range(1, 36)]

#
# SETTINGS
#
#regression_version = '03-4' 
#RDM_version = '03-1'
# no_phase_neurons = 3
plot_RDMs = True 
save_RDMs = True
EV_string = 'simple-clean_loc-fut-rews-state'
#models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)
   
# import pdb; pdb.set_trace()
        
for sub in subjects:
    # load the cleaned behavioural table.
    beh_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh"
    RDM_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/beh/modelled_EVs"
    if os.path.isdir(beh_dir):
        print(f"Running on laptop, now subject {sub}")
    else:
        beh_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/beh"
        RDM_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/beh/modelled_EVs"
        print(f"Running on Cluster, setting {beh_dir} as data directory")
            
                 
    beh_df = pd.read_csv(f"{beh_dir}/{sub}_beh_fmri_clean.csv")
    tasks = beh_df['task_config_ex'].unique()
    states = beh_df['state'].unique()
    bin_type = beh_df['time_bin_type'].unique()
    
    locations = sorted(beh_df['curr_loc'].unique())
    loc_to_row = {loc: i for i, loc in enumerate(locations)}
    
    
    # define regressors.
    regs = sorted(beh_df['unique_time_bin_type'].unique())
    regressors = {}
    for reg in regs:
        regressors[reg] = np.zeros(len(beh_df))
        regressors[reg][beh_df['unique_time_bin_type'] == reg] = 1


    # define models.
    models = {}
    # ['location', 'curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew', 'state', 'clocks']
    models['state'] = np.zeros((len(states), len(beh_df)))
    for s_i, state in enumerate(states):
        models['state'][s_i][beh_df['state'] == state] = 1
    
    for key in ["location", "curr_rew", "next_rew", "two_next_rew", "three_next_rew"]:
        models[key] = np.zeros((len(locations), len(beh_df)), dtype=float)
    
    for loc in locations:
        models['location'][loc-1][beh_df['curr_loc'] == loc] = 1
    
    for loc in locations:
        models['curr_rew'][loc-1][beh_df['curr_rew'] == loc] = 1
    
    
    # this is for the future reward location models.
    # rotates the reward values by k, but keeps time-bin-length in place.
    def rotate_runs(arr, k):
        """Rotate the values of consecutive runs by k, preserving run lengths."""
        changes = np.r_[True, arr[1:] != arr[:-1]]
        starts  = np.flatnonzero(changes)
        lens    = np.diff(np.r_[starts, arr.size])
        vals    = arr[starts]
        rot_vals = np.roll(vals, - (k % len(vals)))   # left-roll so first run takes next run's value
        return np.repeat(rot_vals, lens)


    for task in tasks:
        idx  = (beh_df["task_config_ex"] == task)
        cols = np.flatnonzero(idx)
        rews = beh_df.loc[idx, "curr_rew"].to_numpy()
    
        fut1 = rotate_runs(rews, 1)  # +1 run
        fut2 = rotate_runs(rews, 2)  # +2 runs
        fut3 = rotate_runs(rews, 3)  # +3 runs
    
        for fut, name in [(fut1,"next_rew"), (fut2,"two_next_rew"), (fut3,"three_next_rew")]:
            rows = np.fromiter((loc_to_row[v] for v in fut), dtype=int, count=fut.size)
            models[name][rows, cols] = 1.0
    
    
    # create regressors.
    EVs = {}
    for model in models:
        EVs[model] = {}
        for reg in regressors:
            EVs[model][reg] = np.zeros((len(models[model])))
            for index, row in enumerate(models[model]):
                # Note I don't include an intercept by default.
                # this is because the way I use ithem, the regressors would be a linear combination of the intercept ([11111] vector)
                EVs[model][reg][index] = LinearRegression(fit_intercept=False).fit(regressors[reg].reshape(-1,1), row.reshape(-1,1)).coef_
        
        
        if plot_RDMs == True:
            ev_array = np.zeros((int(len(EVs[model])/2), len(models[model])))
            idx = -1
            y_labels = []
            for ev in EVs[model]:
                if ev.endswith('reward'):
                    idx = idx +1
                    y_labels.append(ev)
                    ev_array[idx] = EVs[model][ev]
            
            ev_array_all = np.zeros((int(len(EVs[model])), len(models[model])))
            y_labels_all = []
            for idx, ev in enumerate(EVs[model]):
                y_labels_all.append(ev)
                ev_array_all[idx] = EVs[model][ev]
                    
                   
            plt.figure(); plt.imshow(np.corrcoef(ev_array), aspect = 'auto')
            plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
            # plt.xticks(ticks=range(len(y_labels)), labels=y_labels)
            intervalline = 4
            for interval in range(0, len(ev_array), intervalline):
                plt.axvline(interval-0.5, color='white', ls='dashed')
                plt.axhline(interval-0.5, color='white', ls='dashed')
            plt.title(model)
            
    
        
    # import pdb; pdb.set_trace()          
    if save_RDMs: 
        # then save these matrices.
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        
        with open(f"{RDM_dir}/{sub}_modelled_EVs_{EV_string}_.pkl", 'wb') as file:
            pickle.dump(EVs, file)
            
        print(f"saved EV dictionary as {RDM_dir}/{sub}_modelled_EVs_{EV_string}.pkl")
    
    

    
    
    