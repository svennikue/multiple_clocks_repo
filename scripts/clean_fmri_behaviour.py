#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:21:08 2025

@author: Svenja Küchenhoff

This script takes the raw .csv file from PsychoPy and cleans it such that I can
easily read out the relevant behavioural details (combine pt 1 and 2!):
    1) task_config [e.g. E1]
    3) forwards/backwards
    4) config type [e.g. E1_forw]
    5) curr rew_loc [location 1-9]
    6) time start reward
    7) curr_loc [location 1-9]
    8) time start at curr_loc
    9) button press for respective loc [1/4 keys]
    10) state
    11) time_bin_type (state+reward/path) [e.g. E1_forw_A_path]
    12) repeat [1-5]
    13) session [1/2]
    

    
"""

import pandas as pd
import os
import sys


if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'


#subjects = [f"sub-{subj_no}"]
subjects = subs_list = [f'sub-{i:02}' for i in range(1, 36)]
# import pdb; pdb.set_trace()
        
for sub in subjects:
    both_halves = []   # collect cleaned tables for both halves
    for task_half in [1,2]:
        data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        out_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        if os.path.isdir(data_dir_beh):
            print(f"Running on laptop, now subject {sub}")
        else:
            data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
            out_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/beh/"
            print(f"Running on Cluster, setting {data_dir_beh} as data directory")
            
        file = data_dir_beh + f"{sub}_fmri_pt{task_half}.csv"
        if not os.path.exists(file):
            print(f"This file doesn't exist: {file}")
            continue  # skip to next loop iteration

        df = pd.read_csv(file)
        
        # create a new df 
        beh_clean = pd.DataFrame()

        # every valid row has a 'type' entry. filter for none-valid rows.
        beh_raw = df[df['type'].notna()].copy()
        
        # for mapping locations
        # mapping dictionary: (x,y) -> grid number
        coord_to_loc = {
            (-0.21,  0.29): 1, (0.0,  0.29): 2, (0.21,  0.29): 3,
            (-0.21,  0.0 ): 4, (0.0,  0.0 ): 5, (0.21,  0.0 ): 6,
            (-0.21, -0.29): 7, (0.0, -0.29): 8, (0.21, -0.29): 9,
        }

        # 1) task_config [e.g. E1]
        beh_clean['task_config_seq'] = beh_raw['task_config'].ffill()
        
        # 3) forwards/backwards
        beh_clean['instruction'] = beh_raw['type']
        
        # 4) config type [e.g. E1_forw]
        beh_clean['task_config_ex'] = beh_clean['task_config_seq'] + '_' + beh_clean['instruction']
        beh_raw['task_config_ex'] = beh_clean['task_config_seq'] + '_' + beh_clean['instruction']
        
        # 5) curr rew_loc [location 1-9]
        beh_clean['curr_rew'] = beh_raw.apply(
            lambda row: coord_to_loc.get((row['curr_rew_x'], row['curr_rew_y']), pd.NA),
            axis=1
        )
        
        
        # 6) time start reward
        beh_clean['t_curr_rew'] = beh_raw['t_reward_start']
        
        # 7) curr_loc [location 1-9]
        beh_clean['curr_loc'] = beh_raw.apply(
            lambda row: coord_to_loc.get((row['curr_loc_x'], row['curr_loc_y']), pd.NA),
            axis=1
        )
        # 8) time start at curr_loc
        beh_clean['t_curr_loc'] = beh_raw['t_step_end_global'].fillna(beh_raw['start_ABCD_screen'])
        
        # 9) button press for respective loc [1/4 keys]
        # leave this for now!
        # a bit more complicated given there might have been more buttons stored
        # than only the ones executed.
        # x = beh_raw[beh_clean['repeat'] == 1][beh_clean['task_config_ex']=='B1_backw']
        #timings = beh_clean[beh_clean['repeat'] == 1][beh_clean['task_config_ex']=='B1_backw']['t_curr_loc'].to_numpy()
        #import ast; press_timings_local = ast.literal_eval(x['nav_key_task.rt'][x['nav_key_task.rt'].notna()].iloc[0])
        
        
        # 10) state
        beh_clean['state'] = beh_raw['state'].fillna('A')
        
        # 11) time_bin_type (state+reward/path) [e.g. E1_forw_A_path]
        rewards_mask = beh_clean['t_curr_rew'].notna()
        beh_clean['time_bin_type'] = 'path'
        beh_clean.loc[rewards_mask, 'time_bin_type'] = 'reward'
        beh_clean['unique_time_bin_type'] = beh_clean['task_config_ex'] + '_' + beh_clean['state'] + '_path'
        beh_clean.loc[rewards_mask, 'unique_time_bin_type'] = beh_clean.loc[rewards_mask, 'task_config_ex'] + '_' + beh_clean.loc[rewards_mask, 'state']+ '_reward'

        # 12) repeat [1-5]
        beh_raw['task_config'] = beh_raw['task_config'].ffill()
        beh_clean['repeat'] = beh_raw.groupby('task_config_ex')['start_ABCD_screen'].transform(lambda s: s.notna().cumsum())
        
        # 13) session [1/2]
        beh_clean['task_half'] = task_half
        
    
        both_halves.append(beh_clean)
    
    # concatenate both halves and save
    beh_both = pd.concat(both_halves, ignore_index=True)
    out_file = os.path.join(out_dir, f"{sub}_beh_fmri_clean.csv")
    beh_both.to_csv(out_file, index=False)
    print(f"Saved {out_file}")

        
        
        
        