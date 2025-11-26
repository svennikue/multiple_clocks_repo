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
import mc
import numpy as np

from glob import glob

# for mapping locations
# mapping dictionary: (x,y) -> grid number
coord_to_loc = {
    (-0.21,  0.29): 1, (0.0,  0.29): 2, (0.21,  0.29): 3,
    (-0.21,  0.0 ): 4, (0.0,  0.0 ): 5, (0.21,  0.0 ): 6,
    (-0.21, -0.29): 7, (0.0, -0.29): 8, (0.21, -0.29): 9,
}

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

alternative_regs = False

# Find the source dir first, outside of the loop
data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/"
out_dir      = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/"
if os.path.isdir(data_dir_beh):
    print(f"Running on laptop, now subject {sub}")
else:
    data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/"
    out_dir      = f"/home/fs0/xpsy1114/scratch/data/derivatives/"
    print(f"Running on Cluster, setting {data_dir_beh} as data directory")

subjects = glob(f"{data_dir_beh}/sub-*")
subjects = [s.split("/")[-1] for s in subjects]
# import pdb; pdb.set_trace()
        
for sub in subjects:
    both_halves = []   # collect cleaned tables for both halves

    # Then here inside the loop, we know which subject we are looking at so we can define the correct folders

    for task_half in [1,2]:
        file = data_dir_beh + f"{sub}_fmri_pt{task_half}.csv"
        if not os.path.exists(file):
            print(f"This file doesn't exist: {file}")
            continue  # skip to next loop iteration

        df = pd.read_csv(file)
        
        # create a new df 
        beh_clean = pd.DataFrame()

        # every valid row has a 'type' entry. filter for none-valid rows.
        beh_raw = df[df['type'].notna()].copy()
        
        # 1) task_config [e.g. E1]
        beh_clean['task_config_seq'] = beh_raw['task_config'].ffill()
        
        # 2) forwards/backwards
        beh_clean['instruction'] = beh_raw['type']
        
        # 3) config type [e.g. E1_forw]
        beh_raw['task_config_ex']   = beh_clean['task_config_seq'] + '_' + beh_clean['instruction']
        beh_clean['task_config_ex'] = beh_clean['task_config_seq'] + '_' + beh_clean['instruction']
        
        # 4) curr rew_loc [location 1-9]
        # Sometimes a trial ended early and coordinates are not defined so we put NA in the df
        beh_clean['curr_rew'] = beh_raw.apply(
            lambda row: coord_to_loc.get((row['curr_rew_x'], row['curr_rew_y']), pd.NA),
            axis=1
        )
        
        
        # 5) time start reward and length
        beh_clean['t_curr_rew']   = beh_raw['t_reward_start']
        beh_clean['reward_delay'] = beh_raw['reward_delay']
        
        # 6) curr_loc [location 1-9]
        beh_clean['curr_loc'] = beh_raw.apply(
            lambda row: coord_to_loc.get((row['curr_loc_x'], row['curr_loc_y']), pd.NA),
            axis=1
        )
        
        # 7) time start at curr_loc
        beh_clean['t_curr_loc'] = beh_raw['t_step_end_global'].fillna(beh_raw['start_ABCD_screen'])
        
        # 8) button press for respective loc [1/4 keys]
        # leave this for now!
        # a bit more complicated given there might have been more buttons stored
        # than only the ones executed.
        # x = beh_raw[beh_clean['repeat'] == 1][beh_clean['task_config_ex']=='B1_backw']
        #timings = beh_clean[beh_clean['repeat'] == 1][beh_clean['task_config_ex']=='B1_backw']['t_curr_loc'].to_numpy()
        #import ast; press_timings_local = ast.literal_eval(x['nav_key_task.rt'][x['nav_key_task.rt'].notna()].iloc[0])
        
        # for now just keep the relevant rows.
        beh_clean['button_rts']  = beh_raw['nav_key_task.rt']
        beh_clean['button_keys'] = beh_raw['nav_key_task.keys']
        

        # 9) state
        # Sometimes at the start state is empty but since we know things start at A we just fill it in
        beh_clean['state'] = beh_raw['state'].fillna('A')
        
        # 10) time_bin_type (state+reward/path) [e.g. E1_forw_A_path]
        rewards_mask = beh_clean['t_curr_rew'].notna()
        beh_clean['time_bin_type'] = 'path'
        beh_clean.loc[rewards_mask, 'time_bin_type'] = 'reward'
        beh_clean['unique_time_bin_type'] = beh_clean['task_config_ex'] + '_' + beh_clean['state'] + '_path'
        beh_clean.loc[rewards_mask, 'unique_time_bin_type'] = beh_clean.loc[rewards_mask, 'task_config_ex'] + '_' + beh_clean.loc[rewards_mask, 'state']+ '_reward'

        # 11) repeat [1-5]
        beh_clean['repeat'] = beh_raw.groupby('task_config_ex')['start_ABCD_screen'].transform(lambda s: s.notna().cumsum())
        
        # 12) session [1/2]
        beh_clean['task_half'] = task_half
        
        # if wanted, you can analyse Brooke's alternative regressors here.
        if alternative_regs == True:
            # beh_clean['task_half'] = mc.analyse.extract_and_clean.define_futsteps_x_locs_regressors(beh_clean)
            df = mc.analyse.extract_and_clean.define_futsteps_x_locs_regressors(beh_clean)
            for step in ['curr','one_fut','two_fut','three_fut']:
                for loc in range(1, 10):
                    row_name = f'loc_{loc}_{step}'
                    print(f"{step} {loc} has {np.sum(df[row_name])} occurances")

        both_halves.append(beh_clean)
    
    # concatenate both halves and save
    beh_both = pd.concat(both_halves, ignore_index=True)
    
    # import pdb; pdb.set_trace()
    # store where same reward-states appear at the same locations for later masking
    mc.analyse.extract_and_clean.store_same_locs_in_same_state(beh_both, out_dir)
    
    out_file = os.path.join(out_dir, f"{sub}_beh_fmri_clean.csv")
    beh_both.to_csv(out_file, index=False)
    print(f"Saved {out_file}")
