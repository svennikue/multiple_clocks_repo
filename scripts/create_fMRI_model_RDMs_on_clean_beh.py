#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:06:25 2025
Based on clean behavioural tables,
create regressors that I want to use for the fMRI.

I will store a standard set of models 

state # abstract task structure
A-state # visual control for feedback print
duration # distance between how long each bin takes
path_rew # are we comparing same-with-same or different conditions
next_buttons # buttons that you'll press next
prev_buttons # buttons that brough you here
buttons_out # buttons that bring you out of the current state
location # phase-encoded location
state_action_loc # current phase-action
phys_abstr_space # physical space (location) x abstract task space (prev. state) combination
curr_rew
next_rew
two_next_rew
three_next_rew
l2_norm
curr_path
DSR_onefut
DSR_twofut
DSR_threefut
DSR_fourfut
DSR_fivefut
DSR_sixfut
DSR_sevenfut
rewDSR
pathDSR
rew_stateactionDSR
path_stateactionDSR
curr_quarter
next_quarter
next2_quarter
next3_quarter
DSR
state_action_DSR
action_DSR


in all possible regressors: both task halves, path x rewards x unique_tasks

note on 05th of feb 2026:
    there are now quite a lot of models. At the same time, this script is very fast
    and the stored models aren't very big. so In a way, it doesn't matter in which folder they
    are stored and how many i produce here.
    maybe adjust such that it's just always the same folder
    and always produce all?
    because you select later anyways

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
import json
from fnmatch import fnmatch
import ast
from collections import Counter


if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

subjects = [f"sub-{subj_no}"]
# subjects = subs_list = [f'sub-{i:02}' for i in range(1, 36)]
# subjects.remove('sub-29')
# subjects.remove('sub-21')

# --- Load configuration ---
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")

config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_DSR_bias-path-rew-splitfuts_combos.json"
#config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_DSR_rew_vs_path_stepwise_combos.json"
#config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_DSR_rew_stepwise_combos.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

#
# SETTINGS
#
plot_RDMs = True 
save_RDMs = True
EV_string = config.get("load_EVs_from", "DSR_loc-fut-rews-state-dur-type")
plot_DSR_task_matrices = False
plot_DSR_tasks = [] # fill this with eg tasks[14]
plot_DSR_rotation_bins = None
len_standardised_path = 12

coord_to_loc = {
    (-0.21,  0.29): 1, (0.0,  0.29): 2, (0.21,  0.29): 3,
    (-0.21,  0.0 ): 4, (0.0,  0.0 ): 5, (0.21,  0.0 ): 6,
    (-0.21, -0.29): 7, (0.0, -0.29): 8, (0.21, -0.29): 9,
}
loc_to_coord = {v:k for k,v in coord_to_loc.items()}

state_onehot_to_str = {0:'A', 1:'B', 2:'C', 3:'D'}


def resample_locations(path, T=len_standardised_path):
    n = len(path)
    reps = [T // n + (i < T % n) for i in range(n)]
    return np.repeat(path, reps)


# import pdb; pdb.set_trace()
        
for sub in subjects:
    # load the cleaned behavioural table.
    beh_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/beh"
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
    buttons = ['left', 'up', 'down', 'right']
    bin_type = beh_df['time_bin_type'].unique()

    locations = sorted(beh_df['curr_loc'].unique())
    coordinates = np.array([loc_to_coord[loc] for loc in locations])
    
    loc_to_row = {loc: i for i, loc in enumerate(locations)}
    
    
    # define regressors. unique_time_bin_type look like E1_forw_A_reward etc.
    regs = sorted(beh_df['unique_time_bin_type'].unique())
    regressors = {}
    for reg in regs:
        regressors[reg] = np.zeros(len(beh_df))
        regressors[reg][beh_df['unique_time_bin_type'] == reg] = 1


    # select which EVs are included in the RDM (same logic as fMRI_run_RSA_without_rsatoolbox_clean.py)
    conditions = config.get("EV_condition_selection", {})
    parts_to_use = conditions.get("parts")
    if parts_to_use:
        for _p in ("task", "direction", "state", "phase"):
            if _p not in parts_to_use:
                raise ValueError(f"Missing selection.parts['{_p}'] in config.")
        EV_keys = []
        for ev in sorted(regs):
            task, direction, state, phase = ev.split('_')
            for name, value in zip(
                ["task", "direction", "state", "phase"],
                [task, direction, state, phase],
            ):
                part = parts_to_use[name]
                includes = part.get("include", [])
                excludes = part.get("exclude", [])
                if any(fnmatch(value, pat) for pat in excludes):
                    break
                if includes and not any(fnmatch(value, pat) for pat in includes):
                    break
            else:
                EV_keys.append(ev)
    else:
        EV_keys = list(regs)


    # define models.
    models = {}
    models['state'] = np.zeros((len(states), len(beh_df)))
    models['A-state'] = np.zeros((len(states), len(beh_df)))
    models['duration'] = np.expand_dims(beh_df['t_spent_at_curr_loc'].to_numpy(), axis=0)
    models['path_rew'] = np.expand_dims(beh_df['time_bin_type'].to_numpy(), axis=0)
    
    for s_i, state in enumerate(states):
        # import pdb; pdb.set_trace()
        if state == 'A':
            models['A-state'][s_i][(beh_df['state'] == state)& (beh_df['time_bin_type'] == 'reward')] = 1
        models['state'][s_i][beh_df['state'] == state] = 1
    
    # these models are order-preserving encodings which will be computed based on hamming-distance
    for key in ['next_buttons', 'prev_buttons', 'buttons_out', 'location', 'state_action_loc', 'phys_abstr_space',
                "curr_path","DSR_onefut", "DSR_twofut", "DSR_threefut","DSR_fourfut", "DSR_fivefut", "DSR_sixfut", "DSR_sevenfut"]:
        models[key] = np.zeros((len_standardised_path, len(beh_df)), dtype=float)

    # these models are one-hot encodings of locations which will be computes based on cosine similarity
    for key in ["curr_rew", "next_rew", "two_next_rew", "three_next_rew", "l2_norm"]:
        models[key] = np.zeros((len(locations), len(beh_df)), dtype=float)

    
    for i_loc, loc in enumerate(locations):
        # models['location'][i_loc][beh_df['curr_loc'] == loc] = 1
        models['curr_rew'][i_loc][beh_df['curr_rew'] == loc] = 1
        # models['path'][i_loc][(beh_df['curr_loc'] == loc) & (beh_df['time_bin_type'] == 'path')] = 1
        # models['rew'][i_loc][(beh_df['curr_rew'] == loc) & (beh_df['time_bin_type'] == 'reward')] = 1
        for idx_inner_loc, inner_loc in enumerate(locations):
            models['l2_norm'][idx_inner_loc][beh_df['curr_loc'] == loc] = -np.linalg.norm(coordinates[i_loc] - coordinates[idx_inner_loc])


    # this is for the future reward location models.
    # rotates the reward values by k, but keeps time-bin-length in place.
    def rotate_runs(arr, k):
        """Rotate the values of consecutive runs by k, preserving run lengths."""
        # Finds the points at which a new value starts and turn them into indices
        changes = np.r_[True, arr[1:] != arr[:-1]]
        starts  = np.flatnonzero(changes)
        # Count number of identical consecutive items
        lens    = np.diff(np.r_[starts, arr.size])
        # Find which values are repeated
        vals    = arr[starts]
        # rols, and then repeat and return
        rot_vals = np.roll(vals, - (k % len(vals)))   # left-roll so first run takes next run's value
        return np.repeat(rot_vals, lens)


    # future reward encodings 
    for task in tasks:
        idx  = (beh_df["task_config_ex"] == task)
        cols = np.flatnonzero(idx)
        rews = beh_df.loc[idx, "curr_rew"].to_numpy()
        locs = beh_df.loc[idx, "curr_loc"].to_numpy()
        
        fut1_rew = rotate_runs(rews, 1)  # +1 run
        fut2_rew = rotate_runs(rews, 2)  # +2 runs
        fut3_rew = rotate_runs(rews, 3)  # +3 runs

        for fut, name in [(fut1_rew,"next_rew"), (fut2_rew,"two_next_rew"), (fut3_rew,"three_next_rew")]:
            rows = np.fromiter((loc_to_row[v] for v in fut), dtype=int, count=fut.size)
            models[name][rows, cols] = 1.0
    
    
    # create regressors.
    EVs, raw_loc_dict, raw_button_dict = {},{}, {}
    for model in models:
        EVs[model] = {}
        # if model == 'prev_buttons' or model == 'buttons_out':
        #     continue
        for reg in regressors:
            # import pdb; pdb.set_trace()
            df_reg = beh_df[beh_df['unique_time_bin_type']==reg]
            if model == 'path_rew':
                label = 'reward' if reg.endswith('reward') else 'path' if reg.endswith('path') else None
                EVs[model][reg] = np.full(len(models[model]), label, dtype=object)
            elif model == 'location':
                # import pdb; pdb.set_trace()
                raw_loc_dict[reg] = []
                EVs[model][reg]=[]
                for rep in range(0,int(np.max(df_reg['repeat']))+1):
                    raw_loc_dict[reg].append(df_reg[df_reg['repeat']==rep]['curr_loc'].to_numpy())
                # instead of choosing the average, choose the path that occured most often
                # as the represenative plan.
                most_common_path = np.array(Counter(map(lambda x: tuple(x), raw_loc_dict[reg])).most_common(1)[0][0])
                # next, upsample the path to a shared length = 12.
                EVs[model][reg] = resample_locations(most_common_path)
            
            elif model == 'buttons_out':
                raw_button_dict[reg] = []
                EVs[model][reg]=[]
                # NEW
                for rep in range(0,int(np.max(df_reg['repeat']))+1):
                    raw_button_dict[reg].append(df_reg[df_reg['repeat']==rep]['button_exec'].to_list())
                most_common_button_sequence = np.array(Counter(map(lambda x: tuple(x), raw_button_dict[reg])).most_common(1)[0][0])
                EVs[model][reg] = resample_locations(most_common_button_sequence)

            else:
                EVs[model][reg] = np.zeros((len(models[model])))  
                for index, row in enumerate(models[model]):
                    if model == 'duration':
                        # sum up the durations of each regressor and divide by how often they were 'on'
                        n_times_regressor_active = np.sum(np.diff(regressors[reg]) == 1) + (regressors[reg][0] == 1)
                        EVs[model][reg][index] = models[model].transpose()[regressors[reg].astype(bool)].sum()/n_times_regressor_active
                    else:
                        # Note I don't include an intercept by default.
                        # this is because the way I use ithem, the regressors would be a linear combination of the intercept ([11111] vector)
                        EVs[model][reg][index] = LinearRegression(fit_intercept=False).fit(regressors[reg].reshape(-1,1), row.reshape(-1,1)).coef_


    # make a new state-location regressor. First, make state a string.
    for r in regs:
        for idx in state_onehot_to_str:
            if round(EVs['state'][r][idx]) == 1:
                state_str = state_onehot_to_str[idx]
        
        locs_str = EVs['location'][r].astype(str)
        EVs['state_action_loc'][r] = np.char.add(np.char.add(locs_str, '-'), EVs['buttons_out'][r])
        EVs['phys_abstr_space'][r] = np.char.add(np.char.add(locs_str, '-'), state_str)

    
    # additionally, add the simple musicbox: at each of the 8 timebins, the future is already encoded.
    # order inside a task
    temp_order = ["A_path", "A_reward","B_path", "B_reward","C_path", "C_reward","D_path", "D_reward"]
    temp_order_rew = ["A_reward","B_reward","C_reward","D_reward"]
    temp_order_path = ["A_path","B_path","C_path","D_path"]
    
    temp_order_shifted_prev = np.roll(temp_order, 1).copy()
    temp_order_shifted_next = np.roll(temp_order, -1).copy()
    # this is to do the rotation as well for the buttons:
    # previous buttons are the ones you pressed on the previous locations.
    
    # initialise split models
    models['DSR'], models['state_action_DSR'], models['action_DSR']  = np.zeros((len(temp_order)*len_standardised_path)), np.zeros((len(temp_order)*len_standardised_path)), np.zeros((len(temp_order)*len_standardised_path))
    
    split_models = ['rewDSR', 'pathDSR', 'rew_stateactionDSR', 'path_stateactionDSR']
    for s in split_models:
        models[s] = np.zeros((int(len(temp_order)/2)*len_standardised_path))
        EVs[s] = {}
        
    # split_models = ['rew_stateactionDSR', 'path_stateactionDSR']
    # for s in split_models:
    #     models[s] = np.zeros((int(len(temp_order)/2)*len_standardised_path*4))
    #     EVs[s] = {}
        
    
    split8_DSR_keys = ["location", "DSR_onefut", "DSR_twofut", "DSR_threefut","DSR_fourfut", "DSR_fivefut", "DSR_sixfut", "DSR_sevenfut"]
    for s8 in split8_DSR_keys:
        if s8 == 'location':
            continue
        else:
            models[s] = np.zeros((int(len(temp_order)/8)*len_standardised_path))
            EVs[s8] = {}
    
    split4_DSR_keys = ["curr_quarter", "next_quarter", "next2_quarter", "next3_quarter"]
    for s4 in split4_DSR_keys:
        models[s4] = np.zeros((int(len(temp_order)/4)*len_standardised_path))
        EVs[s4] = {}
        
    rot_split4_DSR_keys = ["rot_curr_quarter", "rot_next_quarter", "rot_next2_quarter", "rot_next3_quarter"]
    for rs4 in rot_split4_DSR_keys:
        models[rs4] = np.zeros((int(len(temp_order)/4)*len_standardised_path))
        EVs[rs4] = {}
    
    EVs['DSR'], EVs['state_action_DSR'], EVs['action_DSR'] = {}, {}, {}
    for task in tasks:
        # build base matrix (8 x 9) in canonical order
        bins_curr_task = [f"{task}_{temp_bin}" for temp_bin in temp_order]
        bins_curr_task_rew = [f"{task}_{temp_bin}" for temp_bin in temp_order if temp_bin.endswith('reward')]
        bins_curr_task_path = [f"{task}_{temp_bin}" for temp_bin in temp_order if temp_bin.endswith('path')]
    
        try:
            # concatenate the 8 bins x 12-element vectors into a single 96-element vector
            # this will read: 0-12 = now. 12-24 = next subpath. 24-36 = subpath after, etc.
            # each EVs['location'][k] has 9 location 
            DSR_firing_for_subpath_A = np.concatenate([EVs['location'][k] for k in bins_curr_task], axis=0)  # shape (96,)

            rewDSR_firing_for_subpath_A = np.concatenate([EVs['location'][k] for k in bins_curr_task_rew], axis=0)  # shape (48,)
            pathDSR_firing_for_subpath_A = np.concatenate([EVs['location'][k] for k in bins_curr_task_path], axis=0)  # shape (48,)
            
            actionDSR_firing_for_subpath_A = np.concatenate([EVs['buttons_out'][k] for k in bins_curr_task], axis=0)  # shape (96,)
            
            stact_firing_for_subpath_A = np.concatenate([EVs['state_action_loc'][k] for k in bins_curr_task], axis=0)  # shape (96,)
            rew_stact_firing_for_subpath_A = np.concatenate([EVs['state_action_loc'][k] for k in bins_curr_task_rew], axis=0)  # shape (48,)
            path_stact_firing_for_subpath_A = np.concatenate([EVs['state_action_loc'][k] for k in bins_curr_task_path], axis=0)  # shape (48,)
            
        except KeyError:
            continue
        
        n_bins = len(temp_order)  # 8 (4 x subpaths, 4x rewards)
        # for each position, rotate by whole blocks of `block_len` so subpath-chunks move together
        for pos, temp_bin in enumerate(temp_order):
            bin_curr_task = f"{task}_{temp_bin}"
            # for the buttons:
            bin_shifted_task_prev = f"{task}_{temp_order_shifted_prev[pos]}"
            bin_shifted_task_next = f"{task}_{temp_order_shifted_next[pos]}"
            EVs['prev_buttons'][bin_shifted_task_prev] = EVs['buttons_out'][bin_curr_task].copy()
            EVs['next_buttons'][bin_shifted_task_next] = EVs['buttons_out'][bin_curr_task].copy()

            # left-roll by pos blocks: multiply by block_len to rotate whole 12-element blocks
            rotated_loc = np.roll(DSR_firing_for_subpath_A, -pos * len_standardised_path).copy()
            EVs['DSR'][bin_curr_task] = rotated_loc
        
            # also create the split DSRs 
            for idx8, s8 in enumerate(split8_DSR_keys):
                if s8 == 'location':
                    continue
                else:
                    start_idx = idx8*12
                    end_idx = idx8*12+12
                    EVs[s8][bin_curr_task] = rotated_loc[start_idx:end_idx]
        
            
            for idx4, s4 in enumerate(split4_DSR_keys):
                start_idx = idx4*24
                end_idx = idx4*12*2+24
                EVs[s4][bin_curr_task] = rotated_loc[start_idx:end_idx]
            
            # import pdb; pdb.set_trace()
            for idxr4, rs4 in enumerate(rot_split4_DSR_keys):
                r_start_idx = idxr4*24
                r_end_idx = idxr4*12*2+24
                EVs[rs4][bin_curr_task] = rotated_loc[r_start_idx:r_end_idx]
                    
            # and the same for the state-action DSR
            rotated_state_action = np.roll(stact_firing_for_subpath_A, -pos * len_standardised_path).copy()
            EVs['state_action_DSR'][bin_curr_task] = rotated_state_action
            
            # and the same for action DSR
            rotated_action = np.roll(actionDSR_firing_for_subpath_A, -pos * len_standardised_path).copy()
            EVs['action_DSR'][bin_curr_task] = rotated_action
            
        for pos, rew_bin in enumerate(temp_order_rew):
            # import pdb; pdb.set_trace()
            bin_curr_rew = f"{task}_{rew_bin}"
            bin_curr_path = f"{task}_{temp_order_path[pos]}"
            
            # left-roll by pos blocks: multiply by block_len to rotate whole 12-element blocks
            rotated_rew_loc = np.roll(rewDSR_firing_for_subpath_A, -pos * len_standardised_path).copy()
            rotated_path_loc = np.roll(pathDSR_firing_for_subpath_A, -pos * len_standardised_path).copy()
            # fill rewards and paths times with the same model
            EVs['rewDSR'][bin_curr_path] = rotated_rew_loc
            EVs['rewDSR'][bin_curr_rew] = rotated_rew_loc
            
            EVs['pathDSR'][bin_curr_path] = rotated_path_loc
            EVs['pathDSR'][bin_curr_rew] = rotated_path_loc
            
            rotated_state_action_rew = np.roll(rew_stact_firing_for_subpath_A, -pos * len_standardised_path).copy()
            rotated_state_action_path = np.roll(path_stact_firing_for_subpath_A, -pos * len_standardised_path).copy()
            # fill rewards and path times with the same model
            EVs['rew_stateactionDSR'][bin_curr_path] = rotated_state_action_rew
            EVs['rew_stateactionDSR'][bin_curr_rew] = rotated_state_action_rew
            
            EVs['path_stateactionDSR'][bin_curr_path] = rotated_state_action_path
            EVs['path_stateactionDSR'][bin_curr_rew] = rotated_state_action_path
    

    if plot_DSR_task_matrices and plot_DSR_tasks:
        mc.plotting.results.plot_dsr_task_matrices(
            EVs,
            tasks=plot_DSR_tasks,
            temp_order=temp_order,
            rotation_bins=plot_DSR_rotation_bins,
        )

       
    if plot_RDMs == True:
        for model in models:
            if model == 'path_rew':
                continue
            if 'button' in model:
                continue
            #ev_array = np.zeros((int(len(EVs[model])/2), len(models[model])))
            # if model == 'DSR':
            #     ev_array = np.zeros((int(len(EVs[model])), len(EVs['DSR'][bin_curr_task])))
            # else:
            evs_for_model = [ev for ev in EV_keys if ev in EVs[model]]
            if model in ['location', 'DSR', 'prev_buttons', 'buttons_out', 'next_buttons', 'phys_abstr_space', 'action_DSR', 'state_action_DSR',
                           'state_action_glob', 'state_action_loc', "rewDSR", "pathDSR", "rew_stateactionDSR", "path_stateactionDSR",
                           'DSR_onefut', 'DSR_twofut','DSR_threefut','DSR_fourfut','DSR_fivefut','DSR_sixfut','DSR_sevenfut',
                           'curr_quarter','next_quarter','next2_quarter','next3_quarter']:
                ev_array = np.zeros((int(len(evs_for_model)), len(models[model])), dtype = object)
                method = 'hamming_distance'
            else:
                ev_array = np.empty((int(len(evs_for_model)), len(models[model])), dtype = float)
                method = 'crosscorr'
            idx = -1
            y_labels = []
            for ev in evs_for_model:
                #if ev.endswith('reward'):
                idx = idx +1
                y_labels.append(ev)
                ev_array[idx] = EVs[model][ev]
                
            print(f"now plotting RDM for {model} model")
            mc.plotting.results.plot_model_rdm_half(
                ev_array,
                labels=y_labels,
                method=method,
                label_half="first",
                group_size=8,
                title=model,
            )
    

    # import pdb; pdb.set_trace()          
    if save_RDMs: 
        # then save these matrices.
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        
        with open(f"{RDM_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'wb') as file:
            pickle.dump(EVs, file)
            
        print(f"saved EV dictionary as {RDM_dir}/{sub}_modelled_EVs_{EV_string}.pkl")
