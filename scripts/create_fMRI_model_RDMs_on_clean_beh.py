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
#subjects = subs_list = [f'sub-{i:02}' for i in range(1, 36)]
#subjects.remove('sub-21')
#subjects.remove('sub-29')

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

# Publication figures: rendered ONCE for a single example subject. Uses the
# RDMs and EVs already computed by this script (no recomputation).
PUBLICATION_FIGURES       = True
EXAMPLE_SUBJECT_FOR_FIGS  = 'sub-02'
PUB_MODELS = [
    'state', 'A-state', 'path_rew', 'next_buttons', 'buttons_out',
    'location', 'l2_norm',
    'curr_quarter', 'next_quarter', 'next2_quarter', 'next3_quarter',
    'DSR', 'rewDSR', 'curr_rew', 'next_rew', 'two_next_rew', 'three_next_rew']

PUB_FIG_WIDTH_CM   = 4.0
PUB_FIG_HEIGHT_CM  = 4.0
PUB_FIG_FONT_PT    = 8

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

       
    # Build the "task code → executed-reward-sequence" lookup used for
    # publication figures. We derive each task's goal label from the
    # *location* model at its four reward phases — i.e. the modal location
    # that was actually visited at the X_reward bin of ``{task}_X_reward``.
    # This reflects what the subject executed (direction-aware, robust to
    # any trial-by-trial deviation from the configured reward).
    def _modal_loc_at_reward(task, state):
        ev_key = f'{task}_{state}_reward'
        v = EVs.get('location', {}).get(ev_key)
        if v is None or len(v) == 0:
            return '?'
        return int(round(Counter(np.asarray(v).tolist()).most_common(1)[0][0]))

    task_to_goal_label = {}
    for task in tasks:
        rew_seq = [_modal_loc_at_reward(task, st) for st in ('A', 'B', 'C', 'D')]
        task_to_goal_label[task] = '-'.join(str(r) for r in rew_seq)

    # All RDMs below are computed via ``mc.analyse.my_RSA.build_across_halves_model_RDM``
    # — the SAME pairing + dispatch that the downstream RSA script
    # ``scripts/fMRI_run_RSA_without_rsatoolbox_clean.py`` uses. Each model's
    # ``X1_<dir>_<state>_<phase>`` row is paired with its same-goal partner
    # ``X2_<flipped_dir>_<state>_<phase>``, stacked, and turned into a
    # symmetric n_pairs × n_pairs across-halves RDM — exactly what enters the
    # searchlight regression downstream.

    if plot_RDMs == True:
        for model in models:
            if model == 'path_rew':
                continue
            if 'button' in model:
                continue
            if model not in EVs:
                continue
            try:
                res = mc.analyse.my_RSA.build_across_halves_model_RDM(
                    model, EVs[model], EV_keys)
            except (AssertionError, ValueError, KeyError):
                # leaked-loop-variable models, inconsistent shapes, etc.
                continue
            if res is None:
                continue
            rdm_full, th1_keys, method, vrange = res
            print(f"now plotting across-halves RDM for {model} model "
                  f"(n_pairs = {rdm_full.shape[0]}, method = {method})")
            mc.plotting.results.plot_model_rdm_pub(
                rdm_full, th1_keys, task_to_goal_label,
                title=model, mask_lower=True,
                fig_width_cm=8.0, fig_height_cm=8.0, font_pt=10,
                **vrange,
                show=True,
            )

    # ── Publication figures (example subject only) ───────────────────────
    if PUBLICATION_FIGURES and sub == EXAMPLE_SUBJECT_FOR_FIGS:
        pub_fig_dir = os.path.join(beh_dir, 'pub_figures')
        os.makedirs(pub_fig_dir, exist_ok=True)
        print(f"\nBuilding publication figures for {sub} -> {pub_fig_dir}")

        for model in PUB_MODELS:
            if model not in EVs:
                print(f"  (skip) {model!r}: not in EVs")
                continue
            try:
                res = mc.analyse.my_RSA.build_across_halves_model_RDM(
                    model, EVs[model], EV_keys)
            except (AssertionError, ValueError, KeyError) as exc:
                print(f"  (skip) {model!r}: {exc}")
                continue
            if res is None:
                print(f"  (skip) {model!r}: no pairs after across-halves matching")
                continue
            rdm_full, th1_keys, method, vrange = res
            mc.plotting.results.plot_model_rdm_pub(
                rdm_full, th1_keys, task_to_goal_label,
                save_stem=os.path.join(pub_fig_dir, f'model_RDM_{model}'),
                title=model,
                fig_width_cm=PUB_FIG_WIDTH_CM,
                fig_height_cm=PUB_FIG_HEIGHT_CM,
                font_pt=PUB_FIG_FONT_PT,
                **vrange,
                show=False,
            )

        # Side-by-side schematic of each model for ONE example task. Prefer
        # ``A1_forw`` if available so the goal-config label is straightforward.
        example_task = ('A1_forw' if 'A1_forw' in task_to_goal_label
                        else sorted(task_to_goal_label)[0])
        mc.plotting.results.plot_model_activation_examples(
            EVs, PUB_MODELS, example_task, task_to_goal_label,
            save_stem=os.path.join(pub_fig_dir, 'model_activation_examples'),
            show=False,
        )
        print(f"  saved RDM figures + model-activation schematics")
    

    # import pdb; pdb.set_trace()          
    if save_RDMs: 
        # then save these matrices.
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        
        with open(f"{RDM_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'wb') as file:
            pickle.dump(EVs, file)
            
        print(f"saved EV dictionary as {RDM_dir}/{sub}_modelled_EVs_{EV_string}.pkl")
