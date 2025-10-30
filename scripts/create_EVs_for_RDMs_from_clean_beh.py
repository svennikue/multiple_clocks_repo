#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 13:31:21 2025

It saves EV files for FEAT, as well as an .fsf file that can be used as an input for the EVs,
making sure to order the EVs correctly.

based on 
1. clean_fmri_behaviour
2. EV_config file


@author: Svenja Küchenhoff
"""


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import mc
import pickle
import sys
import json
import shutil

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    data_dir = f"{source_dir}/data/pilot"
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    data_dir = f"{source_dir}/data/derivatives"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
# import pdb; pdb.set_trace()      
# --- Load configuration ---
# config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_simple.json"

config_file = sys.argv[2] if len(sys.argv) > 2 else "EV_config_all_paths_rews_split-buttons.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
version = config.get("name")
split_buttons = config.get("split_buttons", False)
regress_rewards = config.get("regress_rewards", True)
rewards_as_stick_function = config.get("rewards_as_stick_function", False)
regress_subpaths = config.get("regress_subpaths", False)
tasks_included = config.get("tasks_included", ['A', 'B', 'C', 'D', 'E'])
repeats_included = config.get("repeats_included", [1,2,3,4,5])

fut_step_x_loc_regs = config.get("fut_step_x_loc_regs", False)

states_included = config.get("states_included", ['A', 'B', 'C', 'D'])
duration_state = config.get("duration_state", False)
state_regs = config.get("state_regs", False)

# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]



for sub in subjects:
    # load the cleaned behavioural table.
    beh_dir = f"{data_dir}/{sub}/beh"
    beh_df = pd.read_csv(f"{beh_dir}/{sub}_beh_fmri_clean.csv")
    
    # define and make paths
    for th in [1,2]:
        EV_folder = f'{data_dir}/{sub}/func/EVs_{version}_half0{th}/'
        if os.path.exists(EV_folder):
            print("careful, the EV folder does exist- there might be other EVs and thus not all files will be output correctly! Deleting dir.")
            shutil.rmtree(EV_folder)
            os.makedirs(EV_folder)
        if not os.path.exists(EV_folder):
            os.makedirs(EV_folder)
    
        file_all = f"{sub}_fmri_pt{th}_all.csv"
        
        # load behavioural file
        df_all = pd.read_csv(beh_dir+'/'+file_all)
        first_TR_at = df_all['TR_received_no0'].dropna().unique().tolist()[0]
        beh_th1 = beh_df[beh_df['task_half'] == th].copy()


        # 
        #
        # FIRST:
        # button press regressors.
        
        # Button press EV -> will be a nuisance regressor.
        # for button press EVs I need to add the entries in nav_key_task.rt to 
        end_task = beh_th1[(~beh_th1['button_rts'].isna())]
        end_task_idx = end_task.index.to_list()
        start_task_idx = [0] + [e + 1 for e in end_task_idx]
        end_task = end_task.reset_index(drop = True)
        on_press, key_press = [], []
        for i, row in end_task.iterrows():
            onset_curr_task = beh_th1.at[start_task_idx[i], 't_curr_loc']
            # extract button presses from the rt item with all presses
            presses_curr_task = row['button_rts'].strip('[""]').split(', ') # Split the string into a list using a comma as the separator
            buttons_curr_task = row['button_keys'].strip('[""]').split(', ')
            # Convert the elements to floats and add to the point in time where they actually started
            presses_curr_task = [(float(time)+onset_curr_task) for time in presses_curr_task]
            buttons_curr_task = [button.strip("''") for button in buttons_curr_task]
            
            on_press=on_press+presses_curr_task
            key_press=key_press+buttons_curr_task

        if split_buttons == True:
            mapping = {'1':'left', '2':'up', '3':'down', '4':'right'}
            for button_val, button_name in mapping.items():
                # pick times where the key matches this button
                on_press = [t for k, t in zip(key_press, on_press) if str(k) == button_val]
                dur_press = np.full(len(on_press), 0.02)
                mag_press = np.ones(len(on_press))
                button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(on_press, dur_press, mag_press, button_name, EV_folder, first_TR_at) 
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(button_press_EV)
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cutted EV {button_name} file. Happened for subject {sub} in task half {th}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"{button_name}" + '.txt', array, delimiter="    ", fmt='%f')
        else:
            dur_press = np.ones(len(on_press)) * 0.02
            mag_press = np.ones(len(on_press))
            
            button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(on_press, dur_press, mag_press, 'press_EV', EV_folder, first_TR_at)
            
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(button_press_EV)
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV buttonpress file. Happened for subject {sub} in task half {th}")
                np.savetxt(str(EV_folder) + 'ev_' + "button" + '.txt', array, delimiter="    ", fmt='%f')
        

        if fut_step_x_loc_regs == True:
            # defines each regressor as one hot encoding
            # timings are in 't_curr_loc' column.
            fut_step_regs = mc.analyse.extract_and_clean.define_futsteps_x_locs_regressors(beh_th1)
            future_step_EV_names = [c for c in fut_step_regs.columns if c.startswith('loc')]
            for fut_step_EV in future_step_EV_names:
                fut_step_regs_curr_EV = fut_step_regs[(fut_step_regs[fut_step_EV]==1) & (~fut_step_regs['t_curr_rew'].isna())].copy()
                on_fut_step_EV = fut_step_regs_curr_EV['t_curr_rew'].to_list()
                if rewards_as_stick_function == True:
                    dur_fut_step_EV = np.ones(len(on_fut_step_EV))
                else:
                    dur_fut_step_EV = fut_step_regs_curr_EV['reward_delay'].to_list()
                mag_fut_step_EV = np.ones(len(on_fut_step_EV))
                fut_step_EV = mc.analyse.analyse_MRI_behav.create_EV(on_fut_step_EV, dur_fut_step_EV, mag_fut_step_EV, fut_step_EV, EV_folder, first_TR_at)
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(fut_step_EV)
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cutted future step EV {fut_step_EV} file. Happened for subject {sub} in task half {th}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"{fut_step_EV}" + '.txt', array, delimiter="    ", fmt='%f')

        if state_regs == True:
            states_included = ['A', 'B', 'C', 'D']
            # 1) Runs: increment when the state changes
            run_id = beh_th1['state'].ne(beh_th1['state'].shift()).cumsum()
        
            # 2) For each run: state, first time (onset), last time (offset), duration
            runs = (beh_th1.groupby(run_id, as_index=False)   # <- as_index=False
                      .agg(state=('state','first'),
                           onset=('t_curr_loc','first'),
                           offset=('t_curr_loc','last')))
            runs['duration'] = runs['offset'] - runs['onset']
        
            # 3) Per-state lists in state dicts
            onsets_by_state    = runs.groupby('state', sort=False)['onset'].apply(list).to_dict()
            
            if duration_state == 'full_path':
                durations_by_state = runs.groupby('state', sort=False)['duration'].apply(list).to_dict()
            # elif duration_state == 'only_rew':
            # add other options!
            
            # 4) Create one EV file per state in states_included
            for curr_state in states_included:
                onsets = onsets_by_state[curr_state]
                durs   = durations_by_state[curr_state]
                mags   = np.ones(len(onsets))
                state_EV = mc.analyse.analyse_MRI_behav.create_EV(onsets, durs, mags, f"state_{curr_state}", EV_folder, first_TR_at)
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(state_EV)
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cut state EV {curr_state} file. Happened for subject {sub} in task half {th}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"{curr_state}" + '.txt', array, delimiter="    ", fmt='%f')
                
         
        #
        #
        #
        #CONTINUE HERE!!!
                        
        import pdb; pdb.set_trace()
       
        # then, lastly, adjust the .fsf file I will use for the regression.
        if version in ['01', f"01-TR{version_TR}" , '02','02-e', '02-l', '03', '03-e', '03-l', '03-rep1', '03-rep2', '03-rep3', '03-rep4', '03-rep5', '03-2', '03-3', '03-4', '04', '05', '03-99', '03-999', '03-9999', '06', '06-rep1', '07']: 
            print('start loop 2')
            # collect all filepaths I just created.
            # this is a bit risky in case there have been other EVs in there that I didnt want...
            # optimise if you have time!
            files_in_EV_folder = os.listdir(EV_folder) 
            EV_paths = []
            for EV in files_in_EV_folder:
                if EV.startswith("ev_") and EV.endswith(".txt"):
                    EV_path = os.path.join(EV_folder, EV)
                    EV_paths.append(os.path.join(EV_folder, EV)) 
            print(f"I collected {len(EV_paths)} EVs to put into the fsf file.")
            sorted_EVs = sorted(EV_paths)
            
            text_to_write = []
            with open(f"{EV_folder}task-to-EV.txt", 'w') as file:
                for i, EV_path in enumerate(sorted_EVs): 
                    EV_file_name = EV_path.split('/')[-1].replace('.txt', '')
                    file.write(f'{i} {EV_file_name}\n')
                    
            if sub in ['sub-04', 'sub-06', 'sub-30', 'sub-31', 'sub-34']:
                template_name = 'my_RDM_GLM_v2.fsf'
            elif sub in ['sub-35'] and task_half == '1':
                template_name = 'my_RDM_GLM_v2.fsf'
            else:
                template_name = 'my_RDM_GLM_pnm.fsf'
                
            with open(f"{analysisDir}/templates/{template_name}", "r") as fin:                    
                for line in fin:
                    for i, EV_path in enumerate(sorted_EVs): 
                        if line.startswith(f"set fmri(custom{i+1})"):
                            # print(f"my old line was: {line}")
                            line = f'set fmri(custom{i+1}) "{EV_path}"\n'
                        if line.startswith(f"set fmri(evtitle{i+1})"):
                            EV_name_ext = os.path.basename(EV_path)
                            EV_name = EV_name_ext.rsplit('.',1)[0]
                            # print(f"changing evtitle{i+1} to {EV_name}")
                            line = f'set fmri(evtitle{i+1}) "{EV_name}"\n'
                        if line.startswith("set fmri(evs_orig)"):
                            line = f"set fmri(evs_orig) {len(EV_paths)}\n"
                        if line.startswith("set fmri(evs_real)"):
                            line = f"set fmri(evs_real) {len(EV_paths)+1}\n"   
                            # import pdb; pdb.set_trace();
                    text_to_write.append(line)
            
            # then, in the next round, delete all the EVs that I don't actually include.
            # first, do this for the orthogonalisation of the EVs + contrasts you want with the ones you don't.
            skip = 0
            text_to_write_half_cleaned = []
            for line in text_to_write:
                if skip > 0:
                    # if the counter is increased, skip next line and decrease counter
                    skip -= 1
                    continue
                if (line.startswith("# Orthogonalise EV") and int(line[-3:-1]) > len(EV_paths)) or (line.startswith("# Real contrast_orig") and int(line[-3:-1]) > len(EV_paths)) or (line.startswith("# Real contrast_real vector") and int(line[-3:-1]) > len(EV_paths)):
                    #print(f"end of line is {line[-3:-1]}, so skip these next 3")
                    skip = 2
                else:
                    #import pdb; pdb.set_trace();
                    text_to_write_half_cleaned.append(line)
                    
            # then, delete all the configurations of the actual EVs don't want.
            skip_until_marker = False
            marker_line = "# Contrast & F-tests mode"
            text_to_write_cleaned = []
            for line in text_to_write_half_cleaned:
                if skip_until_marker:
                    if line.strip() == marker_line:
                        # add marker line to text and stop skipping
                        text_to_write_cleaned.append(line)
                        skip_until_marker = False
                    continue
                if line.startswith("# EV") and int(line[5:7]) > len(EV_paths):
                    skip_until_marker = True
                else:
                    text_to_write_cleaned.append(line)
        
            with open(f"{funcDir}/{sub}_draft_GLM_0{task_half}_{version}.fsf", "w") as fout:
                for line in text_to_write_cleaned:
                    fout.write(line)
           
    # --- SETTINGS SUMMARY (per subject) ---
    summary = {
        "subject": sub,
        "name": version,
        "split_buttons": split_buttons,
        "repeats_included": repeats_included,
        "regress_rewards": regress_rewards,
        "rewards_as_stick_function": rewards_as_stick_function,
        "regress_subpaths": regress_subpaths,
        "tasks_included": tasks_included,
        "states_included": states_included,
        "fut_step_x_loc_regs": fut_step_x_loc_regs,
        "state_regs": state_regs,
        "data_dir": data_dir,
        "EV_folder": EV_folder,
    }
    
    print("\n=== SETTINGS SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    
    # Save a copy alongside results for provenance
    with open(os.path.join(EV_folder, f"{sub}_settings_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"(Saved summary → {os.path.join(EV_folder, f'{sub}_settings_summary.json')})\n")

#
#
#



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:25:55 2023
creates the EVs for the RDM conditions.

This is the first script that has to be run on the behavioural data to rund the RSA.
As an input, it requires the complete behavioural result file (to extract the TR), 
and the custom-created one (for the rest of the analysis).
One needs to set the subject list it needs to run for, the task-halves, which EVs
it should create and give the GLM a version number. 
It saves EV files for FEAT, as well as an .fsf file that can be used as an input for the EVs,
making sure to order the EVs correctly.

NEW:
GLM ('regression') settings (creating the 'bins'):
    01 - instruction EVs
    01-TR1 - instruction EV, first TR- modelled as a stick function
    02 - 80 regressors; every task is divided into 4 rewards + 4 paths
    03 - 40 regressors; for every tasks, only the rewards are modelled [using a stick function]
    03-e 40 regressors; for evert task, only take the first 2 repeats.
    03-l 40 regressors; for every task, only take the last 3 repeats.
        careful! sometimes, some trials are not finished and thus don't have any last runs. these are then empty regressors.
    03-rep1 40 regressors; for every task, only take the first repeat
    03-rep2 40 regressors; for every task, only take the second repeat
    03-rep3 40 regressors; for every task, only take the third repeat
    03-rep4 40 regressors; for every task, only take the fourth repeat
    03-rep5 40 regressors; for every task, only take the fifth repeat
    03-2 - 40 regressors; for every task, only the rewards are modelled (in their original time)
    03-3 - 30 regressors; for every task, only the rewards are modelled (in their original time), except for A (because of visual feedback)
    03-4 - 40 regressors; for every task, only the rewards are modelled; and NO button-press regressor!
    03-99 - 40 regressors; no button press; I allocate the reward onsets randomly to different state/task combos  -> shuffled through whole task; [using a stick function]
    03-999 - 40 regressors; no button press; created a random but sorted sample of onsets that I am using -> still somewhat sorted by time, still [using a stick function]
    03-9999 - 40 regressors; no button press; shift all regressors 6 seconds earlier
    04 - 40 regressors; for every task, only the paths are modelled
    05 - locations + button presses 
    06 - collapsed task period -> average per task, for the reactivation analysis
    06-rep1 - collapsed tasks, only first repeat. -> average of first task, for the reactivation analysis 
    07 - entire path and reward period, collapsed (= 03 + 04)
    
    

@author: Svenja Küchenhoff, 2024
"""

        
        # then, lastly, adjust the .fsf file I will use for the regression.
        if version in ['01', f"01-TR{version_TR}" , '02','02-e', '02-l', '03', '03-e', '03-l', '03-rep1', '03-rep2', '03-rep3', '03-rep4', '03-rep5', '03-2', '03-3', '03-4', '04', '05', '03-99', '03-999', '03-9999', '06', '06-rep1', '07']: 
            print('start loop 2')
            # collect all filepaths I just created.
            # this is a bit risky in case there have been other EVs in there that I didnt want...
            # optimise if you have time!
            files_in_EV_folder = os.listdir(EV_folder) 
            EV_paths = []
            for EV in files_in_EV_folder:
                if EV.startswith("ev_") and EV.endswith(".txt"):
                    EV_path = os.path.join(EV_folder, EV)
                    EV_paths.append(os.path.join(EV_folder, EV)) 
            print(f"I collected {len(EV_paths)} EVs to put into the fsf file.")
            sorted_EVs = sorted(EV_paths)
            
            text_to_write = []
            with open(f"{EV_folder}task-to-EV.txt", 'w') as file:
                for i, EV_path in enumerate(sorted_EVs): 
                    EV_file_name = EV_path.split('/')[-1].replace('.txt', '')
                    file.write(f'{i} {EV_file_name}\n')
                    
            if sub in ['sub-04', 'sub-06', 'sub-30', 'sub-31', 'sub-34']:
                template_name = 'my_RDM_GLM_v2.fsf'
            elif sub in ['sub-35'] and task_half == '1':
                template_name = 'my_RDM_GLM_v2.fsf'
            else:
                template_name = 'my_RDM_GLM_pnm.fsf'
                
            with open(f"{analysisDir}/templates/{template_name}", "r") as fin:                    
                for line in fin:
                    for i, EV_path in enumerate(sorted_EVs): 
                        if line.startswith(f"set fmri(custom{i+1})"):
                            # print(f"my old line was: {line}")
                            line = f'set fmri(custom{i+1}) "{EV_path}"\n'
                        if line.startswith(f"set fmri(evtitle{i+1})"):
                            EV_name_ext = os.path.basename(EV_path)
                            EV_name = EV_name_ext.rsplit('.',1)[0]
                            # print(f"changing evtitle{i+1} to {EV_name}")
                            line = f'set fmri(evtitle{i+1}) "{EV_name}"\n'
                        if line.startswith("set fmri(evs_orig)"):
                            line = f"set fmri(evs_orig) {len(EV_paths)}\n"
                        if line.startswith("set fmri(evs_real)"):
                            line = f"set fmri(evs_real) {len(EV_paths)+1}\n"   
                            # import pdb; pdb.set_trace();
                    text_to_write.append(line)
            
            # then, in the next round, delete all the EVs that I don't actually include.
            # first, do this for the orthogonalisation of the EVs + contrasts you want with the ones you don't.
            skip = 0
            text_to_write_half_cleaned = []
            for line in text_to_write:
                if skip > 0:
                    # if the counter is increased, skip next line and decrease counter
                    skip -= 1
                    continue
                if (line.startswith("# Orthogonalise EV") and int(line[-3:-1]) > len(EV_paths)) or (line.startswith("# Real contrast_orig") and int(line[-3:-1]) > len(EV_paths)) or (line.startswith("# Real contrast_real vector") and int(line[-3:-1]) > len(EV_paths)):
                    #print(f"end of line is {line[-3:-1]}, so skip these next 3")
                    skip = 2
                else:
                    #import pdb; pdb.set_trace();
                    text_to_write_half_cleaned.append(line)
                    
            # then, delete all the configurations of the actual EVs don't want.
            skip_until_marker = False
            marker_line = "# Contrast & F-tests mode"
            text_to_write_cleaned = []
            for line in text_to_write_half_cleaned:
                if skip_until_marker:
                    if line.strip() == marker_line:
                        # add marker line to text and stop skipping
                        text_to_write_cleaned.append(line)
                        skip_until_marker = False
                    continue
                if line.startswith("# EV") and int(line[5:7]) > len(EV_paths):
                    skip_until_marker = True
                else:
                    text_to_write_cleaned.append(line)
        
            with open(f"{funcDir}/{sub}_draft_GLM_0{task_half}_{version}.fsf", "w") as fout:
                for line in text_to_write_cleaned:
                    fout.write(line)
   

            
#
#
#
            
