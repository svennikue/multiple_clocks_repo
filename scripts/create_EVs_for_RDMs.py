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
    02 - 80 regressors; every task is divided into 4 rewards + 4 paths
    03 - 40 regressors; for every tasks, only the rewards are modelled [using a stick function]
    03-e 40 regressors; for evert task, only take the first 2 repeats.
    03-l 40 regressors; for every task, only take the last 3 repeats.
        careful! sometimes, some trials are not finished and thus don't have any last runs. these are then empty regressors.
    03-2 - 40 regressors; for every task, only the rewards are modelled (in their original time)
    03-3 - 30 regressors; for every task, only the rewards are modelled (in their original time), except for A (because of visual feedback)
    03-4 - 40 regressors; for every task, only the rewards are modelled; and NO button-press regressor!
    03-99 - 40 regressors; no button press; I allocate the reward onsets randomly to different state/task combos  -> shuffled through whole task; [using a stick function]
    03-999 - 40 regressors; no button press; created a random but sorted sample of onsets that I am using -> still somewhat sorted by time, still [using a stick function]
    03-9999 - 40 regressors; no button press; shift all regressors 6 seconds earlier
    04 - 40 regressors; for every task, only the paths are modelled
    05 - locations + button presses 
    

OLD
06.12.2023: version 06 for RDM GLM. TR is different, made the script nicer. Everything else should be the same. 

GLM settings (creating the 'bins'):
    03 was location_EVs. (very long back)
    06 is reward + path phase per task [80 EVs] new, better script is now 06 #'04_pt01+_that_worked' 
    07 is only button press and rewards.
    08 is rewards only and without A (because of the visual feedback)
    09 is the instruction period only.
    10 is only paths
    11 is only rewards as a stick function
    

@author: Svenja Küchenhoff, 2024
"""


import pandas as pd
import os
import numpy as np
import mc
import matplotlib.pyplot as plt
import pickle
import re
import sys
import shutil
import random


#import pdb; pdb.set_trace()

version = '02-l'

# plotting = True
# to debug task_halves = ['1']


task_halves = ['1', '2']
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

# subjects = ['sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18','sub-19', 'sub-20',  'sub-22', 'sub-23','sub-24']
#subjects = ['sub-01']    
subjects = [f"sub-{subj_no}"]

    
if version == '05':
    split_buttons = True
else:
    split_buttons = False
    
    
if version in ['03-4', '03-99', '03-999', '03-9999']:
    no_buttons = True
else:
    no_buttons = False
    
    
analyse_behav = True
    


for sub in subjects:
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
        

        # define and make paths
        EV_folder = f'{funcDir}/EVs_{version}_pt0{task_half}/'
        if os.path.exists(EV_folder):
            print("careful, the EV folder does exist- there might be other EVs and thus not all files will be output correctly! Deleting dir.")
            shutil.rmtree(EV_folder)
            os.makedirs(EV_folder)
        if not os.path.exists(EV_folder):
            os.makedirs(EV_folder)
        
        # load behavioural file
        df = pd.read_csv(data_dir_beh + f"{file}.csv")
        df_all = pd.read_csv(data_dir_beh+file_all)
        
        # Keep in case I want to look at the behaviour at some point, but not really needed for now.
        if analyse_behav: 
            df_analysed, df_clean = mc.analyse.analyse_MRI_behav.analyse_pathlength_beh(df)
            
            
        # Identify where the first trigger was collected.
        # this is the value you have to substract from all other timings.
        first_TR_at = df_all['TR_received_no0'].dropna().unique().tolist()[0]

        # note, these generally are timings I can trust, based on recordings with the global clock:
        # key_resp_test.rt; scanner_prompt_start & end; TR_received_no0; start_ABCD_screen
        # TR updated 06.12.2023 I believe this needs to be TR_received_no0, actually.

        # Button press EV -> will be a nuisance regressor.
        # for button press EVs I need to add the entries in nav_key_task.rt to 
        # the global time when this state started: start_ABCD_screen a few rows before.
        new_task = df[(~df['start_ABCD_screen'].isna())] 
        new_task = new_task.reset_index(drop=True)
        end_task = df[(~df['nav_key_task.rt'].isna())]
        end_task = end_task.reset_index(drop=True)
        
        
        # in case there are started tasks that have not been ended:
        # delete the last row of new_task to make it equally long
        while len(new_task) > len(end_task):
            drop_last_row_index = new_task.index[-1]
            new_task = new_task.drop(drop_last_row_index)
            new_task = new_task.reset_index(drop=True)
        
        if no_buttons == False:
            on_press = []
            key_press = []
            for i, row in end_task.iterrows():
                curr_presses = row['nav_key_task.rt'] # extract button presses from the rt item with all presses
                presses_curr_task = curr_presses.strip('[]').split(', ') # Split the string into a list using a comma as the separator
                curr_buttons = row['nav_key_task.keys']
                buttons_curr_task = curr_buttons.strip('[]').split(', ') 
                # Convert the elements to floats and add to the point in time where they actually started
                presses_curr_task = [(float(time)+new_task.at[i, 'start_ABCD_screen']) for time in presses_curr_task]
                buttons_curr_task = [button.strip("''") for button in buttons_curr_task]
                
                on_press=on_press+presses_curr_task
                key_press=key_press+buttons_curr_task
    
            
            if split_buttons == False:    
                # the duration can just be something like 20 ms
                dur_press = np.ones(len(on_press)) * 0.02
                mag_press = np.ones(len(on_press))
                
                button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(on_press, dur_press, mag_press, 'press_EV', EV_folder, first_TR_at)
                
            
            # make one that differentiates between buttons
            
            # make 4 more specific button-press regressors, instead of the one unspecific one.
            # think about this.
            # which times do I want to take for the task-space clocks model???
            # how do I de-correlate this from the button presses?
            # probably no button-press nuisance regressor for task-space model.
            # is locaton model = button press regressors??
            
            # careful! Now I need to put 4 additional instead of only 1 in the subject-level GLM.
            if split_buttons == True:
                buttons_I_want = ['left', 'up', 'down', 'right']
                button_press_dict = {f"on_{button}": [] for button in buttons_I_want}
                for i, time in enumerate(on_press):
                    if key_press[i] == '1':
                        button_press_dict['on_left'].append(time)
                    elif key_press[i] == '2':
                        button_press_dict['on_up'].append(time)
                    elif key_press[i] == '3':
                        button_press_dict['on_down'].append(time)
                    elif key_press[i] == '4':
                        button_press_dict['on_right'].append(time)
                
                for button in buttons_I_want:
                    button_press_dict[f"dur_{button}"] = np.ones(len(button_press_dict[f"on_{button}"])) * 0.02
                    button_press_dict[f"mag_{button}"] = np.ones(len(button_press_dict[f"on_{button}"]))
                
                for button in buttons_I_want:
                    button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(button_press_dict[f"on_{button}"], button_press_dict[f"dur_{button}"], button_press_dict[f"mag_{button}"], f"{button}", EV_folder, first_TR_at)
                                  
                

            # check there are no nans 
            deleted_x_rows, button_press_EV_to_save = mc.analyse.analyse_MRI_behav.check_for_nan(button_press_EV)
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV button press file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'press_EV' + '.txt', button_press_EV_to_save, delimiter="    ", fmt='%f')
        
        # import pdb; pdb.set_trace()
        if version == '05':
            # import pdb; pdb.set_trace()
            # # Location EVs.
            location_EVs_dict = {}
            list_coords_x = [-0.21, 0, 0.21, -0.21, 0, 0.21, -0.21, 0, 0.21]
            list_coords_y = [-0.29, -0.29, -0.29, 0, 0, 0, 0.29, 0.29, 0.29]
            list_names = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            
            for i, name in enumerate(list_names):
                coord_x = list_coords_x[i]
                coord_y = list_coords_y[i]
                loc_on, loc_dur, loc_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x, coord_y)
                loc_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_on, loc_dur, loc_mag, f"loc_{name}_EV", EV_folder, first_TR_at)
                deleted_x_rows, location_EVs_dict[name] = mc.analyse.analyse_MRI_behav.check_for_nan(loc_EV)
                location_EVs_dict[name]
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cutted EV loc_{name}_EV file. Happened for subject {sub} in task half {task_half}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"loc_{name}_EV" + '.txt', location_EVs_dict[name], delimiter="    ", fmt='%f')
 
        
 
        if version == '01': # instruction
            # extract the timings of where a task has ended: t_reward_afterwait & repeat == '4' 
            # + 3.5 reward text + instruction period lasts 12 seconds; 
            # so t_reward_afterwait + 3.5 rew + 12 sec should be ca. start_ABCD screen
            # first, for the first task do:
            df.loc[df.index[~df['start_ABCD_screen'].isna()][0], 'instruct_start'] = df.loc[df.index[~df['start_ABCD_screen'].isna()][0], 'start_ABCD_screen'] - 12
            # for the other tasks, loop through table:
            for index, row in df.iterrows():
                if (row['rep_runs.thisN'] == 5) and (~pd.isna(row['t_reward_afterwait'])):
                   df.at[index+1, 'instruct_start'] = row['t_reward_afterwait'] + 3.5
                                                             

            # create a reward type to filter for same tasks
            # column which allows to differentiate all trials
            df['config_type'] = df['task_config'] + '_' + df['type']
            df['config_type'] = df['config_type'].fillna(method='ffill')
            task_names = df['config_type'].dropna().unique().tolist()
            
            # then make regressors based on that.
            instruc_EV_dic = {}
            for i, task in enumerate(task_names):
                EVname_instruction_onset = f"{task}_instruction_onset"
                partial_df = df[((df['config_type'] == task) & (~df['instruct_start'].isna()))]
                instruc_EV_dic[EVname_instruction_onset] = partial_df['instruct_start'].tolist()
                
                dur_instruct = [12] # duration is always 12 seconds
                mag_instruct = np.ones(len(instruc_EV_dic[f"{task}_instruction_onset"]))
                
                instruction_EV = mc.analyse.analyse_MRI_behav.create_EV(instruc_EV_dic[f"{task}_instruction_onset"], dur_instruct , mag_instruct, f"{task}_instruction_onset", EV_folder, first_TR_at)
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(instruction_EV)
                # import pdb; pdb.set_trace()
                
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cutted EV {task} file. Happened for subject {sub} in task half {task_half}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"{task}_instruction_onset" + '.txt', array, delimiter="    ", fmt='%f')
            # additionally check if I made a regressor for each task.
            if len(instruc_EV_dic) < 10:
                print(f"careful! Less instruction periods than tasks (10) have been saved. Happened for subject {sub} in task half {task_half}")
                
                
            
        if version in ['02','02-e', '02-l', '03', '03-e', '03-l', '03-2', '03-3', '03-4','03-99','03-999','03-9999', '04']: #06 is subpath and reward, 07 only reward, 08 is reward without A reward
            # 10 is only paths
            # identify where the next task begins by iterating through the DataFrame 
            # and collecting the indices where the column is not empty
            index_next_task = []
            for index, row in df.iterrows():
                if not pd.isna(row['start_ABCD_screen']):
                    index_next_task.append(index)
                 
            # compute the task length for each task - careful! this only works if the task was completed.
            for i, index in enumerate(index_next_task):
                df.at[index, 'task_onset'] = df.at[index, 'start_ABCD_screen'] 
                df.at[index+1, 'subpath_onset'] = df.at[index, 'start_ABCD_screen'] 
     
            # identify where the next reward starts by iterating through the DataFrame 
            # and collecting the indices where the column is not empty
            index_next_reward = []
            for index, row in df.iterrows():
                if not pd.isna(row['t_reward_start']):
                    index_next_reward.append(index)
            
            for i, index in enumerate(index_next_reward):
                df.at[index, 'reward_onset'] = df.at[index, 't_reward_start']
                df.at[index+1, 'subpath_onset'] = df.at[index, 't_step_press_global']
                if df.at[index, 'state'] == 'D':
                    if df.at[index, 'rep_runs.thisN'] == 5: #if the next task starts, this is way more precise.
                        df.at[index, 'reward_duration'] = df.at[index, 'reward_delay']
                    else:
                        df.at[index, 'reward_duration'] = df.at[index + 1, 't_step_press_global'] - df.at[index, 't_reward_start']
                else:
                    df.at[index, 'reward_duration'] = df.at[index, 't_step_press_global'] - df.at[index, 't_reward_start']
             
            index_next_subpath = []
            for index, row in df.iterrows():
                if not pd.isna(row['subpath_onset']):
                    index_next_subpath.append(index)
                              
            for i, index in enumerate(index_next_subpath):
                if i+1 < len(index_next_reward):
                    df.at[index, 'subpath_dur_with_rew'] = df.at[index_next_subpath[i+1], 'subpath_onset'] - df.at[index, 'subpath_onset']
                    df.at[index, 'subpath_dur_without_rew'] = df.at[index_next_reward[i], 'reward_onset'] - df.at[index, 'subpath_onset']
                
            # I need 8 regressors per task (e.g. C1). I have 5 * 2 tasks.
            # actually, C1 forward = C2 backward. For now, don't put together
            
            # look at to check if its really the same task. For this, create a reward type 
            # column which allows to differentiate all trials
            df['config_type'] = df['task_config'] + '_' + df['type']
            df['config_type'] = df['config_type'].fillna(method='ffill')
            task_names = df['config_type'].dropna().unique().tolist()
            state_names = df['state'].dropna().unique().tolist()
            
            if version == '03-3': # without the A-state because of visual feedback
                state_names.remove('A')
            
            if version in ['03-99']:
                shuffled_df = df['reward_onset'].dropna().to_list()
                np.random.shuffle(shuffled_df)
            
            if version in ['03-999']:
                valid_onset_times = df['reward_onset'].dropna().to_list()
                earliest_onset = min(valid_onset_times)
                latest_onset = max(valid_onset_times)
                random_onsets = sorted(random.sample(range(int(earliest_onset), int(latest_onset)), len(valid_onset_times)))
            
            
            taskEV_dic = {}
            # e.g. for 06 I want 80 EVs in the end -> 160 elements in the dictionary (duration + onset)
            counter = 0
            for i, task in enumerate(task_names):
                # if task == 'A2_backw':
                #     import pdb; pdb.set_trace()
                for s, state in enumerate(state_names):
                    # import pdb; pdb.set_trace()
                    if version in ['02','02-e', '02-l', '03',  '03-e', '03-l','03-3', '03-2', '03-4', '03-99', '03-999', '03-9999']:
                        EV_rewardname_onset = f"{task}_{state}_reward_onset"
                        EV_rewardname_dur = f"{task}_{state}_reward_dur"
                    if version in ['02','02-e', '02-l', '04']: # inlude subpaths
                        EV_subpathname_onset = f"{task}_{state}_subpath_onset"
                        EV_subpathname_dur = f"{task}_{state}_subpath_dur"

                    partial_df = df[((df['config_type'] == task) & (df['state'] == state))]
                    
                    if version in ['02','02-e', '02-l', '03', '03-e', '03-l', '03-2', '03-3', '03-4', '03-99', '03-999', '03-9999']:
                        # import pdb; pdb.set_trace()
                        if version in ['02-e','03-e']:
                            taskEV_dic[EV_rewardname_onset] = partial_df['reward_onset'].dropna().to_list()[0:2]
                        elif version in ['02-l','03-l']:
                            taskEV_dic[EV_rewardname_onset] = partial_df['reward_onset'].dropna().to_list()[2:]
                        else:
                            taskEV_dic[EV_rewardname_onset] = partial_df['reward_onset'].dropna().to_list()
                        if version in ['03-99']:
                            taskEV_dic[EV_rewardname_onset] = shuffled_df[counter: counter+len(taskEV_dic[EV_rewardname_onset])]
                        if version in ['03-999']:
                            taskEV_dic[EV_rewardname_onset] = random_onsets[counter:counter+len(taskEV_dic[EV_rewardname_onset])]
                        if version in ['03-9999']:
                            taskEV_dic[EV_rewardname_onset] = [elem - 6 for elem in taskEV_dic[EV_rewardname_onset]]
                            # # be careful to not make this longer than the actual fMRI file!
                            # if taskEV_dic[EV_rewardname_onset][-1] > df['reward_onset'].dropna().to_list()[-1]:
                            #     taskEV_dic[EV_rewardname_onset][-1] =  df['reward_onset'].dropna().to_list()[-1]
                        if version in ['02', '02-l', '02-e', '03', '03-e', '03-l','03-99', '03-999', '03-9999']: # reward as stick-function: duration of all rewards to 500ms -> all regressors will be equally long.
                            taskEV_dic[EV_rewardname_dur] = np.ones(len(taskEV_dic[EV_rewardname_onset])) * 0.5
                        elif version in ['03-2', '03-3', '03-4']:
                            taskEV_dic[EV_rewardname_dur] = partial_df['reward_duration'].dropna().to_list()
                        mag_reward = np.ones(len(taskEV_dic[EV_rewardname_onset]))
                        # if version in ['03-99']:
                        #     # maybe better than this is to just take the entire dataset and shuffle it, because then there
                        #     # is not the possibility of creating crazy overlapping regressors.
                            
                        #     task_start = min(taskEV_dic[EV_rewardname_onset])
                        #     task_end = max(taskEV_dic[EV_rewardname_onset]) + taskEV_dic[EV_rewardname_dur][-1]
                        #     new_onset = random.sample(range(int(task_start), int(task_end)), 5)
                            
                        # if len(mag_reward) < 3:
                        #     print(f"Careful! {task} x {state} reward is not complete and will be excluded.")
                        #     excluded = excluded + 1
                        #     continue
                        reward_EV = mc.analyse.analyse_MRI_behav.create_EV(taskEV_dic[f"{task}_{state}_reward_onset"], taskEV_dic[f"{task}_{state}_reward_dur"], mag_reward, f"{task}_{state}_reward", EV_folder, first_TR_at)
                        deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(reward_EV)
                        if deleted_x_rows > 0:
                            print(f"careful! I am saving a cutted EV {task}{state} reward file. Happened for subject {sub} in task half {task_half}")
                            np.savetxt(str(EV_folder) + 'ev_' + f"{task}_{state}_reward" + '.txt', array, delimiter="    ", fmt='%f')
                        
                    if version in ['02', '02-e', '02-l', '04']: #include subpaths
                        if version in ['02-e']:
                            taskEV_dic[EV_subpathname_onset] = partial_df['subpath_onset'].dropna().to_list()[0:2]
                            taskEV_dic[EV_subpathname_dur] = partial_df['subpath_dur_without_rew'].dropna().to_list()[0:2]
                        elif version in ['02-l']:
                            taskEV_dic[EV_subpathname_onset] = partial_df['subpath_onset'].dropna().to_list()[2:]
                            taskEV_dic[EV_subpathname_dur] = partial_df['subpath_dur_without_rew'].dropna().to_list()[2:]
                        else:
                            taskEV_dic[EV_subpathname_onset] = partial_df['subpath_onset'].dropna().to_list()
                            taskEV_dic[EV_subpathname_dur] = partial_df['subpath_dur_without_rew'].dropna().to_list()
                        mag_subpath = np.ones(len(taskEV_dic[EV_subpathname_onset]))
                        # if len(mag_subpath) < 3:
                        #     print(f"Careful! {task} x {state} subpath and reward is not complete and will be excluded.")
                        #     excluded = excluded + 2 # bc reward will also be exluded
                        #     continue
                        subpath_EV = mc.analyse.analyse_MRI_behav.create_EV(taskEV_dic[f"{task}_{state}_subpath_onset"], taskEV_dic[f"{task}_{state}_subpath_dur"], mag_subpath, f"{task}_{state}_path", EV_folder, first_TR_at)
                        deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(subpath_EV)
                        if deleted_x_rows > 0:
                            print(f"careful! I am saving a cutted EV {task}{state} subpath file. Happened for subject {sub} in task half {task_half}")
                            np.savetxt(str(EV_folder) + 'ev_' + f"{task}_{state}_path" + '.txt', array, delimiter="    ", fmt='%f')
                    
                    counter = counter + 1  
                    
            # lastly, save the taskEV_dic so that I can also use it as data regressors.
            with open(f"{EV_folder}my_EV_dict", 'wb') as f:
                pickle.dump(taskEV_dic, f)
     
        # then, lastly, adjust the .fsf file I will use for the regression.
        if version in ['01', '02','02-e', '02-l', '03', '03-e', '03-l','03-2', '03-3', '03-4', '04', '05', '03-99', '03-999', '03-9999']: #06 is subpath and reward, 07 only reward, 08 is reward without A reward, 09 is instruction period
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
   


        # if plotting == True:
        #     # create a list of all EV variables
        #     allEVnames = [var for var in globals() if '_EV' in var]
        #     comboEVs = np.vstack([globals()[var] for var in allEVnames])
        #     # but also store the indices when a new EV starts in the combo file
        #     EVend_indices = [0] + [v.shape[0] for v in [globals()[var] for var in allEVnames]]
        #     EVend_indices_cum = np.cumsum(EVend_indices)
        #     title = ['onset', 'duration', 'magnitude']
        #     fig, axes = plt.subplots(1,3, figsize = (15,5))
        #     # loop through each columm amd create a bar plot
        #     for i, ax in enumerate(axes):
        #         column_data = comboEVs[:,i]
        #         ax.imshow(column_data.reshape(-1,1), cmap = 'viridis', aspect = 'auto')
        #         # ax.imshow(column_data, cmap = 'viridis', aspect = 'auto')
        #         ax.set_xticks([]) # remove x-axis ticks
        #         ax.set_title(f'EV {title[i]}')
        #         for line_pos in EVend_indices_cum[1:]:
        #             ax.axhline(y = line_pos-0.5, color = 'red', linestyle ='--')

            

            
            
