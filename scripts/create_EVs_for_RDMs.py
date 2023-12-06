#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:25:55 2023
creates the EVs for the RDM conditions.

ToDo: rewrite with input args.

@author: Svenja KUechenhoff
"""


import pandas as pd
import os
import numpy as np
import mc
import matplotlib.pyplot as plt

import pdb; pdb.set_trace()

#subjects = ['sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
subjects = ['sub-01']
version = '04' # GLM number
# to debug
#task_halves = ['1']
task_halves = ['1', '2']
locationEVs = False

# NEXT DOUBLE CHECK THESE
# THEY ARE NOT SPLIT YET BY BACKW FORW!!
time_binEVs = True
plotting = False
analyse_behav = False

for sub in subjects:
    for task_half in task_halves:
        data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        file = f"{sub}_fmri_pt{task_half}"
        file_all = f"{sub}_fmri_pt{task_half}_all.csv"
        
        # This is for location EVs:
        if locationEVs:
            EV_folder = f'/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/EVs_{version}_pt0{task_half}_press_and_loc/'
        elif time_binEVs:
            EV_folder = f'/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/EVs_{version}_pt0{task_half}/'
        if not os.path.exists(EV_folder):
            os.makedirs(EV_folder)
    
        df = pd.read_csv(data_dir + f"{file}.csv")
        df_all = pd.read_csv(data_dir+file_all)
        
        
        if analyse_behav: 
            # identify where the next task begins by iterating through the DataFrame 
            # and collecting the indices where the column is not empty
            index_next_task = []
            for index, row in df.iterrows():
                if not pd.isna(row['start_ABCD_screen']):
                    index_next_task.append(index)
            
            # compute the task length for each task
            # careful! this only works if the task was completed.
            # also this isn't super precise since it doesn't actually show where they 
            # walked but where they were able to move away from reward
            for i, index in enumerate(index_next_task):
                if i+1 < len(index_next_task):
                    df.at[index, 'task_length'] = df.at[index_next_task[i+1] - 1 , 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen']   
                    if 'type' in df.columns:
                        df.at[index, 'type'] = df.at[index+ 1, 'type']
                elif i+1 == len(index_next_task):
                    df.at[index, 'task_length'] = df.at[len(df)-1, 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen'] 
                            
            # not sure why I included this... seems wrong.      
            # index_next_task = index_next_task[1:]
                            
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
            # do the same for the task_config 
            df['task_config'] = df['task_config'].fillna(method='ffill')
            # and create a reward type column which allows to differentiate all trials
            df['config_type'] = df['task_config'] + '_' + df['type']
                        
            
            # import pdb; pdb.set_trace()
            # create a new column in which you plot how long ever subpath takes (with rew)
            j = 0
            for i, task_index in enumerate(index_next_task):
                if task_index > 1:
                    while (len(index_next_reward) > j) and (index_next_reward[j] < task_index):
                        df.at[index_next_reward[j], 'cum_subpath_length_without_rew'] = df.at[index_next_reward[j], 't_step_press_curr_run'] + df.at[index_next_reward[j]-1, 'length_step'] 
                        df.at[index_next_reward[j], 'cum_subpath_length_with_rew'] = df.at[index_next_reward[j], 't_step_press_curr_run'] + df.at[index_next_reward[j]-1, 'length_step'] + df.at[index_next_reward[j], 'reward_delay'] 
                        j += 1
                    df.at[task_index-1, 'cum_subpath_length_without_rew'] = df.at[index_next_task[i-1], 'task_length'] - df.at[task_index-1, 'reward_delay']
                    df.at[task_index-1, 'cum_subpath_length_with_rew'] = df.at[index_next_task[i-1], 'task_length']
                    # df.at[task_index-1, 't_step_press_curr_run'] + df.at[task_index-2, 'length_step'] + df.at[task_index-1, 'reward_delay'] 
                # for the next reward count backwards
                if task_index == index_next_task[-1]:
                    for i in range(4,0, -1):
                        df.at[index_next_reward[-i], 'cum_subpath_length_without_rew']= df.at[index_next_reward[-i], 't_step_press_curr_run'] + df.at[index_next_reward[-i]-1, 'length_step'] 
                        df.at[index_next_reward[-i], 'cum_subpath_length_with_rew']= df.at[index_next_reward[-i], 't_step_press_curr_run'] + df.at[index_next_reward[-i]-1, 'length_step'] + df.at[index_next_reward[-i], 'reward_delay']
        
            states = ['A', 'B', 'C', 'D']*len(index_next_task)
            
            
            
            # then, write the not- cumulative columns.
            for i, reward_index in enumerate(index_next_reward):
                if i < len(states):
                    df.at[reward_index, 'state'] = states[i]
                if i > 0:
                    df.at[reward_index, 'subpath_length_without_rew'] = df.at[reward_index, 'cum_subpath_length_without_rew'] - df.at[index_next_reward[i-1], 'cum_subpath_length_with_rew']
                    df.at[reward_index, 'subpath_length_with_rew'] = df.at[reward_index, 'cum_subpath_length_with_rew'] - df.at[index_next_reward[i-1], 'cum_subpath_length_with_rew']
        
            for i in range(0, len(index_next_reward), 4):
                df.at[index_next_reward[i], 'subpath_length_without_rew'] = df.at[index_next_reward[i], 'cum_subpath_length_without_rew'] 
                df.at[index_next_reward[i], 'subpath_length_with_rew'] = df.at[index_next_reward[i], 'cum_subpath_length_with_rew']
        
            
            #first reduce to only including those rows that have values for rewards.
            df_clean = df.dropna(subset = ['subpath_length_with_rew'])
            
            
            
        # the first scanner trigger is in row 3, scanner_trigger_key.started
        # it will be the first index of this 
        # and collecting where the first TRs have been received
        # first_TR_at = df_all['TR_received_no0'].dropna().unique().tolist()[0]
        # index_first_TRs = []
        # for index, row in df_all.iterrows():
        #     if not pd.isna(row['TR_received_no0']):
        #         index_first_TRs.append(index)
        
        # this is the value you have to substract from all other timings.

        # hmmm why is this not the first TR_received???
        # timings I can trust are the ones recorded with the global clock:
        # key_resp_test.rt; scanner_prompt_start & end; TR_received_no0; start_ABCD_screen
        # 06.12.2023 I believe this needs to be TR_received_no0, actually.
        #first_TR_at = df_all.at[index_first_TRs[0], 'time_scanner_prompt_end']
    
        
     
        # First, always compute button press EVs.
        
        # for button press EVs I need to add the entries in nav_key_task.rt to 
        # the global time when this state started: start_ABCD_screen a few rows before.
        new_task = df[(~df['start_ABCD_screen'].isna())] # but take the t_step_press_global as value!
        new_task = new_task.reset_index(drop=True)
        end_task = df[(~df['nav_key_task.rt'].isna())]
        end_task = end_task.reset_index(drop=True)
        
        # in case there are started tasks that have not been ended:
            # delete the last row of new_task to make it equally long
        while len(new_task) > len(end_task):
            drop_last_row_index = new_task.index[-1]
            new_task = new_task.drop(drop_last_row_index)
            new_task = new_task.reset_index(drop=True)
            
        on_press = []
        for i, row in end_task.iterrows():
            curr_presses = row['nav_key_task.rt']
            # Split the string into a list using a comma as the separator
            presses_curr_task = curr_presses.strip('[]').split(', ') 
            # Convert the elements to floats
            
            global_timing_curr_task = new_task.at[i, 't_step_press_global']
            presses_curr_task = [(float(time)+global_timing_curr_task) for time in presses_curr_task]
            on_press=on_press+presses_curr_task
            
        # the duration can just be something like 20 ms
        dur_press = np.ones(len(on_press)) * 0.02
        mag_press = np.ones(len(on_press))
        button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(on_press, dur_press, mag_press, 'press_EV', EV_folder, first_TR_at)
            
        mc.analyse.analyse_MRI_behav.check_for_nan(button_press_EV)
        
        
        if locationEVs:
            # # Location EVs.

            coord_x_one = -0.21
            coord_y_one = -0.29
            loc_one_on, loc_one_dur, loc_one_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_one, coord_y_one)
            loc_one_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_one_on, loc_one_dur, loc_one_mag, 'loc_one_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_one_EV)
            
            coord_x_two = 0
            coord_y_two = -0.29
            loc_two_on, loc_two_dur, loc_two_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_two, coord_y_two)
            loc_two_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_two_on, loc_two_dur, loc_two_mag, 'loc_two_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_two_EV)
                    
            coord_x_three = 0.21
            coord_y_three = -0.29
            loc_three_on, loc_three_dur, loc_three_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_three, coord_y_three)
            loc_three_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_three_on, loc_three_dur, loc_three_mag, 'loc_three_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_three_EV)       

            coord_x_four = -0.21
            coord_y_four = 0
            loc_four_on, loc_four_dur, loc_four_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_four, coord_y_four)
            loc_four_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_four_on, loc_four_dur, loc_four_mag, 'loc_four_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_four_EV)      
            
            
            coord_x_five = 0
            coord_y_five = 0
            loc_five_on, loc_five_dur, loc_five_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_five, coord_y_five)
            loc_five_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_five_on, loc_five_dur, loc_five_mag, 'loc_five_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_five_EV)       
                    
            coord_x_six = 0.21
            coord_y_six = 0
            loc_six_on, loc_six_dur, loc_six_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_six, coord_y_six)
            loc_six_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_six_on, loc_six_dur, loc_six_mag, 'loc_six_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_six_EV)       
             
            coord_x_seven = -0.21
            coord_y_seven = 0.29
            loc_seven_on, loc_seven_dur, loc_seven_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_seven, coord_y_seven)
            loc_seven_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_seven_on, loc_seven_dur, loc_seven_mag, 'loc_seven_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_seven_EV)        
            
            coord_x_eight = 0
            coord_y_eight = 0.29
            loc_eight_on, loc_eight_dur, loc_eight_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_eight, coord_y_eight)
            loc_eight_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_eight_on, loc_eight_dur, loc_eight_mag, 'loc_eight_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_eight_EV)        
            
            coord_x_nine = 0.21
            coord_y_nine = 0.29
            loc_nine_on, loc_nine_dur, loc_nine_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_nine, coord_y_nine)
            loc_nine_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_nine_on, loc_nine_dur, loc_nine_mag, 'loc_nine_EV', EV_folder, first_TR_at)
            mc.analyse.analyse_MRI_behav.check_for_nan(loc_nine_EV)

        
        if time_binEVs:
            # identify where the next task begins by iterating through the DataFrame 
            # and collecting the indices where the column is not empty
            index_next_task = []
            for index, row in df.iterrows():
                if not pd.isna(row['start_ABCD_screen']):
                    index_next_task.append(index)
                 

            # compute the task length for each task
            # careful! this only works if the task was completed.
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
                    df.at[index, 'reward_duration'] = df.at[index + 1, 't_step_press_global'] - df.at[index, 't_reward_start']
                else:
                    df.at[index, 'reward_duration'] = df.at[index, 't_step_press_global'] - df.at[index, 't_reward_start']
             
            index_next_subpath = []
            for index, row in df.iterrows():
                if not pd.isna(row['subpath_onset']):
                    index_next_subpath.append(index)
                    
            #index_next_subpath = index_next_subpath[1:]
                        
            for i, index in enumerate(index_next_subpath):
                if i+1 < len(index_next_reward):
                    df.at[index, 'subpath_dur_with_rew'] = df.at[index_next_subpath[i+1], 'subpath_onset'] - df.at[index, 'subpath_onset']
                    df.at[index, 'subpath_dur_without_rew'] = df.at[index_next_reward[i], 'reward_onset'] - df.at[index, 'subpath_onset']
                
            
            # now. I need 8 regressors per task (e.g. C1). I have 5 * 2 tasks.
            # actually, C1 forward = C2 backward. For now, don't put together
            
            # look at to check if its really the same task. For this, create a reward type column which allows to differentiate all trials
            df['config_type'] = df['task_config'] + '_' + df['type']
            task_names = df['config_type'].dropna().unique().tolist()
            
            # check if that works??
            # task_names = df['task_config'].dropna().unique().tolist()
            
            # I want 80 EVs!!
            
            state_names = df['state'].dropna().unique().tolist()
            taskEV_dic = {}
            for i, task in enumerate(task_names):
                for s, state in enumerate(state_names):
                    EV_subpathname_onset = f"{task}_{state}_subpath_onset"
                    EV_subpathname_dur = f"{task}_{state}_subpath_dur"
                    EV_rewardname_onset = f"{task}_{state}_reward_onset"
                    EV_rewardname_dur = f"{task}_{state}_reward_dur"
                    
                    partial_df = df[((df['config_type'] == task) & (df['state'] == state))]
                    #partial_df = df[((df['task_config'] == task) & (df['state'] == state))]
                    
                    taskEV_dic[EV_subpathname_onset] = partial_df['subpath_onset'].dropna().to_list()
                    taskEV_dic[EV_subpathname_dur] = partial_df['subpath_dur_without_rew'].dropna().to_list()
                    
                    taskEV_dic[EV_rewardname_onset] = partial_df['reward_onset'].dropna().to_list()
                    taskEV_dic[EV_rewardname_dur] = partial_df['reward_duration'].dropna().to_list()
                    
                
            for i, task in enumerate(task_names):
                for s, state in enumerate(state_names):
                    mag_subpath = np.ones(len(taskEV_dic[f"{task}_{state}_subpath_onset"]))
                    subpath_EV = mc.analyse.analyse_MRI_behav.create_EV(taskEV_dic[f"{task}_{state}_subpath_onset"], taskEV_dic[f"{task}_{state}_subpath_dur"], mag_subpath, f"{task}_{state}_subpath", EV_folder, first_TR_at)
                    mag_reward = np.ones(len(taskEV_dic[f"{task}_{state}_reward_onset"]))
                    reward_EV = mc.analyse.analyse_MRI_behav.create_EV(taskEV_dic[f"{task}_{state}_reward_onset"], taskEV_dic[f"{task}_{state}_reward_dur"], mag_reward, f"{task}_{state}_reward", EV_folder, first_TR_at)
                    mc.analyse.analyse_MRI_behav.check_for_nan(reward_EV)
                    mc.analyse.analyse_MRI_behav.check_for_nan(subpath_EV)
            
            if 'C2_forw_A_reward_dur' in taskEV_dic:
                print(f"Now done with {sub} and task half {task_half}. Made {len(taskEV_dic)/2} EVs, each with length {len(taskEV_dic['C2_forw_A_reward_dur'])}")       
            elif 'C1_forw_A_reward_dur' in taskEV_dic:
                print(f"Now done with {sub} and task half {task_half}. Made {len(taskEV_dic)/2} EVs, each with length {len(taskEV_dic['C1_forw_A_reward_dur'])}")       
            
            # I don't want combined ones.
            
            # # lastly, create EVs where the backwards and forwards task are together in one EV (i.e. 10 repeats per task, 10 tasks)
            # # I put a backwards version of the reversed (A1 = orig, A2 = reversed) in the same task.
            # # they thus did the same order twice within the task halfes. Thus, all A can be same EV
            # # eg:
            #     # A2 backw = -0.21 0.29, -0.21 -0.29, 0 0, 0.21 0.29 
            #     # A1 forw = -0.21 0.29, -0.21 -0.29, 0 0, 0.21 0.29
            #     # B1 forw = -0.21 0.29, 0.21 -0.29, -0.21 0.29, 0 -0.29
            #     # B2 back = -0.21 0.29, 0.21 -0.29, -0.21 0.29, 0 -0.29
                
            # EV_folder_combined = f'/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/EVs_combined_{version}_pt0{task_half}/'
            # if not os.path.exists(EV_folder_combined):
            #     os.mkdir(EV_folder_combined)
                
            # df_all = pd.read_csv(data_dir+file_all)
            
            # # I want to remove the variants: C1 forw = C2 backw
            # # careful, this is another level. I don't compare the actual reward order.
            # # todo: that would be more elegant
            # unique_letters = set()  # To track unique letters
            # tasks_backw_forw = []
            # for item in task_names:
            #     letter = ''.join(filter(str.isalpha, item))  # Remove numbers from the string
            #     if letter not in unique_letters:
            #         unique_letters.add(letter)
            #         tasks_backw_forw.append(letter)
            
            # #import pdb; pdb.set_trace()
            # taskEV_combined_dic = {}
            # for i, task in enumerate(tasks_backw_forw):
            #     for s, state in enumerate(state_names):
            #         EV_subpathname_onset = f"{task}_{state}_subpath_onset"
            #         EV_subpathname_dur = f"{task}_{state}_subpath_dur"
            #         EV_rewardname_onset = f"{task}_{state}_reward_onset"
            #         EV_rewardname_dur = f"{task}_{state}_reward_dur"
                    
            #         # partial_df = df[((df['task_config'].str.contains(task)) & (df['state'] == state))]
            #         partial_df = df[((df['config_type'].str.contains(task)) & (df['state'] == state))]
                    
            #         taskEV_combined_dic[EV_subpathname_onset] = partial_df['subpath_onset'].dropna().to_list()
            #         taskEV_combined_dic[EV_subpathname_dur] = partial_df['subpath_dur_without_rew'].dropna().to_list()
                    
            #         taskEV_combined_dic[EV_rewardname_onset] = partial_df['reward_onset'].dropna().to_list()
            #         taskEV_combined_dic[EV_rewardname_dur] = partial_df['reward_duration'].dropna().to_list()
                    
    
            # for i, task in enumerate(tasks_backw_forw):
            #     for s, state in enumerate(state_names):
            #         mag_subpath = np.ones(len(taskEV_combined_dic[f"{task}_{state}_subpath_onset"]))
            #         subpath_comb_EV = mc.analyse.analyse_MRI_behav.create_EV(taskEV_combined_dic[f"{task}_{state}_subpath_onset"], taskEV_combined_dic[f"{task}_{state}_subpath_dur"], mag_subpath, f"{task}_{state}_subpath", EV_folder_combined, first_TR_at)
            #         mag_reward = np.ones(len(taskEV_combined_dic[f"{task}_{state}_reward_onset"]))
            #         reward_comb_EV = mc.analyse.analyse_MRI_behav.create_EV(taskEV_combined_dic[f"{task}_{state}_reward_onset"], taskEV_combined_dic[f"{task}_{state}_reward_dur"], mag_reward, f"{task}_{state}_reward", EV_folder_combined, first_TR_at)
            #         mc.analyse.analyse_MRI_behav.check_for_nan(subpath_comb_EV)
            #         mc.analyse.analyse_MRI_behav.check_for_nan(reward_comb_EV)
            
        if plotting == True:
            # create a list of all EV variables
            allEVnames = [var for var in globals() if '_EV' in var]
            comboEVs = np.vstack([globals()[var] for var in allEVnames])
            # but also store the indices when a new EV starts in the combo file
            EVend_indices = [0] + [v.shape[0] for v in [globals()[var] for var in allEVnames]]
            EVend_indices_cum = np.cumsum(EVend_indices)
            title = ['onset', 'duration', 'magnitude']
            fig, axes = plt.subplots(1,3, figsize = (15,5))
            # loop through each columm amd create a bar plot
            for i, ax in enumerate(axes):
                column_data = comboEVs[:,i]
                ax.imshow(column_data.reshape(-1,1), cmap = 'viridis', aspect = 'auto')
                # ax.imshow(column_data, cmap = 'viridis', aspect = 'auto')
                ax.set_xticks([]) # remove x-axis ticks
                ax.set_title(f'EV {title[i]}')
                for line_pos in EVend_indices_cum[1:]:
                    ax.axhline(y = line_pos-0.5, color = 'red', linestyle ='--')

            

            
            
