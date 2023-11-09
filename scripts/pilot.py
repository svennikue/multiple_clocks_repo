#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:33:36 2023

@author: xpsy1114
here, I am having a look a the pilot data.
# my goals are: 
# 1. plot how long every task takes
# 2. plot how long every subpath takes
# 3. look at the interaction between fields and pathlength -> which paths are easiest?
# 4. check if they tend to walk the same routes

"""

import pandas as pd
import os
import numpy as np
import mc
import matplotlib.pyplot as plt
#import pdb; pdb.set_trace()

subjects = ['sub-01']
plotting = False
version = '04' # GLM number
task_halves = ['1', '2']

# filenames = ['00_pilot.csv', '01_pilot.csv', '02_pilot.csv', '03_pilot.csv', '04_pilot.csv', '05_pilot.csv', '06_pilot.csv', '07_pilot.csv', ]
#filenames = ['05_pilot.csv', '06_pilot.csv', '07_pilot.csv', '08_pilot.csv', '09_pilot.csv']
#filenames = ['01_practice.csv', '01_MRI_pt1.csv', '01_MRI_pt2.csv']


for sub in subjects:
    for task_half in task_halves:
        data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        file = f"{sub}_fmri_pt{task_half}"
        file_all = f"{sub}_fmri_pt{task_half}_all.csv"
        
        #filenames = [f"{sub}_fmri_pt1", f"{sub}_fmri_pt2"] 
        #file_all = [f"{sub}_fmri_pt1_all.csv", f"{sub}_fmri_pt2_all.csv"]
        
        #
        # no 1. How long does every task take?
        #
    
        df = pd.read_csv(data_dir+f"{file}.csv")
        
        #drop some unnecessary columns
        to_drop = ['rep_runs.thisRepN', 'rep_runs.thisTrialN', 'rep_runs.thisIndex', 'sand_box.started', 'sand_box.stopped', 'foot.started', 'foot.stopped']
        df.drop(to_drop, inplace=True, axis = 1)
    
        
        # identify where the next task begins by iterating through the DataFrame 
        # and collecting the indices where the column is not empty
        index_next_task = []
        for index, row in df.iterrows():
            if not pd.isna(row['start_ABCD_screen']):
                index_next_task.append(index)
        
        # compute the task length for each task
        # careful! this only works if the task was completed.
        for i, index in enumerate(index_next_task):
            if i+1 < len(index_next_task):
                df.at[index, 'task_length'] = df.at[index_next_task[i+1] - 1 , 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen']   
                if 'type' in df.columns:
                    df.at[index, 'type'] = df.at[index+ 1, 'type']
            elif i+1 == len(index_next_task):
                # start_ABCD screen really is where they can move for the first time.
                df.at[index, 'task_length'] = df.at[len(df)-1, 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen'] 
                
        df_clean = df.dropna(subset = ['round_no'])
    
        
        if plotting:
            # plot path lenghts per task configuration
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
                
            plt.axhline(30, color='black', linestyle='--', label='Reference 30 secs')
            
            plt.xticks(x_axis)
            plt.xlabel('Repeats')
            plt.ylabel('Time in seconds')
            plt.legend()
            plt.title(f'Task Length by Task Config for {file}')
            plt.show()
    
        
        #
        # no 2: plot how long every subpath takes, across participants and separetly per task configuration.
        #
        # filenames = ['00_pilot.csv', '01_pilot.csv', '02_pilot.csv', '03_pilot.csv', '04_pilot.csv', '05_pilot.csv', '06_pilot.csv', '07_pilot.csv', ]
        # to do that, just look at column 't_step_press_curr_run', filtered for those rows that have a reward.
        # plot reward times per subpath by looking at column 'reward_delay'. 
        # CAREFUL!!
        
        # I believe something went wrong with the timings here. See the big below (more correct)
    
        df = pd.read_csv(data_dir + f"{file}.csv")
        
        # identify where the next task begins by iterating through the DataFrame 
        # and collecting the indices where the column is not empty
        index_next_task = []
        for index, row in df.iterrows():
            if not pd.isna(row['start_ABCD_screen']):
                index_next_task.append(index)
        
        # compute the task length for each task
        # careful! this only works if the task was completed.
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
        
        # import pdb; pdb.set_trace()
        if plotting:
            # group by task config and subpath
            grouped = df_clean.groupby(['round_no', 'state'])['subpath_length_with_rew'].mean().unstack()
            # Plot the bar chart
            ax = grouped.plot(kind='bar', figsize=(10, 6))
            # Customize the plot
            plt.xlabel('Task configuration')
            plt.ylabel('Average subpath lenght (reward included)')
            plt.title('Average subpath length (reward included) per subpath in each task')
            plt.legend(title='States', loc='upper right')
            plt.axhline(5, color='black', linestyle='--', label='Baseline at 5.5 secs')
            
            # Show the plot
            plt.show()
            
            # group by task config and subpath
            grouped_without = df_clean.groupby(['round_no', 'state'])['subpath_length_without_rew'].mean().unstack()
            # Plot the bar chart
            ax = grouped_without.plot(kind='bar', figsize=(10, 6))
            # Customize the plot
            plt.xlabel('Task configuration')
            plt.ylabel('Average subpath lenght (reward excluded)')
            plt.title('Average subpath length (reward excluded) per subpath in each task')
            plt.legend(title='States', loc='upper right')
            plt.axhline(5, color='black', linestyle='--', label='Baseline at 4 secs')
        
        
            # Show the plot
            plt.show()
    
        
        # import pdb; pdb.set_trace()
    # no 3: extract some EVs. These first ones will be behaviour based.
    #
    # an EV has to be a 3-column .txt file, onset(in seconds), duration, magnitude
        #EV_folder = f'/Users/xpsy1114/Documents/projects/multiple_clocks/output/{file}_EVs/'
        EV_folder = f'/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/EVs_{version}_pt0{task_half}/'
        if not os.path.exists(EV_folder):
            os.mkdir(EV_folder)
            
        df_all = pd.read_csv(data_dir+file_all)
        
        # the first scanner trigger is in row 3, scanner_trigger_key.started
        # it will be the first index of this 
        # and collecting where the first TRs have been received
        index_first_TRs = []
        for index, row in df_all.iterrows():
            if not pd.isna(row['TR_received_no0']):
                index_first_TRs.append(index)
        
        # this is the value you have to substract from all other timings.

        # double check which value here is the correct one!!!
        first_TR_at = df_all.at[index_first_TRs[0], 'time_scanner_prompt_end']
        
        
        # NOT FOR NOW SINCE THEY WOULD TAKE VARIANCE AWAY FROM THE OTHER EVs
        # # STATE EVs.
        # on: t_reward_afterwait[-1]
        # dur: start - t_reward_afterwait
        # mag = ones(len(on))
        
        # on_A = []
        # for index, row in df.iterrows():
        #     if not pd.isna(row['start_ABCD_screen']):
        #         on_A.append(df.at[index, 'start_ABCD_screen'])
        
        # dur_A = []
        # A_ends = df[(~df['t_reward_afterwait'].isna()) & (df['state'] == 'A')]
        
        # j = 0
        # for i,row in A_ends.iterrows():
        #     if j < len(on_A):
        #         dur_A.append(row['t_reward_afterwait'] - on_A[j])
        #         j += 1
        
        # if len(on_A) > len(dur_A):
        #     on_A = on_A[:-1]
        
        # mag_state = np.ones(len(dur_A))
        
        # state_A_EV = mc.analyse.analyse_MRI_behav.create_EV(on_A, dur_A, mag_state, 'state_A_EV', EV_folder, first_TR_at)
        
    
        # # one could also define the onset of the other states as the timepoint when
        # # the step is made to continue walking from the last reward: t_step_press_global at
        # # t_reward_start not nan and state = 'A' (for B), 'B' (for C), 'C' (for D)
        
        # step_to_next_B = df[(~df['t_reward_start'].isna()) & (df['state'] == 'A')]
        # on_B = step_to_next_B['t_step_press_global'].to_list()
        
        # dur_B = []
        # B_ends = df[(~df['t_reward_afterwait'].isna()) & (df['state'] == 'B')]
        
        # j = 0
        # for i,row in B_ends.iterrows():
        #     if j < len(on_B):
        #         dur_B.append(row['t_reward_afterwait'] - on_B[j])
        #         j += 1
                
                    
        # state_b_EV = mc.analyse.analyse_MRI_behav.create_EV(on_B, dur_B, mag_state, 'state_B_EV', EV_folder, first_TR_at)
            
    
        # step_to_next_C = df[(~df['t_reward_start'].isna()) & (df['state'] == 'B')]
        # on_C = step_to_next_C['t_step_press_global'].to_list()
        
        # dur_C = []
        # C_ends = df[(~df['t_reward_afterwait'].isna()) & (df['state'] == 'C')]
        
        # j = 0
        # for i,row in C_ends.iterrows():
        #     if j < len(on_C):
        #         dur_C.append(row['t_reward_afterwait'] - on_C[j])
        #         j += 1
                
        # state_c_EV = mc.analyse.analyse_MRI_behav.create_EV(on_C, dur_C, mag_state, 'state_C_EV', EV_folder, first_TR_at)
            
        # step_to_next_D = df[(~df['t_reward_start'].isna()) & (df['state'] == 'C')]
        # on_D = step_to_next_D['t_step_press_global'].to_list()
        
        # dur_D = []
        # D_ends = df[(~df['t_reward_afterwait'].isna()) & (df['state'] == 'D')]
        
        # j = 0
        # for i,row in D_ends.iterrows():
        #     if j < len(on_D):
        #         dur_D.append(row['t_reward_afterwait'] - on_D[j])
        #         j += 1
                
        # state_d_EV = mc.analyse.analyse_MRI_behav.create_EV(on_D, dur_D, mag_state, 'state_D_EV', EV_folder, first_TR_at)
        
    
    
    
        
        # # button press EVs.
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
        
        # DOUBLE CHECK THIS AS WELL!! Why is it slightly off from the t_step_press_global times???
        
            
            
        #import pdb; pdb.set_trace()
        # CONTINUE HERE!!!
        
        # # Location EVs.
        df[(~df['t_reward_start'].isna()) & (df['state'] == 'C')]
        loc_one_on = df[(df['curr_loc_x'] == '-0.21') & (df['curr_loc_y'] == '-0.29')]
        loc_two_on = df[(df['curr_loc_x'] == '0') & (df['curr_loc_y'] == '-0.29')]
        loc_three_on = df[(df['curr_loc_x'] == '0.21') & (df['curr_loc_y'] == '-0.29')]
        loc_four_on = df[(df['curr_loc_x'] == '- 0.21') & (df['curr_loc_y'] == '0')]
        loc_five_on = df[(df['curr_loc_x'] == '0') & (df['curr_loc_y'] == '0')]
        loc_six_on = df[(df['curr_loc_x'] == '0.21') & (df['curr_loc_y'] == '0')]
        loc_seven_on = df[(df['curr_loc_x'] == '-0.21') & (df['curr_loc_y'] == '0.29')]
        loc_eight_on = df[(df['curr_loc_x'] == '0') & (df['curr_loc_y'] == '0.29')]
        loc_nine_on = df[(df['curr_loc_x'] == '0.21') & (df['curr_loc_y'] == '0.29')]
        # and then in these rows onset is 't_step_press_global'
        # and duration is 't_step_end_global' at index i+1 - 't_step_press_global' at index i
        
        # on
        # dur 
        # mag = ones(len(df))
        
        # state_A_EV = mc.analyse.analyse_MRI_behav.create_EV(on, dur, mag, 'state_A_EV')
        
        # Actually, the thing that I need is my task time regressors. Build those based on the behaviour!
        # start by creating subpath columns.
        
        
        
        # so what is easily doable is the subpath reward vs no-reward.
        # I already created the columns for this: subpath_length_without_rew and 
        # subpath_length_with_rew
        # loop through the task_config labels.
        # For each task_config, I want to make 8 regressors:
            # onset = 
            # dur = 
    
        
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
        task_names = df['task_config'].dropna().unique().tolist()
        state_names = df['state'].dropna().unique().tolist()
        task_EV_dic = {}
        for i, task in enumerate(task_names):
            for s, state in enumerate(state_names):
                EV_subpathname_onset = f"{task}_{state}_subpath_onset"
                EV_subpathname_dur = f"{task}_{state}_subpath_dur"
                EV_rewardname_onset = f"{task}_{state}_reward_onset"
                EV_rewardname_dur = f"{task}_{state}_reward_dur"
                
                partial_df = df[((df['task_config'] == task) & (df['state'] == state))]
                
                task_EV_dic[EV_subpathname_onset] = partial_df['subpath_onset'].dropna().to_list()
                task_EV_dic[EV_subpathname_dur] = partial_df['subpath_dur_without_rew'].dropna().to_list()
                
                task_EV_dic[EV_rewardname_onset] = partial_df['reward_onset'].dropna().to_list()
                task_EV_dic[EV_rewardname_dur] = partial_df['reward_duration'].dropna().to_list()
                
            
        
        
        for i, task in enumerate(task_names):
            for s, state in enumerate(state_names):
                mag_subpath = np.ones(len(task_EV_dic[f"{task}_{state}_subpath_onset"]))
                EV_subpath = mc.analyse.analyse_MRI_behav.create_EV(task_EV_dic[f"{task}_{state}_subpath_onset"], task_EV_dic[f"{task}_{state}_subpath_dur"], mag_subpath, f"{task}_{state}_subpath", EV_folder, first_TR_at)
                mag_reward = np.ones(len(task_EV_dic[f"{task}_{state}_reward_onset"]))
                EV_reward = mc.analyse.analyse_MRI_behav.create_EV(task_EV_dic[f"{task}_{state}_reward_onset"], task_EV_dic[f"{task}_{state}_reward_dur"], mag_reward, f"{task}_{state}_reward", EV_folder, first_TR_at)
                
                
        # lastly, create EVs where the backwards and forwards task are together in one EV (i.e. 10 repeats per task, 10 tasks)
        # I put a backwards version of the reversed (A1 = orig, A2 = reversed) in the same task.
        # they thus did the same order twice within the task halfes. Thus, all A can be same EV
        # eg:
            # A2 backw = -0.21 0.29, -0.21 -0.29, 0 0, 0.21 0.29 
            # A1 forw = -0.21 0.29, -0.21 -0.29, 0 0, 0.21 0.29
            # B1 forw = -0.21 0.29, 0.21 -0.29, -0.21 0.29, 0 -0.29
            # B2 back = -0.21 0.29, 0.21 -0.29, -0.21 0.29, 0 -0.29
            
        EV_folder_combined = f'/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/EVs_combined_{version}_pt0{task_half}/'
        if not os.path.exists(EV_folder_combined):
            os.mkdir(EV_folder_combined)
            
        df_all = pd.read_csv(data_dir+file_all)
        
        # I want to remove the variants: C1 forw = C2 backw
        # careful, this is another level. I don't compare the actual reward order.
        # todo: that would be more elegant
        unique_letters = set()  # To track unique letters
        tasks_backw_forw = []
        for item in task_names:
            letter = ''.join(filter(str.isalpha, item))  # Remove numbers from the string
            if letter not in unique_letters:
                unique_letters.add(letter)
                tasks_backw_forw.append(letter)
        
        #import pdb; pdb.set_trace()
        task_EV_combined_dic = {}
        for i, task in enumerate(tasks_backw_forw):
            for s, state in enumerate(state_names):
                EV_subpathname_onset = f"{task}_{state}_subpath_onset"
                EV_subpathname_dur = f"{task}_{state}_subpath_dur"
                EV_rewardname_onset = f"{task}_{state}_reward_onset"
                EV_rewardname_dur = f"{task}_{state}_reward_dur"
                
                partial_df = df[((df['task_config'].str.contains(task)) & (df['state'] == state))]
                
                task_EV_combined_dic[EV_subpathname_onset] = partial_df['subpath_onset'].dropna().to_list()
                task_EV_combined_dic[EV_subpathname_dur] = partial_df['subpath_dur_without_rew'].dropna().to_list()
                
                task_EV_combined_dic[EV_rewardname_onset] = partial_df['reward_onset'].dropna().to_list()
                task_EV_combined_dic[EV_rewardname_dur] = partial_df['reward_duration'].dropna().to_list()
                

        for i, task in enumerate(tasks_backw_forw):
            for s, state in enumerate(state_names):
                mag_subpath = np.ones(len(task_EV_combined_dic[f"{task}_{state}_subpath_onset"]))
                EV_subpath_comb = mc.analyse.analyse_MRI_behav.create_EV(task_EV_combined_dic[f"{task}_{state}_subpath_onset"], task_EV_combined_dic[f"{task}_{state}_subpath_dur"], mag_subpath, f"{task}_{state}_subpath", EV_folder_combined, first_TR_at)
                mag_reward = np.ones(len(task_EV_combined_dic[f"{task}_{state}_reward_onset"]))
                EV_reward_comb = mc.analyse.analyse_MRI_behav.create_EV(task_EV_combined_dic[f"{task}_{state}_reward_onset"], task_EV_combined_dic[f"{task}_{state}_reward_dur"], mag_reward, f"{task}_{state}_reward", EV_folder_combined, first_TR_at)
                    
            
# PART 2 #        
##### now this part is to prepare the behavioural data for the simulations
# goal: to create the model RDMs for each model.
# step 1: extract the walked paths.
# step 2: extract the respective timings of these walked paths.
# step 3: create the simulation neural responses.
# step 4: make it an fMRI response.
# step 5: make it a RDM.

# make all of this similar to the ephys data.
# create one dictionary per subject that you also safe.
# walked_path which is a list of x = number-of-tasks lists.
# timings which is a list of x = number-of-tasks list
# reward_configs which is a (x,4) np.array x = number-of-tasks times 4 rewards
# condition which is a x = number-of-tasks list of forward or backward

# CONTINUE HERE!!
# reward_index which is a (x,4*repeats) np.array x = number-of-tasks times 4*repeats rewards
# reward_timing which is a (x,4*repeats) np.array x = number-of-tasks times 4*repeats rewards



# step 1, extract the walked paths.
for index, row in df.iterrows():
    df.at[index, 'curr_loc_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_y'], is_y=True, is_x = False)
    df.at[index, 'curr_loc_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_x'], is_x=True, is_y = False)
    df.at[index, 'curr_rew_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_y'], is_y=True, is_x = False)
    df.at[index, 'curr_rew_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_x'], is_x=True, is_y = False)
    
walked_path = df[['curr_loc_x_coord', 'curr_loc_y_coord']].values.astype(int).tolist()

walked_path = []
timings = []
current_task = None
rew_list = []
rew_timing = []
rew_index = []
condition = []
for index, row in df.iterrows():
    if row['task_config'] != current_task:
        walked_path.append([])  # Start a new list entry for a new task
        timings.append([])
        rew_timing.append([])
        rew_list.append([])
        rew_index.append([])
        condition.append([])
        current_task = row['task_config']    
        #import pdb; pdb.set_trace()
    walked_path[-1].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
    timings[-1].append(row['t_step_press_global'])
    condition[-1].append([row['task_config'], row['type']])
    if not pd.isna(row['rew_loc_x']):
        rew_list[-1].append([row['curr_rew_x_coord'], row['curr_rew_y_coord']])
        rew_timing[-1].append(row['t_reward_start'])
        rew_index.append(index)
        

for i in range(1, len(rew_list)):
    rew_list[i] = rew_list[i][0:4]
    
# in this case, the reward index has to be 1

# step 2: extract the respective timings of these walked paths.
# timings = df['t_step_press_global'].values.tolist()

# walked_path and timings now still include nans always when the next task started.
# use this to create multiple arrays for different task configs

# rew configs I'll extract out of the df_all table.
rew_configs = []
for cond in condition[1:-1]:
    row = df_all[df_all['Config']== cond[0][0]].iloc[0] # Find the first row with matching 'Config'
    all_curr_rews = [row['rew_x_A'], row['rew_y_A']],[row['rew_x_B'], row['rew_y_B']],[row['rew_x_C'], row['rew_y_C']],[row['rew_x_D'], row['rew_y_D']]
    rew_configs.append(all_curr_rews)                                                                                                                        

# ok and now I should somehow be able to generate my models!!
# lesssego.

# this needs to happen per task configuration. So, loop!


min_trialno = 60
for task_number in timings_all:
    curr_trialno = len(task_number)
    if curr_trialno < min_trialno:
        min_trialno = curr_trialno

for no_trial_in_each_task in range(0, min_trialno):
    for task_no, task_config in enumerate(task_configs):
        # to take the final runs not the first ones.
        run_no = -1*(no_trial_in_each_task + 1)
        trajectory, timings_curr_run, index_make_step, step_number, curr_neurons = mc.simulation.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, run_no, task_no, task_config, neurons)
        
location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.create_model_RDMs_fmri(trajectory, timings_curr_run, index_make_step, step_number, no_phase_neurons= number_phase_neurons, plot = False)
 

















