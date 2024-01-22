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

06.12.2023: version 06 for RDM GLM. TR is different, made the script nicer. Everything else should be the same. 

@author: Svenja KUechenhoff
"""


import pandas as pd
import os
import numpy as np
import mc
import matplotlib.pyplot as plt
import pickle
import re
import sys


#import pdb; pdb.set_trace()


if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'
    
    
subjects = [f"sub-{subj_no}"]
# subjects = ['sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18','sub-19', 'sub-20',  'sub-22', 'sub-23','sub-24']
#subjects = ['sub-01']
version = '07' # GLM number -> 07 is only button press and rewards. | new, better script is now 06. first GLM was 04.

# to debug task_halves = ['1']
task_halves = ['1', '2']
locationEVs = False

time_binEVs = True
plotting = False
analyse_behav = False

for sub in subjects:
    for task_half in task_halves:
        data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        funcDir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func"
        analysisDir = "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
        if os.path.isdir(data_dir_beh):
            print("Running on laptop.")
        else:
            data_dir_beh = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/beh/"
            funcDir = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/func"
            analysisDir = "/home/fs0/xpsy1114/scratch/analysis"
            print(f"Running on Cluster, setting {data_dir_beh} as data directory")

        file = f"{sub}_fmri_pt{task_half}"
        file_all = f"{sub}_fmri_pt{task_half}_all.csv"
        

        # define and make paths
        if locationEVs:
            EV_folder = f'{funcDir}/EVs_{version}_pt0{task_half}_press_and_loc/'
        elif time_binEVs:
            EV_folder = f'{funcDir}/EVs_{version}_pt0{task_half}/'
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
            
        on_press = []
        for i, row in end_task.iterrows():
            curr_presses = row['nav_key_task.rt'] # extract button presses from the rt item with all presses
            # Split the string into a list using a comma as the separator
            presses_curr_task = curr_presses.strip('[]').split(', ') 
            # Convert the elements to floats and add to the point in time where they actually started
            presses_curr_task = [(float(time)+new_task.at[i, 'start_ABCD_screen']) for time in presses_curr_task]
            on_press=on_press+presses_curr_task
            
        # the duration can just be something like 20 ms
        dur_press = np.ones(len(on_press)) * 0.02
        mag_press = np.ones(len(on_press))
        
        button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(on_press, dur_press, mag_press, 'press_EV', EV_folder, first_TR_at)
        
        # check there are no nans 
        deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(button_press_EV)
        if deleted_x_rows > 0:
            print(f"careful! I am saving a cutted EV button press file. Happened for subject {sub} in task half {task_half}")
            np.savetxt(str(EV_folder) + 'ev_' + 'press_EV' + '.txt', array, delimiter="    ", fmt='%f')
        
        # import pdb; pdb.set_trace()
        if locationEVs:
            # # Location EVs.

            coord_x_one = -0.21
            coord_y_one = -0.29
            loc_one_on, loc_one_dur, loc_one_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_one, coord_y_one)
            loc_one_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_one_on, loc_one_dur, loc_one_mag, 'loc_one_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_one_EV)
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_one_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_one_EV' + '.txt', array, delimiter="    ", fmt='%f')
            
            coord_x_two = 0
            coord_y_two = -0.29
            loc_two_on, loc_two_dur, loc_two_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_two, coord_y_two)
            loc_two_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_two_on, loc_two_dur, loc_two_mag, 'loc_two_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_two_EV)
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_two_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_two_EV' + '.txt', array, delimiter="    ", fmt='%f')
            
                    
            coord_x_three = 0.21
            coord_y_three = -0.29
            loc_three_on, loc_three_dur, loc_three_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_three, coord_y_three)
            loc_three_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_three_on, loc_three_dur, loc_three_mag, 'loc_three_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_three_EV)  
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_three_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_three_EV' + '.txt', array, delimiter="    ", fmt='%f')
            

            coord_x_four = -0.21
            coord_y_four = 0
            loc_four_on, loc_four_dur, loc_four_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_four, coord_y_four)
            loc_four_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_four_on, loc_four_dur, loc_four_mag, 'loc_four_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_four_EV)  
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_four_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_four_EV' + '.txt', array, delimiter="    ", fmt='%f')
            
                     
            coord_x_five = 0
            coord_y_five = 0
            loc_five_on, loc_five_dur, loc_five_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_five, coord_y_five)
            loc_five_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_five_on, loc_five_dur, loc_five_mag, 'loc_five_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_five_EV) 
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_five_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_five_EV' + '.txt', array, delimiter="    ", fmt='%f')
            
                    
            coord_x_six = 0.21
            coord_y_six = 0
            loc_six_on, loc_six_dur, loc_six_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_six, coord_y_six)
            loc_six_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_six_on, loc_six_dur, loc_six_mag, 'loc_six_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_six_EV)       
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_six_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_six_EV' + '.txt', array, delimiter="    ", fmt='%f')
            
            coord_x_seven = -0.21
            coord_y_seven = 0.29
            loc_seven_on, loc_seven_dur, loc_seven_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_seven, coord_y_seven)
            loc_seven_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_seven_on, loc_seven_dur, loc_seven_mag, 'loc_seven_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_seven_EV) 
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_seven_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_seven_EV' + '.txt', array, delimiter="    ", fmt='%f')
            
            
            coord_x_eight = 0
            coord_y_eight = 0.29
            loc_eight_on, loc_eight_dur, loc_eight_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_eight, coord_y_eight)
            loc_eight_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_eight_on, loc_eight_dur, loc_eight_mag, 'loc_eight_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_eight_EV)
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_eight_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_eight_EV' + '.txt', array, delimiter="    ", fmt='%f')
            
            
            coord_x_nine = 0.21
            coord_y_nine = 0.29
            loc_nine_on, loc_nine_dur, loc_nine_mag = mc.analyse.analyse_MRI_behav.make_loc_EV(df, coord_x_nine, coord_y_nine)
            loc_nine_EV = mc.analyse.analyse_MRI_behav.create_EV(loc_nine_on, loc_nine_dur, loc_nine_mag, 'loc_nine_EV', EV_folder, first_TR_at)
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(loc_nine_EV)
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV loc_nine_EV file. Happened for subject {sub} in task half {task_half}")
                np.savetxt(str(EV_folder) + 'ev_' + 'loc_nine_EV' + '.txt', array, delimiter="    ", fmt='%f')
            

        
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
                
            
            # I need 8 regressors per task (e.g. C1). I have 5 * 2 tasks.
            # actually, C1 forward = C2 backward. For now, don't put together
            
            # look at to check if its really the same task. For this, create a reward type 
            # column which allows to differentiate all trials
            df['config_type'] = df['task_config'] + '_' + df['type']
            df['config_type'] = df['config_type'].fillna(method='ffill')
            task_names = df['config_type'].dropna().unique().tolist()
            
            # separate by state and subpath/reward for now.
            state_names = df['state'].dropna().unique().tolist()
            
            taskEV_dic = {}
            # I want 80 EVs in the end -> 160 elements in the dictionary (duration + onset)
            for i, task in enumerate(task_names):
                for s, state in enumerate(state_names):
                    #EV_subpathname_onset = f"{task}_{state}_subpath_onset"
                    #EV_subpathname_dur = f"{task}_{state}_subpath_dur"
                    EV_rewardname_onset = f"{task}_{state}_reward_onset"
                    EV_rewardname_dur = f"{task}_{state}_reward_dur"
                    
                    partial_df = df[((df['config_type'] == task) & (df['state'] == state))]
                    #partial_df = df[((df['task_config'] == task) & (df['state'] == state))]
                    
                    #taskEV_dic[EV_subpathname_onset] = partial_df['subpath_onset'].dropna().to_list()
                    #taskEV_dic[EV_subpathname_dur] = partial_df['subpath_dur_without_rew'].dropna().to_list()
                    
                    taskEV_dic[EV_rewardname_onset] = partial_df['reward_onset'].dropna().to_list()
                    taskEV_dic[EV_rewardname_dur] = partial_df['reward_duration'].dropna().to_list()
            
            # I need a stratgey for this on how to include empty regressors.
            # in the future: only include those regressors that actually have more than 2 activations.
            # excluded = 0
            for i, task in enumerate(task_names):
                for s, state in enumerate(state_names):
                    # mag_subpath = np.ones(len(taskEV_dic[f"{task}_{state}_subpath_onset"]))
                    # if len(mag_subpath) < 3:
                    #     print(f"Careful! {task} x {state} subpath and reward is not complete and will be excluded.")
                    #     excluded = excluded + 2 # bc reward will also be exluded
                    #     continue
                    # subpath_EV = mc.analyse.analyse_MRI_behav.create_EV(taskEV_dic[f"{task}_{state}_subpath_onset"], taskEV_dic[f"{task}_{state}_subpath_dur"], mag_subpath, f"{task}_{state}_path", EV_folder, first_TR_at)
                    mag_reward = np.ones(len(taskEV_dic[f"{task}_{state}_reward_onset"]))
                    # if len(mag_reward) < 3:
                    #     print(f"Careful! {task} x {state} reward is not complete and will be excluded.")
                    #     excluded = excluded + 1
                    #     continue
                    reward_EV = mc.analyse.analyse_MRI_behav.create_EV(taskEV_dic[f"{task}_{state}_reward_onset"], taskEV_dic[f"{task}_{state}_reward_dur"], mag_reward, f"{task}_{state}_reward", EV_folder, first_TR_at)
                    deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(reward_EV)
                    if deleted_x_rows > 0:
                        print(f"careful! I am saving a cutted EV {task}{state} reward file. Happened for subject {sub} in task half {task_half}")
                        np.savetxt(str(EV_folder) + 'ev_' + f"{task}_{state}_reward" + '.txt', array, delimiter="    ", fmt='%f')
                    
                    #deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(subpath_EV)
                    #if deleted_x_rows > 0:
                    #    print(f"careful! I am saving a cutted EV {task}{state} subpath file. Happened for subject {sub} in task half {task_half}")
                    #   np.savetxt(str(EV_folder) + 'ev_' + f"{task}_{state}_path" + '.txt', array, delimiter="    ", fmt='%f')
                    
                    
            if 'C2_forw_A_reward_dur' in taskEV_dic:
                print(f"Now done with {sub} and task half {task_half}. Made {len(taskEV_dic)/2} EVs, each with length {len(taskEV_dic['C2_forw_A_reward_dur'])}")       
            elif 'C1_forw_A_reward_dur' in taskEV_dic:
                print(f"Now done with {sub} and task half {task_half}. Made {len(taskEV_dic)/2} EVs, each with length {len(taskEV_dic['C1_forw_A_reward_dur'])}")       
        

            
            # lastly, save the taskEV_dic so that I can also use it as data regressors.
            # this has to be like this bc its a dictionary
            with open(f"{EV_folder}my_EV_dict", 'wb') as f:
                pickle.dump(taskEV_dic, f)
            
            
            # then, lastly, adjust the .fsf file I will use for the regression.
            # original_fsf_file=f"{analysisDir}/templates/my_RDM_GLM_v2.fsf"
            # with open(original_fsf_file, 'r') as file:
            #     fsf_content = file.read()
                
            # collect all filepaths I just created.
            files_in_EV_folder = os.listdir(EV_folder) 
            EV_paths = []
            for EV in files_in_EV_folder:
                if EV.startswith("ev_") and EV.endswith(".txt"):
                    EV_path = os.path.join(EV_folder, EV)
                    EV_paths.append(os.path.join(EV_folder, EV)) 
            print(f"I collected {len(EV_paths)} EVs to put into the fsf file.")
            sorted_EVs = sorted(EV_paths)
            
            text_to_write = []
            with open(f"{analysisDir}/templates/my_RDM_GLM_v2.fsf", "r") as fin:                    
                for line in fin:
                    for i, EV_path in enumerate(sorted_EVs): 
                        if line.startswith(f"set fmri(custom{i+1})"):
                            # print(f"my old line was: {line}")
                            line = f'set fmri(custom{i+1}) "{EV_path}"\n'
                        if line.startswith(f"set fmri(evtitle{i+1})"):
                            EV_name_ext = os.path.basename(EV_path)
                            EV_name = EV_name_ext.rsplit('.',1)[0]
                            print(f"changing evtitle{i+1} to {EV_name}")
                            line = f'set fmri(evtitle{i+1}) "{EV_name}"\n'
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
                    
                    
                    

            with open(f"{funcDir}/{sub}_my_RDM_GLM_0{task_half}_{version}.fsf", "w") as fout:
                for line in text_to_write_cleaned:
                    fout.write(line)

                


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

            

            
            
