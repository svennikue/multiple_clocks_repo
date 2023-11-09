#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:34:17 2023

extract behaviour such that I can create my model RDMs for the fMRI data.


@author: xpsy1114
"""

import pandas as pd
import os
import numpy as np
import mc
import matplotlib.pyplot as plt

# import pdb; pdb.set_trace()

subjects = ['sub-01']
task_halves = ['1', '2']


for sub in subjects:
    for task_half in task_halves:
        data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        file = f"{sub}_fmri_pt{task_half}"
        file_all = f"{sub}_fmri_pt{task_half}_all.csv"
        
        df = pd.read_csv(data_dir + f"{file}.csv")
        # the first row is empty so delete to get indices right
        df = df.iloc[1:].reset_index(drop=True)
        df_all = pd.read_csv(data_dir+file_all)
        
        
        # fill gaps in the round_no column
        df['round_no'] = df['round_no'].fillna(method='ffill')
        # do the same for the task_config 
        df['task_config'] = df['task_config'].fillna(method='ffill')
        df['repeat'] = df['repeat'].fillna(method='ffill')
        
        
        # transform the locations
        for index, row in df.iterrows():
            df.at[index, 'curr_loc_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_y'], is_y=True, is_x = False)
            df.at[index, 'curr_loc_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_x'], is_x=True, is_y = False)
            df.at[index, 'curr_rew_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_y'], is_y=True, is_x = False)
            df.at[index, 'curr_rew_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_x'], is_x=True, is_y = False)
        
        for index, row in df_all.iterrows():
            df_all.at[index, 'rew_y_A_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_y_A'], is_y=True, is_x = False)
            df_all.at[index, 'rew_x_A_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_x_A'], is_x=True, is_y = False)
            df_all.at[index, 'rew_y_B_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_y_B'], is_y=True, is_x = False)
            df_all.at[index, 'rew_x_B_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_x_B'], is_x=True, is_y = False)
            df_all.at[index, 'rew_y_C_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_y_C'], is_y=True, is_x = False)
            df_all.at[index, 'rew_x_C_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_x_C'], is_x=True, is_y = False)
            df_all.at[index, 'rew_y_D_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_y_D'], is_y=True, is_x = False)
            df_all.at[index, 'rew_x_D_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_x_D'], is_x=True, is_y = False)
            
            
        # walked_path = df[['curr_loc_x_coord', 'curr_loc_y_coord']].values.astype(int).tolist()

        walked_path = []
        timings = []
        current_task = None
        rew_list = []
        rew_timing = []
        rew_index = []
        subpath_after_steps = []
        condition = []
        for index, row in df.iterrows():
            if row['task_config'] != current_task:
                walked_path.append([])  # Start a new list entry for a new task
                timings.append([])
                rew_timing.append([])
                rew_list.append([])
                rew_index.append([])
                subpath_after_steps.append([])
                condition.append([])
                current_task = row['task_config']    
                # import pdb; pdb.set_trace()
            if not np.isnan(row['t_step_press_global']):
                walked_path[-1].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                timings[-1].append(row['t_step_press_global'])
                condition[-1].append([row['task_config'], row['type']])
            # rew_index[-1].append(index)
            if not pd.isna(row['rew_loc_x']):
                rew_list[-1].append([row['curr_rew_x_coord'], row['curr_rew_y_coord']])
                rew_timing[-1].append(row['t_reward_start'])
                rew_index[-1].append(index)
                subpath_after_steps[-1].append(int(index-row['repeat']))
                # CONTINUE HERE!!!
                # this is not quite right if I do want to count the steps based on this.
                # I have a location repeated here. thats why I then suddenly jump one row 
                # and the index is +1 than it should be. e.g. next task starts with 14,
                # but if I want to count steps it needs to be 13!
                # rew_index.append(index)
                

        for i in range(0, len(rew_list)):
            rew_list[i] = [[int(value) for value in sub_list] for sub_list in rew_list[i][0:4]]
        # in this case, the reward index has to be 1

        # step 2: extract the respective timings of these walked paths.
        # timings = df['t_step_press_global'].values.tolist()

        # walked_path and timings now still include nans always when the next task started.
        # use this to create multiple arrays for different task configs

        # rew configs I'll extract out of the df_all table.
        rew_configs = []
        for cond in condition:
            row = df_all[df_all['Config']== cond[0][0]].iloc[0] # Find the first row with matching 'Config'
            all_curr_rews = [row['rew_x_A_coord'], row['rew_y_A_coord']],[row['rew_x_B_coord'], row['rew_y_B_coord']],[row['rew_x_C_coord'], row['rew_y_C_coord']],[row['rew_x_D_coord'], row['rew_y_D_coord']]
            rew_configs.append(all_curr_rews)                                                                                                                        

        
        # create subpath files with rew_index and how many steps there are per subpath.
        steps_subpath_alltasks = []
        # so for this I need to actually start at the first step, not at index.
        
        for i, task in enumerate(subpath_after_steps):
            steps_per_subpath = []
            # import pdb; pdb.set_trace()
            for r in range(0, len(task), 4):
                subpath = task[r:r+4]
                steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                if r == 0:
                    if i == 0:
                        steps.insert(0, subpath[0])
                    else:
                        # import pdb; pdb.set_trace()
                        steps.insert(0, (subpath[0] - (rew_index[i-1][-1]+1)))
                if r > 0:
                    steps.insert(0, (subpath[0]- task[r-1]))
                steps_per_subpath.append(steps)
            steps_subpath_alltasks.append(steps_per_subpath)
    
        # SOMETHING GOES WRONG HERE. why are there more steps???
            
            
        # ok and now I should somehow be able to generate my models!!
        # lesssego.

        # this needs to happen per task configuration. So, loop!
        # this will happen five times, since this is the no of repeats I have.
        
        for task_no, task_config in enumerate(rew_configs[:-2]):
                # run_no = -1*(no_trial_in_each_task + 1)
                # import pdb; pdb.set_trace()
                trajectory = walked_path[task_no]
                trajectory = [[int(value) for value in sub_list] for sub_list in trajectory]
    
                timings_curr_run = timings[task_no]
                
                step_number = steps_subpath_alltasks[task_no]
                step_number = [[int(value) for value in sub_list] for sub_list in step_number]
                cumsteps_per_run = [np.cumsum(task)[-1] for task in step_number]
                cumsteps_task = np.cumsum(cumsteps_per_run)
                
                for no_run in range(5):
                    if no_run == 0:
                        curr_trajectory = trajectory[0:cumsteps_task[no_run]]
                        curr_timings = timings_curr_run[0:cumsteps_task[no_run]]
                        curr_stepnumber = step_number[no_run]
                    elif no_run > 0:
                        curr_trajectory = trajectory[cumsteps_task[no_run-1]:cumsteps_task[no_run]]
                        curr_timings = timings_curr_run[cumsteps_task[no_run-1]:cumsteps_task[no_run]]
                        curr_stepnumber = step_number[no_run]
                        
                    location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber)
                    
                    
                
                    
                    # then timebin
                    regs_phase_state_run = mc.simulation.predictions.create_x_regressors_per_state(trajectory, timings_curr_run, step_number, no_regs_per_state = no_bins_per_state)
                    # then use these regressors to timebin
                    curr_neurons = mc.simulation.predictions.transform_data_to_betas(curr_neurons, regs_phase_state_run)
                    clocks_model = mc.simulation.predictions.transform_data_to_betas(clocks_model, regs_phase_state_run)
                    midnight_model= mc.simulation.predictions.transform_data_to_betas(midnight_model, regs_phase_state_run)
                    location_model = mc.simulation.predictions.transform_data_to_betas(location_model, regs_phase_state_run)
                    phase_model = mc.simulation.predictions.transform_data_to_betas(phase_model, regs_phase_state_run)
                   
                    # these need to be concatenated for each run and task
                    if task_no == 0:
                    #if task_no == 0 and trial_no == 0:
                        neurons_between = curr_neurons.copy()
                        clocks_between = clocks_model.copy()
                        midnight_between = midnight_model.copy()
                        location_between = location_model.copy()
                        phase_between = phase_model.copy()
                        
                        
                # then create the RDMs
                # now create the model RDMs
                RSM_location = mc.simulation.RDMs.within_task_RDM(location_model, plotting = False, titlestring = 'Location RDM')
                RSM_clock = mc.simulation.RDMs.within_task_RDM(clocks_model, plotting = False, titlestring = 'Clock RDM')
                RSM_midnight = mc.simulation.RDMs.within_task_RDM(midnight_model, plotting = False, titlestring = 'Midnight RDM')
                RSM_phase = mc.simulation.RDMs.within_task_RDM(phase_model, plotting = False, titlestring = 'Phase RDM')
                
                    
        # min_trialno = 60
        # for task_number in timings_all:
        #     curr_trialno = len(task_number)
        #     if curr_trialno < min_trialno:
        #         min_trialno = curr_trialno

        # for no_trial_in_each_task in range(0, min_trialno):
        #     for task_no, task_config in enumerate(task_configs):
        #         # to take the final runs not the first ones.
        #         run_no = -1*(no_trial_in_each_task + 1)
        #         trajectory, timings_curr_run, index_make_step, step_number, curr_neurons = mc.simulation.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, run_no, task_no, task_config, neurons)
                
        # location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.create_model_RDMs_fmri(trajectory, timings_curr_run, index_make_step, step_number, no_phase_neurons= number_phase_neurons, plot = False)
         
