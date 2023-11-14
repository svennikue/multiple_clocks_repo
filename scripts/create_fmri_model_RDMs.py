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

subjects = ['sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
task_halves = ['1', '2']
no_bins_per_state = 2
plotting = False
RDM_version = '01'

for sub in subjects:
    for task_half in task_halves:
        data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
        data_dir_beh = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/{sub}/beh/"
        file = f"{sub}_fmri_pt{task_half}"
        file_all = f"{sub}_fmri_pt{task_half}_all.csv"
        
        df = pd.read_csv(data_dir_beh + f"{file}.csv")
        # the first row is empty so delete to get indices right
        df = df.iloc[1:].reset_index(drop=True)
        df_all = pd.read_csv(data_dir_beh+file_all)
        

        # fill gaps in the round_no column
        df['round_no'] = df['round_no'].fillna(method='ffill')
        # do the same for the task_config 
        df['task_config'] = df['task_config'].fillna(method='ffill')
        df['repeat'] = df['repeat'].fillna(method='ffill')
        
        # don't do the state thing and instead use it to know when a new repeat starts. ignore these ones, only include steps!
        #df['state'] = df['state'].fillna(method = 'bfill')
        # in case there is a nan in the very end also fill forward
        #df['state'] = df['state'].fillna(method = 'ffill')
        
        # transform the locations
        # current locations
        for index, row in df.iterrows():
            df.at[index, 'curr_loc_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_y'], is_y=True, is_x = False)
            df.at[index, 'curr_loc_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_x'], is_x=True, is_y = False)
            df.at[index, 'curr_rew_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_y'], is_y=True, is_x = False)
            df.at[index, 'curr_rew_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_x'], is_x=True, is_y = False)
            if not pd.isna(row['state']):
                if not np.isnan(row['rew_loc_x']):
                    df.at[index, 'time_bin_type'] =  df.at[index, 'task_config'] + '_' + df.at[index, 'state'] + '_reward'
                elif np.isnan(row['rew_loc_x']):
                    df.at[index, 'time_bin_type'] = df.at[index, 'task_config'] + '_' + df.at[index, 'state'] + '_subpath'
                    # THIS DOESNT WORK YET!!!!

        # import pdb; pdb.set_trace()
        # reward specs
        for index, row in df_all.iterrows():
            df_all.at[index, 'rew_y_A_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_y_A'], is_y=True, is_x = False)
            df_all.at[index, 'rew_x_A_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_x_A'], is_x=True, is_y = False)
            df_all.at[index, 'rew_y_B_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_y_B'], is_y=True, is_x = False)
            df_all.at[index, 'rew_x_B_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_x_B'], is_x=True, is_y = False)
            df_all.at[index, 'rew_y_C_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_y_C'], is_y=True, is_x = False)
            df_all.at[index, 'rew_x_C_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_x_C'], is_x=True, is_y = False)
            df_all.at[index, 'rew_y_D_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_y_D'], is_y=True, is_x = False)
            df_all.at[index, 'rew_x_D_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df_all.at[index,'rew_x_D'], is_x=True, is_y = False)
            
            
        # create a dictionnary with all future regressors.
        time_bin_types = df['time_bin_type'].dropna().unique()
        regressors = {}
        for time_bin_type in time_bin_types:
            regressors[time_bin_type] = []
            
        
        
        walked_path = []
        timings = []
        current_task = None
        rew_list = []
        rew_timing = []
        rew_index = []
        subpath_after_steps = []
        condition = []
        # import pdb; pdb.set_trace()
        for index, row in df.iterrows():
            task_config = row['task_config']
            time_bin_type = row['time_bin_type']
            
            #iterate through the regression dictionary first
            for key in regressors.keys():
                # check if the key starts with the task_config value
                # CAREFUL!!! THE VERY LAST STEP IS MISSING!!!
                if key.startswith(task_config):
                    if time_bin_type == key:
                        regressors[key].append(1)
                    elif pd.notna(time_bin_type):
                        regressors[key].append(0)
                        
                    #regressors[key].append(1 if time_bin_type == key else 0 if not pd.notna(time_bin_type) else 0)

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
            
            
            # CHECK WHY THE VERY LAST STEP DOES NOT GET ADDED TO WALKED PATH!
            
            # in case a new task has just started
            if not np.isnan(row['next_task']): 
                # first check if this is the first task of several repeats.
                if (index == 0) or (row['task_config'] != df.at[index -1, 'task_config']):
                    timings[-1].append(row['next_task'])
                else: # if it isnt, then take the reward start time from last rew D as start field.
                    timings[-1].append(df.at[index -1, 't_reward_start'])
                condition[-1].append([row['task_config'], row['type']])
                walked_path[-1].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                      
            # last option: if this is just a normal walking field
            elif not np.isnan(row['t_step_press_global']): # always except if this is reward D 
                # if its reward D, then it will be covered by the first if: if not np.isnan(row['next_task']): 
                timings[-1].append(df.at[index - 1, 't_step_press_global'])  # Extract value from index-1
                condition[-1].append([row['task_config'], row['type']])
                walked_path[-1].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
           
            # next check if its a reward field
            if not np.isnan(row['rew_loc_x']): # if this is a reward field.
                # first check if this is where the task stopped, or if the next reward is where the task stopped.
                # ignore these as they are not complete.
                if index+2 < len(df):
                    rew_timing[-1].append(row['t_reward_start'])
                    rew_list[-1].append([row['curr_rew_x_coord'], row['curr_rew_y_coord']])
                    subpath_after_steps[-1].append(int(index-row['repeat']))  
                    if row['state'] == 'D':
                        rew_index[-1].append(len(walked_path[-1])) #bc step has not been added yet
                        if (row['task_config'] != df.at[index +1, 'task_config']):
                            walked_path[-1].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            condition[-1].append([row['task_config'], row['type']])
                            timings[-1].append(df.at[index -1, 't_reward_start'])
                    else:
                        rew_index[-1].append(len(walked_path[-1])-1) 
                else:
                    continue
                    # # if it is the last reward, then still add to walked_path!
                    # if row['state'] == 'D': # Tis didnt work!!!! continue here!!!
                    #     print('entered D')
                    #     walked_path[-1].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                    #     print(f"adding {row['curr_loc_x_coord']} and {row['curr_loc_y_coord']}")
                    #     print(f" so that walked path is not {walked_path}")
                    #     condition[-1].append([row['task_config'], row['type']])
                    #     timings[-1].append(df.at[index -1, 't_reward_start'])
                    # else: 
                    #     continue # if this is the last row, then ignore- this is where the task stopped.
        
        # import pdb; pdb.set_trace()           
        # overview of the reward fields per task.
        for i in range(0, len(rew_list)):
            rew_list[i] = [[int(value) for value in sub_list] for sub_list in rew_list[i][0:4]]
        
        configs = df['task_config'].dropna().unique()
        rew_index_per_config = dict(zip(configs, rew_index))
        subpath_per_config = dict(zip(configs, subpath_after_steps))
            
        if plotting:
            # check if the regressor dictionary worked!
            # CHECK HERE: the length of each regressor should be equal to len(elem) for elem in walked_path
            plot_dict = np.concatenate((regressors['C1_A_subpath'], regressors['C1_A_reward'], regressors['C1_B_subpath'], regressors['C1_B_reward'], regressors['C1_C_subpath'], regressors['C1_C_reward'], regressors['C1_D_subpath'], regressors['C1_D_reward'])).reshape(8, -1)
            plt.figure(); plt.imshow(plot_dict, aspect = 'auto')           
             
            # write a test figure for the regressors   
            for config in configs:
                #config = 'C1'
                count = 0
                to_plot = np.zeros((8,100)) + 3
                for key in regressors.keys():
                    if key.startswith(config):
                        to_plot[count, 0 : len(regressors[key])] = regressors[key]
                        count += 1
                plt.figure(); plt.imshow(to_plot, aspect = 'auto')   
                for index in rew_index_per_config[config]:
                    plt.plot([index, index,index,index], [1,3,5,7], color = 'black', marker = 'x', markersize =8)
                      


        
        # now create subpath files with rew_index and how many steps there are per subpath.
        steps_subpath_alltasks = []
        for i, task in enumerate(subpath_after_steps):
            steps_per_subpath = []
            if (len(task)%4) == 0:
                for r in range(0, len(task), 4):
                    subpath = task[r:r+4]
                    steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                    if r == 0:
                        steps.insert(0, rew_index[i][r])
                    if r > 0:
                        steps.insert(0, (subpath[0]- task[r-1]))
                    steps_per_subpath.append(steps)
            elif (len(task)%4) > 0:
                completed_tasks = len(task)-(len(task)%4)
                for r in range(0, completed_tasks, 4):
                    subpath = task[r:r+4]
                    steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                    if r == 0:
                        steps.insert(0, rew_index[i][r])
                    if r > 0:
                        steps.insert(0, (subpath[0]- task[r-1]))
                    steps_per_subpath.append(steps)    
            steps_subpath_alltasks.append(steps_per_subpath)
    

        # this needs to happen per task configuration. So, loop!
        # this will happen five times, since this is the no of repeats I have per task.
        #for task_no, task_config in enumerate(rew_list[:-2]):
        for task_no, task_config in enumerate(rew_list):
                # import pdb; pdb.set_trace()

                # select the config name -> needed for dictionaries
                config_name = configs[task_no]
                
                # select complete trajectory of current task.
                trajectory = walked_path[task_no]
                trajectory = [[int(value) for value in sub_list] for sub_list in trajectory]
                
                # maybe delete: select step timings for current task.
                timings_curr_run = timings[task_no]
                
                # select file that shows step no per subpath
                step_number = steps_subpath_alltasks[task_no]
                step_number = [[int(value) for value in sub_list] for sub_list in step_number]
                # make file that shows cumulative steps per subpath
                cumsteps_per_run = [np.cumsum(task)[-1] for task in step_number]
                cumsteps_task = np.cumsum(cumsteps_per_run)
                
                # then start looping through each run within one task
                for no_run in range(len(step_number)):
                    # first check if the run is not completed. if so, skip the uncomplete part.
                    if len(subpath_after_steps[task_no]) < 20:
                        stop_after_x_runs = len(subpath_after_steps[task_no]) // 4
                        if no_run >= stop_after_x_runs:
                            continue
                    
                    if no_run == 0:
                        # careful: fields is always one more than the step number
                        curr_trajectory = trajectory[0:cumsteps_task[no_run]+1]
                        curr_timings = timings_curr_run[0:cumsteps_task[no_run]+1]
                        curr_stepnumber = step_number[no_run]
                    elif no_run > 0:
                        # careful: fields is always one more than the step number
                        curr_trajectory = trajectory[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
                        curr_timings = timings_curr_run[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
                        curr_stepnumber = step_number[no_run]
                        curr_cumsumsteps = cumsteps_task[no_run]
                    
                    # now create the models for this particular run
                    # DOUBLE CHECK IF THIS IS ALL AS IT SHOULD!!!
                    # at some point. Not prio.
                    # 11.11.2023
                    # CAREFUL!!!
                    # for some reason, I ignore the last step location!!
                    # but only the very last...
                    # why??
                    location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.create_model_RDMs_fmri(curr_trajectory, curr_timings, curr_stepnumber, plot=False)
                    # these need to be concatenated for each run and task
                    # import pdb; pdb.set_trace() 
                    if no_run == 0:
                        clocks_repeats = clocks_model.copy()
                        midnight_repeats = midnight_model.copy()
                        location_repeats = location_model.copy()
                        phase_repeats = phase_model.copy()
                        state_repeats = state_model.copy()
                    else:
                        clocks_repeats = np.concatenate((clocks_repeats, clocks_model), 1)
                        midnight_repeats = np.concatenate((midnight_repeats, midnight_model), 1)
                        location_repeats = np.concatenate((location_repeats, location_model), 1)
                        phase_repeats = np.concatenate((phase_repeats, phase_model), 1)
                        state_repeats = np.concatenate((state_repeats, state_model), 1)
                    
                # for some reason, the regressor matrix has one field more than the combined matrices....
                # DOUBLE CHECK WHY!
                if plotting:
                    plt.figure(); plt.imshow(clocks_repeats, aspect = 'auto')
                    plt.figure(); plt.imshow(midnight_repeats, aspect = 'auto')
                    plt.figure(); plt.imshow(phase_repeats, aspect = 'auto')
                    plt.figure(); plt.imshow(location_repeats, aspect = 'auto')
                    plt.figure(); plt.imshow(state_repeats, aspect = 'auto')
                    
                # then select the correct regressors.
                # Filter keys starting with 'A1'
                regressors_curr_task = {key: value for key, value in regressors.items() if key.startswith(config_name)}
                
                # check that all regressors have the same length in case the task wasn't completed.
                if len(subpath_after_steps[task_no]) < 20:
                    # if I cut the task short, then also cut the regressors short.
                    for config_name, regressor_list in regressors_curr_task.items():
                    # Truncate the list if its length is greater than the maximum length
                        regressors_curr_task[config_name] = regressor_list[:(np.shape(clocks_repeats)[1])]

                
                # Ensure all lists have the same length
                list_lengths = set(len(value) for value in regressors_curr_task.values())
                if len(list_lengths) != 1:
                    raise ValueError("All lists must have the same length.")
                
                
                
                # Create a NumPy matrix
                # CONTINUE HERE!!!
                # for some reason, the regressor matrix has one field more than the combined matrices....
                # DOUBLE CHECK WHY!
                regressors_matrix = np.array(list(regressors_curr_task.values()))
                
                # then timebin the run like it happened for the actual data -> now several repeats within one timebin.
                # this can be done just based on the steps! I don't need the timings for this.
                clocks_binned = mc.simulation.predictions.transform_data_to_betas(clocks_repeats, regressors_matrix)
                midnight_binned = mc.simulation.predictions.transform_data_to_betas(midnight_repeats, regressors_matrix)
                location_binned = mc.simulation.predictions.transform_data_to_betas(location_repeats, regressors_matrix)
                phase_binned = mc.simulation.predictions.transform_data_to_betas(phase_repeats, regressors_matrix)
                state_binned = mc.simulation.predictions.transform_data_to_betas(state_repeats, regressors_matrix)
               
                

                # these need to be concatenated for each run and task
                if task_no == 0:
                #if task_no == 0 and trial_no == 0:
                    # neurons_between = curr_neurons.copy() this was the ephys data previously
                    clocks_between = clocks_binned.copy()
                    midnight_between = midnight_binned.copy()
                    location_between = location_binned.copy()
                    phase_between = phase_binned.copy()
                    state_between = state_binned.copy()
                else:
                    clocks_between = np.concatenate((clocks_between, clocks_binned),1)
                    midnight_between = np.concatenate((midnight_between, midnight_binned),1)
                    location_between = np.concatenate((location_between, location_binned),1)
                    phase_between = np.concatenate((phase_between, phase_binned),1)
                    state_between = np.concatenate((state_between, state_binned),1)

            
        # import pdb; pdb.set_trace()               
        # then, in a last step, create the RDMs
        # now create the model RDMs
        RSM_location = mc.simulation.RDMs.within_task_RDM(location_between, plotting = True, titlestring = 'Location RDM')
        RSM_clock = mc.simulation.RDMs.within_task_RDM(clocks_between, plotting = True, titlestring = 'Clock RDM')
        RSM_midnight = mc.simulation.RDMs.within_task_RDM(midnight_between, plotting = True, titlestring = 'Midnight RDM')
        RSM_phase = mc.simulation.RDMs.within_task_RDM(phase_between, plotting = True, titlestring = 'Phase RDM')
        RSM_state = mc.simulation.RDMs.within_task_RDM(state_between, plotting = True, titlestring= 'State RDM')
        
        # then save these matrices.
        RDM_dir = f"{data_dir}/derivatives/{sub}/beh/RDMs_{RDM_version}"
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        np.save(os.path.join(RDM_dir, f"RSM_location_{sub}_fmri_pt{task_half}"), RSM_location)
        np.save(os.path.join(RDM_dir, f"RSM_clock_{sub}_fmri_pt{task_half}"), RSM_clock)
        np.save(os.path.join(RDM_dir, f"RSM_midnight_{sub}_fmri_pt{task_half}"), RSM_midnight)
        np.save(os.path.join(RDM_dir, f"RSM_phase_{sub}_fmri_pt{task_half}"), RSM_phase)
        np.save(os.path.join(RDM_dir, f"RSM_state_{sub}_fmri_pt{task_half}"), RSM_state)
        
        # also save the regression files
        np.save(os.path.join(RDM_dir, f"data_location_{sub}_fmri_pt{task_half}"), location_between)
        np.save(os.path.join(RDM_dir, f"data_clock_{sub}_fmri_pt{task_half}"), clocks_between)
        np.save(os.path.join(RDM_dir, f"data_midnight_{sub}_fmri_pt{task_half}"), midnight_between)
        np.save(os.path.join(RDM_dir, f"data_phase_{sub}_fmri_pt{task_half}"), phase_between)
        np.save(os.path.join(RDM_dir, f"data_state_{sub}_fmri_pt{task_half}"), state_between)
        
