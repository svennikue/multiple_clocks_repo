#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:25:52 2023

@author: Svenja Küchenhoff

This script includes optimisation function for different means and with different parameters.
"""


# this will be a function that optimises for low correlation between two different neural predictions.
# it will always be over time, but one can choose which ones: location, clocks, 0-angle
# one can choose if hrf convolved or not
# one can choose which parameters to optimise for: grid size, times per step, amounts of reward
# and lastly, one can determine the number of permutations to try these optimisations.

import mc
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def optimise_task_for(prediction_one, prediction_two, hrf = True, grid_size = 3, step_time = 15, reward_no = 4, perms = 1, plot = False):
    # import pdb; pdb.set_trace()    
    # idea is to vary parameters oneself.
    # output is the optimal similarity after x permutations, with the respective parameters.
    
    if prediction_one != 'clocks' and prediction_one != 'location' and prediction_one != 'phase_loc':
        raise TypeError("Please enter 'location', 'phase_loc' or 'clocks'") 
    if prediction_two != 'clocks' and prediction_two != 'location' and prediction_two != 'phase_loc':
        raise TypeError("Please enter 'location', 'phase_loc' or 'clocks'")
    
    # start the optimisation process.
    # steps:
    # 1. create a task configuration
    # 2. create the predictions for both models
    # 2.1 check if HRF convolution wanted?
    # 3. create the RSM and similarity between the two models
    # 4. change the task configuration, do steps 2-3 and compare 
    # prep plot
    dissimilarity_values = []
    countgood_corr = 0
    maximally_dissimilar = []
    best_reward_coords = []
    
    for perm_no in range(perms):
        # 1. create a task configuration
        rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False)
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = False)
        
        # 2. create the predictions for both models
        if prediction_one == 'clocks' or prediction_two == 'clocks':
            clocksm, neuroncl, clocks_model = mc.simulation.predictions.set_clocks_bytime_one_neurone(walk, steps_per_walk, step_time, grid_size)
            # DELTE THIS AFTER DEBUGGING!!
            # ALSO CHANGE PLOTTING OPTIONS BACK TO FALSE
            #mc.simulation.predictions.plot_without_legends(clocks_model)
    
        if prediction_one == 'phase_loc' or prediction_two == 'phase_loc':
            clocksm, neuroncl, clocks_model_dummy = mc.simulation.predictions.set_clocks_bytime_one_neurone(walk, steps_per_walk, step_time, grid_size)
            phase_loc_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model_dummy, steps_per_walk, grid_size)
            # DELTE THIS AFTER DEBUGGING!!
            # ALSO CHANGE PLOTTING OPTIONS BACK TO FALSE
            #mc.simulation.predictions.plot_without_legends(phase_loc_model)
            
        if prediction_one == 'location' or prediction_two == 'location':
            locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, step_time, grid_size)
            # DELTE THIS AFTER DEBUGGING!!
            # ALSO CHANGE PLOTTING OPTIONS BACK TO FALSE
            #mc.simulation.predictions.plot_without_legends(location_model)
            
            
        # 2.1 check if HRF convolution wanted?
        if hrf:
            if 'location_model' in locals():
                location_model = mc.simulation.predictions.convolve_with_hrf(location_model, steps_per_walk, step_time, plotting = False)
            if 'clocks_model' in locals():
                clocks_model = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = False)
            if 'phase_loc_model' in locals():
                clocks_model_dummy = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = False)
                phase_loc_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model_dummy, steps_per_walk, grid_size)
        
        # 3. create the RSM and similarity between the two models
        # make the names the same again
        if 'location_model' in locals():
            model_one = location_model
            if 'clocks_model' in locals():
                model_two = clocks_model
            elif 'phase_loc_model' in locals():
                model_two = phase_loc_model
        # if this statement isnt true, then there is no location model, thus:
        else:
            model_one = clocks_model[:]
            model_two = phase_loc_model[:]
    
        # create a string for the columns
        count_columns = list(range(0,len(model_one[0])))
        col_names = count_columns.copy()
        for i in count_columns:
            col_names[i] = str(i) 
     
        RSM_one = mc.simulation.RDMs.within_task_RDM(model_one, col_names)
        RSM_two = mc.simulation.RDMs.within_task_RDM(model_two, col_names)
        
        
        similarity = mc.simulation.RDMs.corr_matrices(RSM_one, RSM_two)
        # store for the histogram
        dissimilarity_values.append(1 - similarity[0,1])
        
        if perm_no == 0:
            best_sim_value = similarity[:]
            best_walk = walk[:]
            best_rewards = rew_coords[:]
            best_model_one = model_one[:]
            best_model_two = model_two[:]
        # 4. change the task configuration, do steps 2-3 and compare > loop
        else:
            if similarity[1,0] < best_sim_value[1,0]:
                best_sim_value = similarity[:]
                best_walk = walk[:]
                best_rewards = rew_coords[:]
                best_model_one = model_one[:]
                best_model_two = model_two[:]
        
        # save the configurations if the similarity is really good.
        # save those configurations that have are maximally dissimlar. 
        if similarity[0,1] <  0.70:   
            path = pd.DataFrame(walk)
            coef = pd.DataFrame(similarity[0]) 
            rewards = pd.DataFrame(rew_coords)
            if countgood_corr == 0:
                # to save as .csv file
                maximally_dissimilar = pd.concat([coef, path, rewards], axis = 1)
                # this is to have the coords in a np file in py
                temp_best_reward_coords = np.array(rew_coords)
                best_reward_coords = np.expand_dims(temp_best_reward_coords, axis = 0)
            else:
                maximally_dissimilar = pd.concat([maximally_dissimilar, coef, path, rewards], axis = 1)
                # best_reward_coords = np.stack([best_reward_coords, np.array(rew_coords)])     
                # this is to have the coords in a np file in py
                curr_coords = np.array(rew_coords)
                best_reward_coords = np.concatenate([curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]]), best_reward_coords], axis=0)
                #best_reward_coords = np.stack([best_reward_coords, np.array(reward_coords)])
            countgood_corr += 1 
            
    
    if plot == True:
        # include the settings on the graps
        mc.simulation.predictions.plot_without_legends(best_model_one, 'clocks_model', hrf, grid_size, step_time, reward_no, perms)
        mc.simulation.predictions.plot_without_legends(best_model_two, 'phase_loc_model', hrf, grid_size, step_time, reward_no, perms)
        
        walk, steps_per_walk = mc.simulation.grid.walk_paths(best_rewards, grid_size, plotting = True)
        # might also be nice to plot the distribution of correlation values for these settings...
        plt.figure()
        ax2 = plt.axes()
        plt.hist(dissimilarity_values)
        titletext = (f'Variance of {prediction_one} beyond {prediction_two}, hrf is {hrf}, grid is {grid_size} x {grid_size}, one step lasts {step_time} ms, over {perms} perms')
        plt.title(titletext)
        plt.ylabel = 'frequency'
        plt.xlabel = '1 - Similarity'
        
        
        

    return best_sim_value, best_walk, best_rewards, best_reward_coords, maximally_dissimilar, dissimilarity_values


def optimise_several_task_configs(prediction_one, prediction_two, no_tasks, hrf = True, grid_size = 4, step_time = 15, reward_no = 4, perms = 1):
    # import pdb; pdb.set_trace()    
    # output is the optimal X optimal task configurations, with XY as similarity for the given parameters.   
    if prediction_one != 'clocks' and prediction_one != 'location' and prediction_one != 'phase_loc':
        raise TypeError("Please enter 'location', 'phase_loc' or 'clocks'") 
    if prediction_two != 'clocks' and prediction_two != 'location' and prediction_two != 'phase_loc':
        raise TypeError("Please enter 'location', 'phase_loc' or 'clocks'")
    dissimilarity_values = []
    countgood_corr = 0
    maximally_dissimilar = []
    best_reward_coords = []
    
    # first, create X random task configurations for model 1 and model 2.
    for i in range(no_tasks):
        # 1. create a task configuration
        rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False)
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = False)
        # 2. create the predictions for both models
        if prediction_one == 'clocks' or prediction_two == 'clocks':
            clocksm, neuroncl, clocks_model = mc.simulation.predictions.set_clocks_bytime_one_neurone(walk, steps_per_walk, step_time, grid_size)
            # DELTE THIS AFTER DEBUGGING!!
            # ALSO CHANGE PLOTTING OPTIONS BACK TO FALSE
            #mc.simulation.predictions.plot_without_legends(clocks_model)
    
        if prediction_one == 'phase_loc' or prediction_two == 'phase_loc':
            clocksm, neuroncl, clocks_model_dummy = mc.simulation.predictions.set_clocks_bytime_one_neurone(walk, steps_per_walk, step_time, grid_size)
            phase_loc_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model_dummy, steps_per_walk, grid_size)
            # DELTE THIS AFTER DEBUGGING!!
            # ALSO CHANGE PLOTTING OPTIONS BACK TO FALSE
            #mc.simulation.predictions.plot_without_legends(phase_loc_model)
            
        if prediction_one == 'location' or prediction_two == 'location':
            locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, step_time, grid_size)
            # DELTE THIS AFTER DEBUGGING!!
            # ALSO CHANGE PLOTTING OPTIONS BACK TO FALSE
            #mc.simulation.predictions.plot_without_legends(location_model)
        
        # 2.1 check if HRF convolution wanted?
        if hrf:
            if 'location_model' in locals():
                location_model = mc.simulation.predictions.convolve_with_hrf(location_model, steps_per_walk, step_time, plotting = False)
            if 'clocks_model' in locals():
                clocks_model = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = False)
            if 'phase_loc_model' in locals():
                clocks_model_dummy = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = False)
                phase_loc_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model_dummy, steps_per_walk, grid_size)
        
        # make the names the same again
        if 'location_model' in locals():
            model_one = location_model
            if 'clocks_model' in locals():
                model_two = clocks_model
            elif 'phase_loc_model' in locals():
                model_two = phase_loc_model
        # if this statement isnt true, then there is no location model, thus:
        else:
            model_one = clocks_model[:]
            model_two = phase_loc_model[:]
        
        #now, concatenate these X matrices
        # CONTINUE HERE!! FIND OUT HOW TO CONCAT THESE MATRICES.
        
        
        
        
#     # create a string for the columns
#     count_columns = list(range(0,len(model_one[0])))
#     col_names = count_columns.copy()
#     for i in count_columns:
#         col_names[i] = str(i) 
 
#     RSM_one = mc.simulation.RDMs.within_task_RDM(model_one, col_names)
#     RSM_two = mc.simulation.RDMs.within_task_RDM(model_two, col_names)
    
        
    
    
    
#     # start the optimisation process.
#     # steps:
#     # 1. create a task configuration
#     # 2. create the predictions for both models
#     # 2.1 check if HRF convolution wanted?
#     # 3. create the RSM and similarity between the two models
#     # 4. change the task configuration, do steps 2-3 and compare 
    
    
    
    
# def find_best_tasks(loop_no, no_columns, column_names): 
# #    import pdb; pdb.set_trace()
# #     # this needs to be something like:
# #         # 1. create 10 random tasks and the between-task corr maps.
# #         # 2. compute similarity between those 2 big matrices (this needs to be exclude_diag = False!! bc thats the within task one)
# #         # 3. stepwise go through each task configuration and check if replacing it with 
# #         #      a new one reduces the similarity value
# #         # do this a number of loops
# #         # always store the current configurations/ toss the one I am replacing
#     # first, create one 10 tasks x 10 tasks matrix for clocks and locations
#     task_config_no = 10
#     clock_RSM_matrix, loc_RSM_matrix, df_clock, df_loc, task_configs = mc.simulation.RDMs.between_task_RDM(task_config_no, column_names, plotting = False)
#     # and get the similarity between those. 
#     similarity_between = mc.simulation.RDMs.corr_matrices(loc_RSM_matrix, clock_RSM_matrix)
#     # based on this, try to optimize the correlation coefficient (similarity_between)
#     for i in range(0, loop_no):        
#         # then first take the first 10 columns of df_clock and df_loc and replace it with a new config
#         # create new configuration
#         reward_coords = mc.simulation.grid.create_grid()
#         reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords) 
#         df_rewards = pd.DataFrame(reward_coords)
#         df_task_configs = pd.DataFrame(reshaped_visited_fields)
#         df_temp_task_configs = pd.concat([df_rewards, df_task_configs], axis = 1)
#         # create new neural predictions for this task config
#         clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
#         loc_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0)
#         # turn those into dataframe
#         temp_clocks = pd.DataFrame(clocks_matrix)
#         temp_locs = pd.DataFrame(loc_matrix)
#         temp_clocks.columns = column_names
#         temp_locs.columns = column_names
#         # prepare loop here       
#         temp_similarity = np.ones((2,2))
        
#         count = 1 # change back to 1 once debugging is done
#         # then, replace each of the 10 tasks with the new config and text if similarity is now less (= better)
#         # step out of the loop either way once looped through all columns, or when temp_similarity is lower
#         while (temp_similarity[0,1] > similarity_between[0,1]) and (count < task_config_no):
#             # have a counter for all columns   
#             temp_df_loc = df_loc.copy()
#             # replace the first (count) 12 columns with the new configuration
#             temp_df_loc.iloc[:, ((count-1)*no_columns):(count*no_columns)] = temp_locs
#             # temp_df_loc.iloc[:, (count*no_columns):((count*no_columns)+no_columns)] = temp_locs
#             temp_df_loc.fillna(0, inplace = True)
#             temp_df_clock = df_clock.copy()
#             temp_df_clock.iloc[:, ((count-1)*no_columns):(count*no_columns)] = temp_clocks
#             temp_df_clock.fillna(0, inplace = True)
#             # create new correlation matrices for the new clocks and location matrix
#             temp_corr_clocks = temp_df_clock.corr()
#             temp_corr_locs = temp_df_loc.corr()
#             temp_clocks_RSM = temp_corr_clocks.to_numpy()
#             temp_locs_RSM = temp_corr_locs.to_numpy() 
#             # test the new similarity between the new RSMs
#             temp_similarity = mc.simulation.RDMs.corr_matrices(temp_locs_RSM, temp_clocks_RSM)
#             if temp_similarity[0,1] < similarity_between[0,1]:
#                 # task_configs is structured a little different, its always x,y of paths and then x,y of rewards.
#                 # -> 4 columns per task config              
#                 task_configs.iloc[:,((count-1)*4):(count*4)] = df_temp_task_configs
#                 # if the new RSMs correlate less, replace the current configuration and RSM with the new
#                 # and continue to optimize further.
#                 df_clock = temp_df_clock.copy()
#                 df_loc = temp_df_loc.copy()
#                 similarity_between = temp_similarity.copy()
#             count += 1 
#             del temp_df_loc
#             del temp_df_clock
#             del temp_corr_clocks
#             del temp_corr_locs
#             del temp_similarity
#             temp_similarity = np.ones((2,2))
        
#     return df_clock, df_loc, task_configs, similarity_between     
        
        
        