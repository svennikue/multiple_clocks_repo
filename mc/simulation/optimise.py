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
    similarity_values = []
    
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
            model_one = clocks_model
            model_two = phase_loc_model
    
        # create a string for the columns
        count_columns = list(range(0,len(model_one[0])))
        col_names = count_columns.copy()
        for i in count_columns:
            col_names[i] = str(i) 
     
        RSM_one = mc.simulation.RDMs.within_task_RDM(model_one, col_names)
        RSM_two = mc.simulation.RDMs.within_task_RDM(model_two, col_names)
        
        similarity = mc.simulation.RDMs.corr_matrices(RSM_one, RSM_two)
        # store for the histogram
        similarity_values.append(1 - similarity[0,1])
        
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
    
    if plot == True:
        # include the settings on the graps
        mc.simulation.predictions.plot_without_legends(model_one, 'clocks_model', hrf, grid_size, step_time, reward_no, perms)
        mc.simulation.predictions.plot_without_legends(model_two, 'phase_loc_model', hrf, grid_size, step_time, reward_no, perms)
        
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = True)
        # might also be nice to plot the distribution of correlation values for these settings...
        plt.figure()
        ax2 = plt.axes()
        plt.hist(similarity_values)
        titletext = (f'Variance of {prediction_one} beyond {prediction_two}, hrf is {hrf}, grid is {grid_size} x {grid_size}, one step lasts {step_time} ms, over {perms} perms')
        plt.title(titletext)
        plt.ylabel = 'frequency'
        plt.xlabel = '1 - Similarity'
        
        
        

    return best_sim_value, best_walk, best_rewards

        
        
        
        