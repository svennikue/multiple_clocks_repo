#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:35:48 2023



@author: Svenja Küchenhoff

This script identifies good task configurations and plots these.


"""
import mc
import pandas as pd
from datetime import datetime 
from matplotlib import pyplot as plt
import numpy as np


######### SETTINGS ############

# Section 1
# to get task configurations that are dissimilar between 0-angle-neurons and 
# clocks.
section_one_one = 0  #0-angle and clocks
section_one_two = 0 # plotting 0-angle and clocks
section_one_three = 0 # 0-angle and clocks, convolved with HRF
section_one_four = 0 # plotting 0-angle and clocks, convolved with HRF

## now section 2: optimise more flexibly.
section_two_one = 1 # playing around with different task parameters (steptime, gridsize, reward amount)

##############################
##### SECTION 1 ##############
#
#
#### Section 1.1: find low correlation between 0-angle and clocks predictions.
if section_one_one == 1:
    similarity_values = []
    countmax = 0    
    perms = 10000
    for count in range(perms):
        reward_coords = mc.simulation.grid.create_grid(plot = False)
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = False)
        
        # section 2.2
        steps_in_ms = 15
        clocksm, neuroncl, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, steps_in_ms)
        
        # section 2.3
        zero_phase_clocks_m = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_over_time, all_stepnums, 3)
        
        # section 4.2
        counter = list(range(0,len(clocks_over_time[0])))
        seconds = counter.copy()
        for i in counter:
            seconds[i] = str(i)  
            
        zero_clock_RSM = mc.simulation.RDMs.within_task_RDM(zero_phase_clocks_m, seconds)
        
        clock_RSM = mc.simulation.RDMs.within_task_RDM(clocks_over_time, seconds)  
        similarity = mc.simulation.RDMs.corr_matrices(zero_clock_RSM, clock_RSM)  
        similarity_values.append(1 - similarity [0,1])
        
        # save those configurations that have are maximally dissimlar. 
        if similarity[0,1] <  0.70:          
            path = pd.DataFrame(reshaped_visited_fields)
            coef = pd.DataFrame(similarity[0])
            rewards = pd.DataFrame(reward_coords)
            if countmax == 0:
                # this is to save as csv
                maximally_dissimilar = pd.concat([coef, path, rewards], axis = 1)
                # this is to have the coords in a np file in py
                best_reward_coords = np.array(reward_coords)
            elif countmax == 1:
                maximally_dissimilar = pd.concat([maximally_dissimilar, coef, path, rewards], axis = 1)
                best_reward_coords = np.stack([best_reward_coords, np.array(reward_coords)])     
            else: 
                # this is to save as csv
                maximally_dissimilar = pd.concat([maximally_dissimilar, coef, path, rewards], axis = 1)
                # this is to have the coords in a np file in py
                curr_coords = np.array(reward_coords)
                best_reward_coords = np.concatenate([curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]]), best_reward_coords], axis=0)
                #best_reward_coords = np.stack([best_reward_coords, np.array(reward_coords)])      
            countmax += 1 
            
    folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_0phase_with_clocks'
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = '.csv'
    filename = folder + time + file
    if countmax > 0:
        maximally_dissimilar.to_csv(filename)
 
        
 
#
#        
##### Section 1.2: plot the distribution of task similarity and best task configs
# RUN SECTION 1.1 FIRST!
if section_one_two == 1:
    # plot distribution.
    plt.figure()
    ax = plt.axes()
    plt.hist(similarity_values)
    plt.gca().set(title= (f'Variance clocks raw beyond 0-phase-clocks ({perms} perms)'), ylabel = ('frequency'), xlabel='1 - Similarity');
    # plot the task configurations
    if countmax > 0:
        # to read out the reward coords, do:
        for i in range(best_reward_coords.shape[0]):
            curr_rew = np.ndarray.tolist(best_reward_coords[i])
            reshaped_visited_fields, morestepstuff = mc.simulation.grid.walk_paths(curr_rew, plotting = True)
    # plot two examplary prediction matrices
    mc.simulation.predictions.plot_phaseloc_pertime(zero_phase_clocks_m, steps_in_ms, all_stepnums)
    mc.simulation.predictions.plotclock_pertime(clocks_over_time, steps_in_ms, all_stepnums)



#
#
#### Section 1.3: find low correlation between 0-angle and clocks predictions, HRF convolved.
if section_one_three == 1:
    similarity_values_hrf = []
    countmax_hrf = 0
    perms_hrf = 1000
    for count in range(perms_hrf):
        reward_coords = mc.simulation.grid.create_grid(plot = False)
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = False)
        
        # section 2.2
        steps_in_ms = 20
        clocksm, neuroncl, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, steps_in_ms)
        # now do the convolution
        clocks_over_time_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_over_time, all_stepnums, steps_in_ms, plotting = False)
        
        # section 2.3
        zero_phase_clocks_m_hrf = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_over_time_hrf, all_stepnums, 3)
        
        # section 4.2
        # create a list for the column names
        counter = list(range(0,len(clocks_over_time[0])))
        mili_seconds = counter.copy()
        for i in counter:
            mili_seconds[i] = str(i)  
            
        zero_clock_RSM_hrf = mc.simulation.RDMs.within_task_RDM(zero_phase_clocks_m_hrf, mili_seconds)
        clock_RSM_hrf = mc.simulation.RDMs.within_task_RDM(clocks_over_time_hrf, mili_seconds) 
        
        similarity_hrf = mc.simulation.RDMs.corr_matrices(zero_clock_RSM_hrf, clock_RSM_hrf)  
        similarity_values_hrf.append(1 - similarity_hrf[0,1])
        
        # save those configurations that have are maximally dissimlar. 
        if similarity_hrf[0,1] <  0.75:          
            path_hrf = pd.DataFrame(reshaped_visited_fields)
            coef_hrf = pd.DataFrame(similarity_hrf[0])
            rewards_hrf = pd.DataFrame(reward_coords)
            if countmax_hrf == 0:
                # to save as .csv file
                maximally_dissimilar_hrf = pd.concat([coef_hrf, path_hrf, rewards_hrf], axis = 1)
                # this is to have the coords in a np file in py
                best_reward_coords_hrf = np.array(reward_coords)
            elif countmax_hrf == 1:
                maximally_dissimilar_hrf = pd.concat([maximally_dissimilar_hrf, coef_hrf, path_hrf, rewards_hrf], axis = 1)
                best_reward_coords_hrf = np.stack([best_reward_coords_hrf, np.array(reward_coords)])     
            else: 
                # this is to save as csv
                maximally_dissimilar_hrf = pd.concat([maximally_dissimilar_hrf, coef_hrf, path_hrf, rewards_hrf], axis = 1)
                # this is to have the coords in a np file in py
                curr_coords = np.array(reward_coords)
                best_reward_coords_hrf = np.concatenate([curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]]), best_reward_coords_hrf], axis=0)
                #best_reward_coords = np.stack([best_reward_coords, np.array(reward_coords)])
            countmax_hrf += 1 
        
        
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = '.csv'
    folder_hrf = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_0phase_with_clocks_hrf'
    steptime = '_' + str(steps_in_ms) + '_ms_per_step' 
    filename_hrf = folder_hrf + time + '_perms' + str(perms_hrf) + steptime + file 
    if countmax_hrf > 0:
        maximally_dissimilar_hrf.to_csv(filename_hrf)

#
#
##### Section 1.4: plot the distribution of task similarity and best task configs
# RUN SECTION 1.4 FIRST!
if section_one_four == 1:
    # plot distribution.
    plt.figure()
    ax2 = plt.axes()
    plt.hist(similarity_values_hrf)
    plt.gca().set(title= (f'Variance clocks_hrf beyond 0-phase-clocks ({perms_hrf} perms)'), ylabel = ('frequency'), xlabel='1 - Similarity');
    # plot the task configurations
    if countmax_hrf > 0:
        for i in range(best_reward_coords_hrf.shape[0]):
            curr_rew = np.ndarray.tolist(best_reward_coords_hrf[i])
            reshaped_visited_fields, morestepstuff = mc.simulation.grid.walk_paths(curr_rew, plotting = True)
            bli, blub, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, steps_in_ms)
            clocks_over_time_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_over_time, morestepstuff, steps_in_ms, plotting = True)
            zero_phase_clocks_m_hrf = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_over_time_hrf, morestepstuff, 3)
            mc.simulation.predictions.plot_phaseloc_pertime(zero_phase_clocks_m_hrf, steps_in_ms, morestepstuff)
            mc.simulation.predictions.plotclock_pertime(clocks_over_time_hrf, steps_in_ms, morestepstuff)
            


###########################################

#
#
### Section 2.1: optimise for low correlation between 0-angle-neurons and complete clocks, with diff parameters
# use: 
# 1. different grid sizes
# 2. different times per step
# 3. different amounts of reward

# use HRF always, because that's the one that I am ultimately interested in.

if section_two_one == 1:
    clock_prediction = 'clocks'
    phase_loc_prediction = 'phase_loc'
    # similarity, walk_coords, reward_coords = mc.simulation.optimise.optimise_task_for(clock_prediction, phase_loc_prediction, perms = 100, plot = False)
    # plot that shit
    grid_size = 4
    step_time = 15
    reward_no = 4
    
    rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = True)
    walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = True)
    
    # check if my location model works with these settings
    locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, step_time, grid_size)
    mc.simulation.predictions.plot_without_legends(location_model)
    
    # check if my clock model works with these settings
    clocksm, neuroncl, clocks_model = mc.simulation.predictions.set_clocks_bytime_one_neurone(walk, steps_per_walk, step_time, grid_size)
    mc.simulation.predictions.plot_without_legends(clocks_model)
    
    # check if my phaseloc model work with these settings
    phase_loc_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model, steps_per_walk, grid_size)
    mc.simulation.predictions.plot_without_legends(phase_loc_model)
    
    


























