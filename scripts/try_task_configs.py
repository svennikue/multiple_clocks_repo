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

## now section 2: optimise with more flexibly.
# playing around with different task parameters (steptime, gridsize, reward amount)
section_two_one = 0 # optimise similarities between different models over many permutations.
section_two_two = 0 # plot the successful reward patterns from section 2.1
section_two_three = 0 # find out if parameters work, but only one run.

## section 3: identify task configurations that correlate low BETWEEN several configs.
section_three_one = 0
section_three_two = 0 # check if the bigger function works.
section_three_three = 0 # plot the best configurations from section_three_one
section_three_four = 0 # plot the model matrices and similarity for several pre-definded task configs.

## section 4: test if single neurons in the matrices are phase-tuned
section_four_one = 0

## section 5: create a step x step RDM with HRF convolution by taking a GLM approach
section_five_one = 1




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
    perms_hrf = 5000
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


# optimise similarities between different models over many permutations.
# THERE SEEM TO BE ERRORS HAPPENING SOMEWHERE. I CANT REPRODUCE THE GOOD PATHS!!

if section_two_one == 1:
    
    clock_prediction = 'clocks'
    phase_loc_prediction = 'midnight'
    
    top_sim, top_walk_coords, top_reward_coords, rewards_better_seventy, tasks_better_seventy, all_dissim_vals, no_auto_kendall, no_auto_pearson = mc.simulation.optimise.optimise_task_for(clock_prediction, phase_loc_prediction, hrf = True, grid_size = 3, step_time= 15, reward_no= 4, perms = 5000, plot = True)
    
    save = 1
    if save == 1:
        steps_in_ms = 15
        perms = 5000
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        file = '.csv'
        folder_hrf = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_0phase_with_clocks_hrf'
        steptime = '_' + str(steps_in_ms) + '_ms_per_step' 
        filename = folder_hrf + time + '_perms' + str(perms) + steptime + file 
        file_to_save = pd.DataFrame(tasks_better_seventy)
        file_to_save.to_csv(filename)
        # also save the np rewards file if you want to reuse it below!
        # e.g. 
        # name = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_4_grid_4_rew'
        # np.save(name, best_rew_collection)
        
# to look at the successful reward patterns from section 2.1:
if section_two_two == 1:
    df = pd.read_csv('/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_0phase_with_clocks_hrf20230327-150013_perms5000_15_ms_per_step.csv')
    
    grid_size = 4
    for i in range(len(rewards_better_seventy)):
        rew_coords = rewards_better_seventy[i]
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = True)
        
    
# OK ALSO TRY IF SHORT PATHS ARE SOMEHOW SPECIAL??? USE THE TABLE I JUST SAVED! 
    
# find out if parameters work, but only one run.
if section_two_three == 1:

    # just in case I want to test the individual functions, use:
    grid_size = 4
    step_time = 15
    reward_no = 4
    perms = 10
    hrf = True
    
    # rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False)
    
    # alternativelty, create a reward vector yourself:
    # OLD CODING
    # rew_coords = [[0,0], [0,1], [0,3], [1,1], [3,3]] #this is super high, .93 ...
    # rew_coords = [[1,3], [1,2], [2,3], [2,2]] #this is .79
    # rew_coords = [[2,1], [0,1], [1,1],[2,2]] # should be .69 yep!
    # rew_coords = [[0,1], [1,0], [1,1], [0,0]] # should be .69 for 3x3 grid -yep!
    #rew_coords = [[1,0], [2,1], [2,0], [1,1]] # should be .69 for 3x3 grid -yep!
    # rew_coords = [[0,2], [0,1], [3,2], [3,1]] # this is .63
    
    # NEW CODING [27.03.23]
    # rew_coords = top_reward_coords[:]
    # rew_coords = [[0,3], [3,3], [0,2], [3,2]] # should be .53
    
    # rew_coords = [[1,2], [3,1], [1,3], [1,0]] # this is .55
    
    # rew_coords = [[0,0], [0,1], [1,1], [1,0]] # this is .91
    # rew_coords = [[0,0], [0,1], [3,3], [3,2]] 
    rew_coords = [[0,0], [0,3], [2,2], [3,2]] # this is .54
    # rew_coords = [[3,3], [1,2], [3,0], [1,0]] # this is .53
    
    # rew_coords = [[0,2], [1,1], [2,1], [0,3]] # this is .67

    walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = True)
    
    
    # try the new clock function. 
    single_clock, midnight_matrix, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, step_time, grid_size)
    mc.simulation.predictions.plot_without_legends(single_clock)
    mc.simulation.predictions.plot_without_legends(midnight_matrix)
    mc.simulation.predictions.plot_without_legends(clocks_model)
    
    # # ok location works.
    # locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, step_time, grid_size)
    # mc.simulation.predictions.plot_without_legends(location_model, 'location_model', hrf, grid_size, step_time, reward_no, perms)
    
    # # ok I believe it works.
    # clocksm, neuroncl, clocks_model = mc.simulation.predictions.set_clocks_bytime_one_neurone(walk, steps_per_walk, step_time, grid_size)
    # mc.simulation.predictions.plot_without_legends(clocks_model, 'clocks_model', hrf, grid_size, step_time, reward_no, perms)

    # # ok I believe it works.
    # phase_loc_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model, steps_per_walk, grid_size)
    # mc.simulation.predictions.plot_without_legends(phase_loc_model, 'phase_loc_model', hrf, grid_size, step_time, reward_no, perms)
    
    # # HRF add-on
    clocks_model_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = True)
    midnight_model_hrf = mc.simulation.predictions.convolve_with_hrf(midnight_matrix, steps_per_walk, step_time, plotting = True)
    # clocks_model_dummy = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = True)
    # phase_loc_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model_dummy, steps_per_walk, grid_size)
    # # plot those as well
    mc.simulation.predictions.plot_without_legends(clocks_model_hrf)
    mc.simulation.predictions.plot_without_legends(midnight_model_hrf)
    
    
    # and what's the similarity?
    # create a string for the columns
    model_one = clocks_model_hrf[:]
    model_two = midnight_model_hrf[:]
    count_columns = list(range(0,len(model_one[0])))
    col_names = count_columns.copy()
    for i in count_columns:
        col_names[i] = str(i) 
    RSM_one = mc.simulation.RDMs.within_task_RDM(model_one, col_names, plotting = True)
    RSM_two = mc.simulation.RDMs.within_task_RDM(model_two, col_names, plotting = True)
    
    # best_model_one_df = pd.DataFrame(model_one)
    # best_model_two_df = pd.DataFrame(model_two)
    # mc.simulation.RDMs.df_based_RDM(best_model_one_df, plotting = True)
    # mc.simulation.RDMs.df_based_RDM(best_model_two_df, plotting = True)
        
    correlation_sim = mc.simulation.RDMs.corr_matrices(RSM_one, RSM_two)
    similarity = correlation_sim.correlation
##
##
## SECTION 3 ##
##
##
# this selects reward configurations that have low correlations over the course of 10 different tasks.
if section_three_one == 1:
    
    save = 1
    plot = 1
    
    model_one = 'clocks'
    model_two = 'midnight'
    task_repeats = 10
    grid = 4
    hrf_stg = True 
    time_per_step = 10
    rewards = 4
    permutations = 1000
    
    # takes a while with 300, but gets to around .56, ans 1800 timepoints for 10 tasks

    pearson_between, model_one_between, model_two_between, optimal_task_configs, betw_rew_coords = mc.simulation.optimise.optimise_several_task_configs(model_one, model_two, task_repeats, hrf_stg, grid, time_per_step, rewards, permutations)
    # Cso this doesnt work yet
    # lowest corr so far: .523 (pearson)
    
    
    
    if save == 1:
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        file = '.csv'
        folder_hrf = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf'
        steptime = '_' + str(time_per_step) + '_ms_per_step' 
        filename = folder_hrf + time + '_perms' + str(permutations) + steptime + file 
        file_to_save = pd.DataFrame(optimal_task_configs)
        file_to_save.to_csv(filename)
        
        filename_rew = folder_hrf + time + '_perms' + str(permutations) + steptime
        np.save(filename_rew, betw_rew_coords)
        
    if plot == 1:
        plt.figure()
        plt.imshow(model_one_between)
        plt.title('best_%s' %model_one)
        
        plt.figure()
        plt.imshow(model_two_between)
        plt.title('best_%s' %model_two)
        
    

# Write one script which identifies the top 10 task configurations for certain settings.
# Then write a script which computes the between-task similarity over those 10 good tasks (and maybe continues to optimise them)
# Then check if we can do both: optimise for space + 

if section_three_two == 1:
    test_pearson_between, test_model_one_between, test_model_two_between, test_optimal_task_configs = mc.simulation.optimise.testing_several_task_configs(10, perms = 10000)


if section_three_three == 1:
    
    df = pd.read_csv('/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf20230407-150212_perms1000_15_ms_per_step.csv')
    betw_rew_coords = np.load('/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf20230407-150212_perms1000_15_ms_per_step.npy')
    
    grid = 4
    hrf_stg = True 
    time_per_step = 15
    rewards = 4
    
    for count, rew_conf in enumerate(betw_rew_coords):
        if count > 3:
            break
            
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_conf, grid, plotting = True)
        
        single_clock, midnight_matrix, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, time_per_step, grid)
        mc.simulation.predictions.plot_without_legends(single_clock, titlestring = 'single_clock')
        #mc.simulation.predictions.plot_without_legends(midnight_matrix)
        #mc.simulation.predictions.plot_without_legends(clocks_model)
        
        # # HRF add-on
        clocks_model_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, time_per_step, plotting = False)
        midnight_model_hrf = mc.simulation.predictions.convolve_with_hrf(midnight_matrix, steps_per_walk, time_per_step, plotting = False)
        locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, time_per_step, grid)
        location_hrf = mc.simulation.predictions.convolve_with_hrf(location_model, steps_per_walk, time_per_step, plotting = False)
        
        # # plot those as well
        mc.simulation.predictions.plot_without_legends(clocks_model_hrf, titlestring = 'Clock model')
        mc.simulation.predictions.plot_without_legends(midnight_model_hrf, titlestring = 'Midnight model')
        mc.simulation.predictions.plot_without_legends(location_hrf, titlestring = 'Location model')
        
        # and what's the within-task similarity?
        
        # 2.0 prepare the column names 
        # create a string for the columns
        model_one = clocks_model_hrf[:]
        model_two = midnight_model_hrf[:]
        model_three = location_hrf[:]
        count_columns = list(range(0,len(model_one[0])))
        col_names = count_columns.copy()
        for i in count_columns:
            col_names[i] = str(i) 
        RSM_one = mc.simulation.RDMs.within_task_RDM(model_one, col_names, plotting = True, titlestring = 'Clock model RSM')
        RSM_two = mc.simulation.RDMs.within_task_RDM(model_two, col_names, plotting = True, titlestring = 'Midnight model RSM')
        RSM_three = mc.simulation.RDMs.within_task_RDM(model_three, col_names, plotting = True, titlestring = 'Location model RSM')
        
        # best_model_one_df = pd.DataFrame(model_one)
        # best_model_two_df = pd.DataFrame(model_two)
        # mc.simulation.RDMs.df_based_RDM(best_model_one_df, plotting = True)
        # mc.simulation.RDMs.df_based_RDM(best_model_two_df, plotting = True)
         
        sim_clock_midnight = mc.simulation.RDMs.corr_matrices_pearson(RSM_one, RSM_two)
        print(f'Similarity between clock and mignight for task {count} is {sim_clock_midnight[0,1]}')
        sim_clock_location = mc.simulation.RDMs.corr_matrices_pearson(RSM_one, RSM_three)
        print(f'Similarity between clock and location for task {count} is {sim_clock_location[0,1]}')
        sim_midnight_location = mc.simulation.RDMs.corr_matrices_pearson(RSM_two, RSM_three)
        print(f'Similarity between mignight and location for task {count} is {sim_midnight_location[0,1]}')
        
        
        
if section_three_four == 1:
    # 4x4 grid, 4 rewards, 15 ms per step
    # betw_rew_coords = np.load('/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf20230407-150212_perms1000_15_ms_per_step.npy')
    # similarity_dict = mc.simulation.optimise.show_several_taskconfigs(betw_rew_coords, prediction_one = 'location', gridsize = 4, timeperstep = 15, reward_no = 4, prediction_two= 'clocks', prediction_three='midnight', hrf = True)

    # 4x4 grid, 4 rewards, 10ms per step
    betw_rew_coords = np.load('/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf20230411-152557_perms1000_10_ms_per_step.npy')
    similarity_dict = mc.simulation.optimise.show_several_taskconfigs(betw_rew_coords, prediction_one = 'location', gridsize = 4, timeperstep = 10, reward_no = 4, prediction_two= 'clocks', prediction_three='midnight', hrf = True)
    
        
    # 4x4 grid, 4 rewards, 20ms per step
    # betw_rew_coords = np.load('/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf20230411-150723_perms1000_20_ms_per_step.npy')
    # similarity_dict = mc.simulation.optimise.show_several_taskconfigs(betw_rew_coords, prediction_one = 'location', gridsize = 4, timeperstep = 20, reward_no = 4, prediction_two= 'clocks', prediction_three='midnight', hrf = True)
    
    
    # 3x3 grid, 4 rewards, 15ms per step
    #betw_rew_coords = np.load('/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf20230411-122214_perms1000_15_ms_per_step.npy')
    #similarity_dict = mc.simulation.optimise.show_several_taskconfigs(betw_rew_coords, prediction_one = 'location', gridsize = 3, timeperstep = 15, reward_no = 4, prediction_two= 'clocks', prediction_three='midnight', hrf = True)
    
    # 3x3 grid, 4 rewards, 10ms per step
    # betw_rew_coords = np.load('/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf20230411-134215_perms1000_10_ms_per_step.npy')
    # similarity_dict = mc.simulation.optimise.show_several_taskconfigs(betw_rew_coords, prediction_one = 'location', gridsize = 3, timeperstep = 10, reward_no = 4, prediction_two= 'clocks', prediction_three='midnight', hrf = True)
    
    # 3x3 grid, 4 rewards, 20ms per step
    # betw_rew_coords = np.load('/Users/xpsy1114/Documents/projects/multiple_clocks/results/between_configs_midnight_x_clocks_hrf20230411-141148_perms1000_20_ms_per_step.npy')
    # similarity_dict = mc.simulation.optimise.show_several_taskconfigs(betw_rew_coords, prediction_one = 'location', gridsize = 3, timeperstep = 20, reward_no = 4, prediction_two= 'clocks', prediction_three='midnight', hrf = True)
    
    
# it seems as if the most successful paths are those that overlap spatially a lot.
# this means there are a lot of clocks reactivated at different timepoints/phases??

if section_four_one == 1:
    from matplotlib import cm
    repeats = 100
    grid_size = 4
    step_time = 15
    reward_no = 4
    perms = 10
    hrf = False
    
    for i in range(0, repeats): 
        rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False)
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = False)
        single_clock, phase_vector = mc.simulation.predictions.set_single_clock(walk, steps_per_walk, step_time, grid_size)
        single_clock, midnight_matrix, clock_matrix = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, step_time, grid_size)
        
        
        df = pd.DataFrame(np.transpose(single_clock))
        df.insert(0,"Phases", phase_vector)
        if i == 0:
            df_all_configs = df.copy()
        if i > 0:
            df_all_configs = df_all_configs.append(df)
        
        # do the same for the clock matrix
        df_clock = pd.DataFrame(np.transpose(clock_matrix))
        df_clock.insert(0,"Phases", phase_vector)
        if i == 0:
            df_clock_all_configs = df_clock.copy()
        if i > 0:
            df_clock_all_configs = df_clock_all_configs.append(df_clock)
        
        
    # now plot those
    # take only subsets of the clocks, e.g. 2 clocks
    df_first_clock = df_clock_all_configs.iloc[:, 0:13]
    
    df_second_clock = df_clock_all_configs.iloc[:, 13:25]
    df_second_clock["Phases"] = df_first_clock.iloc[:,0]
    
    df_third_clock = df_clock_all_configs.iloc[:, 25:37]
    df_third_clock["Phases"]= df_first_clock.iloc[:,0]
    
    
    colormap = cm.get_cmap('Set3', (len(df_first_clock.columns)-1))  
    df_first_clock.groupby(["Phases"]).mean().plot.barh(color = colormap.colors)
    df_second_clock.groupby(["Phases"]).mean().plot.barh(color = colormap.colors)
    df_third_clock.groupby(["Phases"]).mean().plot.barh(color = colormap.colors)
    # yes, the clocks are phase-selective across many different configurations.
    # df_all_configs.groupby(["Phases"]).mean().plot.barh(color = colormap.colors)



#########################
#### SECTION 5###########
# cares about step-wise predictions and more realistic fMRI modelling
# takes a GLM approach to get there 
"""
1. create vectors with 1s for each step and 0 for the other steps (in ms res)
2. convolve these vectors with hrf
    2.1 optional: subsample both, predictors and current neural models by taking 
    an average across 13 columns 
3. find GLM function and run GLM -> output will be b per step and neuron
4. correlate to create a step x step RDM
5. check the similarity between different models and check if the task configs that are good now are still good


"""  

if section_five_one == 1:
    from sklearn import linear_model
    
    grid = 4
    hrf_stg = True 
    time_per_step = 15
    rewards = 4
    subsample = True
    
    rew_coords = [[0,0], [0,3], [2,2], [3,2]]
    walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid, plotting = True)
    
    # 1. + 2: create regressors, convolve with HRF.
    regressor_matrix = mc.simulation.predictions.gen_regressors_per_step(walk, steps_per_walk, time_per_step, hrf = True)
      
    single_clock, midnight_matrix, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, time_per_step, grid)
    
    # 2.1: subsample
    if subsample == True:
        clocks_model = mc.simulation.predictions.subsample(clocks_model, subsample_factor = 13)
        regressor_matrix = mc.simulation.predictions.subsample(regressor_matrix, subsample_factor = 13)
    
    # 3: GLM per neuron
    coefs_per_neuron = np.zeros((len(clocks_model), len(regressor_matrix)))
    for neuron in range(0, len(clocks_model)):
        x = regressor_matrix
        y = clocks_model[neuron]
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        coefs_per_neuron[neuron] = regr.coef_
        print('Intercept: \n', regr.intercept_)
        print('Coefficients: \n', regr.coef_)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  



