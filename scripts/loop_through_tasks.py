#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:18:05 2023

@author: Svenja Küchenhoff

This script aims at identifying task configurations that have a low correlation
between clocks and location predicitons.

"""

import mc
import pandas as pd
from datetime import datetime 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

# settings for the rest of the script
find_low_similarity_within = 0
find_low_similarity_between = 0
find_low_sim_zerophase_clocks = 1
distr_zero_phase_clocks_optimal = 0
plot_optimal_paths = 0
run_PCA_on_repetitions = 0


# prep the scrip input
phases = ['first_early', 'first_late', 'first_reward', 'scnd_early', 'scnd_late', 'scnd_reward', 'third_early', 'third_late', 'third_reward', 'fourth_early', 'fourth_late', 'fourth_reward']
pathlengths = []
all_pathlengths = []
indices_test = []

##################################################################

### Between Task RSMs.
if find_low_similarity_between == 1:
    loc_RSM_matrix, clock_RSM_matrix, df_clock, df_loc, df_task_configs = mc.simulation.RDMs.between_task_RDM(20, phases, plotting = False)
    similarity_between = mc.simulation.RDMs.corr_matrices(loc_RSM_matrix, clock_RSM_matrix) 

    df_clock_opt, df_loc_opt, task_configs_opt, similarity_opt = mc.simulation.RDMs.find_best_tasks(50000, 12, phases)
    folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_between_tasks'
    sim = str(similarity_opt[0,1])
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = '.csv'
    filename = folder + time + sim[0:5] + file
    task_configs_opt.to_csv(filename)


##################################################################

### Within one task config RSMs.
if find_low_similarity_within == 1:   
    # start the loop.
    cut_off = 0.45
    count = 0
    for i in range(0, 1000):
        ## Section 1.
        ## Create the task and paths
        ##
        reward_coords = mc.simulation.grid.create_grid()
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords)
        
        ## Section 2. 
        ## Setting the Clocks and Location Matrix. 
        ##
        
        # now create the two matrices 
        first_clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
        location_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0) 


        # Section 3. Create RDMs.
        
        loc_RSM = mc.simulation.RDMs.within_task_RDM(location_matrix, phases)
        clock_RSM = mc.simulation.RDMs.within_task_RDM(first_clocks_matrix, phases)
        similarity = mc.simulation.RDMs.corr_matrices(loc_RSM, clock_RSM)    
        all_pathlengths= np.append(all_pathlengths, len(reshaped_visited_fields))
        
        if similarity[1,0] < cut_off:
            # save task configuration and coefficient
            pathlengths= np.append(pathlengths, len(reshaped_visited_fields))
            indices_test = np.append(indices_test, i)
            path = pd.DataFrame(reshaped_visited_fields)
            coef = pd.DataFrame(similarity[0])
            rewards = pd.DataFrame(reward_coords)
            if count == 0:
                save_task = pd.concat([coef, path, rewards], axis = 1)
            else: 
                save_task = pd.concat([save_task, coef, path, rewards], axis = 1)
            count += 1 
        
    folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_within_task'
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = '.csv'
    filename = folder + time + file
    if count > 0:
        save_task.to_csv(filename)
        
##################################################################        

## plot distribution of random configs for 0phase clocks and clocks
# do this for both convoluted predictions and without 
if find_low_sim_zerophase_clocks == 1:
    similarity_values = []
    similarity_values_hrf = []
    # maximally_dissimilar = []
    # maximally_dissimilar_hrf = []
    countmax = 0
    countmax_hrf = 0
    for count in range(0,10000):
        # section 1.1
        reward_coords = mc.simulation.grid.create_grid(plot = False)
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = False)
        
        # section 2.2
        secs_per_step = 3.8
        clocksm, neuroncl, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, secs_per_step)
        
        # now do the convolution
        clocks_over_time_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_over_time, all_stepnums, secs_per_step, plotting = False)
    
        # section 2.3
        zero_phase_clocks_m_hrf = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_over_time_hrf, all_stepnums, 3)
        zero_phase_clocks_m = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_over_time, all_stepnums, 3)
        
        # section 4.2
        counter = list(range(0,len(clocks_over_time[0])))
        seconds = counter.copy()
        for i in counter:
            seconds[i] = str(i)  
            
        zero_clock_RSM = mc.simulation.RDMs.within_task_RDM(zero_phase_clocks_m, seconds)
        zero_clock_RSM_hrf = mc.simulation.RDMs.within_task_RDM(zero_phase_clocks_m_hrf, seconds)
        
        clock_RSM_hrf = mc.simulation.RDMs.within_task_RDM(clocks_over_time_hrf, seconds)        
        similarity_hrf = mc.simulation.RDMs.corr_matrices(zero_clock_RSM_hrf, clock_RSM_hrf)  
        similarity_values_hrf.append(1 - similarity_hrf[0,1])
        
        clock_RSM = mc.simulation.RDMs.within_task_RDM(clocks_over_time, seconds)  
        similarity = mc.simulation.RDMs.corr_matrices(zero_clock_RSM, clock_RSM)  
        similarity_values.append(1 - similarity [0,1])
        
        # # save those configurations that have are maximally dissimlar. 
        # if similarity[0,1] <  0.75:          
        #     path = pd.DataFrame(reshaped_visited_fields)
        #     coef = pd.DataFrame(similarity[0])
        #     rewards = pd.DataFrame(reward_coords)
        #     if countmax == 0:
        #         maximally_dissimilar = pd.concat([coef, path, rewards], axis = 1)
        #     else: 
        #         maximally_dissimilar = pd.concat([maximally_dissimilar, coef, path, rewards], axis = 1)
        #     countmax += 1 
            
        # same for the HRF convolved ones.
        if similarity_hrf[0,1] <  0.75:          
            path_hrf = pd.DataFrame(reshaped_visited_fields)
            coef_hrf = pd.DataFrame(similarity_hrf[0])
            rewards_hrf = pd.DataFrame(reward_coords)
            if countmax_hrf == 0:
                maximally_dissimilar_hrf = pd.concat([coef_hrf, path_hrf, rewards_hrf], axis = 1)
            else: 
                maximally_dissimilar_hrf = pd.concat([maximally_dissimilar_hrf, coef_hrf, path_hrf, rewards_hrf], axis = 1)
            countmax_hrf += 1 
            
    
    # # plot distribution.
    # plt.figure()
    # ax = plt.axes()
    # plt.hist(similarity_values)
    # plt.gca().set(title='Variance clocks_raw beyond 0-phase-clocks 10.000 perms)', ylabel = ('frequency'), xlabel='1 - Similarity');
    # # and save.
    # folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_0phase_with_clocks'
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = '.csv'
    # filename = folder + time + file
    # if countmax > 0:
    #     maximally_dissimilar.to_csv(filename)
    
    # same for the HRF one.  
    # plot distribution.
    plt.figure()
    ax2 = plt.axes()
    plt.hist(similarity_values_hrf)
    plt.gca().set(title='Variance clocks_hrf beyond 0-phase-clocks (10.000 perms)', ylabel = ('frequency'), xlabel='1 - Similarity');
    # and save.
    folder_hrf = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_0phase_with_clocks_hrf'
    seconds = '_' + str(secs_per_step) + '_secs_per_step'
    filename_hrf = folder_hrf + time + seconds + file 
    if countmax_hrf > 0:
        maximally_dissimilar_hrf.to_csv(filename_hrf)


###########################################
# optimise for low correlation between 0-angle-neurons and complete clocks.
# use: 
# 1. different grid sizes
# 2. different times per step
# 3. different amounts of reward








## plot distribution of correlation between RDM for 0phase clocks and clocks
## and use the configurations that work well in terms of space
if distr_zero_phase_clocks_optimal == 1:
    similarity_values = []
    filename = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_between_tasks20230131-1032590.244.csv'
    optimal_data_df = pd.read_csv(filename)
    optimal_data = optimal_data_df.to_numpy() 
    optimal_data = optimal_data[:,1:None]
    secs_per_step = 2
    total_configs = int(len(optimal_data[0])/4)
    for task_no in range(0, total_configs):
        reward_coords = optimal_data[0:4, (task_no*4):((task_no*4)+2)]
        reward_coords = reward_coords.tolist()
        reward_list = [[int(x) for x in lst] for lst in reward_coords]       
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_list, plotting = False)
        clocksm, neuroncl, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, secs_per_step)
        clocks_over_time_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_over_time, all_stepnums, secs_per_step, plotting = False)
        zero_phase_clocks_m = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_over_time_hrf, all_stepnums, 3)
        counter = list(range(0,len(clocks_over_time[0])))
        seconds = counter.copy()
        for i in counter:
            seconds[i] = str(i)  
        zero_clock_RSM = mc.simulation.RDMs.within_task_RDM(zero_phase_clocks_m, seconds)
        clock_RSM = mc.simulation.RDMs.within_task_RDM(clocks_over_time_hrf, seconds)   
        similiarty = mc.simulation.RDMs.corr_matrices(zero_clock_RSM, clock_RSM)  
        similarity_values.append(1 - similiarty[0,1])
   
    plt.figure()
    plt.hist(similarity_values)
    plt.gca().set(title='Dissimilarity 0phase clocks vs clocks, for 10 configs (decorrelated by space)', ylabel = ('frequency'), xlabel='1 - Similarity');
    
    
    
##################################################################       
        
# plot the optimal between-task configurations
# run a PCA for these configurations
       
if plot_optimal_paths == 1:
    filename = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs_between_tasks20230131-1032590.244.csv'
    optimal_data_df = pd.read_csv(filename)
    optimal_data = optimal_data_df.to_numpy() 
    optimal_data = optimal_data[:,1:None]
    secs_per_step = 2
    total_configs = int(len(optimal_data[0])/4)
    for task_no in range(0, total_configs):
        reward_coords = optimal_data[0:4, (task_no*4):((task_no*4)+2)]
        reward_coords = reward_coords.tolist()
        reward_list = [[int(x) for x in lst] for lst in reward_coords]       
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_list, plotting = False)
        clocksm, neuroncl, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, secs_per_step)
        clocks_over_time_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_over_time, all_stepnums, secs_per_step, plotting = False)
        if task_no == 0:
            concatenated_clocks = clocks_over_time_hrf.copy()
        else:
            concatenated_clocks = np.append(concatenated_clocks,clocks_over_time_hrf, axis = 1)
    concatenated_clocks = np.nan_to_num(concatenated_clocks, nan=0)
    concatenated_clocks_long = np.ndarray.transpose(concatenated_clocks)
    # some pandas playaround
    feat_cols = ['neuron'+str(i) for i in range(0, len(concatenated_clocks))]
    neuron_dataset = pd.DataFrame(concatenated_clocks_long, columns=feat_cols)
    x = neuron_dataset.loc[:, feat_cols].values
    x = StandardScaler().fit_transform(x) # normalizing features / neuron values
    normalised_neurons_df = pd.DataFrame(x, columns= feat_cols)
    normalised_neurons_df.tail()
    
    # now run the PCA.
    pca_neurons = PCA()
    principal_components_neurons = pca_neurons.fit_transform(x)
    principal_components_neurons_df = pd.DataFrame(data = principal_components_neurons)
    
    print('Explained variation per principal component: {}'.format(pca_neurons.explained_variance_))
    
    # scree plot of variance explained
    plt.figure()
    sns.set()  
    plt.plot(
        range(1,len(pca_neurons.explained_variance_)+1),
        np.cumsum(pca_neurons.explained_variance_),
        c='red',
        label='Cumulative Explained Variance')
     
    plt.legend(loc='upper left')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance (eignenvalues)')
    plt.title('10 decorrelated tasks, r = .244 between location and clocks')
     
    plt.show()


    # scree plot of variance explained
    plt.figure()
    sns.set()  
    plt.plot(
        range(1,len(pca_neurons.explained_variance_)+1),
        pca_neurons.explained_variance_,
        c='red',
        label='Eigenvalues')
     
    plt.legend(loc='upper left')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance (eignenvalues)')
    plt.title('10 decorrelated tasks, r = .244 between location and clocks')
     
    plt.show()

##################################################################



if run_PCA_on_repetitions == 1:
    no_path_repeats = 40
    concatenated_clocks = np.tile(concatenated_clocks, (1, no_path_repeats))
    concatenated_clocks_long = np.ndarray.transpose(concatenated_clocks)
    # some pandas playaround
    feat_cols = ['neuron'+str(i) for i in range(0, len(concatenated_clocks))]
    neuron_dataset = pd.DataFrame(concatenated_clocks_long, columns=feat_cols)
    x = neuron_dataset.loc[:, feat_cols].values
    x = StandardScaler().fit_transform(x) # normalizing features / neuron values
    normalised_neurons_df = pd.DataFrame(x, columns= feat_cols)
    normalised_neurons_df.tail()
    
    # now run the PCA.
    pca_neurons = PCA()
    principal_components_neurons = pca_neurons.fit_transform(x)
    principal_components_neurons_df = pd.DataFrame(data = principal_components_neurons)
    
    print('Explained variation per principal component: {}'.format(pca_neurons.explained_variance_))
    
    # scree plot of variance explained
    plt.figure()
    sns.set()  
    plt.plot(
        range(1,len(pca_neurons.explained_variance_)+1),
        np.cumsum(pca_neurons.explained_variance_),
        c='red',
        label='Cumulative Explained Variance')
     
    plt.legend(loc='upper left')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance (eignenvalues)')
    plt.title('10 decorrelated tasks, 40 repeats')
     
    plt.show()


    # scree plot of variance explained
    plt.figure()
    sns.set()  
    plt.plot(
        range(1,len(pca_neurons.explained_variance_)+1),
        pca_neurons.explained_variance_,
        c='red',
        label='Eigenvalues')
     
    plt.legend(loc='upper left')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance (eignenvalues)')
    plt.title('10 decorrelated tasks, 40 repeats')
     
    plt.show()





