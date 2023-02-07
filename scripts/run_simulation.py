#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:05:51 2023

@author: Svenja Kuechenhoff

this script calls on several simulation functions to eventually create RDMs
and check if predictions of location neuron activations are sufficiently 
distinct from phase clock neuron activation patterns.

It is mainly thought to check if the simulations are correct. It therefore 
creates loads of figures for task configuration, paths, resulting matrices,
and neuron activations.

This script only creates the similarity between clocks and locations for one task
configuration. 

"""

# %reset -f

import mc
import pandas as pd
from matplotlib import pyplot as plt
import statistics as stats
import numpy as np
import seaborn as sns

## SETTINGS
section_oneone = 1 # Create the task
section_onetwo = 0 # Create a distribution of most common pathlengths
section_twoone = 0 # Setting the Clocks and Location Matrix. 
section_twotwo = 1 # Setting the Clocks + locs but in 'real time' + HRF convolve
section_twothree = 0 # Setting 0-phase clocks in 'real time'
section_twofour = 0 # concatenate 400 HRF convolved clocks and PCA
section_three = 0 # Create 'neuron plots'
section_fourone = 0 # Create RDMs.
section_fourtwo = 0 # create RDMS between 0 phase clock and clocks (HRF + by time)


## Section 1.1
## Create the task
##
if section_oneone == 1:
    reward_coords = mc.simulation.grid.create_grid(plot = False)
    reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = False)

############## 

## Section 1.2
## Create a distribution of most common pathlengths
##
if section_onetwo == 1:
    pathlenghts = list()

    for i in range(1,10000):
        reward_coords = mc.simulation.grid.create_grid(plot = False)
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = False)
        pathlenghts.append(len(reshaped_visited_fields))
    plt.figure()    
    plt.hist(pathlenghts) 
    print(stats.mean(pathlenghts))
    print(stats.mode(pathlenghts))
    print(stats.median(pathlenghts))
    print(max(pathlenghts))
############## 


 
## Section 2.1
## Setting the Clocks and Location Matrix. 
##

# now create the two matrices, print them and plot them. 
if section_twoone == 1:
    first_clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
    location_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0) 
     
    print(first_clocks_matrix)
    print(location_matrix)
     
    mc.simulation.predictions.plotlocation(location_matrix)
    mc.simulation.predictions.plotclocks(first_clocks_matrix)
    
    one_clock = first_clocks_matrix[144:156,:]
    mc.simulation.predictions.plot_one_clock(one_clock)

# #########################


## Section 2.2
## Setting the Clocks and locations but in 'real time'
##
if section_twotwo == 1:   
    deci_secs_per_step = 15
    # needs section 1.1
    
    location_m, location_over_time = mc.simulation.predictions.set_location_by_time(reshaped_visited_fields, all_stepnums, deci_secs_per_step)
    clocksm, neuroncl, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, deci_secs_per_step)    

    
    # PLOTTING PART 
    # plotting the location matrix
    mc.simulation.predictions.plotlocation_pertime(location_over_time, deci_secs_per_step, all_stepnums)
        
    # plotting the whole matrix: clocks
    # mc.simulation.predictions.plotclock_pertime(clocks_over_time, deci_secs_per_step, all_stepnums)
    # # plotting only one anchor 
    # one_anch_clocks_over_time = clocks_over_time[0:35,:]
    # mc.simulation.predictions.plot_one_anchor_all_clocks_pertime(one_anch_clocks_over_time, deci_secs_per_step, all_stepnums)
    # scnd_anch_clocks_over_time = clocks_over_time[288:324,:]
    # mc.simulation.predictions.plot_one_anchor_all_clocks_pertime(scnd_anch_clocks_over_time, deci_secs_per_step, all_stepnums)


    # now do the convolution
    # first of the location matrix
    location_over_time_hrf = mc.simulation.predictions.convolve_with_hrf(location_over_time, all_stepnums, deci_secs_per_step, plotting = False)
    
    # then of the clock matrix
    clocks_over_time_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_over_time, all_stepnums, deci_secs_per_step, plotting = True)
    
    # plotting the convolved matrix
    # plotting the location matrix
    mc.simulation.predictions.plotlocation_pertime(location_over_time_hrf, deci_secs_per_step, all_stepnums)
    
    # plotting the whole clock matrix
    mc.simulation.predictions.plotclock_pertime(clocks_over_time_hrf, deci_secs_per_step, all_stepnums)
    # # plotting only one anchor 
    # one_anch_clocks_over_time_hrf = clocks_over_time_hrf[0:35,:]
    # mc.simulation.predictions.plot_one_anchor_all_clocks_pertime(one_anch_clocks_over_time_hrf, deci_secs_per_step, all_stepnums)
    
    # scnd_anch_clocks_over_time_hrf = clocks_over_time_hrf[288:324,:]
    # mc.simulation.predictions.plot_one_anchor_all_clocks_pertime(scnd_anch_clocks_over_time_hrf, deci_secs_per_step, all_stepnums)


## Section 2.3
## Setting 0-phase clocks in 'real time'
## (run section 2.2 first!)
##
if section_twothree == 1:
    zero_phase_clocks_m = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_over_time_hrf, all_stepnums, 3)
    print(zero_phase_clocks_m)

    # next step: correlate these matrices with the full clock matrix!!!

## Section 2.4
## Creating a concatenated version of 400 different tasks using hte hrf convolved by time matrix,
## then running a PCA and extracting the components.
##

if section_twofour == 1:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    for i in range(0,400):
        secs_per_step = 2
        reward_coords = mc.simulation.grid.create_grid(plot = False)
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = False)
        clocksm, neuroncl, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, secs_per_step)
        clocks_over_time_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_over_time, all_stepnums, secs_per_step, plotting = False)
        if i == 0:
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
    plt.title('Scree plot')
     
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
    plt.title('Scree plot')
     
    plt.show()



# #########################


## Section 3. 
## Create 'neuron plots'
##

# needs section 1.1 and 2.1
if section_three == 1:
    # this is the activity of one neuron based on a single run.
    plt_neurontwo = first_clocks_matrix[146,:]
    randomneuronsin = pd.DataFrame({'value': plt_neurontwo, #these will be averages
                          'bearing': range(0, 360, 30),
                          'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})   
    mc.simulation.predictions.plot_neurons(randomneuronsin)
     
    # to plot the average activity of neuron across many runs, create data of many runs
    loops = 1000
    matrixtype = 'location'
    first_average_loc = mc.simulation.predictions.many_configs_loop(loops, matrixtype)
    matrixtypetwo = 'clocks'
    first_average_clock = mc.simulation.predictions.many_configs_loop(loops, matrixtypetwo)
    # 0,36,72,108,144,180,216,252,288 are respective field-anchors. 144 might be most interesting
    
    # PLOT MATRICES
    # Location
    mc.simulation.predictions.plotlocation(first_average_loc)
    # Clocks
    mc.simulation.predictions.plotclocks(first_average_clock)
    # Clocks, one anchor = 3 clocks
    anchor_five = first_clocks_matrix[144:179,:]
    # Clocks, one clock = 12 neurons
    mc.simulation.predictions.plot_one_anchor_all_clocks(anchor_five)
    
    one_clock = first_clocks_matrix[144:156,:]
    mc.simulation.predictions.plot_one_clock(one_clock)
    
    # ALIGN BY PHASE - PLOT NEURONS (Polar plots)
    # plot two CLOCK neuron examples, based on average.
    # example 1
    plt_neuron = first_average_clock[145,:]
    randomneuron = pd.DataFrame({'value': plt_neuron, #these will be averages
                          'bearing': range(0, 360, 30),
                          'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})   
    mc.simulation.predictions.plot_neurons(randomneuron)
    
    #example 2
    plt_neurontwo = first_average_clock[146,:]
    randomneurontwo = pd.DataFrame({'value': plt_neurontwo, #these will be averages
                          'bearing': range(0, 360, 30),
                          'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})   
    mc.simulation.predictions.plot_neurons(randomneurontwo)
    
    # plot LOCATION neuron average like this
    loc_neuron = first_average_loc[2,:]
    plt_loc_neuron = pd.DataFrame({'value': loc_neuron, #these will be averages
                          'bearing': range(0, 360, 30),
                          'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})   
    mc.simulation.predictions.plot_neurons(plt_loc_neuron)
    

#########################
## Section 4.1
## Create RDMs.
##

# needs section 1.1 and 2.1
if section_fourone == 1:
    phases = ['first_early', 'first_late', 'first_reward', 'scnd_early', 'scnd_late', 'scnd_reward', 'third_early', 'third_late', 'third_reward', 'fourth_early', 'fourth_late', 'fourth_reward']
    
    loc_RSM = mc.simulation.RDMs.within_task_RDM(location_matrix, phases)
    clock_RSM = mc.simulation.RDMs.within_task_RDM(first_clocks_matrix, phases)
    similarity = mc.simulation.RDMs.corr_matrices(loc_RSM, clock_RSM)
    
    print(similarity)


## Section 4.2
## Create RDMs between 0 phase clocks (HRF convoluted, by time) and 
## whole clocks matrices (HRF convoluted, by time)
##

# needs section 1.1, 2.2 and 2.3
if section_fourtwo == 1:
    counter = list(range(0,len(clocks_over_time[0])))
    seconds = counter.copy()
    for i in counter:
        seconds[i] = str(i)  
    zero_clock_RSM = mc.simulation.RDMs.within_task_RDM(zero_phase_clocks_m, seconds)
    clock_RSM = mc.simulation.RDMs.within_task_RDM(clocks_over_time_hrf, seconds)   
    similiarty = mc.simulation.RDMs.corr_matrices(zero_clock_RSM, clock_RSM)    
    print(1- similiarty)




      
