#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:34:15 2023

@author: Svenja Kuechenhoff

This module generates hypothesis matrices of neural activity based on 
predictions for location and phase-clock neurons.

This is getting messy.
I am trying to follow this order:
    0. Helper (field to number, convolve HRF, subsample matrix, regressors per phase state)
    1. simulation models: clocks - midnight - phase - location
    2. continuous models: all - midnight - phase - location
    3. ephys models: continuous - clocks - midnight - phase - location
    4. plotting

Currently in-use models will be first in a section in case there are several versions.
    

First, you will find different versions to model the clock/midnight neurons
Then, there will be different versions of modelling the location neurons
Third, there are functions to plot those

Lastly, there are some functions to play around with the matrices.


# If you successfully ran mc.simulation.grid.create_grid() and mc.simulation.grid.walk_paths(reward_coords),
# you now have 4 locations, 4 states, and paths between the locations/states,
# including going back to A after D.
# 4 locations: coordinates in points[0:4]
# 4 states: points[0:4]
# 4 paths between locations/states: all_paths[0:4]
# A-B, B-C, C-D, D-A
# (number of steps for each path: all_stepnums)

# the next step is thus to now create a prediction for neural firing of location neurons,
# and the same thing for phase neurons based on the multiple clock model.



"""
import colormaps as cmaps
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib.gridspec import GridSpec
import mc
import scipy.signal
from matplotlib.patches import Circle
import scipy
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal
from scipy.stats import norm
from itertools import product
from scipy.signal import argrelextrema
from scipy import interpolate
import math

# PART 1
############ Helpers ################
#####################################
#############################################
##### Playing around with the matrices. #####
##############################################


def field_to_number(currentfield, size_of_grid):
    y = currentfield[0]
    x = currentfield[1]
    fieldnumber = x + y * size_of_grid
    return fieldnumber


############ HRF #############
def convolve_with_hrf(clocks_per_sec, step_number, step_time, plotting = False):
    # import pdb; pdb.set_trace()
    # now do the convolution
    # take the arrays around the activity bumbps. the 1s need to be the peak of the HRF.
    # take the HRF. Convolve both arrays using np.convolve()
    # to do more efficient convolution and big arrays, use scipy.signal.fftconvolve(arr1, arr2)
    # (although this isnt reaaaally necessary in my case)
    # also be careful to get same sized output!
    # one example:
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1]
    def hrf(t):
        "A hemodynamic response function"
        return t ** 8.6 * np.exp(-t / 0.547)
    hrf_times = np.arange(0, step_time*total_steps, 0.1)
    hrf_signal = hrf(hrf_times)
    # to mimic a cyclic function, put double:
    clocks_per_sec_hrf_double = np.concatenate([clocks_per_sec, clocks_per_sec], axis = 1)   
    clocks_per_sec_hrf = clocks_per_sec.copy()
    for row in range(0, len(clocks_per_sec)):
        if np.isnan(clocks_per_sec[row,0])== False:
            neuron = clocks_per_sec_hrf_double[row,:]
            convolve = scipy.signal.convolve(neuron, hrf_signal) 
            clocks_per_sec_hrf[row,:] = convolve[len(clocks_per_sec[0]):(len(clocks_per_sec[0])*2)]
    if plotting == True:
        plt.figure()
        plt.plot(hrf_times, hrf_signal)
        plt.xlabel('time (seconds)')
        plt.ylabel('BOLD signal')
        plt.title('Estimated BOLD signal for event at time 0') 
    return clocks_per_sec_hrf


def gen_regressors_per_step(walked_path, step_number, step_time, hrf = True):
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1]   
    regressors = np.zeros((step_time*total_steps, total_steps))
    for i in range(0, len(regressors[0])):
        regressors[i*step_time: i*step_time + step_time , i] = 1
    
    regressors = np.transpose(regressors)
    
    if hrf == True:
        regressors = mc.simulation.predictions.convolve_with_hrf(regressors, step_number, step_time)
        
    return regressors
    

def subsample(matrix, subsample_factor):
    # import pdb; pdb.set_trace() 
    subsampled_matrix = np.zeros((len(matrix), len(matrix[0])//subsample_factor ))
    for row in range(len(subsampled_matrix)):
        for col in range(len(subsampled_matrix[0])):
            subsampled_matrix[row,col] = np.mean((matrix[row, col*subsample_factor: col*subsample_factor + subsample_factor]))
            
    return subsampled_matrix



def interpolate_neurons(matrix, interpol_len):
    # import pdb; pdb.set_trace() 
    interpol_matrix = np.zeros((len(matrix), interpol_len))
    for i, row in enumerate(matrix):
        x = np.linspace(0,1,len(row))
        new_x = np.linspace(0,1, interpol_len)
        f = interpolate.interp1d(x,row, kind = 'linear')
        interpol_matrix[i] = f(new_x)
    return interpol_matrix
        
        

# loop function to create an average prediction.
def many_configs_loop(loop_no, which_matrix):
    # import pdb; pdb.set_trace()
    if which_matrix != 'clocks' and which_matrix != 'location':
        raise TypeError("Please enter 'location' or 'clocks' to create the correct matrix")   
    for loop in range(loop_no):
        reward_coords = mc.simulation.grid.create_grid()
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords)
        if which_matrix == 'location':
            temp_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 3) 
        elif which_matrix == 'clocks':
            temp_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)
        if loop < 1:
            sum_matrix = temp_matrix[:]
        else:
            sum_matrix = np.nansum(np.dstack((sum_matrix[:],temp_matrix[:])),2)
    average_matrix = sum_matrix[:]/loop_no
    return average_matrix


def create_x_regressors_per_state_simulation(walked_path, step_per_subpath, time_per_step ,no_regs_per_state, grid_size = 3):
    # import pdb; pdb.set_trace()
    # first test if bins are smaller than time_per_step
    if time_per_step < no_regs_per_state:
        raise Exception(f"Sorry, you can't have less bins  per states (curr. {no_regs_per_state}) than the resolution of 1 stepn(curr. {time_per_step}).")
    # trajectory = []
    # for elem in walked_path:
    #     trajectory.append(mc.simulation.predictions.field_to_number(elem, grid_size))
    # trajectory = np.repeat(trajectory, repeats = time_per_step)
    
    timebins_per_step = [step*time_per_step for step in step_per_subpath]
    
    n_states = len(step_per_subpath)                       
    cols_to_fill_previous = 0
    cumsumsteps = np.cumsum(timebins_per_step)
    
    regressors = np.zeros([n_states*no_regs_per_state, cumsumsteps[-1]])
    
    for count_paths, (pathlength) in enumerate(timebins_per_step):
        # create a string that tells me how many columns are one nth of a state
        time_per_state_in_nth_part = ([pathlength // no_regs_per_state + (1 if x < pathlength % no_regs_per_state else 0) for x in range (no_regs_per_state)])
        time_per_state_in_nth_part_cum = np.cumsum(time_per_state_in_nth_part)
        for nth_part in range(0, no_regs_per_state):
            if nth_part == 0:
                regressors[nth_part+(count_paths*no_regs_per_state), cols_to_fill_previous+ 0: cols_to_fill_previous+ time_per_state_in_nth_part_cum[nth_part]] = 1   
            else:
                regressors[nth_part+(count_paths*no_regs_per_state), cols_to_fill_previous + time_per_state_in_nth_part_cum[nth_part-1]: cols_to_fill_previous + time_per_state_in_nth_part_cum[nth_part]] = 1
        cols_to_fill_previous = cols_to_fill_previous + pathlength
    return regressors

def find_start_end_indices(locations, index):
    # Value at the specified index
    target_value = locations[index]
    
    # Initialize start_idx and end_idx
    start_idx = index
    end_idx = index
    
    if index == -1:
        # if reward is found at last index, only test
        # if it was actually found slightly earlier
        while locations[start_idx - 1] == target_value:
            start_idx -= 1
    else:
        # Search backwards to find where the target_value starts
        while start_idx > 0 and locations[start_idx - 1] == target_value:
            start_idx -= 1
        
        # Search forwards to find where the target_value ends
        while end_idx < len(locations) - 1 and locations[end_idx + 1] == target_value:
            end_idx += 1
    
    return start_idx, end_idx
    

def create_x_regressors_per_state(beh_data_curr_rep_dict, no_regs_per_state=5, only_for_rewards = False):
    # import pdb; pdb.set_trace()
    step_no = beh_data_curr_rep_dict['step_number']
    subpath_timings = beh_data_curr_rep_dict['timings_repeat']
    walked_path = beh_data_curr_rep_dict['trajectory']
    n_states = len(step_no)
    
    if only_for_rewards == True:
        # do it differently. instead of dividing into 3 phases, determine 
        # for which timebins the agent is at the reward location.
        #regressors = np.zeros([n_states, len(walked_path)-1])
        regressors = np.zeros([n_states, len(walked_path)])
        for rew_idx in range(0, n_states):
            #import pdb; pdb.set_trace()
            reward_found_at = subpath_timings[rew_idx+1]
            if reward_found_at < len(walked_path)+1:
                if reward_found_at == len(walked_path):
                    reward_found_at = -1
                start_curr_rew, end_at_curr_rew = mc.simulation.predictions.find_start_end_indices(walked_path, reward_found_at)
            regressors[rew_idx, start_curr_rew:end_at_curr_rew] = 1
        # import pdb; pdb.set_trace()
        # THIS IS WHERE THE ERROR COMES FROM
        # CHANGE HOW I DEAL WITH THE LAST REWARD!!!
        
    else:
        # such that dimensions match!!!
        regressors = np.zeros([n_states*no_regs_per_state, len(walked_path)])
        # right so this is a bit more complicated. because of the indexing, 
        # the last repeat actually has to be treated differently. I think
        # that the trajectory there has to basically be 1 shorter- but only for the last!
        # what if i do this?
        length_matrix = []
        for count_paths, (pathlength) in enumerate(step_no):
            length_matrix.append(len(walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]))
        regressors = np.zeros([n_states*no_regs_per_state, np.sum(length_matrix)])
            
        
        cols_to_fill_previous = 0
        for count_paths, (pathlength) in enumerate(step_no):
            # identify subpaths
            # if count_paths == 3:
            #     curr_path = walked_path[subpath_timings[count_paths]:]
            # else:
            #     curr_path = walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]
                
            curr_path = walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]
            
            cols_to_fill = len(curr_path)
            # create a string that tells me how many columns are one nth of a state
            time_per_state_in_nth_part = ([cols_to_fill // no_regs_per_state + (1 if x < cols_to_fill % no_regs_per_state else 0) for x in range (no_regs_per_state)])
            time_per_state_in_nth_part_cum = np.cumsum(time_per_state_in_nth_part)
            for nth_part in range(0, no_regs_per_state):
                if nth_part == 0:
                    regressors[nth_part+(count_paths*no_regs_per_state), cols_to_fill_previous+ 0: cols_to_fill_previous+ time_per_state_in_nth_part_cum[nth_part]] = 1   
                else:
                    regressors[nth_part+(count_paths*no_regs_per_state), cols_to_fill_previous + time_per_state_in_nth_part_cum[nth_part-1]: cols_to_fill_previous + time_per_state_in_nth_part_cum[nth_part]] = 1
            cols_to_fill_previous = cols_to_fill_previous + cols_to_fill

    return regressors


# TO TEST IF EVERYTHING IS GOING OK
def simulate_fake_data(beh_data_curr_rep_dict, model_to_simulate, repeat_idx):
    # import pdb; pdb.set_trace()
    step_no = beh_data_curr_rep_dict['step_number']
    subpath_timings = beh_data_curr_rep_dict['timings_repeat']
    walked_path = beh_data_curr_rep_dict['trajectory']
    n_states = len(step_no)
    
    # basically this needs to be the model that I am expecting (i.e. for state, the entire state is the same)
    # plus added noise. I want to add more noise the higher the repeat. Repeat 0,1,2 shall have comparably litte
    # random noise, while repeats 3,4,5,6,7,8,9 shall have a lot of random noise.
    # add noise depending on the repeat.
    fake_neurons = np.random.rand(n_states*10, len(walked_path)) * repeat_idx*10
    for count_paths, (pathlength) in enumerate(step_no):
        fake_neurons[count_paths*10:count_paths*10+10, subpath_timings[count_paths]:subpath_timings[count_paths+1]] = fake_neurons[count_paths*10:count_paths*10+10, subpath_timings[count_paths]:subpath_timings[count_paths+1]] + 1

    return fake_neurons
    

# CREATE A NEW, MORE FLEXIBLE ONE FOR MY FMRI DATA.
def create_regressors_flexi(walked_path, subpath_timings, step_no, no_regs_per_state):
    # import pdb; pdb.set_trace()
    n_states = len(step_no)
    regressors = np.zeros([n_states*no_regs_per_state, len(walked_path)])
    cols_to_fill_previous = 0
    for count_paths, (pathlength) in enumerate(step_no):
        # identify subpaths
        curr_path = walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]
        cols_to_fill = len(curr_path)
        # create a string that tells me how many columns are one nth of a state
        time_per_state_in_nth_part = ([cols_to_fill // no_regs_per_state + (1 if x < cols_to_fill % no_regs_per_state else 0) for x in range (no_regs_per_state)])
        time_per_state_in_nth_part_cum = np.cumsum(time_per_state_in_nth_part)
        for nth_part in range(0, no_regs_per_state):
            if nth_part == 0:
                regressors[nth_part+(count_paths*no_regs_per_state), cols_to_fill_previous+ 0: cols_to_fill_previous+ time_per_state_in_nth_part_cum[nth_part]] = 1   
            else:
                regressors[nth_part+(count_paths*no_regs_per_state), cols_to_fill_previous + time_per_state_in_nth_part_cum[nth_part-1]: cols_to_fill_previous + time_per_state_in_nth_part_cum[nth_part]] = 1
        cols_to_fill_previous = cols_to_fill_previous + cols_to_fill
    return regressors


def define_start_and_end_of_repeat_regressors(version, example_model, repeat_idx):
    version_spec_start = 0
    version_spec_end = example_model.shape[1] 
    
    if version in ['03-e', '03-4-e', '03-rep1', '03-4-rep1']:
        if version in ['03-e', '03-4-e']:
            version_spec_end = repeat_idx[2]
        else:
            version_spec_end = repeat_idx[1]
    # NOTE: I DONT THINK THIS WAS CORRECT BEFORE FOR LATE REP/REP 5
    # DOUBLE CHECK MY RESULTS HERE!!!   
    if version in ['03-l', '03-4-l', '03-rep5', '03-4-rep5']:
        if version in ['03-l', '03-4-l']:
            version_spec_start = repeat_idx[2]
        else:
            version_spec_start = repeat_idx[4]

    if version in ['03-rep2', '03-4-rep2']:
        version_spec_start = repeat_idx[1]
        version_spec_end = repeat_idx[2]

    if version in ['03-rep3', '03-4-rep3']:
        version_spec_start = repeat_idx[2]
        version_spec_end = repeat_idx[3]

    if version in ['03-rep4', '03-4-rep4']:
        version_spec_start = repeat_idx[3]
        version_spec_end = repeat_idx[4]
    
    return version_spec_start, version_spec_end
    


def create_regressors_per_state_phase_ephys(walked_path, subpath_timings, step_no, grid_size = 3, phases=3, plotting = False, field_no_given = None, ax = None):
    # import pdb; pdb.set_trace()
    n_states = len(step_no)
    regressors = np.zeros([phases*n_states,len(walked_path)])
    cols_to_fill_previous = 0
    for count_paths, (pathlength) in enumerate(step_no):
        # identify subpaths
        curr_path = walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]
        cols_to_fill = len(curr_path)
        # create a string that tells me how many columns are one phase clock
        # this is where I make the assumption of how long every phase actually takes, if not splittable evenly.
        # i.e. > phase is NOT linked to steps taken, but to TIME
        time_per_phase_in_clock = ([cols_to_fill // phases + (1 if x < cols_to_fill % phases else 0) for x in range (phases)])
        time_per_phase_in_clock_cum = np.cumsum(time_per_phase_in_clock)
        for phase in range(0, phases):
            if phase == 0:
                regressors[phase+(count_paths*phases), cols_to_fill_previous+ 0: cols_to_fill_previous+ time_per_phase_in_clock_cum[phase]] = 1
            elif phase == 1:
                regressors[phase+(count_paths*phases), cols_to_fill_previous + time_per_phase_in_clock_cum[phase-1]: cols_to_fill_previous + time_per_phase_in_clock_cum[phase]] = 1
            elif phase == 2:
                regressors[phase+(count_paths*phases), cols_to_fill_previous + time_per_phase_in_clock_cum[phase-1]: cols_to_fill_previous + time_per_phase_in_clock_cum[phase]] = 1
                
        cols_to_fill_previous = cols_to_fill_previous + cols_to_fill
    return regressors


def transform_data_to_betas(data_matrix, regressors, intercept = False):
    # import pdb; pdb.set_trace()
    
    # FIND OUT WHY THE STATE THING DOESNT WORK. 
    # TIMINGS????
    
    # careful! I now don't include an intercept by default.
    # this is because the way I use it, the regressors would be a linear combination of the intercept ([11111] vector)
    beta_matrix = np.zeros((len(data_matrix), len(regressors)))
    # first check if there are any nans in the data, and if so, replace with 0
    data_matrix = np.nan_to_num(data_matrix)
    if intercept == False:
        for index, row in enumerate(data_matrix): 
            beta_matrix[index] = LinearRegression(fit_intercept=False).fit(np.transpose(regressors), row).coef_
    elif intercept == True:
        for index, row in enumerate(data_matrix): 
            beta_matrix[index] = LinearRegression().fit(np.transpose(regressors), row).coef_         
    return beta_matrix



def normalise_neurons(data):
    norm_data = np.empty((data.shape[0], data.shape[1]))
    data = np.nan_to_num(data)
    for i, neuron in enumerate(data):
        mean = np.mean(neuron)
        std_dv = np.std(neuron)
        normal_neuron = (neuron - mean)/std_dv
        norm_data[i] = normal_neuron

    return norm_data

##############################################
############### PART 2 #######################
################ SIMULATION Models ###########
##############################################


# 1.1.: Simulation models - clocks + midnight
# 1.2.: Simulation models - MIDNIGHT

# CURRENTLY USED MODEL  
# Fancy matrix multiplication by Jacob inlcuded :)
# generates clocks and midnight model per milisecond
def set_clocks_bytime(walked_path, step_number, step_time, grid_size = 3, phases=3):
    # try one more time, slight adjustments.
    # based on Mohammadys comment: "In the simulation, the agent will progress one phase 
    # for every spatial step and then wait n time steps till all 5 phases have passed"
    # every phase -clock will always be activated between 2 rewards, but after each other.
    # this is possible because I have 100ms time-steps. This means I first have to 
    # multiply the session by the timestep, then divide the timesteps by the 3 phases.
    # then activate the phase-anchored clocks on the respective fields, and deactivate
    # them once the next phase is activated. 
    
    # import pdb; pdb.set_trace()
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1] 
    n_states = len(step_number)
    # and number of rows is locations*phase*neurons per clock
    no_fields = grid_size*grid_size
    n_rows = no_fields*phases  
    cols_to_fill_previous = 0
    whole_path_matrix = np.empty((grid_size*grid_size*phases,0))
    
    for count_paths, (pathlength) in enumerate(step_number):
        cols_to_fill = pathlength*step_time
        # create a string that tells me how many columns are one phase clock
        time_per_phase_in_clock = ([cols_to_fill // phases + (1 if x < cols_to_fill % phases else 0) for x in range (phases)])
        time_per_phase_in_clock_cum = np.cumsum(time_per_phase_in_clock)
        cols_to_fill_previous = cols_to_fill_previous + cols_to_fill

        # part 2:
        # to identify which phase*anchor clocks to activate, identify subpaths
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]

        # Jacobs way:
        # 1. create a phase matrix
                # create a matrix based on the time_per_phase_in_clock list
        phase_matrix_subpath = np.zeros([phases, len(curr_path)*step_time])
        # activate the matrix
        for phase in range(0, len(time_per_phase_in_clock)):
            if phase == 0:
                phase_matrix_subpath[phase, 0:time_per_phase_in_clock_cum[phase]] = 1
            else:
                phase_matrix_subpath[phase, time_per_phase_in_clock_cum[phase-1]:time_per_phase_in_clock_cum[phase]] = 1
        
        # 2. create a field matrix
            # create a matrix based on the fields I am walking subpaths and field identifier
            # use the way to identify the correct fields as for the location thing
            # then use kroneker product 
        steps = np.ones((1,step_time))
        fields_to_activate = np.zeros([grid_size*grid_size, len(curr_path)])
        for currfield in range(0, len(curr_path)):
            fieldnumber = mc.simulation.predictions.field_to_number(curr_path[currfield], grid_size)
            fields_to_activate[fieldnumber, currfield] = 1
        # kroneker product
        field_matrix_subpath = np.kron(fields_to_activate, steps)
        # test this by plotting!! (not needed in normal function so commenting out here)
        # plt.figure(); plt.subplot(1,3,1); plt.imshow(steps); plt.subplot(1,3,2); plt.imshow(fields_to_activate); plt.subplot(1,3,3); plt.imshow(field_matrix_subpath)
     
        # 3. multiply the two matrices
        repeated_fields = np.kron(field_matrix_subpath,np.ones((phases,1)))
        repeated_phases = np.kron(np.ones((grid_size*grid_size,1)), phase_matrix_subpath)
        subpath_matrix = repeated_fields * repeated_phases
        # to plot, but not really needed
        # plt.figure(); plt.subplot(1,5,1); plt.imshow(field_matrix_subpath); plt.title('Fields (4 steps)'); plt.subplot(1,5,2); plt.imshow(phase_matrix_subpath); plt.title('Phases (E,M,L)'); plt.subplot(1,5,3); plt.imshow(repeated_fields); plt.title('Repeated fields (4 steps x 3 phases)'); plt.subplot(1,5,4); plt.imshow(repeated_phases); plt.title('Repeated phases (3 phases x 4 steps)'); plt.subplot(1,5,5); plt.imshow(repeated_fields * repeated_phases); plt.title('Fields * phases result')
            
        # 4. repeat this for every subpath, and stick the subpaths behind each other
        whole_path_matrix = np.concatenate((whole_path_matrix, subpath_matrix), axis = 1)
    
        
    # 5. stick the neuron-clock matrices in 
    # note.
    # The way I code for the clocks now, it can happen that there are 2 clocks of the same phase anchored
    # to different fields. If all neurons of a clock need to be phase-locked, these two phase-clocks
    # of different fields that come after each other in time have to start at the same point in time.
    # this is the slight difference between midnight and clock model now: in the midnight model, no
    # field*phase anchors overlap, they get turned on and turned off consecutively. 
    # The clock model, however, has some clocks overlapping in time (firing 'too early/too long')
    
 
    clock_neurons_per_ms, phase_vector = mc.simulation.predictions.set_single_clock(walked_path, step_number, step_time, grid_size)
    full_clock_matrix_dummy = np.empty([n_rows*phases*n_states,total_steps*step_time]) # fields times phases.
    full_clock_matrix_dummy[:] = np.nan # if field 3x3, 3 phases and 12 neurons per clock > 324 x stepnum (e.g. 7)  
    # for ever 12th row, stick a row of the midnight matrix in (corresponds to the respective first neuron of the clock)
    for row in range(0, len(whole_path_matrix)):
         full_clock_matrix_dummy[row*phases*n_states,:]= whole_path_matrix[row,:]
        
    # copy the neuron per clock firing pattern
    # I will manipulate clocks_per_step, and use clocks_per_step.dummy as control to check for overwritten stuff.
    full_clock_matrix =  full_clock_matrix_dummy.copy()
    
    # now loop through the already filled columns (every 12th one) and fill the clocks if activated.
    cols_to_shift = []
    for row in range(0, len(full_clock_matrix), len(clock_neurons_per_ms)):
        # for every clock that has been activated, identify the first column where it is '1'
        if sum(full_clock_matrix_dummy[row,:]) > 0: 
            column = np.where(full_clock_matrix_dummy[row,:] == 1)[0][0]
            # check in which row this particular column is '1' in the clock
            horizontal_shift_by = np.where(clock_neurons_per_ms[:,column] == 1)[0][0]
            shifted_clock = np.roll(clock_neurons_per_ms, horizontal_shift_by*-1, axis = 0)
            full_clock_matrix[row:(row+(len(clock_neurons_per_ms))), :] = shifted_clock
            # additionally test if there was a double-activation in this row.
            all_activations = np.where(full_clock_matrix_dummy[row,:] == 1)[:][0]
            for index, activated_ms in enumerate(all_activations):
                if index > 0:
                    # if there is a gap (diff between two indices next to each other > 1)
                    if all_activations[index]-all_activations[index-1] > 1:
                        # store the element so I can activate it after
                        cols_to_shift.append(activated_ms)
            if len(cols_to_shift) > 0:
                # if there are double activations, do the same horizontal shift but only copy the 1s.
                for column in cols_to_shift:
                    horizontal_shift_by = np.where(clock_neurons_per_ms[:,column] == 1)[0][0]
                    shifted_clock = np.roll(clock_neurons_per_ms, horizontal_shift_by*-1, axis = 0)
                    for col in range(0, len(shifted_clock[0])):
                        for rw in range(0, len(shifted_clock)):
                            if shifted_clock[rw, col] == 1:
                                full_clock_matrix[row+rw, col] = 1
                cols_to_shift = []
    
    return clock_neurons_per_ms, whole_path_matrix, full_clock_matrix

# for setting clocks
# there will be a matrix of 9*3*12 (field-anchors * phases * neurons) x 12 (3phase*4rewards)
# activate: phase1-field-clock for an early field
# activate: phase2-field-clock for a -1reward field (just before).
# activate: phase3-field-clock for a reward-field.
# advance every activated clock for progressing 'one phase'

# input is: reshaped_visited_fields and all_stepnums from mc.simulation.grid.walk_paths(reward_coords)

def set_clocks(walked_path, step_number, phases = 3, peak_activity = 1, neighbour_activation = 0.5, size_grid = 3):
    # import pdb; pdb.set_trace()
    n_states = len(step_number)
    n_columns = phases*n_states
    # and number of rows is locations*phase*neurons per clock
    # every field (9 fields) -> can be the anchor of 3 phase clocks
    # -> of 12 neurons each. 9 x 3 x 12 
    # to make it easy, the number of neurons = number of one complete loop (12)
    no_fields = size_grid*size_grid
    n_rows = no_fields*phases*(phases*n_states)    
    clocks_matrix = np.empty([n_rows,n_columns]) # fields times phases.
    clocks_matrix[:] = np.nan
    phase_loop = list(range(0,phases))
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1]
    all_phases = list(range(n_columns))  
    # set up neurons of a clock.
    clock_neurons = np.zeros([phases*n_states,phases*n_states])
    for i in range(0,len(clock_neurons)):
        clock_neurons[i,i]= 1   
    # Now the big loop starts. Going through all subpathes, and interpolating
    # phases and steps.                  
    # if pathlength < phases:
    # it can be either pathlength == 1 or == 2. In both cases,
    # so dublicate the field until it matches length phases
    # if pathlength > phases:
    # dublicate the first phase so it matches length of path
    # so that, if finally, pathlength = phases
    # zip both lists and loop through them together.
    for count_paths, (pathlength) in enumerate(step_number):
        phasecount = len(phase_loop) #this needs to be reset for every subpath.
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
        if pathlength < phasecount: 
            finished = False
            while not finished:
                curr_path.insert(0, curr_path[0]) # dublicate first field 
                pathlength = len(curr_path)
                finished = pathlength == phasecount
        elif pathlength > phasecount:
            finished = False
            while not finished:
                phase_loop.insert(0,phase_loop[0]) #make more early phases
                phasecount = len(phase_loop)
                finished = pathlength == phasecount
        if pathlength == phasecount:    
            for phase, step in zip(phase_loop, curr_path):
                x = step[0]
                y = step[1]
                anchor_field = x + y*size_grid
                anchor_phase_start = (anchor_field * n_columns * phases) + (phase * n_columns)
                initiate_at_phase = all_phases[phase+(count_paths*phases)]
                # check if clock has been initiated already
                is_initiated = clocks_matrix[anchor_phase_start, 0]
                if np.isnan(is_initiated): # if not initiated yet
                    # slice the clock neuron filler at the phase we are currently at.
                    first_split = clock_neurons[:, 0:(n_columns-initiate_at_phase)]
                    second_split = clock_neurons[:, n_columns-initiate_at_phase:None]
                    # then add another row of 1s and fill the matrix with it
                    fill_clock_neurons = np.concatenate((second_split, first_split), axis =1)
                    # and only the second part will be filled in.
                    clocks_matrix[anchor_phase_start:(anchor_phase_start+12), 0:None]= fill_clock_neurons
                else:
                    # if the clock has already been initiated and thus is NOT nan
                    # first take the already split clock from before within the whole matrix
                    # then identify the initiation point to fill in the new activation pattern
                    # (this is anchor_phase_start)
                    # now simply change the activity already in the matrix with a neuron-loop
                    for neuron in range(0, n_columns-initiate_at_phase):
                        # now the indices will be slightly more complicated...
                        # [row, column]
                        clocks_matrix[(anchor_phase_start + neuron), (neuron + initiate_at_phase)] = 1
                    if initiate_at_phase > 0:
                        for neuron in range(0, initiate_at_phase):
                            clocks_matrix[(anchor_phase_start + n_columns - initiate_at_phase + neuron), neuron] = 0 
            phase_loop = list(range(0,phases))
                    
    return clocks_matrix, total_steps  

# OLD 
# problem: sometimes, phases were on at the same time. probably a wrong assumption!
# this was based on lazy interpolation  
# Creates a neurons x time matrix. Neurons are gridsize (anchors) x phases (phase-clocks per anchor) x reward*phases (neurons per clock)
# BUT: if there are less steps than phases, then the clocks will be activated at the same time, which is probably wrong.
# input is: reshaped_visited_fields and all_stepnums from mc.simulation.grid.walk_paths(reward_coords)
def set_clocks_bytime_one_neurone(walked_path, step_number, step_time, grid_size = 3, phases=3):
    # CAREFUL! step_time needs to be in 100ms scale. 1.5 secs eg would be 15
    
    # for simplicity, I will just use the same number of neurons as before.
    # this is a bit annoying though because now I can't propagate the signal
    # by 1 neuron per step, because the step lengths will be different.
    # so. I need some sort of interpolation over the clock neurons. At the same
    # time, the interpolation for phases will be obsolete.
    # first, set up the same matrix structure as before.
    # import pdb; pdb.set_trace()
    phase_loop = list(range(0,phases))
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1] 
    n_states = len(step_number)
    n_columns = total_steps    
    # and number of rows is locations*phase*neurons per clock
    no_fields = grid_size*grid_size
    n_rows = no_fields*phases  
    clocks_matrix = np.empty([n_rows,n_columns]) # fields times phases.
    clocks_matrix[:] = np.nan # 324 x stepnum (e.g. 7)  
    clock_neurons_prep = np.zeros([phases*n_states,n_columns])
    # I will use the same logic as with the clocks. The first step is to take
    # each subpath isolated, since the phase-neurons are aligned with the phases (ie. reward)
    # then, I check if the pathlength is the same as the phase length.
    # if not, I will adjust either length, and then use the zip function 
    # to loop through both together and fill the matrix.
    # I will do the same to identify the neuron-level firing pattern, which 
    # needs to be interpolated now!
    for count_paths, (pathlength) in enumerate(step_number):
        phasecount = len(phase_loop) #this needs to be reset for every subpath.
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
        # if pathlength < phases -> 
        # it can be either pathlength == 1 or == 2. In both cases,
        # dublicate the field until it matches length phases
        # if pathlength > phases
        # dublicate the first phase so it matches length of path
        # so that, if finally, pathlength = phases
        # zip both lists and loop through them together.
        if pathlength < phasecount: 
            finished = False
            while not finished:
                curr_path.insert(0, curr_path[0]) # dublicate first field 
                pathlength = len(curr_path)
                finished = pathlength == phasecount
        elif pathlength > phasecount:
            finished = False
            while not finished:
                phase_loop.insert(0,phase_loop[0]) #make more early phases
                phasecount = len(phase_loop)
                finished = pathlength == phasecount
        if pathlength == phasecount:
            for currstep, (phase, step) in enumerate(zip(phase_loop, curr_path)):
                x = step[0]
                y = step[1]
                fieldnumber = x + y*grid_size
                # fieldnumber tells me the current anchor.
                # first fill in a dummy-matrix where each clock only has one neuron. 
                if currstep >= step_number[count_paths]:
                    clocks_matrix[(fieldnumber * phases) + phase ,(cumsumsteps[count_paths]-1)] = 1  
                    clock_neurons_prep[phase+(count_paths*phases),(cumsumsteps[count_paths]-1)] = 1
                elif count_paths > 0: 
                    clocks_matrix[(fieldnumber * phases) + phase ,currstep+cumsumsteps[count_paths-1]] = 1
                    clock_neurons_prep[phase+(count_paths*phases),currstep+cumsumsteps[count_paths-1]] = 1
                elif count_paths == 0:
                    clocks_matrix[(fieldnumber * phases) + phase ,currstep] = 1  
                    clock_neurons_prep[phase+(count_paths*phases),currstep] = 1
                #location_matrix[fieldnumber, ((phases*count_paths)+phase)] = 1 # currstep = phases
            phase_loop = list(range(0,phases)) 
            
    # now that I have the 0 degree neurons activated, as well as the neuron
    # pattern matrix, construct the whole matrix out both
    # stick the matrix in whenever a clock is activated ('1'), and split it at that column.
    # 1., create matrix with the right dimensions.
    clocks_per_step_dummy = np.empty([n_rows*phases*n_states,n_columns]) # fields times phases.
    clocks_per_step_dummy[:] = np.nan # 324 x stepnum (e.g. 7)  
    # for ever 12th row, stick a row of the small matrix in
    for row in range(0, len(clocks_matrix)):
        clocks_per_step_dummy[row*phases*n_states,:]= clocks_matrix[row,:]
        
    # copy the neuron per clock firing pattern
    # I will manipulate clocks_per_step, and use clocks_per_step.dummy as control to check for overwritten stuff.
    clocks_per_step = clocks_per_step_dummy.copy()
    
    # now loop through all columns and rows and input clock-neurons.   
    for column in range(0, len(clocks_per_step[0])):
        for row in range(0, len(clocks_per_step)):
            clock_neurons = clock_neurons_prep.copy()
            # first test if clocks_per step also has a 1 at the current position -> if not, it will be overwritten!
            if (clocks_per_step_dummy[row,column] == 1) and (clocks_per_step[row,column] == 1):
                # stick the neuron activation in.
                # but first slice the neuron matrix correctly
                first_split = clock_neurons[:, 0:(n_columns-column)]
                second_split = clock_neurons[:, (n_columns-column):None]
                fill_clock_neurons = np.concatenate((second_split, first_split), axis =1)
                # DOUBLE CHECK IF THE SLICING WORKS ALRIGHT!!               
                clocks_per_step[row:(row+(len(clock_neurons_prep))), :] = fill_clock_neurons
            elif (clocks_per_step_dummy[row,column] == 1):
                # loop through the clocks neurons and only copy the ones
                first_split = clock_neurons[:, 0:(n_columns-column)]
                second_split = clock_neurons[:, (n_columns-column):None]
                fill_clock_neurons = np.concatenate((second_split, first_split), axis =1)
                # DOUBLE CHECK IF THE SLICING WORKS ALRIGHT!!
                for col in range(0, len(fill_clock_neurons[0])):
                    for rw in range(0, len(fill_clock_neurons)):
                        if fill_clock_neurons[rw, col] == 1:
                            clocks_per_step[row+rw, col] = 1
            
        # at the end: multiply by how ever many seconds a step should take.
    clocks_per_sec = np.repeat(clocks_per_step, repeats= step_time, axis=1)
    return clocks_matrix, clock_neurons_prep, clocks_per_sec


def set_single_clock(walked_path, step_number, step_time, grid_size = 3, phases = 3):
   # import pdb; pdb.set_trace()
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1] 
    n_states = len(step_number)    
    clock_neurons_per_ms = np.zeros([phases*n_states,total_steps*step_time])
    phase_array = np.zeros([total_steps*step_time])
    # clock_neurons_per_ms = np.repeat(clock_neurons_prep, repeats = step_time, axis = 1)
    # now, for every sub-path, divide the timesteps by the number of phases.
    # e.g. 1 step, 1sec > 3 cols, 3cols, 4 cols per phase.
    # e.g. 2 step, 1 sec > 7 cols, 7 cols, 6 cols per phase.
    # define step length:
    cols_to_fill_previous = 0
    # phase_list = [None] * len(clock_neurons_per_ms[0])
    for count_paths, (pathlength) in enumerate(step_number):
        cols_to_fill = pathlength*step_time
        # create a string that tells me how many columns are one phase clock
        time_per_phase_in_clock = ([cols_to_fill // phases + (1 if x < cols_to_fill % phases else 0) for x in range (phases)])
        time_per_phase_in_clock_cum = np.cumsum(time_per_phase_in_clock)
        # use the elements of cols_per_clock to fill the single-clock matrix.
        # add a category-vector for the phases
        for phase in range(0, phases):
            if phase == 0:
                clock_neurons_per_ms[phase+(count_paths*phases), cols_to_fill_previous+ 0: cols_to_fill_previous+ time_per_phase_in_clock_cum[phase]] = 1
                phase_array[cols_to_fill_previous+ 0: cols_to_fill_previous+ time_per_phase_in_clock_cum[phase]] = 10
            elif phase == 1:
                clock_neurons_per_ms[phase+(count_paths*phases), cols_to_fill_previous + time_per_phase_in_clock_cum[phase-1]: cols_to_fill_previous + time_per_phase_in_clock_cum[phase]] = 1
                phase_array[cols_to_fill_previous + time_per_phase_in_clock_cum[phase-1]: cols_to_fill_previous + time_per_phase_in_clock_cum[phase]]= 20
            elif phase == 2:
                clock_neurons_per_ms[phase+(count_paths*phases), cols_to_fill_previous + time_per_phase_in_clock_cum[phase-1]: cols_to_fill_previous + time_per_phase_in_clock_cum[phase]] = 1
                phase_array[cols_to_fill_previous + time_per_phase_in_clock_cum[phase-1]: cols_to_fill_previous + time_per_phase_in_clock_cum[phase]] = 30
        cols_to_fill_previous = cols_to_fill_previous + cols_to_fill
    return clock_neurons_per_ms, phase_array


# 1.3.: Simulation models - PHASE
def set_phase_model(walked_path, step_number, step_time, grid_size = 3, phases=3):
    # import pdb; pdb.set_trace()
    n_rows = phases
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1]   
    n_columns = total_steps   
    loc_matrix = np.empty([n_rows,n_columns]) # fields times steps
    loc_matrix[:] = np.nan
    cols_to_fill_previous = 0
    for count_paths, pathlength in enumerate(step_number):
        cols_to_fill = pathlength*step_time
        # create a string that tells me how many columns are one phase clock
        time_per_phase_in_clock = ([cols_to_fill // phases + (1 if x < cols_to_fill % phases else 0) for x in range (phases)])
        time_per_phase_in_clock_cum = np.cumsum(time_per_phase_in_clock)
        cols_to_fill_previous = cols_to_fill_previous + cols_to_fill
        
        # part 2:
        # to identify which phase*anchor clocks to activate, identify subpaths
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
            
            
        phase_matrix_subpath = np.zeros([phases, len(curr_path)*step_time])
        # activate the matrix
        for phase in range(0, len(time_per_phase_in_clock)):
            if phase == 0:
                phase_matrix_subpath[phase, 0:time_per_phase_in_clock_cum[phase]] = 1
            else:
                phase_matrix_subpath[phase, time_per_phase_in_clock_cum[phase-1]:time_per_phase_in_clock_cum[phase]] = 1
        
        if count_paths == 0:
            phase_model = phase_matrix_subpath.copy()
        elif count_paths > 0:
            phase_model = np.concatenate((phase_model,phase_matrix_subpath), axis = 1)
        
    return phase_model



def zero_phase_clocks_by_time(clocks_per_sec, step_number, grid_size = 3, phases = 3):
    #import pdb; pdb.set_trace()
    n_states = len(step_number)
    neuron_number = n_states*phases
    field_number = grid_size*grid_size
    clock_number = phases*field_number
    n_columns = len(clocks_per_sec[0])
    zero_phase_clocks_matrix = np.zeros([clock_number,n_columns])
    for i in range(clock_number):
        zero_phase_clocks_matrix[i] = clocks_per_sec[i*neuron_number,:]
    return zero_phase_clocks_matrix


   

# 1.4 Simulation: Location models.
#############################
### Location Models #########
#############################

# CURRENT MODEL
def set_location_by_time(walked_path, step_number, step_time, grid_size = 3, field_no_given = None):
    # import pdb; pdb.set_trace()   
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1]    
    n_columns = total_steps   
    n_rows = grid_size*grid_size
    loc_matrix = np.empty([n_rows,n_columns]) # fields times steps
    loc_matrix[:] = np.nan
    for i in range(0, total_steps):
        if field_no_given is None:
            curr_field = walked_path[i+1] # cut the first field because this is the reward field
            x = curr_field[0]
            y = curr_field[1]
            fieldnumber = x + y* grid_size
        else:
            fieldnumber = walked_path[i+1]
        # test if this has already been activated!
        if loc_matrix[fieldnumber, i] == 0:
            # if so, then don't overwrite it.
            loc_matrix[fieldnumber, i] = 1
        else:   
            loc_matrix[fieldnumber, :] = 0
            loc_matrix[fieldnumber, i] = 1
    loc_per_sec = np.repeat(loc_matrix, repeats = step_time, axis=1)    
    return loc_matrix, loc_per_sec


# next, set location matrix.
# this will be a matrix which is 9 (fields) x  12 (phases).
# every field visit will activate the respective field. 
# since there always have to be 3 phases between 2 reward fields, I need to interpolate.
# my current solution for this:
# 1 step = 2 fields → both are early, late and reward (reward old and reward new)
# 2 steps = 3 fields → leave current field as is; 2nd is early and late; 3rd is reward
# 3 steps = 4 fields → leave current field as is, 2nd is early, 3rd is late, fourth is reward
# 4 steps = 5 fields → leave current field as is, 2nd is early, 3rd is early, 4th is late, 5th is reward

# input is: reshaped_visited_fields and all_stepnums from mc.simulation.grid.walk_paths(reward_coords)
# THIS IS AN OLD MODEL!
def set_location_matrix(walked_path, step_number, phases, size_grid = 3):
    # import pdb; pdb.set_trace()
    n_states = len(step_number)
    n_columns = phases*n_states
    no_fields = size_grid*size_grid
    location_matrix = np.zeros([no_fields,n_columns]) # fields times phases.
    phase_loop = list(range(0,phases))
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1] # DOUBLE CHECK IF THIS IS TRUE
    # I will use the same logic as with the clocks. The first step is to take
    # each subpath isolated, since the phase-neurons are aligned with the phases (ie. reward)
    # then, I check if the pathlength is the same as the phase length.
    # if not, I will adjust either length, and then use the zip function 
    # to loop through both together and fill the matrix.
    for count_paths, (pathlength) in enumerate(step_number):
        phasecount = len(phase_loop) #this needs to be reset for every subpath.
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
        # if pathlength < phases -> 
        # it can be either pathlength == 1 or == 2. In both cases,
        # dublicate the field until it matches length phases
        # if pathlength > phases
        # dublicate the first phase so it matches length of path
        # so that, if finally, pathlength = phases
        # zip both lists and loop through them together.
        if pathlength < phasecount: 
            finished = False
            while not finished:
                curr_path.insert(0, curr_path[0]) # dublicate first field 
                pathlength = len(curr_path)
                finished = pathlength == phasecount
        elif pathlength > phasecount:
            finished = False
            while not finished:
                phase_loop.insert(0,phase_loop[0]) #make more early phases
                phasecount = len(phase_loop)
                finished = pathlength == phasecount
        if pathlength == phasecount:
            for phase, step in zip(phase_loop, curr_path):
                x = step[0]
                y = step[1]
                fieldnumber = x + y* size_grid
                location_matrix[fieldnumber, ((phases*count_paths)+phase)] = 1 # currstep = phases
            phase_loop = list(range(0,phases))
    return location_matrix, total_steps  




##############################################
############### PART 3 #######################
####### CONTINUOUS MODELS ###################
##############################################

# 3. continuous models: all - midnight - phase - location



# 3.1 continuous models: all
# fuck it, I can probably do all continous models in one. LESSSE GOOOOO
def set_continous_models(walked_path, step_number, step_time, grid_size = 3, no_phase_neurons=3, fire_radius = 0.25, wrap_around = 1):
    #import pdb; pdb.set_trace()

    # build all possible coord combinations 
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))] 
    # code up the 2d location neurons. this is e.g. a 3x3 grid tiled with multivatiate
    # gaussians that are centred around the grid locations.
    neuron_loc_functions = []
    for coord in all_coords:
        neuron_loc_functions.append(multivariate_normal(coord, cov = fire_radius))
     
        
    # make the phase continuum
    # set the phases such that the mean is between 0 and 1/no_phase_neurons; 1/no_phase_neurons and 2/no_phase_neurons,
    # and 2/no_phase_neurons and 1.
    neuron_phase_functions = []
    #means_at_phase = np.linspace(0, 1, (no_phase_neurons))
    if wrap_around == 0:
        means_at_phase = np.linspace(0, 1, (no_phase_neurons*2)+1)
        means_at_phase = means_at_phase[1::2].copy()
        for div in means_at_phase: 
            neuron_phase_functions.append(norm(loc = div, scale = 1/(no_phase_neurons/2))) 
    
        # # to plot the functions.
        # x = np.linspace(0,1,1000)
        # plt.figure();
        # for neuron in range(0, len(neuron_phase_functions)):
        #     plt.plot(x, neuron_phase_functions[neuron].pdf(x))
        # to plot the functions.

        
    if wrap_around == 1:
        means_at_phase = np.linspace(-np.pi, np.pi, (no_phase_neurons*2)+1)
        means_at_phase = means_at_phase[1::2].copy()
        
        for div in means_at_phase:
            neuron_phase_functions.append(scipy.stats.vonmises(1/(no_phase_neurons/10), loc=div))
            #neuron_phase_functions.append(scipy.stats.vonmises(1/(no_phase_neurons/2), loc=div))
            # careful! this has to be read differently.
        
        # #to plot the functions.
        # plt.figure(); 
        # for f in neuron_phase_functions:
        #     plt.plot(np.linspace(0,1,1000), f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)/np.max(f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)))
                   
    # make the state continuum
    neuron_state_functions = []
    #if wrap_around == 0:
        # actually, there should not be any smoothness in state.
    means_at_state = np.linspace(0,(len(step_number)-1), (len(step_number)))
    for div in means_at_state:
        neuron_state_functions.append(norm(loc = div, scale = 1/len(step_number)))
        
    # x = np.linspace(0,3,1000)
    # plt.figure();
    # for neuron in range(0, len(neuron_state_functions)):
    #     plt.plot(x, neuron_state_functions[neuron].pdf(x))

    cumsumsteps = np.cumsum(step_number)
    # this time, do it per subpath.
    for count_paths, pathlength in enumerate(step_number):
        # first step: divide into subpaths
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
           
        # second step: location model.
        # build the 'timecourse', assuming that one timestep is one value.
        # first make the coords into numbers
        # import pdb; pdb.set_trace()
        fields_path = []
        for elem in curr_path:
            fields_path.append(mc.simulation.predictions.field_to_number(elem, grid_size))
        locs_over_time = np.repeat(fields_path, repeats = step_time)
        # back to list and to coords for the location model 
        coords_over_time = list(locs_over_time)
        for index, elem in enumerate(coords_over_time):
            coords_over_time[index] = all_coords[elem]
        # make the location matrix
        loc_matrix = np.empty([grid_size*grid_size,len(coords_over_time)])
        loc_matrix[:] = np.nan
        # and then simply fill the matrix with the respective functions
        for timepoint, location in enumerate(coords_over_time):
            for row in range(0, grid_size*grid_size):
                loc_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
            
        
        # third step: make phase neurons
        # fit subpaths into 0:1 trajectory
        samplepoints = np.linspace(-np.pi, np.pi, len(locs_over_time)) if wrap_around == 1 else np.linspace(0, 1, len(locs_over_time))
        
        phase_matrix_subpath = np.empty([len(neuron_phase_functions), len(samplepoints)])
        phase_matrix_subpath[:] = np.nan
        # read out the respective phase coding 
        for timepoint, read_out_point in enumerate(samplepoints):
            for row in range(0, len(neuron_phase_functions)):
                phase_matrix_subpath[row, timepoint] = neuron_phase_functions[row].pdf(read_out_point)


        # fourth: create the state matrix.
        state_subpath = np.empty([len(neuron_state_functions), len(locs_over_time)])
        state_subpath[:] = np.nan
        # only consider the 1/no_states*count_paths+1 part of the functions
        # then sample by timepoint
        # ADJUST THIS!!! DON'T SAMPLE PER ROW. SAMPLE PER SECOND AS WELL!!
        for row in range(0, len(neuron_state_functions)):
            state_subpath[row] = neuron_state_functions[row].pdf(count_paths)
        
        # # CAREFUL! ADJUST THAT I ONLY SAMPLE THE SUBPATHS!! THIS DOESNT WORKU YET!
        # # more precisely sampled state matrix
        # state_subpath = np.empty([len(neuron_state_functions), len(samplepoints)])
        # state_subpath[:] = np.nan
        # # only consider the 1/no_states*count_paths+1 part of the functions
        # # then sample by timepoint
        # # ADJUST THIS!!! DON'T SAMPLE PER ROW. SAMPLE PER SECOND AS WELL!!
        # for timepoint, read_out_point in enumerate(samplepoints):
        #     for row in range(0, len(neuron_state_functions)):
        #         state_subpath[row, timepoint]= neuron_state_functions[row].pdf(read_out_point+count_paths)
        # all of this looks back. go back to what I had at the beginning.       
                
        # fifth step: midnight. = make location neurons phase sensitive.
        midnight_model_subpath = np.repeat(loc_matrix, repeats = no_phase_neurons, axis = 0)
        # multiply three rows of the location matrix (1 location)
        # with the phase_matrix_subpath, respectively
        for location in range(0, len(midnight_model_subpath), no_phase_neurons):
            midnight_model_subpath[location:location+no_phase_neurons] = midnight_model_subpath[location:location+no_phase_neurons] * phase_matrix_subpath
        
        # sixth. make the clock model. 
        # solving 2 (see below): make the neurons within the clock.
        
        # phase state neurons.
        phase_state_subpath = np.repeat(state_subpath, repeats = len(phase_matrix_subpath), axis = 0)
        for phase in range(0, len(phase_state_subpath), len(phase_matrix_subpath)):
            phase_state_subpath[phase: phase+len(phase_matrix_subpath)] = phase_matrix_subpath * phase_state_subpath[phase: phase+len(phase_matrix_subpath)]
        
        # last step: put subpaths together and concat into a bigger matrix.
        if count_paths == 0:
            midn_model = midnight_model_subpath.copy()
            phas_model = phase_matrix_subpath.copy()
            loc_model = loc_matrix.copy()
            phas_stat = phase_state_subpath.copy()
            stat_model = state_subpath.copy()
        elif count_paths > 0:
            midn_model = np.concatenate((midn_model,midnight_model_subpath), axis = 1)
            phas_model = np.concatenate((phas_model, phase_matrix_subpath), axis = 1)
            loc_model = np.concatenate((loc_model, loc_matrix), axis = 1)
            phas_stat = np.concatenate((phas_stat, phase_state_subpath), axis = 1)
            stat_model = np.concatenate((stat_model, state_subpath), axis = 1)
                       
    # ok I'll try something.
    # this might not be the best way to solve this
    # anyways, I'l try the same trick that I did before for the clocks model.
    
    # I am going to fuse the midnight and the phas_stat model. Thus they need to be equally 'strong' > normalise!
    norm_midn = (midn_model.copy()-np.min(midn_model))/(np.max(midn_model)-np.min(midn_model))
    norm_phas_stat = (phas_stat.copy()-np.min(phas_stat))/(np.max(phas_stat)-np.min(phas_stat))
    
    # try with normalised model.
      # just with the adjustment that I will make the firing of the clocks less strong if
      # they the midnight neurons fire very little
      # identify the max firing.
      # If i normalise this I don't need that step; it'll always be 1
    #max_firing = np.amax(norm_midn)
      # translate this into a value between 0 and 1 by doing *value* / max_firing
     
      # 5. stick the neuron-clock matrices in  
    full_clock_matrix_dummy = np.zeros([len(norm_midn)*len(norm_phas_stat),len(norm_midn[0])]) # fields times phases.
    # for ever 12th row, stick a row of the midnight matrix in (corresponds to the respective first neuron of the clock)
    for row in range(0, len(norm_midn)):
        full_clock_matrix_dummy[row*len(norm_phas_stat),:]= norm_midn[row,:].copy()
         
      # copy the neuron per clock firing pattern
      # I will manipulate clocks_per_step, and use clocks_per_step.dummy as control to check for overwritten stuff.
    clo_model =  full_clock_matrix_dummy.copy()
     
      # now loop through the already filled columns (every 12th one) and fill the clocks if activated.
    for row in range(0, len(norm_midn)):
        local_maxima = argrelextrema(norm_midn[row,:], np.greater_equal, order = 5, mode = 'wrap')
        # delete if the local maxima are neighbouring
        local_maxima = local_maxima[0].copy()
        for index, maxima in enumerate(local_maxima):
            if maxima == local_maxima[index-1]+1:
                # print(maxima, index)
                local_maxima = np.delete(local_maxima, index)
            
        for activation_neuron in local_maxima:
            # shift the clock around so that the activation neuron comes first
            # DOES THIS HAVE TO BE *-1 or just activation neuron???
            # CHECK!!!
            shifted_clock = np.roll(norm_phas_stat, activation_neuron, axis = 0)
            # adjust the firing strength according to the local maxima
            firing_factor = norm_midn[row, activation_neuron].copy()
            #firing_factor = norm_midn[row,activation_neuron]/ max_firing
            shifted_adjusted_clock = shifted_clock.copy()*firing_factor
            # delete row 0 bc I already have it from the midnight ones
            shifted_adjusted_clock[0] = 0
            # then add the values to the existing clocks.
            # Q: IS THIS WAY OF DEALING WIHT DOUBLE ACTIVATION OK???
            clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :] = clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :].copy() + shifted_adjusted_clock.copy()
    
    return loc_model, phas_model, stat_model, midn_model, clo_model, phas_stat



# 3.2 continuous models: midnight
# make a continuous MIDNIGHT model.      
# idea:
    # I will first compute the location matrix, and make 3 location neurons instead of one, but do this for subpaths separately
    # then I will compute the phase matrix
    # then I will multiply the respective location with the phase neurons (low will be nearly off, high will be on )  
def set_midnight_contin(walked_path, step_number, step_time, grid_size = 3, no_phase_neurons=3, fire_radius = 0.25):
    # import pdb; pdb.set_trace()
    cumsumsteps = np.cumsum(step_number)
    # build all possible coord combinations 
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))] 
    
    # make the phase continuum
    neuron_phase_functions = []
    means_at = np.linspace(0, 1, (no_phase_neurons))
    for div in means_at: 
        neuron_phase_functions.append(norm(loc = div, scale = 1/no_phase_neurons/2)) 

    # code up the 2d location neurons. this is e.g. a 3x3 grid tiled with multivatiate
    # gaussians that are centred around the grid locations.
    neuron_loc_functions = []
    for coord in all_coords:
        neuron_loc_functions.append(multivariate_normal(coord, cov = fire_radius))
        
    
    # this time, do it per subpath.
    for count_paths, pathlength in enumerate(step_number):
        # first step: divide into subpaths
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
        
        # second step: location model.
        # build the 'timecourse', assuming that one timestep is one value.
        # first make the coords into numbers
        fields_path = []
        for elem in curr_path:
            fields_path.append(mc.simulation.predictions.field_to_number(elem, grid_size))
        
        locs_over_time = np.repeat(fields_path, repeats = step_time)
        
        # back to list and to coords for the location model 
        coords_over_time = list(locs_over_time)
        for index, elem in enumerate(coords_over_time):
            coords_over_time[index] = all_coords[elem]
    
        # make the matrix
        loc_matrix = np.empty([grid_size*grid_size,len(coords_over_time)])
        loc_matrix[:] = np.nan
        # and then simply fill the matrix with the respective functions
        for timepoint, location in enumerate(coords_over_time):
            for row in range(0, grid_size*grid_size):
                loc_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
        
        # third step: make phase neurons
        # fit subpaths into 0:1 trajectory
        samplepoints = np.linspace(0, 1, len(locs_over_time))
        
        # fourth step: make a subpath-matrix
        phase_matrix_subpath = np.empty([len(neuron_phase_functions), len(samplepoints)])
        phase_matrix_subpath[:] = np.nan
        
        # fifth step: activate the neurons for the respective timepoint
        # read out the respective phase coding 
        for timepoint, read_out_point in enumerate(samplepoints):
            for row in range(0, len(neuron_phase_functions)):
                phase_matrix_subpath[row, timepoint] = neuron_phase_functions[row].pdf(read_out_point)
        
        # sixth step: make location neurons phase sensitive.
        # since the location neurons are also phase sensitive, and 
        # I am assuming 3 neuron are encoding for phase, make each loc neuron 3
        midnight_model_subpath = np.repeat(loc_matrix, repeats = no_phase_neurons, axis = 0)
        # multiply three rows of the location matrix (1 location)
        # with the phase_matrix_subpath, respectively
        for location in range(0, len(midnight_model_subpath), no_phase_neurons):
            midnight_model_subpath[location:location+no_phase_neurons] = midnight_model_subpath[location:location+no_phase_neurons] * phase_matrix_subpath
        
        # seventh: put subpaths together and concat into a bigger matrix.
        if count_paths == 0:
            midnight_model = midnight_model_subpath.copy()
        elif count_paths > 0:
            midnight_model = np.concatenate((midnight_model,midnight_model_subpath), axis = 1)    
    
    return midnight_model
        



# 3.3 continuous models: phase
# make a continuuous PHASE model.
def set_phase_contin(walked_path, step_number, step_time, grid_size = 3, no_phase_neurons=3):
    # import pdb; pdb.set_trace()
    cumsumsteps = np.cumsum(step_number)
    # total_steps = cumsumsteps[-1] 

    #x = np.linspace(0, 1 ,1000)
    neuron_functions = []
    # Q: do I want to have the peaks at 0 and 1 or not?
    # means_at = np.linspace(0, 1, (no_phase_neurons + 2))
    # for div in means_at[1:-1]: 
    means_at = np.linspace(0, 1, (no_phase_neurons))
    for div in means_at: 
        neuron_functions.append(norm(loc = div, scale = 1/no_phase_neurons/2)) 
    # ok forget about the differently spaced phases rn. this is a bit annoying 
    # with normal distributions > do I maybe prefer x^2 ? 
    # if phases == 3:
    #     # make this only the functions!!!
    #     early = norm(loc = 0.125, scale = 0.05)
    #     mid = 2*norm(loc = 0.5, scale = 0.1)
    #     late = 1.5*norm(loc = 0.75, scale = 0.075)
    #     neuron_functions = [early, mid, late]
    # elif phases != 3:
    #     means_at = np.linspace(0, 1, (phases + 2))
        # for div in means_at[1:-1]:
        #     neuron_functions.append(norm.pdf(x, loc = div, scale = 0.05))
   
    for count_paths, pathlength in enumerate(step_number):
        # first step: divide into subpaths
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
        # second step: make location-timecourse
        # build the 'timecourse', assuming that one timestep is one value.
        # first make the coords into numbers
        fields_path = []
        for elem in curr_path:
            fields_path.append(mc.simulation.predictions.field_to_number(elem, grid_size))
        locs_over_time = np.repeat(fields_path, repeats = step_time)
        # third step: fit subpaths into 0:1 trajectory
        samplepoints = np.linspace(0, 1, len(locs_over_time))
        # fourth step: make a subpath-matrix
        phase_matrix_subpath = np.empty([len(neuron_functions), len(samplepoints)])
        phase_matrix_subpath[:] = np.nan
        # fifth step: activate the neurons for the respective timepoint
        # read out the respective phase coding 
        for timepoint, read_out_point in enumerate(samplepoints):
            for row in range(0, len(neuron_functions)):
                phase_matrix_subpath[row, timepoint] = neuron_functions[row].pdf(read_out_point)
       
        # sixth step: concatenate subpaths to one bigger matrix.
        if count_paths == 0:
            phase_model = phase_matrix_subpath.copy()
        elif count_paths > 0:
            phase_model = np.concatenate((phase_model,phase_matrix_subpath), axis = 1)           
    return phase_model   



# 3.4 continuous models: location

# Make a continous LOCATION model.
def set_location_contin(walked_path, step_time, grid_size = 3, fire_radius = 0.25):
    # import pdb; pdb.set_trace()
    neuron_no = grid_size*grid_size
    # build all possible coord combinations 
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))] 
    
    # code up the 2d location neurons. this is e.g. a 3x3 grid tiled with multivatiate
    # gaussians that are centred around the grid locations.
    neuron_functions = []
    for coord in all_coords:
        neuron_functions.append(multivariate_normal(coord, cov = fire_radius))
    
    # build the 'timecourse', assuming that one timestep is one value.
    # first make the coords into numbers
    fields_path = []
    for elem in walked_path:
        fields_path.append(mc.simulation.predictions.field_to_number(elem, grid_size))
    
    locs_over_time = np.repeat(fields_path, repeats = step_time)
    # back to list and to coords
    coords_over_time = list(locs_over_time)
    for index, elem in enumerate(coords_over_time):
        coords_over_time[index] = all_coords[elem]

    # make the matrix
    loc_matrix = np.empty([grid_size*grid_size,len(coords_over_time)])
    loc_matrix[:] = np.nan
    # and then simply fill the matrix with the respective functions
    for timepoint, location in enumerate(coords_over_time):
        for row in range(0, grid_size*grid_size):
            loc_matrix[row, timepoint] = neuron_functions[row].pdf(location) # location has to be a coord
        # don't hardcode the row. how to do that?
    return loc_matrix
        
    
 
       


# 4. ephys models: continuous - clocks - midnight - phase - location

##############################################
############### PART 4 #######################
############# EPHYS MODELS ###################
##############################################


# simple model for human cells.
# models_per_repeat[f"rep_{repeat}"] = mc.simulation.predictions.set_simple_models_cells(prep_repeat_dict)

def set_simple_models_cells(data_dict):
    # import pdb; pdb.set_trace()
    # simple models are: location model, current reward, next reward, 
    # 2 future reward, 3 future reward,state

    # a few hard-coded things
    grid_size = 3
    fire_radius = 0.25 # spatial overlap for location cells
    
    # first extract from dict
    step_number = data_dict['step_number']
    subpath_timings = data_dict['timings_repeat']
    # note: I might need to add a start time for the human data. 
    # timings need to be start- findA, findB, findC,findD
    make_step = data_dict['index_make_step']
    walked_path = data_dict['trajectory']
    rewards = np.tile(data_dict['reward_locs'], 2)
    
    cumsumsteps = np.cumsum(step_number)
    # build all possible coord combinations 
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))] 
    # translate rewards to coords
    reward_coords = list(rewards)
    for index, elem in enumerate(reward_coords):
        reward_coords[index] = all_coords[elem]
    
    
    # code up the 2d location neurons. this is e.g. a 3x3 grid tiled with multivatiate
    # gaussians that are centred around the grid locations.
    neuron_loc_functions = []
    for coord in all_coords:
        neuron_loc_functions.append(multivariate_normal(coord, cov = fire_radius))
    
    model_dict = {}
    for count_paths, pathlength in enumerate(step_number):
        subpath_dict = {}
        if count_paths == 0:
            prev_end_state = 0
        else:
            prev_end_state = end_at_curr_rew
        
        # first, fine-tune the timings.
        # DEFINITION NEW STATE = once they leave the reward location.
        reward_found_at = subpath_timings[count_paths+1]
        if reward_found_at > len(walked_path)+1 or reward_found_at == len(walked_path):
            reward_found_at = -1
        # consider this as end of a state.
        start_curr_rew, end_at_curr_rew = mc.simulation.predictions.find_start_end_indices(walked_path, reward_found_at)
        # to test if the paths matches the current reward configuration
        if walked_path[start_curr_rew] != rewards[count_paths]:
            import pdb; pdb.set_trace()
        
        # first step: divide into subpaths
        curr_path = walked_path[prev_end_state:end_at_curr_rew] 
        if count_paths == 3:
            curr_path = walked_path[prev_end_state:]

        # second step: location model.
        coords_over_time = list(curr_path)
        for index, elem in enumerate(coords_over_time):
            coords_over_time[index] = all_coords[elem]
     
        # make the location based matrice
        location_based_matrices = ['location', 'curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew']
        for loc_model in location_based_matrices:
            subpath_dict[loc_model] = np.empty([grid_size*grid_size,len(coords_over_time)])
            subpath_dict[loc_model][:] = np.nan
        
        # for location, simply fill the matrix with the respective functions
        for timepoint, location in enumerate(coords_over_time):
            for row in range(0, grid_size*grid_size):
                subpath_dict['location'][row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
                # make the split clocks matrices.
                subpath_dict['curr_rew'][row, timepoint] = neuron_loc_functions[row].pdf(reward_coords[count_paths]) # location has to be a coord
                subpath_dict['next_rew'][row, timepoint] = neuron_loc_functions[row].pdf(reward_coords[count_paths+1]) # location has to be a coord
                subpath_dict['second_next_rew'][row, timepoint] = neuron_loc_functions[row].pdf(reward_coords[count_paths+2]) # location has to be a coord
                subpath_dict['third_next_rew'][row, timepoint] = neuron_loc_functions[row].pdf(reward_coords[count_paths+3]) # location has to be a coord

        # second: create the state matrix.
        subpath_dict['state'] = np.zeros([len(step_number), len(curr_path)])
        subpath_dict['state'][count_paths] = 1
        
        #last, concatenate.
        for model in subpath_dict:
            if count_paths == 0:
                model_dict[model] = subpath_dict[model].copy()
            else:
                model_dict[model] = np.concatenate((model_dict[model], subpath_dict[model]), axis = 1)
    # import pdb; pdb.set_trace()          
    return model_dict
    
    
    
    
    

#4.1 ephys models: continuous - clocks - midnight - phase - location

# ok now I need the same thing but for my ephys stuff.
def set_continous_models_ephys(beh_data_curr_rep_dict,  grid_size = 3, no_phase_neurons=3, fire_radius = 0.25, wrap_around = 1, plot = False, split_clock = False, only_rew = False, only_path = False, use_orig_timings = False):
    # import pdb; pdb.set_trace()
    # first extract from dict
    step_number = beh_data_curr_rep_dict['step_number']
    subpath_timings = beh_data_curr_rep_dict['timings_repeat']
    # note: I might need to add a start time for the human data. 
    # timings need to be start- findA, findB, findC,findD
    #make_step = beh_data_curr_rep_dict['index_make_step']
    walked_path = beh_data_curr_rep_dict['trajectory']
    
    #cumsumsteps = np.cumsum(step_number)
    # build all possible coord combinations 
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))] 
    # code up the 2d location neurons. this is e.g. a 3x3 grid tiled with multivatiate
    # gaussians that are centred around the grid locations.
    neuron_loc_functions = []
    for coord in all_coords:
        neuron_loc_functions.append(multivariate_normal(coord, cov = fire_radius))
    
    # make the phase continuum
    # set the phases such that the mean is between 0 and 1/no_phase_neurons; 1/no_phase_neurons and 2/no_phase_neurons,
    # and 2/no_phase_neurons and 1.
    neuron_phase_functions = []
    #means_at_phase = np.linspace(0, 1, (no_phase_neurons))
    if wrap_around == 0:
        means_at_phase = np.linspace(0, 1, (no_phase_neurons*2)+1)
        means_at_phase = means_at_phase[1::2].copy()
        for div in means_at_phase: 
            neuron_phase_functions.append(norm(loc = div, scale = 1/(no_phase_neurons/2))) 
        # # to plot the functions.
        # x = np.linspace(0,1,1000)
        # plt.figure();
        # for neuron in range(0, len(neuron_phase_functions)):
        #     plt.plot(x, neuron_phase_functions[neuron].pdf(x))
        # to plot the functions.
 
    if wrap_around == 1:
        means_at_phase = np.linspace(-np.pi, np.pi, (no_phase_neurons*2)+1)
        means_at_phase = means_at_phase[1::2].copy()
        for div in means_at_phase:
            neuron_phase_functions.append(scipy.stats.vonmises(1/(no_phase_neurons/5), loc=div))
            #neuron_phase_functions.append(scipy.stats.vonmises(1/(no_phase_neurons/2), loc=div))
        # # to plot the functions.
        # plt.figure(); 
        # for f in neuron_phase_functions:
        #     plt.plot(np.linspace(0,1,1000), f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)/np.max(f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)))
    
    neuron_state_functions = []
    means_at_state = np.linspace(0,(len(step_number)-2), (len(step_number)-1))
    for div in means_at_state:
        neuron_state_functions.append(norm(loc = div, scale = 1/len(step_number)-1))
               
    # # make the state continuum
    # neuron_state_functions = []
    # #if wrap_around == 0:
    #     # actually, there should not be any smoothness in state.
    # means_at_state = np.linspace(0,(len(step_number)-1), (len(step_number)))
    # for div in means_at_state:
    #     neuron_state_functions.append(norm(loc = div, scale = 1/len(step_number)))
    # import pdb; pdb.set_trace()
    # # THIS NEEDS TO BE DIFFERENT!!!
    # # state is not continous. think about this
    # # it needs to be 0 or 1.
    
    # x = np.linspace(0,3,1000)
    # plt.figure();
    # for neuron in range(0, len(neuron_state_functions)):
    #     plt.plot(x, neuron_state_functions[neuron].pdf(x))
    
    # this time, do it per subpath.
    result_model_dict = {}
    for count_paths, pathlength in enumerate(step_number):
        # import pdb; pdb.set_trace()
        if count_paths == 0:
            prev_end_state = 0
        else:
            prev_end_state = end_at_curr_rew
        
        # first, fine-tune the timings.
        # DEFINITION NEW STATE = once they leave the reward location.
        reward_found_at = subpath_timings[count_paths+1]
        if reward_found_at > len(walked_path)+1 or reward_found_at == len(walked_path) or count_paths==3:
            reward_found_at = -1
        # consider this as end of a state.
        start_curr_rew, end_at_curr_rew = mc.simulation.predictions.find_start_end_indices(walked_path, reward_found_at)
        
        if use_orig_timings:
            curr_path = walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]
            # print(f"curr path is {len(curr_path)} long")
        # first step: divide into subpaths
        else:
            curr_path = walked_path[prev_end_state:end_at_curr_rew]
        
        
        # second step: location model.
        coords_over_time = list(curr_path)
        for index, elem in enumerate(coords_over_time):
            coords_over_time[index] = all_coords[elem]
     
        # make the location matrix
        loc_matrix = np.empty([grid_size*grid_size,len(coords_over_time)])
        loc_matrix[:] = np.nan
        # and then simply fill the matrix with the respective functions
        for timepoint, location in enumerate(coords_over_time):
            for row in range(0, grid_size*grid_size):
                loc_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
                
        # third: create the state matrix.
        # state_matrix = np.zeros([len(step_number), len(curr_path)])
        # state_matrix[count_paths] = 1
        helper_state_matrix = np.zeros([len(step_number), len(curr_path)])
        helper_state_matrix[count_paths] = 1
        
        state_matrix = np.zeros([len(step_number), len(curr_path)])
        if count_paths > 0:
            state_matrix[count_paths-1] = 1
        
        # state_matrix = np.empty([len(neuron_state_functions), len(curr_path)])
        # state_matrix[:] = np.nan
        # import pdb; pdb.set_trace()
        # maybe, instead of state neurons, can I just create a matrix of 0 
        # and then fill in the parts with 1s where we are in the state??
        # for row in range(0, len(neuron_state_functions)):
        #     state_matrix[row] = neuron_state_functions[row].pdf(count_paths)
        
        
        # fourth step: make phase neurons
        # fit subpaths into 0:1 trajectory
        samplepoints = np.linspace(-np.pi, np.pi, len(curr_path)) if wrap_around == 1 else np.linspace(0, 1, len(curr_path))
        phase_matrix_subpath = np.empty([len(neuron_phase_functions), len(samplepoints)])
        phase_matrix_subpath[:] = np.nan
        # read out the respective phase coding 
        for timepoint, read_out_point in enumerate(samplepoints):
            for row in range(0, len(neuron_phase_functions)):
                phase_matrix_subpath[row, timepoint] = neuron_phase_functions[row].pdf(read_out_point)

        # fifth step: prepare the clocks model 
        # phase state neurons - these will be used to fill the musicbox with neurons that track progress.
        phase_state_subpath = np.repeat(helper_state_matrix, repeats = len(phase_matrix_subpath), axis = 0)
        for phase in range(0, len(phase_state_subpath), len(phase_matrix_subpath)):
            phase_state_subpath[phase: phase+len(phase_matrix_subpath)] = phase_matrix_subpath * phase_state_subpath[phase: phase+len(phase_matrix_subpath)]
            
        # sixth step: midnight. = make location neurons phase sensitive.
        # Find the index where the subject arrives at rewarded field.
        for i in range(len(curr_path) - 2, -1, -1):  # Start from second to last element and move backwards
            if curr_path[i] != curr_path[-1]:
                break
        # Index where subject arrives at rewarded field.
        index_rew_starts = -(len(curr_path) - i - 1)
        # build in the option to only 'turn on' clusters that are rewarded.
        if only_rew == True:
            reward_mask = np.ones(len(coords_over_time))
            # 0 all non-reward neurons!
            reward_mask[0:index_rew_starts] = 0

            loc_rew_matrix = np.zeros([grid_size*grid_size,len(coords_over_time)])
            # and then simply fill the matrix with the respective functions
            for timepoint, location in enumerate(coords_over_time):
                if reward_mask[timepoint] > 0:
                    for row in range(0, grid_size*grid_size):
                        loc_rew_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
            midnight_model_subpath = np.repeat(loc_rew_matrix, repeats = no_phase_neurons, axis = 0)
        
        # if this filter is on, there will only be 'bumps' for those rings that are not at a reward (at the path)
        elif only_path == True:
            # 0 all non-path neurons!
            path_mask = np.ones(len(coords_over_time))
            path_mask[index_rew_starts:] = 0
            
            loc_path_matrix = np.zeros([grid_size*grid_size,len(coords_over_time)])
            # and then simply fill the matrix with the respective functions
            for timepoint, location in enumerate(coords_over_time):
                if path_mask[timepoint] > 0:
                    for row in range(0, grid_size*grid_size):
                        loc_path_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
            midnight_model_subpath = np.repeat(loc_path_matrix, repeats = no_phase_neurons, axis = 0)
        else:
            # normal midnight model
            midnight_model_subpath = np.repeat(loc_matrix, repeats = no_phase_neurons, axis = 0)
        # multiply three rows of the location matrix (1 location)
        # with the phase_matrix_subpath, respectively
        for location in range(0, len(midnight_model_subpath), no_phase_neurons):
            midnight_model_subpath[location:location+no_phase_neurons] = midnight_model_subpath[location:location+no_phase_neurons] * phase_matrix_subpath
        
        

        # last step: put subpaths together and concat into a bigger matrix.
        if count_paths == 0:
            result_model_dict['midn_model'] = midnight_model_subpath.copy()
            result_model_dict['phas_model'] = phase_matrix_subpath.copy()
            result_model_dict['loc_model'] = loc_matrix.copy()
            result_model_dict['stat_model'] = state_matrix.copy()
            result_model_dict['phas_stat_model'] = phase_state_subpath.copy()
        elif count_paths > 0:
            result_model_dict['midn_model'] = np.concatenate((result_model_dict['midn_model'],midnight_model_subpath), axis = 1)
            result_model_dict['phas_model'] = np.concatenate((result_model_dict['phas_model'], phase_matrix_subpath), axis = 1)
            result_model_dict['loc_model'] = np.concatenate((result_model_dict['loc_model'], loc_matrix), axis = 1)
            result_model_dict['stat_model'] = np.concatenate((result_model_dict['stat_model'], state_matrix), axis = 1)
            result_model_dict['phas_stat_model'] = np.concatenate((result_model_dict['phas_stat_model'], phase_state_subpath), axis = 1)
    
    #  pdb; pdb.set_trace()
    
    if result_model_dict['midn_model'].shape[1] != subpath_timings[-1]:
        #print(f"length matches - lenght is {result_model_dict['midn_model'].shape[1]}"
        print("careful!!! length of simulation and path doesnt match!!")
        print(f"its {result_model_dict['midn_model'].shape[1]} vs {subpath_timings[-1]}")
        # import pdb; pdb.set_trace()
        
    # sixth. make the CLOCK MODEL by filling the midnight model with progress neurons.
    # 6.1 I am going to fuse the midnight and the phas_stat model. Thus they need to be equally 'strong' > normalise!
    norm_midn = (result_model_dict['midn_model'].copy()-np.min(result_model_dict['midn_model']))/(np.max(result_model_dict['midn_model'])-np.min(result_model_dict['midn_model']))
    norm_phas_stat = (result_model_dict['phas_stat_model'].copy()-np.min(result_model_dict['phas_stat_model']))/(np.max(result_model_dict['phas_stat_model'])-np.min(result_model_dict['phas_stat_model']))

    # 6.2 Stick the neuron-clock matrices in 
    full_clock_matrix_dummy = np.zeros([len(norm_midn)*len(norm_phas_stat),len(norm_midn[0])]) # fields times phases.
    # for ever 12th row, stick a row of the midnight matrix in (corresponds to the respective first neuron of the clock)
    for row in range(0, len(norm_midn)):
        full_clock_matrix_dummy[row*len(norm_phas_stat),:]= norm_midn[row,:].copy()
         
      # copy the neuron per clock firing pattern
      # I will manipulate clocks_per_step, and use clocks_per_step.dummy as control to check for overwritten stuff.
    clo_model =  full_clock_matrix_dummy.copy()
    
    if split_clock == True:
        split_clock_model_dict = {}
        split_clock_strings = ['curr_rings_split_clock_model', 'one_fut_rings_split_clock_model', 'two_fut_rings_split_clock_model', 'three_fut_rings_split_clock_model']
        for model in split_clock_strings:
            # length of the future clock model will be 3x midnight: predicting the subpaths, not only the reward.
               split_clock_model_dict[model] = np.zeros([len(norm_midn)*no_phase_neurons,len(norm_midn[0])]) 
               
      # now loop through the already filled columns (every 12th one) and fill the clocks if activated.
    for row in range(0, len(norm_midn)):
        # find the peaks of the highest activations in the midnight neurons
        local_maxima = argrelextrema(norm_midn[row,:], np.greater_equal, order = 5, mode = 'wrap')
        # ignore if the local maxima are neighbouring
        local_maxima = local_maxima[0].copy()
        for index, maxima in enumerate(local_maxima):
            if maxima == local_maxima[index-1]+1:
                local_maxima = np.delete(local_maxima, index)
        
        # for each clock that has been activated       
        for activation_neuron in local_maxima:
            horizontal_shift_by = np.argmax(norm_phas_stat[:,activation_neuron])
            # shift the clock around so that the activation neuron comes first
            shifted_clock = np.roll(norm_phas_stat, horizontal_shift_by*-1, axis = 0)
            # the first row of a cluster of 12 clock neurons is the 'midnight'
            # or activation neuron. this is when an agent NOW visits a location 
            # (in a certain phase.)
            
            # next, adjust the firing strength according to the local maxima
            firing_factor = norm_midn[row, activation_neuron].copy()
            #firing_factor = norm_midn[row,activation_neuron]/ max_firing
            shifted_adjusted_clock = shifted_clock.copy()*firing_factor
            if split_clock == True:
                split_clock_model_dict['curr_rings_split_clock_model'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[0:no_phase_neurons] + split_clock_model_dict['curr_rings_split_clock_model'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_clock_model_dict['one_fut_rings_split_clock_model'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons:no_phase_neurons*2] + split_clock_model_dict['one_fut_rings_split_clock_model'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_clock_model_dict['two_fut_rings_split_clock_model'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons*2:no_phase_neurons*3] + split_clock_model_dict['two_fut_rings_split_clock_model'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_clock_model_dict['three_fut_rings_split_clock_model'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons*3:] + split_clock_model_dict['three_fut_rings_split_clock_model'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                
            # then add the values to the existing clocks, but also replace the first row by 0!!
            shifted_adjusted_clock[0] = np.zeros((len(shifted_adjusted_clock[0])))
            clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :] = clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :].copy() + shifted_adjusted_clock.copy()
    
    result_model_dict['clo_model'] = clo_model.copy()
    if split_clock == True:
        for model in split_clock_strings:
            result_model_dict[model] = split_clock_model_dict[model].copy()
    
    # # to plot the matrices
    # plt.figure()
    # plt.imshow(loc_model, aspect = 'auto', interpolation='none')
    # for subpath in subpath_timings:
    #     plt.axvline(subpath, color='white', ls='dashed')
    if plot == True:
        for model in result_model_dict:
            mc.simulation.predictions.plot_without_legends(result_model_dict[model], titlestring=model, timings_curr_run = subpath_timings)

    return result_model_dict




#4.2 ephys models: clocks & midnight

# EPHYS VALIDATION MODELS

### IF USING RAW EPHYS DATA: TIMEBINS 
# based on the fancy matrix multiplication by Jacob inlcuded :)
# generates clocks and midnight model per milisecond
def set_clocks_raw_ephys(walked_path, subpath_timings, step_indices, step_number, grid_size = 3, phases=3, plotting = False, field_no_given = None, ax = None):
    # try one more time, slight adjustments.
    # based on Mohammadys comment: "In the simulation, the agent will progress one phase 
    # for every spatial step and then wait n time steps till all 5 phases have passed"
    # every phase -clock will always be activated between 2 rewards, but after each other.
    # this is possible because I have 100ms time-steps. This means I first have to 
    # multiply the session by the timestep, then divide the timesteps by the 3 phases.
    # then activate the phase-anchored clocks on the respective fields, and deactivate
    # them once the next phase is activated. 
    
    # import pdb; pdb.set_trace()
    n_states = len(step_number)
    # and number of rows is locations*phase*neurons per clock
    no_fields = grid_size*grid_size
    n_rows = no_fields*phases  
    cols_to_fill_previous = 0
    midnight_matrix = np.empty((grid_size*grid_size*phases,0)) #potentially rename to midnight!
    clock_neurons = np.zeros([phases*n_states,len(walked_path)])
    
    for count_paths, (pathlength) in enumerate(step_number):
        # identify subpaths and create single-clock matrices.
        curr_path = walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]
        cols_to_fill = len(curr_path)
        
        # create a string that tells me how many columns are one phase clock
        # this is where I make the assumption of how long every phase actually takes, if not splittable evenly.
        # i.e. > phase is NOT linked to steps taken, but to TIME
        time_per_phase_in_clock = ([cols_to_fill // phases + (1 if x < cols_to_fill % phases else 0) for x in range (phases)])
        time_per_phase_in_clock_cum = np.cumsum(time_per_phase_in_clock)
        
        # fill the single clock matrix analoguos to the phase-switches.
        # this means tha t every clock will be phase-matched to the whole matrix.
        # note: btw this is pretty much the same as below for the phase-matrix_subpath, just for all 12 rows 
        # instead of only for 3. kind of obsolete
        for phase in range(0, phases):
            if phase == 0:
                clock_neurons[phase+(count_paths*phases), cols_to_fill_previous+ 0: cols_to_fill_previous+ time_per_phase_in_clock_cum[phase]] = 1
            elif phase == 1:
                clock_neurons[phase+(count_paths*phases), cols_to_fill_previous + time_per_phase_in_clock_cum[phase-1]: cols_to_fill_previous + time_per_phase_in_clock_cum[phase]] = 1
            elif phase == 2:
                clock_neurons[phase+(count_paths*phases), cols_to_fill_previous + time_per_phase_in_clock_cum[phase-1]: cols_to_fill_previous + time_per_phase_in_clock_cum[phase]] = 1
                
        cols_to_fill_previous = cols_to_fill_previous + cols_to_fill

        # 1. create a phase matrix
        # create a matrix based on time_per_phase_in_clock 
        phase_matrix_subpath = np.zeros([phases, len(curr_path)])
        # activate the matrix
        for phase in range(0, len(time_per_phase_in_clock)):
            if phase == 0:
                phase_matrix_subpath[phase, 0:time_per_phase_in_clock_cum[phase]] = 1
            else:
                phase_matrix_subpath[phase, time_per_phase_in_clock_cum[phase-1]:time_per_phase_in_clock_cum[phase]] = 1
        
        # 2. create a location  matrix
        # create a matrix based on the fields I am walking subpaths and field identifier
        fields_to_activate = np.zeros([grid_size*grid_size, len(curr_path)])
        for currfield in range(0, len(curr_path)):
            if field_no_given is None:
                fieldnumber = mc.simulation.predictions.field_to_number(curr_path[currfield], grid_size)      
            elif field_no_given is not None:
                fieldnumber = curr_path[currfield]
            fields_to_activate[fieldnumber, currfield] = 1
        
        # 3. kroneker - multiply the two matrices
        fields_x_phases = np.kron(fields_to_activate,np.ones((phases,1)))
        phases_x_fields = np.kron(np.ones((grid_size*grid_size,1)), phase_matrix_subpath)
        subpath_matrix = fields_x_phases * phases_x_fields
        # to plot, but not really needed
        # plt.figure(); plt.subplot(1,5,1); plt.imshow(field_matrix_subpath); plt.title('Fields (4 steps)'); plt.subplot(1,5,2); plt.imshow(phase_matrix_subpath); plt.title('Phases (E,M,L)'); plt.subplot(1,5,3); plt.imshow(repeated_fields); plt.title('Repeated fields (4 steps x 3 phases)'); plt.subplot(1,5,4); plt.imshow(repeated_phases); plt.title('Repeated phases (3 phases x 4 steps)'); plt.subplot(1,5,5); plt.imshow(repeated_fields * repeated_phases); plt.title('Fields * phases result')
          
        # 4. repeat this for every subpath, and stick the subpaths behind each other
        midnight_matrix = np.concatenate((midnight_matrix, subpath_matrix), axis = 1)
    
        
    # 5. stick the neuron-clock matrices in  
    full_clock_matrix_dummy = np.empty([n_rows*phases*n_states,len(walked_path)]) # fields times phases.
    full_clock_matrix_dummy[:] = np.nan # if field 3x3, 3 phases and 12 neurons per clock > 324 x stepnum (e.g. 7)  
    # for ever 12th row, stick a row of the midnight matrix in (corresponds to the respective first neuron of the clock)
    for row in range(0, len(midnight_matrix)):
         full_clock_matrix_dummy[row*phases*n_states,:]= midnight_matrix[row,:]
        
    # copy the neuron per clock firing pattern
    # I will manipulate clocks_per_step, and use clocks_per_step.dummy as control to check for overwritten stuff.
    full_clock_matrix =  full_clock_matrix_dummy.copy()
    
    # now loop through the already filled columns (every 12th one) and fill the clocks if activated.
    cols_to_shift = []
    for row in range(0, len(full_clock_matrix), len(clock_neurons)): # every 12th row in the clock matrix
        # for every clock that has been activated, identify the first column where it is '1'
        if sum(full_clock_matrix_dummy[row,:]) > 0: # only more then 0 if activated
            column = np.where(full_clock_matrix_dummy[row,:] == 1)[0][0]
            # check in which row this particular column is '1' in the clock
            horizontal_shift_by = np.where(clock_neurons[:,column] == 1)[0][0]
            shifted_clock = np.roll(clock_neurons, horizontal_shift_by*-1, axis = 0)
            full_clock_matrix[row:(row+(len(clock_neurons))), :] = shifted_clock
            # additionally test if there was a double-activation in this row.
            all_activations = np.where(full_clock_matrix_dummy[row,:] == 1)[:][0]
            # from here, everything is about double-activations.
            # UNDO THIS LATER!!
            # check how everything looks like without double activations
            for index, activated_ms in enumerate(all_activations):
                if index > 0:
                    # if there is a gap (diff between two indices next to each other > 1)
                    if all_activations[index]-all_activations[index-1] > 1:
                        # store the element so I can activate it after
                        # (this means there was a double activation of the clock)
                        cols_to_shift.append(activated_ms)
            if len(cols_to_shift) > 0: # cols to shift will only be bigger than 0 if there was a double-activation
                # if there are double activations, do the same horizontal shift but only copy the 1s.
                for column in cols_to_shift:
                    horizontal_shift_by = np.where(clock_neurons[:,column] == 1)[0][0] # find the double-activated columns
                    shifted_clock = np.roll(clock_neurons, horizontal_shift_by*-1, axis = 0)
                    for col in range(0, len(shifted_clock[0])):
                        for rw in range(0, len(shifted_clock)):
                            if shifted_clock[rw, col] == 1:
                                full_clock_matrix[row+rw, col] = 1
                cols_to_shift = []
                
                
    # this is just the first row of the clocks matrix - neurons that are forced to 
    # fire for the whole phase.
    alternative_midnight = full_clock_matrix[::12, :].copy()
    
    # this is the reverse - replacing the midnight neurons in the clocks matrix with
    # neurons that are being turned on consequetively, not simultaneously.
    # for ever 12th row, stick a row of the midnight matrix in (corresponds to the respective first neuron of the clock)
    alternative_clock = full_clock_matrix.copy()
    for row in range(0, len(midnight_matrix)):
         alternative_clock[row*phases*n_states,:]= midnight_matrix[row,:]
         
    # now there are gaps in the clocks when the phases aren't completely filled.
    # actually, do it the other way around. first work with the midnight model.
    compromise_midnight = np.zeros((len(midnight_matrix), len(midnight_matrix[0])))
    #for phase_state, phase_state_index in enumerate(clock_neurons):
    
    
    for row_index, neuron in enumerate(compromise_midnight): 
        double_activation = []
        if sum(midnight_matrix[row_index,:])>0:
            start_fire = np.where(midnight_matrix[row_index,:] == 1)[0][0]
            phase_state = np.where(clock_neurons[:, start_fire] == 1)[0][0]
            end_fire = np.where(clock_neurons[phase_state, :] == 1)[0][-1]
            neuron[start_fire:end_fire] = 1
            # now check for double activations.
            all_activations = np.where(midnight_matrix[row_index,:] == 1)[:][0]
            for index, activated_ms in enumerate(all_activations):
                if index >0:
                    if all_activations[index]- all_activations[index-1] >1:
                        # if there was a gap in the activations
                        double_activation.append(activated_ms)
            for start_fire in double_activation:
                phase_state = np.where(clock_neurons[:,  start_fire] == 1)[0][0]
                end_fire = np.where(clock_neurons[phase_state, :] == 1)[0][-1]
                neuron[start_fire:end_fire] = 1
    compromise_clock = full_clock_matrix.copy()
    
    for row in range(0, len(midnight_matrix)):
         compromise_clock[row*phases*n_states,:]= compromise_midnight[row,:]
                  
                
    if plotting == True:
        if ax is None:
            plt.figure()
            ax = plt.axes()   
        fig, axs = plt.subplots(nrows =1, ncols = 2)
        axs[0].imshow(full_clock_matrix, interpolation = 'none', aspect = 'auto')
        axs[1].imshow(midnight_matrix, interpolation = 'none', aspect = 'auto')
        
        # OUTPUT 
        # midnight_matrix = neurons only on while agent on the respective field
        # full_clock_matrix = all neurons are activated for the entire phase
        # alternative_midnight = all neurons are activated for the entire phase
        # alternative_clock = neurons only on while agent on the respective field > massive gaps!
        # compromise_midnight and clock = neuron are on as soon as agent steps on field, until the phase ends.
    return midnight_matrix, full_clock_matrix, alternative_midnight, alternative_clock, compromise_midnight, compromise_clock


# this is based on 360 timebins > 1 state is 90 bins, 1 phase is 30 bins.
# the steps are given in numbers, not coordinates.
# important: fields need to be between 0 and 8!
# again, problem that steps and phases do not overlap. I will model everything 
# assuming phases are the more relevant bit -> clocks may start earlier/later than when just stepping on that field,
# but will always be phase-aligned. 
def set_clocks_ephys(walked_path, reward_fields, grid_size = 3, phases = 3, plotting = False, ax=None):
    # import pdb; pdb.set_trace()
    no_rewards = len(reward_fields)
    no_fields = grid_size*grid_size
    no_neurons = phases*no_rewards
    n_columns = len(walked_path)
    # and number of rows is fields*phase*neurons per clock 
    n_rows = no_fields*phases*no_neurons 
    
    clocks_matrix = np.empty([n_rows,n_columns]) # fields times phases.
    clocks_matrix[:] = np.nan
    
    bins_per_reward = int(n_columns/no_rewards)
    curr_rew_vector = np.repeat([0,1,2,3], repeats = bins_per_reward)
    bins_per_phase = int(bins_per_reward/phases)
    curr_phase_vector = np.repeat([0,1,2], repeats = bins_per_phase)
    curr_phase_vector = np.tile(curr_phase_vector, reps = no_rewards)
    neuron_vector = np.repeat([0,1,2,3,4,5,6,7,8,9,10,11], repeats = bins_per_phase)
    # set up neurons of a clock.
    clock_neurons = np.zeros([no_neurons,n_columns])
    for i in range(0,len(clock_neurons[0])):
        clock_neurons[neuron_vector[i],i]= int(1)
    
    
    for i, field in enumerate(walked_path):
        # now loop!!!
        # exception for first field.
        if i == 0:
            curr_midnight_clock_neuron = field*no_neurons*phases + curr_phase_vector[i]*no_neurons
            clocks_matrix[curr_midnight_clock_neuron : (curr_midnight_clock_neuron+12), 0:None] = clock_neurons
        # assumption: after the first field, I only turn on a new clock if the phase or the field changes. 
        # first: check if the field changed.
        elif field != walked_path[i-1]:
            print(f"mouse went from field {walked_path[i-1]} to field {field}") # delete later
            # now check the phase vector alignment: if index smaller bins_per_phase > align with this phase,
            # if bigger, align with next phase.
            distance_from_phase_locking = i - (i // bins_per_phase)*bins_per_phase
            if distance_from_phase_locking < int(bins_per_phase/2):
                activate_at_col = np.where(neuron_vector == (i // bins_per_phase))[0][0]
            elif distance_from_phase_locking >= int(bins_per_phase/2):
                if ((i // bins_per_phase)+1) < max(neuron_vector):
                    activate_at_col = np.where(neuron_vector == (i // bins_per_phase)+1)[0][0]
                elif ((i // bins_per_phase)+1) == max(neuron_vector):
                    activate_at_col = neuron_vector[0]         
            # next, identify which row will be the first midnight-clock-neuron
            curr_midnight_clock_neuron = field*no_neurons*phases + curr_phase_vector[activate_at_col]*no_neurons
            # slice the clock neuron filler at the step we are currently at.
            first_split = clock_neurons[:, 0:(n_columns-activate_at_col)]
            second_split = clock_neurons[:, n_columns-activate_at_col:None]
            fill_clock_neurons = np.concatenate((second_split, first_split), axis =1)
            # then check if clock has been initiated already
            is_initiated = clocks_matrix[curr_midnight_clock_neuron, 0]
            if np.isnan(is_initiated): # if not initiated yet
                clocks_matrix[curr_midnight_clock_neuron:(curr_midnight_clock_neuron+12), 0:None]= fill_clock_neurons
            else:
                # if the clock has already been initiated and thus is NOT nan
                # only add the 1s
                for row in range(len(fill_clock_neurons)):
                    activate_at_col = np.where(fill_clock_neurons[row] == 1)[0][0]
                    clocks_matrix[curr_midnight_clock_neuron+row, activate_at_col:activate_at_col+bins_per_phase] = 1
        # second: even if the field didn't change, check if the phase changed.
        elif field == walked_path[i-1]:
            if curr_phase_vector[i] != curr_phase_vector[i-1]:
                print(f"phase changed on field {field} from phase {curr_phase_vector[i-1]} to phase {curr_phase_vector[i]}") # delete later
                # this means i is already phase-aligned and I just need to identify the field.
                curr_midnight_clock_neuron = field*no_neurons*phases + curr_phase_vector[i]*no_neurons
                # slice the clock neuron filler at the step we are currently at.
                first_split = clock_neurons[:, 0:(n_columns-i)]
                second_split = clock_neurons[:, n_columns-i:None]
                fill_clock_neurons = np.concatenate((second_split, first_split), axis =1)
                # then check if clock has been initiated already
                is_initiated = clocks_matrix[curr_midnight_clock_neuron, 0]
                if np.isnan(is_initiated): # if not initiated yet
                    clocks_matrix[curr_midnight_clock_neuron:(curr_midnight_clock_neuron+12), 0:None]= fill_clock_neurons
                else:
                    # if the clock has already been initiated and thus is NOT nan
                    # only add the 1s
                    for row in range(len(fill_clock_neurons)):
                        activate_at_col = np.where(fill_clock_neurons[row] == 1)[0][0]
                        clocks_matrix[curr_midnight_clock_neuron+row, activate_at_col:activate_at_col+bins_per_phase] = 1
    
    # use start:stop:step slicing in np
    # midnight matrix is always the anchored neuron of the clocks matrix > every 12th row
    midnight_matrix = clocks_matrix[0::12,:]
    
    if plotting == True:
        fig, axs = plt.subplots(nrows =1, ncols = 2)
        axs[0].imshow(clocks_matrix, interpolation = 'none', aspect = 'auto')
        axs[0].set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
        axs[0].set_xticks([30,60,90,120,150,180,210,240,270,300,330,360])
        axs[1].imshow(midnight_matrix, interpolation = 'none', aspect = 'auto')
        axs[1].set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
        axs[1].set_xticks([30,60,90,120,150,180,210,240,270,300,330,360]) 
        #plt.xlabel('360 timebins, 90 per state')

    return clocks_matrix, midnight_matrix 



#4.3 ephys models: phase

def set_phase_model_ephys(walked_path, subpath_timings, step_indices, step_number, grid_size = 3, phases=3, plotting = False, field_no_given = None, ax = None):
    # import pdb; pdb.set_trace()
    cols_to_fill_previous = 0
    
    for count_paths, (pathlength) in enumerate(step_number):
        # identify subpaths and create single-clock matrices.
        curr_path = walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]
        cols_to_fill = len(curr_path)
        
        # create a string that tells me how many columns are one phase clock
        # this is where I make the assumption of how long every phase actually takes, if not splittable evenly.
        # i.e. > phase is NOT linked to steps taken, but to TIME
        time_per_phase_in_clock = ([cols_to_fill // phases + (1 if x < cols_to_fill % phases else 0) for x in range (phases)])
        time_per_phase_in_clock_cum = np.cumsum(time_per_phase_in_clock)
        
        # fill the single clock matrix analoguos to the phase-switches.
        # this means that every clock will be phase-matched to the whole matrix.
        cols_to_fill_previous = cols_to_fill_previous + cols_to_fill

        # 1. create a phase matrix
        # create a matrix based on time_per_phase_in_clock 
        phase_matrix_subpath = np.zeros([phases, len(curr_path)])
        # activate the matrix
        for phase in range(0, len(time_per_phase_in_clock)):
            if phase == 0:
                phase_matrix_subpath[phase, 0:time_per_phase_in_clock_cum[phase]] = 1
            else:
                phase_matrix_subpath[phase, time_per_phase_in_clock_cum[phase-1]:time_per_phase_in_clock_cum[phase]] = 1
        
        if count_paths == 0:
            phase_model_ephys = phase_matrix_subpath.copy()
        elif count_paths > 0:
            phase_model_ephys = np.concatenate((phase_model_ephys,phase_matrix_subpath), axis = 1)
    # import pdb; pdb.set_trace()   
    return phase_model_ephys
        
        
#4.4 ephys models: location

# ephys validation model
# this is based on 360 timebins > 1 state is 90 bins, 1 phase is 30 bins.
# the steps are given in numbers, not coordinates.


# MODEL EPHYS RAW DATA 
def set_location_raw_ephys(walked_path, step_time, grid_size = 3, plotting = False, field_no_given = None, ax = None):
    # import pdb; pdb.set_trace()  
    n_rows = grid_size*grid_size
    n_columns = len(walked_path)
    loc_matrix = np.empty([n_rows,n_columns]) # fields times steps
    loc_matrix[:] = np.nan
    total_steps = len(walked_path)
    for i in range(0, total_steps):
        if field_no_given is None:
            curr_field = walked_path[i+1] # cut the first field because this is the reward field
            x = curr_field[0]
            y = curr_field[1]
            fieldnumber = x + y* grid_size
        else:
            fieldnumber = walked_path[i]
        # test if this has already been activated!
        if loc_matrix[fieldnumber, i] == 0:
            # if so, then don't overwrite it.
            loc_matrix[fieldnumber, i] = 1
        else:   
            loc_matrix[fieldnumber, :] = 0
            loc_matrix[fieldnumber, i] = 1  
    if plotting == True:
        if ax is None:
            plt.figure()
            ax = plt.axes()   
        plt.imshow(loc_matrix, interpolation = 'none', aspect = 'auto')
        ax.set_yticks([0,1,2,3,4,5,6,7,8])
        ax.set_yticklabels(['field 1', 'field 2','field 3', 'field 4', 'field 5', 'field 6', 'field 7', 'field 8', 'field 9'])
        plt.title('location model per time bin')
        plt.xlabel('25ms per time bin')
        plt.ylabel('field-neurons')
             
    return loc_matrix 


# important: fields need to be between 0 and 8!
def set_location_ephys(walked_path, reward_fields, grid_size = 3, plotting = False, ax=None):
    # import pdb; pdb.set_trace()
    n_rows = grid_size*grid_size
    n_columns = len(walked_path)
    loc_matrix = np.empty([n_rows,n_columns]) # fields times steps
    loc_matrix[:] = np.nan
    walked_path = [int(field_no) for field_no in walked_path]
    for i, field in enumerate(walked_path):
        # test if this has already been activated!
        if loc_matrix[field, i] == 0:
            # if so, then don't overwrite it.
            loc_matrix[field, i] = 1
        else:   
            loc_matrix[field, :] = 0
            loc_matrix[field, i] = 1
    if plotting == True:
        r0x, r0y = reward_fields[0], 1
        r1x, r1y = reward_fields[1], 90
        r2x, r2y = reward_fields[2], 180
        r3x, r3y = reward_fields[3], 270
        r4x, r4y = reward_fields[0], 359
        rewards = [Circle((r0y, r0x), radius = 0.4, color = 'black'),
                   Circle((r1y, r1x), radius = 0.4, color = 'black'),
                   Circle((r2y, r2x), radius = 0.4, color = 'black'),
                   Circle((r3y, r3x), radius = 0.4, color = 'black'),
                   Circle((r4y, r4x), radius = 0.4, color = 'black'),]
        if ax is None:
            plt.figure()
            ax = plt.axes()   
        plt.imshow(loc_matrix, interpolation = 'none', aspect = 'auto')
        ax.set_yticks([1,2,3,4,5,6,7,8,9])
        ax.set_yticklabels(['field 1', 'field 2','field 3', 'field 4', 'field 5', 'field 6', 'field 7', 'field 8', 'field 9'])
        ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
        plt.xticks(rotation = 45)
        ax.set_xticks([30,60,90,120,150,180,210,240,270,300,330,360])
        plt.title('location model per time bin')
        plt.xlabel('360 timebins, 90 per state')
        plt.ylabel('field-neurons')
        for r in rewards:
            ax.add_patch(r)
    return loc_matrix








##############################################
############### PART 4 #######################
############# fMRI MODELS ###################
##############################################
def create_model_RDMs_fmri_simple(walked_path, timings_per_step, step_number, rewards, temporal_resolution = 10):
    # import pdb; pdb.set_trace()
    # this is ONLY for the following settings:
    # rewards only, state, split clock, no phase-tuning, locations

    # a few hard-coded things
    grid_size = 3
    fire_radius = 0.25 # spatial overlap for location cells
    
    
    rewards_twice = rewards * 2
    
    # this has to be 0 not -1 because I want to ignore the first location
    # (this is the previous reward D and shouldnt count again)
    cumsumsteps = np.insert(np.cumsum(step_number),0,0)
    
    # build all possible coord combinations 
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))] 
    # code up the 2d location neurons. this is e.g. a 3x3 grid tiled with multivatiate
    # gaussians that are centred around the grid locations.
    neuron_loc_functions = []
    for coord in all_coords:
        neuron_loc_functions.append(multivariate_normal(coord, cov = fire_radius))
        
    model_dict = {}
    for count_paths, pathlength in enumerate(step_number):
        subpath_dict = {}
        prev_end_state = cumsumsteps[count_paths]+1
        end_at_curr_rew = cumsumsteps[count_paths+1]
        
        # to test if the paths matches the current reward configuration
        if walked_path[end_at_curr_rew] != rewards[count_paths]:
            print('careful! reward does not match last field of walked path')
            import pdb; pdb.set_trace()
        
        # first step: divide into subpaths
        curr_path = walked_path[prev_end_state:end_at_curr_rew+1] 
        if count_paths == 3:
            curr_path = walked_path[prev_end_state:]

        # second step: prepare models based on locations.
        location_based_matrices = ['location', 'curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew']
        for loc_model in location_based_matrices:
            subpath_dict[loc_model] = np.empty([grid_size*grid_size,len(curr_path)])
            subpath_dict[loc_model][:] = np.nan
            
        # for location, simply fill the matrix with the respective functions
        for timepoint, location in enumerate(curr_path):
            for row in range(0, grid_size*grid_size):
                subpath_dict['location'][row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
                # make the split clocks matrices.
                subpath_dict['curr_rew'][row, timepoint] = neuron_loc_functions[row].pdf(rewards_twice[count_paths]) # location has to be a coord
                subpath_dict['next_rew'][row, timepoint] = neuron_loc_functions[row].pdf(rewards_twice[count_paths+1]) # location has to be a coord
                subpath_dict['second_next_rew'][row, timepoint] = neuron_loc_functions[row].pdf(rewards_twice[count_paths+2]) # location has to be a coord
                subpath_dict['third_next_rew'][row, timepoint] = neuron_loc_functions[row].pdf(rewards_twice[count_paths+3]) # location has to be a coord


        # second: create the state matrix.
        subpath_dict['state'] = np.zeros([len(step_number), len(curr_path)])
        subpath_dict['state'][count_paths] = 1
        
        #last, concatenate.
        for model in subpath_dict:
            if count_paths == 0:
                model_dict[model] = subpath_dict[model].copy()
            else:
                model_dict[model] = np.concatenate((model_dict[model], subpath_dict[model]), axis = 1)

    clocks_string = ['curr_rew', 'next_rew','second_next_rew','third_next_rew']
    model_dict['clocks'] = np.zeros((len(model_dict['curr_rew'])*4, model_dict['curr_rew'].shape[1]))
    for row in range(0, len(model_dict['curr_rew'])):
        for i, c in enumerate(clocks_string):
            model_dict['clocks'][4*row+i] = model_dict[c][row]
            
    # build in the temporal resolution i want
    model_dict_res = {}
    for m in model_dict:
        model_dict_res[m] = np.zeros((model_dict[m].shape[0],model_dict[m].shape[1]*temporal_resolution))
        for i, r in enumerate(model_dict[m]):
            model_dict_res[m][i] = np.repeat(r, repeats = temporal_resolution)
            

    # import pdb; pdb.set_trace()
    return model_dict_res


# to create the model RDMs.
def create_model_RDMs_fmri(walked_path, timings_per_step, step_number, grid_size = 3, no_phase_neurons=1, fire_radius = 0.25, wrap_around = 1, temporal_resolution = 10, plot = False, only_rew = False, only_path = False, split_clock = False, imaginary = False, lag_weighting = False):
    # import pdb; pdb.set_trace()
    cumsumsteps = np.cumsum(step_number)
    
    # code up the 2d location neurons. this is e.g. a 3x3 grid tiled with multivatiate
    # gaussians that are centred around the grid locations.
    # first, build all possible coord combinations 
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))] 
    neuron_loc_functions = []
    for coord in all_coords:
        neuron_loc_functions.append(multivariate_normal(coord, cov = fire_radius))
         
    # make the phase continuum
    # set the phases such that the mean is between 0 and 1/no_phase_neurons; 1/no_phase_neurons and 2/no_phase_neurons,
    # and 2/no_phase_neurons and 1.
    # also prepare the progress continuum
    no_task_progress_neurons = 10
    
    neuron_phase_functions = []
    neuron_task_progress_functions = []
    if wrap_around == 0:
        means_at_phase = np.linspace(0, 1, (no_phase_neurons*2)+1)
        means_at_phase = means_at_phase[1::2].copy()
        for div in means_at_phase: 
            neuron_phase_functions.append(norm(loc = div, scale = 1/(no_phase_neurons/2))) 
        
        means_at_prog = np.linspace(0,1,(no_task_progress_neurons*2)+1)
        means_at_prog = means_at_prog[1::2].copy()
        for div in means_at_prog: 
            neuron_task_progress_functions.append(norm(loc = div, scale = 1/((no_task_progress_neurons)*2))) 
        # x = np.linspace(0,1,1000)
        # plt.figure();
        # for neuron in range(0, len(neuron_task_progress_functions)):
        #     plt.plot(x, neuron_task_progress_functions[neuron].pdf(x))       
        # # to plot the functions.
        # x = np.linspace(0,1,1000)
        # plt.figure();
        # for neuron in range(0, len(neuron_phase_functions)):
        #     plt.plot(x, neuron_phase_functions[neuron].pdf(x))
        # to plot the functions.    
    if wrap_around == 1:
        means_at_phase = np.linspace(-np.pi, np.pi, (no_phase_neurons*2)+1)
        means_at_phase = means_at_phase[1::2].copy() 
        for div in means_at_phase:
            neuron_phase_functions.append(scipy.stats.vonmises(1/(no_phase_neurons/10), loc=div))
            
        means_at_prog = np.linspace(-np.pi, np.pi, (no_task_progress_neurons*2)+1)
        means_at_prog = means_at_prog[1::2].copy() 
        for div in means_at_prog:
            neuron_task_progress_functions.append(scipy.stats.vonmises(1/(no_task_progress_neurons/100), loc=div))
        # careful! this has to be read differently due to vonmises
        # to plot the functions.
        # plt.figure(); 
        # for f in neuron_task_progress_functions:
        #     plt.plot(np.linspace(0,1,1000), f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)/np.max(f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)))                
            
            
        
    # make a task progress model    
    samplepoints = np.linspace(-np.pi, np.pi, (temporal_resolution*cumsumsteps[-1])) if wrap_around == 1 else np.linspace(0, 1, len(temporal_resolution*cumsumsteps[-1]))
    task_prog_matrix = np.empty([len(neuron_task_progress_functions), len(samplepoints)])
    task_prog_matrix[:] = np.nan
    # read out the respective phase coding 
    for timepoint, read_out_point in enumerate(samplepoints):
        for row in range(0, len(neuron_task_progress_functions)):
            task_prog_matrix[row, timepoint] = neuron_task_progress_functions[row].pdf(read_out_point)


    # make the state continuum, no smoothness in state.
    neuron_state_functions = []
    means_at_state = np.linspace(0,(len(step_number)-1), (len(step_number)))
    for div in means_at_state:
        neuron_state_functions.append(norm(loc = div, scale = 1/len(step_number)))     
    # x = np.linspace(0,3,1000)
    # plt.figure();
    # for neuron in range(0, len(neuron_state_functions)):
    #     plt.plot(x, neuron_state_functions[neuron].pdf(x))

    # now loop through all subpaths.
    for count_paths, pathlength in enumerate(step_number):
        # first step: divide into subpaths
        step_index = np.insert(cumsumsteps, 0, 0)
        curr_path = walked_path[step_index[count_paths]+1 : step_index[count_paths+1]+1]
        # second ste: prepare in case I want to have a better 'temporal resolution'
        fields_path = []
        for elem in curr_path:
            fields_path.append(mc.simulation.predictions.field_to_number(elem, grid_size))
        locs_over_time = np.repeat(fields_path, repeats = temporal_resolution)

        coords_over_time = list(locs_over_time)
        for index, elem in enumerate(coords_over_time):
            coords_over_time[index] = all_coords[elem]
        
        
        # third step: location model.
        # make the location matrix
        loc_matrix = np.empty([grid_size*grid_size,len(coords_over_time)])
        loc_matrix[:] = np.nan
        # and then simply fill the matrix with the respective functions
        for timepoint, location in enumerate(coords_over_time):
            for row in range(0, grid_size*grid_size):
                loc_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
 
        # fourth: create the state matrix.
        state_matrix = np.empty([len(neuron_state_functions), len(locs_over_time)])
        state_matrix[:] = np.nan
        for row in range(0, len(neuron_state_functions)):
            state_matrix[row] = neuron_state_functions[row].pdf(count_paths)
        
        # fifth: make phase neurons
        # fit subpaths into 0:1 trajectory
        samplepoints = np.linspace(-np.pi, np.pi, len(locs_over_time)) if wrap_around == 1 else np.linspace(0, 1, len(locs_over_time))
        
        phase_matrix_subpath = np.empty([len(neuron_phase_functions), len(samplepoints)])
        phase_matrix_subpath[:] = np.nan
        # read out the respective phase coding 
        for timepoint, read_out_point in enumerate(samplepoints):
            for row in range(0, len(neuron_phase_functions)):
                phase_matrix_subpath[row, timepoint] = neuron_phase_functions[row].pdf(read_out_point)
        
        # phase state neurons - these will be used to fill the musicbox with neurons that track progress.
        phase_state_subpath = np.repeat(state_matrix, repeats = len(phase_matrix_subpath), axis = 0)
        for phase in range(0, len(phase_state_subpath), len(phase_matrix_subpath)):
            phase_state_subpath[phase: phase+len(phase_matrix_subpath)] = phase_matrix_subpath * phase_state_subpath[phase: phase+len(phase_matrix_subpath)]
        
        # The following is to test if there are PURELY rings of the musicbox model that are 
        # activated at reward locations or only at the path locations that aren't reward
        
        # if this filter is on, there will only be 'bumps' for those rings that are at a reward
        if only_rew == True:
            # 0 all non-reward neurons!
            #reward_mask = np.append(np.zeros(len(fields_path)-1),count_paths+1)
            reward_mask = np.append(np.zeros(len(fields_path)-1),1)
            reward_mask = np.repeat(reward_mask, repeats = temporal_resolution)
            
            loc_rew_matrix = np.zeros([grid_size*grid_size,len(coords_over_time)])
            # and then simply fill the matrix with the respective functions
            for timepoint, location in enumerate(coords_over_time):
                if reward_mask[timepoint] > 0:
                    for row in range(0, grid_size*grid_size):
                        loc_rew_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
            midnight_model_subpath = np.repeat(loc_rew_matrix, repeats = no_phase_neurons, axis = 0)
        # if this filter is on, there will only be 'bumps' for those rings that are not at a reward (at the path)
        elif only_path == True:
            # 0 all non-path neurons!
            path_mask = np.append(np.ones(len(fields_path)-1),0)
            path_mask = np.repeat(path_mask, repeats = temporal_resolution)
            
            loc_path_matrix = np.zeros([grid_size*grid_size,len(coords_over_time)])
            # and then simply fill the matrix with the respective functions
            for timepoint, location in enumerate(coords_over_time):
                if path_mask[timepoint] > 0:
                    for row in range(0, grid_size*grid_size):
                        loc_path_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
            midnight_model_subpath = np.repeat(loc_path_matrix, repeats = no_phase_neurons, axis = 0)
        else:
            # fifth step: midnight. = make location neurons phase sensitive.
            midnight_model_subpath = np.repeat(loc_matrix, repeats = no_phase_neurons, axis = 0)
        
        # then multiply three rows of the location matrix (1 location) with the phase_matrix_subpath, respectively
        for location in range(0, len(midnight_model_subpath), no_phase_neurons):
            midnight_model_subpath[location:location+no_phase_neurons] = midnight_model_subpath[location:location+no_phase_neurons] * phase_matrix_subpath
        
        # last step: put subpaths together and concat into a bigger matrix.
        if count_paths == 0:
            midn_model = midnight_model_subpath.copy()
            phas_model = phase_matrix_subpath.copy()
            loc_model = loc_matrix.copy()
            stat_model = state_matrix.copy()
            phas_stat = phase_state_subpath.copy()
        elif count_paths > 0:
            midn_model = np.concatenate((midn_model,midnight_model_subpath), axis = 1)
            phas_model = np.concatenate((phas_model, phase_matrix_subpath), axis = 1)
            loc_model = np.concatenate((loc_model, loc_matrix), axis = 1)
            stat_model = np.concatenate((stat_model, state_matrix), axis = 1)
            phas_stat = np.concatenate((phas_stat, phase_state_subpath), axis = 1)
                            
    # sixth. make the CLOCK MODEL by filling the midnight model with progress neurons.
    # I am going to fuse the midnight and the phas_stat model. Thus they need to be equally 'strong' > normalise!
    norm_midn = (midn_model.copy()-np.min(midn_model))/(np.max(midn_model)-np.min(midn_model))
    norm_phas_stat = (phas_stat.copy()-np.min(phas_stat))/(np.max(phas_stat)-np.min(phas_stat))
    
    # to weight the neurons depending on their task lag.
    if imaginary == True:
        # weight the 12 neurons by e to the power of i*cos(theta), and one for e i*sin(theta) where theta is the angle to 0 degree task lag.
        # there are 12 angles, one per task-lag neuron. 
        theta = np.linspace(0, 2*np.pi, len(norm_phas_stat)) #angles
        weight_compl = np.exp(1j * theta / 2)
        diag_complex_weights = np.diag(weight_compl)
        #norm_phas_stat = np.transpose(np.dot(norm_phas_stat.T, diag_complex_weights))
        
    if lag_weighting == True:
        # weight the 12 neurons by e to the power of i*cos(theta), and one for e i*sin(theta) where theta is the angle to 0 degree task lag.
        # there are 12 angles, one per task-lag neuron. 
        theta = np.linspace(0, 2*np.pi, len(norm_phas_stat)) #angles
        weights_cos = np.cos(theta)[:, np.newaxis]
        weights_sin = np.sin(theta)[:, np.newaxis]
        # alternative, easier option: do 
        # compl_clock_weights_cos = np.tile((weights_cos), (27,1))
        # and in the end clocks_weighted = clo_model * compl_clock_weights_cos
  
    # stick the neuron-clock matrices in 
    full_clock_matrix_dummy = np.zeros([len(norm_midn)*len(norm_phas_stat),len(norm_midn[0])]) # fields times phases.
    # for ever 12th row, stick a row of the midnight matrix in (corresponds to the respective first neuron of the clock)
    for row in range(0, len(norm_midn)):
        full_clock_matrix_dummy[row*len(norm_phas_stat),:]= norm_midn[row,:].copy()
         
    # copy the neuron per clock firing pattern
    # I will manipulate clocks_per_step, and use clocks_per_step.dummy as control to check for overwritten stuff.
    clo_model =  full_clock_matrix_dummy.copy()
    if lag_weighting == True:
        clo_model =  full_clock_matrix_dummy.copy()
        clo_model_sin =  full_clock_matrix_dummy.copy()
    
    # import pdb; pdb.set_trace()
    # if you also want a split clock, then prepare those as well as a dicitonary
    if split_clock == True:  
        split_clock_strings = ['curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock']
        if lag_weighting == True:
            split_clock_strings = ['curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'curr_rings_split_clock_sin', 'one_fut_rings_split_clock_sin', 'two_fut_rings_split_clock_sin', 'three_fut_rings_split_clock_sin']
        split_clock_model_dict = {}
        for model in split_clock_strings:
            # length of the future clock model will be 3x midnight: predicting the subpaths, not only the reward.
            if imaginary == True:
                split_clock_model_dict[model] = np.zeros([len(norm_midn)*no_phase_neurons,len(norm_midn[0])], dtype=np.complex128) 
            else:
               split_clock_model_dict[model] = np.zeros([len(norm_midn)*no_phase_neurons,len(norm_midn[0])]) 

    # now loop through the already filled columns (every 12th one) and fill the clocks if activated.
    for row in range(0, len(norm_midn)):
        local_maxima = argrelextrema(norm_midn[row,:], np.greater_equal, order = 5, mode = 'wrap')
        # delete if the local maxima are neighbouring
        local_maxima = local_maxima[0].copy()
        for index, maxima in enumerate(local_maxima):
            if maxima == local_maxima[index-1]+1:
                # print(maxima, index)
                local_maxima = np.delete(local_maxima, index)        
        for activation_neuron in local_maxima:
            horizontal_shift_by = np.argmax(norm_phas_stat[:,activation_neuron])
            # shift the clock around so that the activation neuron comes first
            shifted_clock = np.roll(norm_phas_stat, horizontal_shift_by*-1, axis = 0)
            # adjust the firing strength according to the local maxima
            firing_factor = norm_midn[row, activation_neuron].copy()
            #if firing_factor > 0.5:
            #    import pdb; pdb.set_trace()
            #firing_factor = norm_midn[row,activation_neuron]/ max_firing
            shifted_adjusted_clock = shifted_clock.copy()*firing_factor
            # lastly, if weighting by task-lag desired, compute dot product of weight matrix
            if imaginary == True:
                shifted_adjusted_clock = np.transpose(np.dot(shifted_adjusted_clock.T, diag_complex_weights))
            if lag_weighting == True:
                shifted_adjusted_clock_sin = shifted_adjusted_clock*weights_sin
                shifted_adjusted_clock = shifted_adjusted_clock*weights_cos
                
            # before 0ing out the first row (= musicboxneuron), first save the specific rows (i.e. now, next future, 2 future, 3 future) in the split clocks model.
            # if firing_factor > 0.7:
            #     import pdb; pdb.set_trace()
            if split_clock == True:
                split_clock_model_dict['curr_rings_split_clock'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[0:no_phase_neurons] + split_clock_model_dict['curr_rings_split_clock'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_clock_model_dict['one_fut_rings_split_clock'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons:no_phase_neurons*2] + split_clock_model_dict['one_fut_rings_split_clock'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_clock_model_dict['two_fut_rings_split_clock'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons*2:no_phase_neurons*3] + split_clock_model_dict['two_fut_rings_split_clock'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_clock_model_dict['three_fut_rings_split_clock'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons*3:] + split_clock_model_dict['three_fut_rings_split_clock'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                if lag_weighting == True:
                    split_clock_model_dict['curr_rings_split_clock_sin'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock_sin[0:no_phase_neurons] + split_clock_model_dict['curr_rings_split_clock_sin'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                    split_clock_model_dict['one_fut_rings_split_clock_sin'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock_sin[no_phase_neurons:no_phase_neurons*2] + split_clock_model_dict['one_fut_rings_split_clock_sin'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                    split_clock_model_dict['two_fut_rings_split_clock_sin'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock_sin[no_phase_neurons*2:no_phase_neurons*3] + split_clock_model_dict['two_fut_rings_split_clock_sin'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                    split_clock_model_dict['three_fut_rings_split_clock_sin'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock_sin[no_phase_neurons*3:] + split_clock_model_dict['three_fut_rings_split_clock_sin'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]

                    
            # then, for the full clock model, add the values to the existing clocks, but also replace the first row by 0!!
            shifted_adjusted_clock[0] = np.zeros((len(shifted_adjusted_clock[0])))
            clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :] = clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :].copy() + shifted_adjusted_clock.copy()
            
            if lag_weighting == True:
                shifted_adjusted_clock_sin[0] = np.zeros((len(shifted_adjusted_clock_sin[0])))
                clo_model_sin[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :] = clo_model_sin[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :].copy() + shifted_adjusted_clock_sin.copy()
               
    # import pdb; pdb.set_trace()
    if plot == True:
        mc.simulation.predictions.plot_without_legends(loc_model, titlestring='Location_model')
        mc.simulation.predictions.plot_without_legends(phas_model, titlestring='Phase Model')
        mc.simulation.predictions.plot_without_legends(stat_model, titlestring='State Model')
        mc.simulation.predictions.plot_without_legends(midn_model, titlestring='Midnight Model')
        mc.simulation.predictions.plot_without_legends(np.real(clo_model), titlestring='Musicbox model')
        mc.simulation.predictions.plot_without_legends(np.real(norm_phas_stat), titlestring='One ring of musicbox')
        mc.simulation.predictions.plot_without_legends(task_prog_matrix, titlestring='Task progress Model')
        if split_clock == True:
            for model in split_clock_model_dict:
                mc.simulation.predictions.plot_without_legends(np.real(split_clock_model_dict[model]), titlestring=model)
    # import pdb; pdb.set_trace()
    # save results as dict
    result_dict = {}
    result_dict['location'] = loc_model
    result_dict['phase'] = phas_model
    result_dict['phase_state'] = phas_stat
    result_dict['state'] = stat_model
    result_dict['task_prog'] = task_prog_matrix
    

    if split_clock == True:
        result_dict.update(split_clock_model_dict)
    if lag_weighting == True:
        result_dict['midnight_only-rew'] = midn_model
        result_dict['clocks_only-rew'] = clo_model
        result_dict['clocks_only-rew_sin'] = clo_model_sin
    elif only_rew == True:
        # name all affected models different so that one notices the different representation
        result_dict['midnight_only-rew'] = midn_model
        result_dict['clocks_only-rew'] = clo_model
    elif only_path == True:
        # name all affected models different so that one notices the different representation
        result_dict['midnight_no-rew'] = midn_model
        result_dict['clocks_no-rew'] = clo_model
    else:
        result_dict['midnight'] = midn_model
        result_dict['clocks'] = clo_model
  
    return result_dict


def create_action_model_RDMs_fmri(keys_pressed, timings, no_key_executions, temporal_resolution, only_rew = True, only_path= False, split_future_actions = False, no_phase_neurons=1):
    # import pdb; pdb.set_trace()
    
    cumsumkeypresses = np.cumsum(no_key_executions)
    
    no_button_neurons = 4
    # code up overlapping button neurons.
    # they will be like states, but slightly more overlapping.
    neuron_button_functions = []
    means_at_button = np.linspace(0,(no_button_neurons-1), no_button_neurons)
    for div in means_at_button:
        neuron_button_functions.append(norm(loc = div, scale = 1/(no_button_neurons/2)))   
    # x = np.linspace(0,3,1000)
    # plt.figure();
    # for neuron in range(0, len(neuron_button_functions)):
    #     plt.plot(x, neuron_button_functions[neuron].pdf(x))
         
    # make the phase continuum
    # set the phases such that the mean is between 0 and 1/no_phase_neurons; 1/no_phase_neurons and 2/no_phase_neurons,
    # and 2/no_phase_neurons and 1.
    neuron_phase_functions = []
    means_at_phase = np.linspace(-np.pi, np.pi, (no_phase_neurons*2)+1)
    means_at_phase = means_at_phase[1::2].copy() 
    for div in means_at_phase:
        neuron_phase_functions.append(scipy.stats.vonmises(1/(no_phase_neurons/10), loc=div))
        
        # careful! this has to be read differently due to vonmises
        # to plot the functions.
        # plt.figure(); 
        # for f in neuron_task_progress_functions:
        #     plt.plot(np.linspace(0,1,1000), f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)/np.max(f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)))                
    
    # make the state continuum, no smoothness in state.
    neuron_state_functions = []
    means_at_state = np.linspace(0,(len(no_key_executions)-1), (len(no_key_executions)))
    for div in means_at_state:
        neuron_state_functions.append(norm(loc = div, scale = 1/len(no_key_executions)))     
    # x = np.linspace(0,3,1000)
    # plt.figure();
    # for neuron in range(0, len(neuron_state_functions)):
    #     plt.plot(x, neuron_state_functions[neuron].pdf(x))
    # now loop through all subpaths.
    for count_paths, pathlength in enumerate(no_key_executions):
        # first step: divide into subpaths
        press_index = np.insert(cumsumkeypresses, 0, 0)
        curr_path = keys_pressed[press_index[count_paths] : press_index[count_paths+1]]
        buttons_over_time = np.repeat(curr_path, repeats = temporal_resolution)

        # second step: buttons model.
        # make the location matrix
        # import pdb; pdb.set_trace()
        button_matrix = np.empty([no_button_neurons,len(buttons_over_time)])
        button_matrix[:] = np.nan
        # and then simply fill the matrix with the respective functions
        # read out the respective button coding 
        for timepoint, curr_button in enumerate(buttons_over_time):
            for button_neuron in range(0, len(neuron_button_functions)):
                button_matrix[button_neuron, timepoint] = neuron_button_functions[button_neuron].pdf(curr_button-1)
 
        # third: create the state matrix (only for the action-musicbox-model)
        state_matrix = np.empty([len(neuron_state_functions), len(buttons_over_time)])
        state_matrix[:] = np.nan
        for row in range(0, len(neuron_state_functions)):
            state_matrix[row] = neuron_state_functions[row].pdf(count_paths)
        
        # fourth: make phase neurons
        # fit subpaths into 0:1 trajectory
        samplepoints = np.linspace(-np.pi, np.pi, len(buttons_over_time))
        
        phase_matrix_subpath = np.empty([len(neuron_phase_functions), len(samplepoints)])
        phase_matrix_subpath[:] = np.nan
        # read out the respective phase coding 
        for timepoint, read_out_point in enumerate(samplepoints):
            for row in range(0, len(neuron_phase_functions)):
                phase_matrix_subpath[row, timepoint] = neuron_phase_functions[row].pdf(read_out_point)
        
        # phase state neurons - these will be used to fill the musicbox with neurons that track progress.
        phase_state_subpath = np.repeat(state_matrix, repeats = len(phase_matrix_subpath), axis = 0)
        for phase in range(0, len(phase_state_subpath), len(phase_matrix_subpath)):
            phase_state_subpath[phase: phase+len(phase_matrix_subpath)] = phase_matrix_subpath * phase_state_subpath[phase: phase+len(phase_matrix_subpath)]
        
        # The following is to test if there are PURELY rings of the musicbox model that are 
        # activated at when going to a reward locations or only for buttons that lead to locations without reward
        
        # if this filter is on, there will only be 'bumps' for those rings that are at a reward
        if only_rew == True:
            # import pdb; pdb.set_trace()
            # only rewarded steps
            reward_mask = np.append(np.zeros(len(curr_path)-1),1)
            reward_mask = np.repeat(reward_mask, repeats = temporal_resolution)
            
            button_rew_matrix = np.zeros([no_button_neurons,len(buttons_over_time)])
            # and then simply fill the matrix with the respective functions
            for timepoint, curr_button in enumerate(buttons_over_time):
                if reward_mask[timepoint] > 0:
                    for button_neuron in range(0, len(neuron_button_functions)):
                        button_rew_matrix[button_neuron, timepoint] = neuron_button_functions[button_neuron].pdf(curr_button-1)
            midnight_model_subpath = np.repeat(button_rew_matrix, repeats = no_phase_neurons, axis = 0)
        
        # if this filter is on, there will only be 'bumps' for those rings that are NOT rewarded (at the path)
        elif only_path == True:
            # 0 all non-path neurons!
            path_mask = np.append(np.ones(len(curr_path)-1),0)
            path_mask = np.repeat(path_mask, repeats = temporal_resolution)
            
            button_path_matrix = np.zeros([no_button_neurons,len(buttons_over_time)])
            # and then simply fill the matrix with the respective functions
            for timepoint, curr_button in enumerate(buttons_over_time):
                if path_mask[timepoint] > 0:
                    for button_neuron in range(0, len(neuron_button_functions)):
                        button_path_matrix[button_neuron, timepoint] = neuron_button_functions[button_neuron].pdf(curr_button-1)
            midnight_model_subpath = np.repeat(button_path_matrix, repeats = no_phase_neurons, axis = 0)
        
        else:
            # fifth step: midnight. = make location neurons phase sensitive.
            midnight_model_subpath = np.repeat(button_matrix, repeats = no_phase_neurons, axis = 0)
       

        # then multiply three rows of the location matrix (1 location) with the phase_matrix_subpath, respectively
        for button in range(0, len(midnight_model_subpath), no_phase_neurons):
            midnight_model_subpath[button:button+no_phase_neurons] = midnight_model_subpath[button:button+no_phase_neurons] * phase_matrix_subpath
        
        # last step: put subpaths together and concat into a bigger matrix.
        if count_paths == 0:
            midn_model = midnight_model_subpath.copy()
            phas_model = phase_matrix_subpath.copy()
            button_model = button_matrix.copy()
            stat_model = state_matrix.copy()
            phas_stat = phase_state_subpath.copy()
        elif count_paths > 0:
            midn_model = np.concatenate((midn_model,midnight_model_subpath), axis = 1)
            phas_model = np.concatenate((phas_model, phase_matrix_subpath), axis = 1)
            button_model = np.concatenate((button_model, button_matrix), axis = 1)
            stat_model = np.concatenate((stat_model, state_matrix), axis = 1)
            phas_stat = np.concatenate((phas_stat, phase_state_subpath), axis = 1)
    
    
    # sixth. make the CLOCK MODEL by filling the midnight model with progress neurons.
    # I am going to fuse the midnight and the phas_stat model. Thus they need to be equally 'strong' > normalise!
    norm_midn = (midn_model.copy()-np.min(midn_model))/(np.max(midn_model)-np.min(midn_model))
    norm_phas_stat = (phas_stat.copy()-np.min(phas_stat))/(np.max(phas_stat)-np.min(phas_stat))
    
    # stick the neuron-clock matrices in 
    full_action_box_dummy = np.zeros([len(norm_midn)*len(norm_phas_stat),len(norm_midn[0])]) # fields times phases.
    # for ever 12th row, stick a row of the midnight matrix in (corresponds to the respective first neuron of the clock)
    for row in range(0, len(norm_midn)):
        full_action_box_dummy[row*len(norm_phas_stat),:]= norm_midn[row,:].copy()
    
    # copy the neuron per clock firing pattern
    # I will manipulate clocks_per_step, and use clocks_per_step.dummy as control to check for overwritten stuff.
    action_box_model =  full_action_box_dummy.copy()
 
    if split_future_actions == True:
        split_actions_strings = ['curr_subpath_buttons', 'one_future_subp_buttons', 'two_future_subp_buttons', 'three_future_subp_buttons']
        split_future_actions_model_dict = {}
        # for each anchor (e.g. buttons, 4) there can be (3) = no_phase_neurons phases within a subpath to start, 
        # and also (3)=no_phase_neurons neurons per subpath 
        # so basically, every 3rd (nth) neuron tells me which anchor will be active in which phase current,
        # next subpath, 2 future or 3 future.
        for model in split_actions_strings:
            split_future_actions_model_dict[model] = np.zeros([len(norm_midn)*no_phase_neurons,len(norm_midn[0])]) 
            
    # now loop through the already filled columns (every 12th one) and fill the clocks if activated.
    for row in range(0, len(norm_midn)):
        local_maxima = argrelextrema(norm_midn[row,:], np.greater_equal, order = 5, mode = 'wrap')
        # delete if the local maxima are neighbouring
        local_maxima = local_maxima[0].copy()
        for index, maxima in enumerate(local_maxima):
            if maxima == local_maxima[index-1]+1:
                # print(maxima, index)
                local_maxima = np.delete(local_maxima, index)        
        for activation_neuron in local_maxima:
            horizontal_shift_by = np.argmax(norm_phas_stat[:,activation_neuron])
            # shift the clock around so that the activation neuron comes first
            shifted_clock = np.roll(norm_phas_stat, horizontal_shift_by*-1, axis = 0)
            # adjust the firing strength according to the local maxima
            firing_factor = norm_midn[row, activation_neuron].copy()
            #if firing_factor > 0.5:
                #import pdb; pdb.set_trace()
            shifted_adjusted_clock = shifted_clock.copy()*firing_factor
            
            if split_future_actions == True:
                # for each anchor (e.g. buttons, 4) there can be (3) = no_phase_neurons phases within a subpath to start, 
                # and also (3)=no_phase_neurons neurons per subpath 
                # so basically, every 3rd (nth) neuron tells me which anchor will be active in which phase current,
                # next subpath, 2 future or 3 future.
                split_future_actions_model_dict['curr_subpath_buttons'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[0:no_phase_neurons] + split_future_actions_model_dict['curr_subpath_buttons'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_future_actions_model_dict['one_future_subp_buttons'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons:no_phase_neurons*2] + split_future_actions_model_dict['one_future_subp_buttons'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_future_actions_model_dict['two_future_subp_buttons'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons*2:no_phase_neurons*3] + split_future_actions_model_dict['two_future_subp_buttons'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                split_future_actions_model_dict['three_future_subp_buttons'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :] = shifted_adjusted_clock[no_phase_neurons*3:] + split_future_actions_model_dict['three_future_subp_buttons'][row*no_phase_neurons:row*no_phase_neurons+no_phase_neurons, :]
                
            # then, for the full clock model, add the values to the existing clocks, but also replace the first row by 0!!
            shifted_adjusted_clock[0] = np.zeros((len(shifted_adjusted_clock[0])))
            action_box_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :] = action_box_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :].copy() + shifted_adjusted_clock.copy()
    
    # import pdb; pdb.set_trace()
    # save results as dict
    result_dict = {}
    result_dict['buttons'] = button_model
    result_dict['phase'] = phas_model
    result_dict['phase_state'] = phas_stat
    result_dict['state'] = stat_model
        
    if only_rew == True:
        # name all affected models different so that one notices the different representation
        result_dict['buttonsXphase_only-rew'] = midn_model
        result_dict['action-box_only-rew'] = action_box_model
    elif only_path == True:
        # name all affected models different so that one notices the different representation
        result_dict['buttonsXphase_no-rew'] = midn_model
        result_dict['action-box_no-rew'] = action_box_model
    else:
        result_dict['buttonsXphase'] = midn_model
        result_dict['action-box'] = action_box_model
        
    if split_future_actions == True:
        result_dict.update(split_future_actions_model_dict)
        
    return result_dict

    
    


def create_mask_same_tasks(curr_RDM, reward_per_task_per_taskhalf_dict, excluding):
    # import pdb; pdb.set_trace() 
    no_rewards = 4-excluding
    rewards_experiment = mc.analyse.extract_and_clean.flatten_nested_dict(reward_per_task_per_taskhalf_dict)
    ordered_config_names = {half: [] for half in reward_per_task_per_taskhalf_dict}  

    no_duplicates_list = []    
    for i, task_reference in enumerate(sorted(rewards_experiment.keys())):
        if task_reference not in no_duplicates_list:
            for task_comp in rewards_experiment:
                if task_comp not in no_duplicates_list:
                    if not task_reference == task_comp:
                        if rewards_experiment[task_reference] == rewards_experiment[task_comp]:
                            ordered_config_names['1'].append(task_reference)
                            ordered_config_names['2'].append(task_comp)
                            no_duplicates_list.append(task_reference)
                            no_duplicates_list.append(task_comp)
    
    ordered_configs_concat = np.concatenate((ordered_config_names['1'], ordered_config_names['2']))
                      
    mask_RDM = np.ones((len(curr_RDM), len(curr_RDM)))
    len_task = len(curr_RDM)/ len(no_duplicates_list)
    for ix, x in enumerate(ordered_configs_concat):
        for iy, y in enumerate(ordered_configs_concat):
            if x.startswith(y.split("2",1)[0]):
                mask_RDM[ix*no_rewards:ix*no_rewards + no_rewards, iy*no_rewards:iy*no_rewards + no_rewards] = 0
            if x.startswith(y.split("1",1)[0]):
                mask_RDM[ix*no_rewards:ix*no_rewards + no_rewards, iy*no_rewards:iy*no_rewards + no_rewards] = 0
            if y.startswith(x.split("2",1)[0]):
                mask_RDM[ix*no_rewards:ix*no_rewards + no_rewards, iy*no_rewards:iy*no_rewards + no_rewards] = 0
            if y.startswith(x.split("1",1)[0]):
                mask_RDM[ix*no_rewards:ix*no_rewards + no_rewards, iy*no_rewards:iy*no_rewards + no_rewards] = 0
                
    return mask_RDM



def create_run_count_model_fmri(number_of_steps, number_of_runs, norm_number_of_runs=5, wrap_around = 1, temporal_resolution = 10, plot = False): 
    # define where the mean of these function shall be
    # if number_of_runs < 5:
    #     import pdb; pdb.set_trace()
    steps_per_run = [(np.sum(run) * temporal_resolution) for run in number_of_steps]
    cumsumsteps = np.cumsum(steps_per_run)
    cumsumsteps = np.append(0, cumsumsteps)
    half_steps = [int(steps/2) for steps in steps_per_run]
    means_at = []
    for i, elem in enumerate(half_steps):
        means_at.append(elem + cumsumsteps[i])
    
    neuron_run_count = []   
    if wrap_around == 0:
        means_at_norm = [(elem/(cumsumsteps[-1])) for elem in means_at]
        for div in means_at: 
            neuron_run_count.append(norm(loc = div, scale = 1/(norm_number_of_runs/2))) 
        # x = np.linspace(0,1,1000)
        # plt.figure();
        # for neuron in range(0, len(neuron_phase_functions)):
        #     plt.plot(x, neuron_phase_functions[neuron].pdf(x))
        # to plot the functions.  
    if wrap_around == 1:
        # then normalise it to be between between -np.pi and np.pi
        means_at_pi = [((2*np.pi*(elem/(cumsumsteps[-1])))-np.pi) for elem in means_at]
        for div in means_at_pi:
            # neuron_run_count.append(scipy.stats.vonmises(1, loc=div))  
            neuron_run_count.append(scipy.stats.vonmises(1/(norm_number_of_runs/10), loc=div))  
        # plt.figure(); 
        # for f in neuron_run_count:
        #     plt.plot(np.linspace(0,1,1000), f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)/np.max(f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)))                
    
    all_steps_all_runs = sum(steps_per_run)
    run_count_model = np.empty((norm_number_of_runs,all_steps_all_runs))
    samplepoints = np.linspace(-np.pi, np.pi, run_count_model.shape[1]) if wrap_around == 1 else np.linspace(0, 1, run_count_model.shape[1])
    # read out the respective phase coding 
    for timepoint, read_out_point in enumerate(samplepoints):
        for row in range(0, len(neuron_run_count)):
            run_count_model[row, timepoint] = neuron_run_count[row].pdf(read_out_point)
            
    if plot == True:
        fig, ax = plt.subplots(figsize=(5,4))
        plt.imshow(run_count_model, aspect='auto', interpolation = 'none')
        # to plot where a new run starts
        for interval in cumsumsteps:
            ax.axvline(interval, color='white', linewidth=1)
        
    return run_count_model    





# Now. I want to make a model which:
# only encodes the current reward locations (midnight for rewards)
# encodes the current and future reward locations (clocks for rewards/ state x rewards)
# this solves in in one of 2 different ways:
# midnight x reward = like location model which is only active if there is a reward (so at the last step of a task)
# clocks x reward would also have neurons in the middle which would propagate the activity with a task-lag.
# ie. 4 neurons x 9 locations, like a state x location matrix.
def create_reward_model_RDMs_fmri(walked_path, timings_per_step, step_number, grid_size = 3, no_phase_neurons=3, fire_radius = 0.25, wrap_around = 1, temporal_resolution = 10, plot = False):
    
    cumsumsteps = np.cumsum(step_number)
    # code up the 2d location neurons. this is e.g. a 3x3 grid tiled with multivatiate
    # gaussians that are centred around the grid locations.
    # first, build all possible coord combinations 
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))] 
    neuron_loc_functions = []
    for coord in all_coords:
        neuron_loc_functions.append(multivariate_normal(coord, cov = fire_radius))
        
    # make the phase continuum
    # set the phases such that the mean is between 0 and 1/no_phase_neurons; 1/no_phase_neurons and 2/no_phase_neurons,
    # and 2/no_phase_neurons and 1.
    # also prepare the progress continuum
    no_task_progress_neurons = 10
    neuron_task_progress_functions = []
    if wrap_around == 0:
        means_at_prog = np.linspace(0,1,(no_task_progress_neurons*2)+1)
        means_at_prog = means_at_prog[1::2].copy()
        for div in means_at_prog: 
            neuron_task_progress_functions.append(norm(loc = div, scale = 1/((no_task_progress_neurons)*2))) 
        # x = np.linspace(0,1,1000)
        # plt.figure();
        # for neuron in range(0, len(neuron_task_progress_functions)):
        #     plt.plot(x, neuron_task_progress_functions[neuron].pdf(x))       
        # # to plot the functions.
        
    if wrap_around == 1:
        means_at_prog = np.linspace(-np.pi, np.pi, (no_task_progress_neurons*2)+1)
        means_at_prog = means_at_prog[1::2].copy() 
        for div in means_at_prog:
            neuron_task_progress_functions.append(scipy.stats.vonmises(1/(no_task_progress_neurons/100), loc=div))
        # careful! this has to be read differently due to vonmises
        # to plot the functions.
        # plt.figure(); 
        # for f in neuron_task_progress_functions:
        #     plt.plot(np.linspace(0,1,1000), f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)/np.max(f.pdf(np.linspace(0,1,1000)*2*np.pi - np.pi)))                
    
    # make a task progress model    
    samplepoints = np.linspace(-np.pi, np.pi, (temporal_resolution*cumsumsteps[-1])) if wrap_around == 1 else np.linspace(0, 1, len(temporal_resolution*cumsumsteps[-1]))
    task_prog_matrix = np.empty([len(neuron_task_progress_functions), len(samplepoints)])
    task_prog_matrix[:] = np.nan
    # read out the respective phase coding 
    for timepoint, read_out_point in enumerate(samplepoints):
        for row in range(0, len(neuron_task_progress_functions)):
            task_prog_matrix[row, timepoint] = neuron_task_progress_functions[row].pdf(read_out_point)


    # make the state continuum, no smoothness in state.
    neuron_state_functions = []
    means_at_state = np.linspace(0,(len(step_number)-1), (len(step_number)))
    for div in means_at_state:
        neuron_state_functions.append(norm(loc = div, scale = 1/len(step_number)))     
    # x = np.linspace(0,3,1000)
    # plt.figure();
    # for neuron in range(0, len(neuron_state_functions)):
    #     plt.plot(x, neuron_state_functions[neuron].pdf(x))

    # now loop through all subpaths.
    for count_paths, pathlength in enumerate(step_number):
        # first step: divide into subpaths
        step_index = np.insert(cumsumsteps, 0, 0)
        curr_path = walked_path[step_index[count_paths]+1 : step_index[count_paths+1]+1]
        # second step: prepare in case I want to have a better 'temporal resolution'
        fields_path = []
        for elem in curr_path:
            fields_path.append(mc.simulation.predictions.field_to_number(elem, grid_size))
        locs_over_time = np.repeat(fields_path, repeats = temporal_resolution)
        
        reward_mask = np.append(np.zeros(len(fields_path)-1),count_paths+1)
        reward_mask = np.repeat(reward_mask, repeats = temporal_resolution)

        coords_over_time = list(locs_over_time)
        for index, elem in enumerate(coords_over_time):
            coords_over_time[index] = all_coords[elem]
        
        # third step: from the previous location model, create the midnight x reward model.
        # this basically only means: activate location if there is a reward here.
        reward_midn_matrix = np.zeros([grid_size*grid_size,len(coords_over_time)])
        #reward_midn_matrix[:] = np.nan
        # and then simply fill the matrix with the respective functions
        for timepoint, location in enumerate(coords_over_time):
            if reward_mask[timepoint] > 0:
                for row in range(0, grid_size*grid_size):
                    reward_midn_matrix[row, timepoint] = neuron_loc_functions[row].pdf(location) # location has to be a coord
                   

        # fourth: create the state matrix.
        state_matrix = np.empty([len(neuron_state_functions), len(locs_over_time)])
        state_matrix[:] = np.nan
        for row in range(0, len(neuron_state_functions)):
            state_matrix[row] = neuron_state_functions[row].pdf(count_paths)


        # last step: put subpaths together and concat into a bigger matrix.
        if count_paths == 0:
            rew_midn_model = reward_midn_matrix.copy()
            stat_model = state_matrix.copy()
            reward_mask_task = reward_mask.copy()
        elif count_paths > 0:
            rew_midn_model = np.concatenate((rew_midn_model,reward_midn_matrix), axis = 1)
            stat_model = np.concatenate((stat_model, state_matrix), axis = 1)
            reward_mask_task = np.append(reward_mask_task, reward_mask)
       
    
    # for the clocks x reward model, fill in the task-lag neurons that are anchored at that location
    # first create an empty one
    reward_clo_model_empty = np.zeros((len(rew_midn_model) * len(state_matrix), rew_midn_model.shape[1]))
    for row in range(0, len(reward_clo_model_empty), len(state_matrix)):
        # import pdb; pdb.set_trace()
        if row == 0:
            reward_clo_model_empty[row] = rew_midn_model[row]
        else:
            reward_clo_model_empty[row] = rew_midn_model[int(row/len(stat_model))]
    
    reward_clo_model = reward_clo_model_empty.copy()
    # then fill the task-lag neurons.
    for step in range(reward_clo_model_empty.shape[1]):
        # import pdb; pdb.set_trace()
        for row, neuron_activity in enumerate(reward_clo_model_empty[:,step]):
            if neuron_activity > 0:
                # find out which reward has been activated and fill the respective 4 rows accordingly.
                if reward_mask_task[step] == 1 and reward_mask_task[step-1] == 0:
                    reward_clo_model[row+1, reward_mask_task == 2] += neuron_activity
                    reward_clo_model[row+2, reward_mask_task == 3] += neuron_activity
                    reward_clo_model[row+3, reward_mask_task == 4] += neuron_activity
                if reward_mask_task[step] == 2 and reward_mask_task[step-1] == 0:
                    reward_clo_model[row+1, reward_mask_task == 3] += neuron_activity
                    reward_clo_model[row+2, reward_mask_task == 4] += neuron_activity
                    reward_clo_model[row+3, reward_mask_task == 1] += neuron_activity
                if reward_mask_task[step] == 3 and reward_mask_task[step-1] == 0:
                    reward_clo_model[row+1, reward_mask_task == 4] += neuron_activity
                    reward_clo_model[row+2, reward_mask_task == 1] += neuron_activity
                    reward_clo_model[row+3, reward_mask_task == 2] += neuron_activity
                if reward_mask_task[step] == 4 and reward_mask_task[step-1] == 0:
                    reward_clo_model[row+1, reward_mask_task == 1] += neuron_activity
                    reward_clo_model[row+2, reward_mask_task == 2] += neuron_activity
                    reward_clo_model[row+3, reward_mask_task == 3] += neuron_activity


    if plot == True:
        mc.simulation.predictions.plot_without_legends(stat_model, titlestring='State Model')
        mc.simulation.predictions.plot_without_legends(rew_midn_model, titlestring='Midnight Model, only rewards')
        mc.simulation.predictions.plot_without_legends(task_prog_matrix, titlestring='Task progress Model')
        # and the reward x musicbox model
        fig, ax = plt.subplots(figsize=(5,4))
        plt.imshow(reward_clo_model, aspect = 'auto')
        for new_anchor in range(0, len(reward_clo_model_empty), len(state_matrix)):
            ax.axhline(new_anchor-0.5, color='white', linewidth=1)
        
    #return rew_midn_model, reward_clocks_model, stat_model, task_progress_model
    # make everything a dicitonary
    result_dict = {}
    result_dict['reward_midnight'] = rew_midn_model
    result_dict['reward_clocks'] = reward_clo_model
    result_dict['state'] = stat_model
    result_dict['task_prog'] = task_prog_matrix
    
     #['clocks', 'midnight', 'location', 'phase', 'state', 'task_prog']   
    return result_dict

def create_counting_reward_models_fmri(dict_config_by_reward):
    # import pdb; pdb.set_trace()
    from collections import Counter
    # the logic here will be that I create a temporary concatenation of all configs,
    # but in the end give out an array per config so I can sort it afterwards.
    
    # create a reward list as tuples so I can count
    all_rewards_list = [tuple(item) for sublist in dict_config_by_reward.values() for item in sublist]   
    # create counter for each reward coordinate
    rew_frequencies = Counter(all_rewards_list)
   
    # new dicitonary where each config has assigned their frequency per reward
    midnight_dict = {config: np.array([rew_frequencies[tuple(single_rew)] for single_rew in rewards]) for config, rewards in dict_config_by_reward.items()}
    
    # now I need to have another one for the clocks.
    # the only issue I am having with this rn is that I am unsure of the within-between task activations.
    # the midnight is between; while the clocks shifted activity is within.
    # Convert frequency lists to 4-row matrices with shifts
    clocks_dict = {}
    for config, freq_list in midnight_dict.items():
        # Convert the list to a NumPy array for manipulation
        array = np.array(freq_list)
        # Create the matrix with shifted rows
        clocks_frequencies = np.vstack([np.roll(array, -shift) for shift in range(4)])
        # Store the matrix in the new dictionary
        clocks_dict[config] = clocks_frequencies

    result_dict = {}
    result_dict['reward_midnight_count'] = midnight_dict
    result_dict['reward_clocks_count'] = clocks_dict
    
    return result_dict


def create_instruction_model(rewards_of_task, trial_type, grid_size = 3, fire_radius = 0.25):
    #import pdb; pdb.set_trace()

    reward_fields = []
    for elem in rewards_of_task:
        reward_fields.append(mc.simulation.predictions.field_to_number(elem, grid_size))
    
    if trial_type.endswith('backw'):
        reward_fields.reverse()
    # here, I have 4 positions per location: ABC or D; and fields 1-9.
    # if backwards, its a -1, forwards they are 1s  
    rew_loc_matrix = np.zeros([grid_size*grid_size*len(rewards_of_task),1])
    for position, field in enumerate(reward_fields):
        if trial_type.endswith('forw'):
            rew_loc_matrix[position*grid_size*grid_size + field] = 1
        elif trial_type.endswith('backw'):
            rew_loc_matrix[position*grid_size*grid_size + field] = -1
   
    result_dict = {'instruction': rew_loc_matrix}

    
    return result_dict


################################
####### PART 4: cells ############
################################
#### regressors for human cells ####

def state_cells(empty_reg, grid_t_all, reward_locs):
    # import pdb; pdb.set_trace()
    state_regressors = empty_reg.copy()
    grid_t_all = grid_t_all-grid_t_all[0,0]
    for repeat_idx, grid_times in enumerate(grid_t_all):
        if np.isnan(grid_times).any():
            continue
        else:

            # state_regressors[0, int(grid_times[1]):int(grid_times[2])] = 1
            # state_regressors[1, int(grid_times[2]):int(grid_times[3])] = 1
            # state_regressors[2, int(grid_times[3]):int(grid_times[4])] = 1
            state_regressors[0, int(grid_times[0]):int(grid_times[1])] = 1
            state_regressors[1, int(grid_times[1]):int(grid_times[2])] = 1
            state_regressors[2, int(grid_times[2]):int(grid_times[3])] = 1
            if repeat_idx == len(grid_t_all)-1:
                state_regressors[3, int(grid_times[3]):int(grid_times[4]+1)] = 1
            else:  
                state_regressors[3, int(grid_times[3]):int(grid_times[4])] = 1
            
    return state_regressors


def test_timings_rew(subject, locations, grid_t_all, reward_locs, number_of_grid):  
    for repeat_idx, grid_times in enumerate(grid_t_all):
        for state in range(0,4):
            if repeat_idx == len(grid_t_all)-1 and state == 3: #ignore the last state/ repeat
                continue
            elif reward_locs[state] != locations[int(grid_times[state+1])]:
                print(f"careful! timings don't match for {subject}, grid {number_of_grid}; location at state {state+1} and step {grid_times[state+1]} is {locations[int(grid_times[state+1])]}, and should be one of {reward_locs}!")
                import pdb; pdb.set_trace()
    # print(f"all timings matched with finding correct rewarded location for subject {subject} and grid {reward_locs}!")



def single_reward(location, empty_reg, grid_t_all, reward_locs, setting = 'current'): 
    # setting can be ['current', 'next', 'second_next', 'previous']
    # import pdb; pdb.set_trace()
    grid_t_all = grid_t_all-grid_t_all[0,0]
    all_reward_locs = np.tile(reward_locs, 2)
    
    if setting == 'current':
        difference = 0
    elif setting == 'next':
        difference = 1
    elif setting == 'second_next':
        difference = 2
    elif setting == 'previous':
        difference = -1
    musicbox_regressors = empty_reg.copy()
     
    for repeat_idx, rep_times in enumerate(grid_t_all):
        if np.isnan(rep_times).any():
            continue
        for state in range(0,4):
            # curr_rew_loc = int(location[end_state]-1
            # this should be equal to int(reward_locs[state]-1) if not random!
            
            # this can be used for actual locations. int(location[end_next_state]-1)
            rew_loc_of_interest = int(all_reward_locs[state+difference]-1)
            # curr_rew_loc = int(all_reward_locs[state]-1)
            # next_rew_loc = int(all_reward_locs[state+1]-1)
            # second_next_rew_loc = int(all_reward_locs[state+2]-1)
            # prev_rew_loc = int(all_reward_locs[state-1]-1)
            
            # current
            start_state = int(rep_times[state])
            end_state = int(rep_times[state+1])
            musicbox_regressors[rew_loc_of_interest, start_state:end_state] = 1

    # plt.figure()
    # plt.imshow(musicbox_regressors, aspect = 'auto')
    # plt.axhline(8.5, color='white', linewidth=1)
    # plt.axhline(17.5, color='white', linewidth=1)
    # plt.axhline(26.5, color='white', linewidth=1)
    # import pdb; pdb.set_trace()        
    return musicbox_regressors
    
    
    
    
    

def music_box_simple_cells(location, empty_reg, grid_t_all, reward_locs, setting = None): 
    # setting can be ['withoutnow', 'only2and3future','onlynowandnext', 'onlynowand3future', 'onlynextand2future']
    # import pdb; pdb.set_trace()
    grid_t_all = grid_t_all-grid_t_all[0,0]
    all_reward_locs = np.tile(reward_locs, 2)
    
    musicbox_regressors = empty_reg.copy()
    for repeat_idx, rep_times in enumerate(grid_t_all):
        if np.isnan(rep_times).any():
            continue
        if repeat_idx < len(grid_t_all)-1:
            rep_and_next_times = np.concatenate((rep_times, grid_t_all[repeat_idx+1][1:]))
        for state in range(0,4):
            # curr_rew_loc = int(location[end_state]-1
            # this should be equal to int(reward_locs[state]-1) if not random!
            
            # this can be used for actual locations. int(location[end_next_state]-1)
            
            curr_rew_loc = int(all_reward_locs[state]-1)
            next_rew_loc = int(all_reward_locs[state+1]-1)
            second_next_rew_loc = int(all_reward_locs[state+2]-1)
            third_next_rew_loc = int(all_reward_locs[state+3]-1)
            
            # current
            start_state = int(rep_times[state])
            end_state = int(rep_times[state+1])
            if setting not in ['withoutnow', 'only2and3future', 'onlynextand2future']:
                # current reward
                musicbox_regressors[curr_rew_loc, start_state:end_state] = 1
                # plus 1
                end_next_state = int(rep_and_next_times[state+2])
                musicbox_regressors[next_rew_loc+9, start_state:end_state] = 1
    
                if setting not in ['onlynowandnext', 'onlynowand3future']:
                    # plus 2
                    end_secnext_state = int(rep_and_next_times[state+3])
                    musicbox_regressors[second_next_rew_loc+2*9, start_state:end_state] = 1
        
                    # plus 3
                    end_thirnext_state = int(rep_and_next_times[state+4])
                    musicbox_regressors[third_next_rew_loc+3*9, start_state:end_state] = 1
            
            elif setting in ['withoutnow', 'onlynextand2future']:
                #import pdb; pdb.set_trace()
                # plus 1
                end_next_state = int(rep_and_next_times[state+2])
                musicbox_regressors[next_rew_loc, start_state:end_state] = 1
    
                # plus 2
                end_secnext_state = int(rep_and_next_times[state+3])
                musicbox_regressors[second_next_rew_loc+9, start_state:end_state] = 1
                
                if setting == 'withoutnow':
                    # plus 3
                    end_thirnext_state = int(rep_and_next_times[state+4])
                    musicbox_regressors[third_next_rew_loc+2*9, start_state:end_state] = 1
            
            elif setting == 'only2and3future':
                #import pdb; pdb.set_trace()
                # plus 2
                end_secnext_state = int(rep_and_next_times[state+3])
                musicbox_regressors[second_next_rew_loc, start_state:end_state] = 1
    
                # plus 3
                end_thirnext_state = int(rep_and_next_times[state+4])
                musicbox_regressors[third_next_rew_loc+9, start_state:end_state] = 1
             
            elif setting == 'onlynowand3future':  
                # plus 3
                end_thirnext_state = int(rep_and_next_times[state+4])
                musicbox_regressors[third_next_rew_loc+9, start_state:end_state] = 1


    # plt.figure()
    # plt.imshow(musicbox_regressors, aspect = 'auto')
    # plt.axhline(8.5, color='white', linewidth=1)
    # plt.axhline(17.5, color='white', linewidth=1)
    # plt.axhline(26.5, color='white', linewidth=1)
    # import pdb; pdb.set_trace()        
    return musicbox_regressors


def locations_cells(location, empty_reg): 
    location_regressors = empty_reg.copy()
    loc_change_points = np.where(np.diff(location) != 0)[0] + 1  # Find where index changes
    
    for i, step_at in enumerate(loc_change_points):
        curr_loc = int(location[step_at-1]-1) # location before step, -1 because of py indexing
        previous_step_at = loc_change_points[i-1]
        if i == 0:
            location_regressors[curr_loc, 0:step_at] = 1
        elif i == len(loc_change_points)-1:
            location_regressors[curr_loc, previous_step_at:step_at] = 1
            final_loc = int(location[step_at]-1)
            location_regressors[final_loc, step_at:] = 1
        else:
            location_regressors[curr_loc, previous_step_at:step_at] = 1
    
    return location_regressors
     
def musicbox_cells_complete_withoutphase(location, empty_reg, grid_t_all, reward_locs, setting = None):
    # import pdb; pdb.set_trace()
    
    # CAREFUL!
    # this musicbox currently encodes the future.
    # from 3 subpaths back, it predicts when you are going to walk on the fields
    # you are currently walking on. 
    # this is fine if they take the same paths, but it may make a difference if they are not.
    # I also don't include any phase-coding.
    grid_t_all = grid_t_all-grid_t_all[0,0]
    
    state_regressors = empty_reg[0:4, :].copy()
    musicbox_complete = empty_reg.copy() 
    location = [int(l-1) for l in location] # -1 because of py indexing
    location = np.array(location)
    for repeat_idx, grid_times in enumerate(grid_t_all):
        if np.isnan(grid_times).any():
            continue
        else:
            state_regressors[0, int(grid_times[0]):int(grid_times[1])] = 1
            state_regressors[1, int(grid_times[1]):int(grid_times[2])] = 2
            state_regressors[2, int(grid_times[2]):int(grid_times[3])] = 3
            state_regressors[3, int(grid_times[3]):int(grid_times[4])] = 4
    
    state_model_bins = np.sum(state_regressors, axis = 0)
    # Identify change points (boundaries) in the index array
    state_change_points = np.where(np.diff(state_model_bins) != 0)[0] + 1  # Find where index changes
    # import pdb; pdb.set_trace()
    
    
    # this is current location
    loc_change_points = np.where(np.diff(location) != 0)[0] + 1  # Find where index changes
    for i, step_at in enumerate(loc_change_points):
        curr_loc = location[step_at-1] # location before step
        previous_step_at = loc_change_points[i-1]
        if i == 0:
            musicbox_complete[curr_loc, 0:step_at] = 1
        else:
            musicbox_complete[curr_loc, previous_step_at:step_at] = 1

    
    
    # Additionally, propagate locations of current state around in the other states.
    # this will create only 4 musicboxneurons, where 'current' is the most precise.
    # import pdb; pdb.set_trace()
    # For each state segment (as defined by state_change_points), compute future projections.
    # We process each segment and, for each future offset (1,2,3), project the location.
    
    num_future_states = 3  # we want 1-, 2-, and 3-state ahead projections
    
    for no_state_change, state_end in enumerate(state_change_points):
        state_start = int(grid_t_all[0,0]) if no_state_change == 0 else state_change_points[no_state_change-1]
        state_length = state_end-state_start
        if state_length <=0:
            continue
        
        # Values from 0 up to (but not including) 1 for each timepoint in the subpath
        relative_time_in_state = np.linspace(0,1,state_length, endpoint=False)
        
        
        # For each future subpath, if a future state exists, map the current relative time onto it.
        for x_states_in_future in range(1, num_future_states + 1):
            # Check if the future path exists.
            if no_state_change + x_states_in_future < len(state_change_points):
                # For 1 in future: future state is [state_change_points[no_state_change], state_change_points[no_state_change+1])
                # For 2 in future: future state is [state_change_points[no_state_change+1], state_change_points[no_state_change+2])
                # For 3 in future: future state is [state_change_points[no_state_change+2], state_change_points[no_state_change+3])
                future_state_start = state_change_points[no_state_change] if x_states_in_future == 1 else state_change_points[no_state_change + x_states_in_future - 1]
                future_state_end = state_change_points[no_state_change + x_states_in_future]
                future_state_length = future_state_end - future_state_start
                if future_state_length <= 0:
                    continue
                # Map each relative time in the current subpath to an index within the future subpath.
                indices = (relative_time_in_state * future_state_length).astype(int)
                # For each timepoint in the current segment, pick the future location.
                future_state_locs = location[future_state_start + indices]  # 1-indexed locations
    
                # Create one-hot encoded predictions for subpath x_states_in_future
                one_hot = np.zeros((9, state_length), dtype=int)
                one_hot[future_state_locs, np.arange(state_length)] = 1
    
                # Place these predictions into final_matrix.
                # The appropriate row block is: rows offset*9 to (offset+1)*9,
                # and the columns correspond to the current segment [seg_start:seg_end].
                row_start = x_states_in_future * 9
                row_end = (x_states_in_future + 1) * 9
                musicbox_complete[row_start:row_end, state_start:state_end] = one_hot
                # to also fill in the final bits of the shifted musicbox
                if no_state_change + x_states_in_future + 1 == len(state_change_points):
                    # import pdb; pdb.set_trace()
                    # copy the last state repeat in. this will be x_states_in_future long
                    for n_states_to_copy in range(0,x_states_in_future):
                        # For x_states_in_future = 1, we copy the last activation of 4 states ago;
                        # for x_states_in_future = 2, we copy the last two activations of 4 states ago; etc.
                        sim_state_start = state_change_points[no_state_change+n_states_to_copy]
                        sim_state_end = state_change_points[no_state_change+n_states_to_copy+1]
                        sim_state_length = sim_state_end-sim_state_start
                        relative_time_in_sim_state = np.linspace(0,1,sim_state_length, endpoint=False)
                        past_start = state_change_points[no_state_change-(4-x_states_in_future)+n_states_to_copy]  # starting index of the block to copy
                        past_state_end = state_change_points[no_state_change-(3-x_states_in_future)+n_states_to_copy]
                        past_state_length = past_state_end - past_start
                        
                        # this needs to be adjusted to the actual locations
                        # for 3 in future this is like one in past
                        # for 2 in future this is 2 in past
                        # for 1 in future this is 3 in past
                        # and then just move forward on the location vector
                        # adjusting no_state_change presumably.
                        
                        # Stretch/squeeze the current relative time into this smaller block.
                        indices = (relative_time_in_sim_state * past_state_length).astype(int)
                        past_state_locs = location[past_start + indices]
                        
                        # Create one-hot encoded predictions for subpath x_states_in_future
                        one_hot = np.zeros((9, sim_state_length), dtype=int)
                        one_hot[past_state_locs, np.arange(sim_state_length)] = 1
                        musicbox_complete[row_start:row_end, sim_state_start:sim_state_end] = one_hot

    
    # import pdb; pdb.set_trace()
    # figure out if the indexing works correclty!! 8 vs 9 - starting at 0 vs at 1
    # depending on which setting I want, cut the musicbox_complete matrix.
    # ['withoutnow', 'only2and3future','onlynowandnext']
    if setting in ['withoutnow']:
        musicbox_complete = musicbox_complete[9:, :]
    elif setting in ['only2and3future']:
        musicbox_complete = musicbox_complete[18:, :]
    elif setting in ['onlynowandnext']:
        musicbox_complete = musicbox_complete[0:18, :]

    # import pdb; pdb.set_trace()
    return musicbox_complete 
   

def musicbox_cells_complete(location, empty_reg, grid_t_all, reward_locs, setting = None):
    # CAREFUL!
    # this musicbox currently encodes the future.
    # from 3 subpaths back, it predicts when you are going to walk on the fields
    # you are currently walking on. 
    # this is fine if they take the same paths, but it may make a difference if they are not.
    # I also don't include any phase-coding.
    
    # specifically, it does not wrap around- so the last few columns are only filled for
    # current reward, not the future ones.
    
    grid_t_all = grid_t_all-grid_t_all[0,0]
    
    state_regressors = empty_reg[0:4, :].copy()
    musicbox_complete = empty_reg.copy() 
    location = [int(l-1) for l in location] # -1 because of py indexing
    location = np.array(location)
    for repeat_idx, grid_times in enumerate(grid_t_all):
        if np.isnan(grid_times).any():
            continue
        else:
            state_regressors[0, int(grid_times[0]):int(grid_times[1])] = 1
            state_regressors[1, int(grid_times[1]):int(grid_times[2])] = 2
            state_regressors[2, int(grid_times[2]):int(grid_times[3])] = 3
            state_regressors[3, int(grid_times[3]):int(grid_times[4])] = 4
    
    state_model_bins = np.sum(state_regressors, axis = 0)
    # Identify change points (boundaries) in the index array
    state_change_points = np.where(np.diff(state_model_bins) != 0)[0] + 1  # Find where index changes
    # import pdb; pdb.set_trace()
    
    
    # this is current location
    loc_change_points = np.where(np.diff(location) != 0)[0] + 1  # Find where index changes 
    for i, step_at in enumerate(loc_change_points):
        curr_loc = location[step_at-1] # location before step
        previous_step_at = loc_change_points[i-1]
        if i == 0:
            musicbox_complete[curr_loc, 0:step_at] = 1
        elif i == len(loc_change_points)-1:
            musicbox_complete[curr_loc, previous_step_at:step_at] = 1
            final_loc = int(location[step_at]-1)
            musicbox_complete[final_loc, step_at:] = 1
        else:
            musicbox_complete[curr_loc, previous_step_at:step_at] = 1
                
    # Additionally, propagate locations of current state around in the other states.
    # this will create only 4 musicboxneurons, where 'current' is the most precise.
    # import pdb; pdb.set_trace()
    # For each state segment (as defined by state_change_points), compute future projections.
    # We process each segment and, for each future offset (1,2,3), project the location.
    
    num_future_states = 3  # we want 1-, 2-, and 3-state ahead projections
    
    for no_state_change, state_end in enumerate(state_change_points):
        state_start = int(grid_t_all[0,0]) if no_state_change == 0 else state_change_points[no_state_change-1]
        state_length = state_end-state_start
        if state_length <=0:
            continue
        
        # Values from 0 up to (but not including) 1 for each timepoint in the subpath
        relative_time_in_state = np.linspace(0,1,state_length, endpoint=False)
        
        # import pdb; pdb.set_trace()
        # For each future subpath, if a future state exists, map the current relative time onto it.
        for x_states_in_future in range(1, num_future_states + 1):
            # Check if the future path exists.
            if no_state_change + x_states_in_future < len(state_change_points):
                # For 1 in future: future state is [state_change_points[no_state_change], state_change_points[no_state_change+1])
                # For 2 in future: future state is [state_change_points[no_state_change+1], state_change_points[no_state_change+2])
                # For 3 in future: future state is [state_change_points[no_state_change+2], state_change_points[no_state_change+3])
                future_state_start = state_change_points[no_state_change] if x_states_in_future == 1 else state_change_points[no_state_change + x_states_in_future - 1]
                future_state_end = state_change_points[no_state_change + x_states_in_future]
                future_state_length = future_state_end - future_state_start
                if future_state_length <= 0:
                    continue
                # Map each relative time in the current subpath to an index within the future subpath.
                indices = (relative_time_in_state * future_state_length).astype(int)
                # For each timepoint in the current segment, pick the future location.
                future_state_locs = location[future_state_start + indices]  # 1-indexed locations
    
                # Create one-hot encoded predictions for subpath x_states_in_future
                one_hot = np.zeros((9, state_length), dtype=int)
                one_hot[future_state_locs, np.arange(state_length)] = 1
    
                # Place these predictions into final_matrix.
                # The appropriate row block is: rows offset*9 to (offset+1)*9,
                # and the columns correspond to the current segment [seg_start:seg_end].
                row_start = x_states_in_future * 9
                row_end = (x_states_in_future + 1) * 9
                musicbox_complete[row_start:row_end, state_start:state_end] = one_hot
                # to also fill in the final bits of the shifted musicbox
                if no_state_change + x_states_in_future + 1 == len(state_change_points):
                    # import pdb; pdb.set_trace()
                    # copy the last state repeat in. this will be x_states_in_future long
                    for n_states_to_copy in range(0,x_states_in_future):
                        # For x_states_in_future = 1, we copy the last activation of 4 states ago;
                        # for x_states_in_future = 2, we copy the last two activations of 4 states ago; etc.
                        sim_state_start = state_change_points[no_state_change+n_states_to_copy]
                        sim_state_end = state_change_points[no_state_change+n_states_to_copy+1]
                        sim_state_length = sim_state_end-sim_state_start
                        relative_time_in_sim_state = np.linspace(0,1,sim_state_length, endpoint=False)
                        past_start = state_change_points[no_state_change-(4-x_states_in_future)+n_states_to_copy]  # starting index of the block to copy
                        past_state_end = state_change_points[no_state_change-(3-x_states_in_future)+n_states_to_copy]
                        past_state_length = past_state_end - past_start
                        
                        # this needs to be adjusted to the actual locations
                        # for 3 in future this is like one in past
                        # for 2 in future this is 2 in past
                        # for 1 in future this is 3 in past
                        # and then just move forward on the location vector
                        # adjusting no_state_change presumably.
                        
                        # Stretch/squeeze the current relative time into this smaller block.
                        indices = (relative_time_in_sim_state * past_state_length).astype(int)
                        past_state_locs = location[past_start + indices]
                        
                        # Create one-hot encoded predictions for subpath x_states_in_future
                        one_hot = np.zeros((9, sim_state_length), dtype=int)
                        one_hot[past_state_locs, np.arange(sim_state_length)] = 1
                        musicbox_complete[row_start:row_end, sim_state_start:sim_state_end] = one_hot

    
    # import pdb; pdb.set_trace()
    # figure out if the indexing works correclty!! 8 vs 9 - starting at 0 vs at 1
    # depending on which setting I want, cut the musicbox_complete matrix.
    # ['withoutnow', 'only2and3future','onlynowandnext', 'onlynowand3future', 'onlynextand2future']
    if setting in ['withoutnow']:
        musicbox_complete = musicbox_complete[9:, :]
    elif setting in ['only2and3future']:
        musicbox_complete = musicbox_complete[18:, :]
    elif setting in ['onlynowandnext']:
        musicbox_complete = musicbox_complete[0:18, :]
    elif setting in ['onlynowand3future', 'onlynextand2future']:
        musicbox_pt_one = musicbox_complete[0:9, :]
        if setting == 'onlynowand3future':
            musicbox_pt_two = musicbox_complete[27:, :]
        elif setting == 'onlynextand2future':
            musicbox_pt_two = musicbox_complete[18:27, :]
        musicbox_complete = np.concatenate((musicbox_pt_one, musicbox_pt_two), axis=0)

    return musicbox_complete 





def button_box_simple_cells(buttons, empty_reg, grid_t_all): 
    #import pdb; pdb.set_trace() 
    buttonbox_regressors = empty_reg.copy()
    # first convert buttons to integers
    # left = 0; up = 1; right = 2; down = 3; return = 99
    buttons_ints = mc.analyse.helpers_human_cells.buttons_to_ints(buttons)
    
    # for the 'future button presses" I will shift the buttons by the time it takes
    # them to press the next. This means that I need to find out how long until
    # the next button is pressed: 
    next_button_index = np.insert(mc.analyse.helpers_human_cells.button_change_indices(buttons_ints), 0,0)
    for no_button, b_idx in enumerate(next_button_index):
        curr_button = buttons_ints[b_idx]
        if no_button+1 < len(next_button_index):
            if curr_button != 99:
                buttonbox_regressors[curr_button, b_idx: next_button_index[no_button+1]] = 1
        
        if no_button < len(next_button_index)-1:
            next_button = buttons_ints[next_button_index[no_button+1]]
            if next_button != 99:
                buttonbox_regressors[next_button+4, b_idx: next_button_index[no_button+1]] = 1
        
        if no_button < len(next_button_index)-2:
            second_next_button = buttons_ints[next_button_index[no_button+2]]
            if second_next_button != 99:
                buttonbox_regressors[second_next_button+2*4, b_idx: next_button_index[no_button+1]] = 1
        
        if no_button < len(next_button_index)-3:
            third_next_button = buttons_ints[next_button_index[no_button+3]]
            if third_next_button != 99:    
                buttonbox_regressors[third_next_button+3*4, b_idx: next_button_index[no_button+1]] = 1

    # plt.figure()
    # plt.imshow(buttonbox_regressors, aspect = 'auto')
    # plt.axhline(3.5, color='white', linewidth=1)
    # plt.axhline(7.5, color='white', linewidth=1)
    # plt.axhline(11.5, color='white', linewidth=1)      
    return buttonbox_regressors

    

################################
####### PLOTTING ###############
################################
    
#
#
# PART 5: PLOTTING
# create functions to plot the matrices
import textwrap

def plot_without_legends(any_matrix, titlestring = None, prediction = None,  hrf = None, grid_size = None, step_time = None, reward_no = None, perms = None, intervalline = None, timings_curr_run = None, saving_file = None):
    #import pdb; pdb.set_trace()
    fig, ax = plt.subplots(figsize=(5,4))
    cmaps.bilbao
    cmap = plt.get_cmap('bilbao')
    
    plt.imshow(any_matrix, aspect = 'auto', interpolation= 'none', cmap=cmap) 


    # Create a wrapped title with a maximum of 20 characters per line
    title = f"{titlestring}"
    wrapped_title = '\n'.join(textwrap.wrap(title, width=20))
    # Set the wrapped title with larger font size
    ax.set_title(wrapped_title, fontsize=18)



    ax.set_ylabel('simulated neurons', fontsize = 16)
    ax.set_ylabel('neural activity across tasks', fontsize = 16)
    
    
    if hrf:
        hrf_set = '_hrf=' + str(hrf)
    if grid_size:
        grid_set = '_gridsize_' + str(grid_size) + 'x' + str(grid_size)
    if step_time:
        time_set = '_' + str(step_time) + 'ms_per_step_'
    if reward_no:
        rew_set = str(reward_no) + '_rewards_' 
    
    if timings_curr_run:
        for interval in timings_curr_run:
            plt.axvline(interval, color = 'white', ls = 'dashed')
    elif intervalline:
        intervalline = int(intervalline)
        for interval in range(intervalline-1, len(any_matrix[0])-1, intervalline):
            ax.axvline(interval, color='white', linewidth=1)
            
        # Set x-axis and y-axis ticks and labels with 45-degree rotation
        ticks = np.arange((intervalline/2), (len(any_matrix[0])+1), intervalline)
        ax.set_xticks(ticks)
        ax.set_xticklabels(['Task {}'.format(int(i) // 40 + 1) for i in ticks], rotation=45, ha='right', fontsize = 16)
        ax.grid(False)

    plt.tight_layout()
    
    if saving_file:
        fig.savefig(f"{saving_file}{titlestring}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{saving_file}{titlestring}.tiff", dpi=300, bbox_inches='tight')

        
    
    # if 'perms' in locals():
    #     perm_set = '_' + str(perms) + '_perms'
    #     plt.title('settings:_' + prediction + hrf_set + grid_set + time_set + rew_set + perm_set)
        # ax.title = 'settings:_' + prediction + hrf_set + grid_set + time_set + grid_set + perm_set
    

def plotclocks(clocks_matrix):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(clocks_matrix, aspect = 'auto') 
    ax.set_xticks([2,5,8,11])
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,36,72,108,144,180,216,252,288])
    ax.set_yticklabels(['anchor 1', 'anchor 2','anchor 3', 'anchor 4', 'anchor 5', 'anchor 6', 'anchor 7', 'anchor 8', 'anchor 9'])
    #return fig

def plot_one_clock(one_clock_matrix):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(one_clock_matrix, aspect = 'auto')
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
    plt.xticks(rotation = 45)
    plt.xlabel('phases')
    ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_yticklabels(['neuron 1', 'neuron2','neuron 3', 'neuron 4', 'neuron 5', 'neuron 6', 'neuron 7', 'neuron 8', 'neuron 9', 'neuron 10', 'neuron 11', 'neuron 12'])

def plot_one_anchor_all_clocks(one_anchor_matrix):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(one_anchor_matrix, aspect = 'auto')
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
    plt.xticks(rotation = 45)
    plt.xlabel('phases')
    ax.set_yticks([0,12,24])
    ax.set_yticklabels(['early_phase', 'mid_phase','late_phase'])
    
    
def plotclock_pertime(clocks_matrix, step_time, all_stepnums):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(clocks_matrix, aspect = 'auto') 
    cumsumsteps = np.cumsum(all_stepnums)
    total_steps = cumsumsteps[-1]    
    xticks_no = np.arange(0,step_time*total_steps,10) #100 ms steps
    ax.set_xticks(xticks_no)
    plt.xlabel('scale is in 100 ms (10 = 1 sec)')
    # ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,36,72,108,144,180,216,252,288])
    ax.set_yticklabels(['anchor 1', 'anchor 2','anchor 3', 'anchor 4', 'anchor 5', 'anchor 6', 'anchor 7', 'anchor 8', 'anchor 9'])
    #return fig
    
    
def plot_phaseloc_pertime(phase_loc_matrix, step_time, all_stepnums):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(phase_loc_matrix, aspect = 'auto') 
    cumsumsteps = np.cumsum(all_stepnums)
    total_steps = cumsumsteps[-1]    
    xticks_no = np.arange(0,step_time*total_steps,10) #100 ms steps
    ax.set_xticks(xticks_no)
    plt.xlabel('scale is in 100 ms (10 = 1 sec)')
    # ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,3,6,9,12,15,18,21,24])
    ax.set_yticklabels(['anchor 1', 'anchor 2','anchor 3', 'anchor 4', 'anchor 5', 'anchor 6', 'anchor 7', 'anchor 8', 'anchor 9'])
    #return fig
    
    
   
    
def plot_one_anchor_all_clocks_pertime(one_anchor_matrix, step_time, all_stepnums):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(one_anchor_matrix, aspect = 'auto')
    cumsumsteps = np.cumsum(all_stepnums)
    total_steps = cumsumsteps[-1] 
    xticks_no = np.array(range(0,step_time*total_steps))
    ax.set_xticks(xticks_no)
    ax.grid(visible = True)
    plt.xlabel('seconds')
    plt.grid(visible= True)
    # ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,12,24])
    ax.set_yticklabels(['early_phase', 'mid_phase','late_phase'])
    
    
def plotlocation(location_matrix):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(location_matrix, aspect = 'auto')
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back to r1'])
    plt.xticks(rotation = 45)
    plt.xlabel('phases')
    ax.set_yticks([0,1,2,3,4,5,6,7,8])
    ax.set_yticklabels(['field 1', 'field 2','field 3', 'field 4', 'field 5', 'field 6', 'field 7', 'field 8', 'feld 9'])
    #plt.plot([0, total_steps+1],[] )

    
def plotlocation_pertime(location_matrix, step_time, all_stepnums):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(location_matrix, aspect = 'auto') 
    ax.set_xticks([2,5,8,11])
    cumsumsteps = np.cumsum(all_stepnums)
    total_steps = cumsumsteps[-1]    
    xticks_no = np.arange(0,step_time*total_steps,10) # 100ms steps
    ax.set_xticks(xticks_no)
    plt.xlabel('scale is in 100 ms (10 = 1 sec)')
    plt.xticks(rotation = 45)
    ax.set_yticks([0,1,2,3,4,5,6,7,8])
    ax.set_yticklabels(['field 1', 'field 2','field 3', 'field 4', 'field 5', 'field 6', 'field 7', 'field 8', 'feld 9'])
    #return fig
    

# create polar plots that visualize neuron firing per phase
def plot_neurons(data):  
    #import pdb; pdb.set_trace()
    plt.figure()
    data.index = data['bearing'] * 2*pi / 360
    
    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x=data['phases'], height=data['value'], width=1)
    plt.xticks(rotation = 45)
    
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.bar(x=data.index, height=data['value'], width=pi/4)
    ax2.set_xticklabels(data.phases)
    ax2.set_xticks([2*pi*i/12 for i in range(12)])
    




#############



