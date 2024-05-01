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
import colormaps as cmap
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal
from scipy.stats import norm
from itertools import product
from scipy.signal import argrelextrema
from scipy import interpolate

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


def create_x_regressors_per_state(walked_path, subpath_timings, step_no, no_regs_per_state):
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


#4.1 ephys models: continuous - clocks - midnight - phase - location

# ok now I need the same thing but for my ephys stuff.
def set_continous_models_ephys(walked_path, subpath_timings, step_indices, step_number,  grid_size = 3, no_phase_neurons=3, fire_radius = 0.25, wrap_around = 1, plot = False):
    # import pdb; pdb.set_trace()

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
        
        # to plot the functions.
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

        
       
    # cumsumsteps = np.cumsum(step_number)
    # this time, do it per subpath.
    for count_paths, pathlength in enumerate(step_number):
        # first step: divide into subpaths
        curr_path = walked_path[subpath_timings[count_paths]:subpath_timings[count_paths+1]]

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
        state_matrix = np.empty([len(neuron_state_functions), len(curr_path)])
        state_matrix[:] = np.nan
        # only consider the 1/no_states*count_paths+1 part of the functions
        # then sample by timepoint
        #sample_state = means_at_state[count_paths] if wrap_around == 1 else count_paths
        sample_state = count_paths
        
        for row in range(0, len(neuron_state_functions)):
            state_matrix[row] = neuron_state_functions[row].pdf(sample_state)
        
        # fourth step: make phase neurons
        # fit subpaths into 0:1 trajectory
        samplepoints = np.linspace(-np.pi, np.pi, len(curr_path)) if wrap_around == 1 else np.linspace(0, 1, len(curr_path))
        
        phase_matrix_subpath = np.empty([len(neuron_phase_functions), len(samplepoints)])
        phase_matrix_subpath[:] = np.nan
        # read out the respective phase coding 
        for timepoint, read_out_point in enumerate(samplepoints):
            for row in range(0, len(neuron_phase_functions)):
                phase_matrix_subpath[row, timepoint] = neuron_phase_functions[row].pdf(read_out_point)

        # fifth step: midnight. = make location neurons phase sensitive.
        midnight_model_subpath = np.repeat(loc_matrix, repeats = no_phase_neurons, axis = 0)
        # multiply three rows of the location matrix (1 location)
        # with the phase_matrix_subpath, respectively
        for location in range(0, len(midnight_model_subpath), no_phase_neurons):
            midnight_model_subpath[location:location+no_phase_neurons] = midnight_model_subpath[location:location+no_phase_neurons] * phase_matrix_subpath
        
        
        # sixth. make the clock model. 
        # solving 2 (see below): make the neurons within the clock.
        # phase state neurons.
        
        phase_state_subpath = np.repeat(state_matrix, repeats = len(phase_matrix_subpath), axis = 0)
        for phase in range(0, len(phase_state_subpath), len(phase_matrix_subpath)):
            phase_state_subpath[phase: phase+len(phase_matrix_subpath)] = phase_matrix_subpath * phase_state_subpath[phase: phase+len(phase_matrix_subpath)]
        
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
                       
    
    # I am going to fuse the midnight and the phas_stat model. Thus they need to be equally 'strong' > normalise!
    norm_midn = (midn_model.copy()-np.min(midn_model))/(np.max(midn_model)-np.min(midn_model))
    norm_phas_stat = (phas_stat.copy()-np.min(phas_stat))/(np.max(phas_stat)-np.min(phas_stat))
    
     
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
            # import pdb; pdb.set_trace()
            horizontal_shift_by = np.argmax(norm_phas_stat[:,activation_neuron])
            # shift the clock around so that the activation neuron comes first
            shifted_clock = np.roll(norm_phas_stat, horizontal_shift_by*-1, axis = 0)
            
            # THIS IS GOIGN WRONG. I HAVE TO DO A HORIZONTAL SHIFT, but
            # it doesnt make sense to make the shift by the column-neuron. 
            # the defining one should be rwo...
            # try a different approach.
            
            
            # can I read the clocks out only here?
            

            # adjust the firing strength according to the local maxima
            firing_factor = norm_midn[row, activation_neuron].copy()
            #firing_factor = norm_midn[row,activation_neuron]/ max_firing
            shifted_adjusted_clock = shifted_clock.copy()*firing_factor
            
            # then add the values to the existing clocks, but also replace the first row by 0!!
            shifted_adjusted_clock[0] = np.zeros((len(shifted_adjusted_clock[0])))
        
            # Q: IS THIS WAY OF DEALING WIHT DOUBLE ACTIVATION OK???
            clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :] = clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :].copy() + shifted_adjusted_clock.copy()
    
    # # to plot the matrices
    # plt.figure()
    # plt.imshow(loc_model, aspect = 'auto', interpolation='none')
    # for subpath in subpath_timings:
    #     plt.axvline(subpath, color='white', ls='dashed')
    # import pdb; pdb.set_trace()
    if plot == True:
        mc.simulation.predictions.plot_without_legends(loc_model, titlestring='Location_model', timings_curr_run = subpath_timings)
        mc.simulation.predictions.plot_without_legends(phas_model, titlestring='Phase Model', timings_curr_run = subpath_timings)
        mc.simulation.predictions.plot_without_legends(stat_model, titlestring='State Model',timings_curr_run = subpath_timings)
        mc.simulation.predictions.plot_without_legends(midn_model, titlestring='Midnight Model', timings_curr_run = subpath_timings)
        mc.simulation.predictions.plot_without_legends(clo_model, titlestring='Musicbox model',timings_curr_run = subpath_timings)
        mc.simulation.predictions.plot_without_legends(phas_stat, titlestring='One ring of musicbox', timings_curr_run = subpath_timings)
        
    return loc_model, phas_model, stat_model, midn_model, clo_model, phas_stat


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
        # this means that every clock will be phase-matched to the whole matrix.
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



# to create the model RDMs.
def create_model_RDMs_fmri(walked_path, timings_per_step, step_number, grid_size = 3, no_phase_neurons=3, fire_radius = 0.25, wrap_around = 1, temporal_resolution = 10, plot = False, only_rew = False, only_path = False, split_clock = False):
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
    # stick the neuron-clock matrices in 
    full_clock_matrix_dummy = np.zeros([len(norm_midn)*len(norm_phas_stat),len(norm_midn[0])]) # fields times phases.
    # for ever 12th row, stick a row of the midnight matrix in (corresponds to the respective first neuron of the clock)
    for row in range(0, len(norm_midn)):
        full_clock_matrix_dummy[row*len(norm_phas_stat),:]= norm_midn[row,:].copy()
         
    # copy the neuron per clock firing pattern
    # I will manipulate clocks_per_step, and use clocks_per_step.dummy as control to check for overwritten stuff.
    clo_model =  full_clock_matrix_dummy.copy()
    
    
    # if you also want a split clock, then prepare those as well as a dicitonary
    if split_clock == True:  
        split_clock_strings = ['curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock']
        split_clock_model_dict = {}
        for model in split_clock_strings:
            # length of the future clock model will be 3x midnight: predicting the subpaths, not only the reward.
            split_clock_model_dict[model] = np.zeros([len(norm_midn)*3,len(norm_midn[0])])

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
            # import pdb; pdb.set_trace()
            horizontal_shift_by = np.argmax(norm_phas_stat[:,activation_neuron])
            # shift the clock around so that the activation neuron comes first
            shifted_clock = np.roll(norm_phas_stat, horizontal_shift_by*-1, axis = 0)
            # adjust the firing strength according to the local maxima
            firing_factor = norm_midn[row, activation_neuron].copy()
            #firing_factor = norm_midn[row,activation_neuron]/ max_firing
            shifted_adjusted_clock = shifted_clock.copy()*firing_factor
            # before 0ing out the first row (= musicboxneuron), first save the specific rows (i.e. now, next future, 2 future, 3 future) in the split clocks model.
            # import pdb; pdb.set_trace()
            if split_clock == True:
                split_clock_model_dict['curr_rings_split_clock'][row*3:row*3+3, :] = shifted_adjusted_clock[0:3] + split_clock_model_dict['curr_rings_split_clock'][row*3:row*3+3, :]
                split_clock_model_dict['one_fut_rings_split_clock'][row*3:row*3+3, :] = shifted_adjusted_clock[3:6] + split_clock_model_dict['one_fut_rings_split_clock'][row*3:row*3+3, :]
                split_clock_model_dict['two_fut_rings_split_clock'][row*3:row*3+3, :] = shifted_adjusted_clock[6:9] + split_clock_model_dict['two_fut_rings_split_clock'][row*3:row*3+3, :]
                split_clock_model_dict['three_fut_rings_split_clock'][row*3:row*3+3, :] = shifted_adjusted_clock[9:] + split_clock_model_dict['three_fut_rings_split_clock'][row*3:row*3+3, :]
                
            # then, for the full clock model, add the values to the existing clocks, but also replace the first row by 0!!
            shifted_adjusted_clock[0] = np.zeros((len(shifted_adjusted_clock[0])))
            clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :] = clo_model[row*len(norm_phas_stat): row*len(norm_phas_stat)+len(norm_phas_stat), :].copy() + shifted_adjusted_clock.copy()
        
    # import pdb; pdb.set_trace()
    if plot == True:
        mc.simulation.predictions.plot_without_legends(loc_model, titlestring='Location_model')
        mc.simulation.predictions.plot_without_legends(phas_model, titlestring='Phase Model')
        mc.simulation.predictions.plot_without_legends(stat_model, titlestring='State Model')
        mc.simulation.predictions.plot_without_legends(midn_model, titlestring='Midnight Model')
        mc.simulation.predictions.plot_without_legends(clo_model, titlestring='Musicbox model')
        mc.simulation.predictions.plot_without_legends(phas_stat, titlestring='One ring of musicbox')
        mc.simulation.predictions.plot_without_legends(task_prog_matrix, titlestring='Task progress Model')
        if split_clock == True:
            for model in split_clock_model_dict:
                mc.simulation.predictions.plot_without_legends(split_clock_model_dict[model], titlestring=model)
    
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
        
    # if only_rew == True:
    #     result_dict['midn_model'] = midn_model
    #     result_dict['clo_model'] = clo_model
    #     result_dict['state'] = stat_model
    #     result_dict['task_prog'] = task_prog_matrix
    #     result_dict['location'] = loc_model
    if only_rew == True:
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
####### PLOTTING ###############
################################
    
#
#
# PART 4: PLOTTING
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



