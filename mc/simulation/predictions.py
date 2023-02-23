#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:34:15 2023

@author: Svenja Kuechenhoff

This module generates hypothesis matrices of neural activity based on 
predictions for location and phase-clock neurons.
It also includes functions to plot those.

"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib.gridspec import GridSpec
import mc
import scipy.signal

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

def convolve_with_hrf(clocks_per_sec, step_number, step_time, plotting = True):
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


# for i, j in enumerate([0, 10, 40, 45]):
#     plt.subplot(8,1,i*2+1)
#     plt.plot(np.eye(50)[j])
#     plt.subplot(8,1,i*2+2)
#     plt.plot(np.convolve(np.concatenate([np.eye(50)[j] for _ in range(2)]), hrf_signal)[50:100])


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
def set_location_matrix(walked_path, step_number, phases, size_grid = 3):
    #import pdb; pdb.set_trace()
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



def set_location_by_time(walked_path, step_number, step_time, grid_size = 3):
    # import pdb; pdb.set_trace()   
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1]    
    n_columns = total_steps   
    n_rows = grid_size*grid_size
    loc_matrix = np.empty([n_rows,n_columns]) # fields times steps
    loc_matrix[:] = np.nan
    for i in range(0, total_steps):
        curr_field = walked_path[i+1] # cut the first field because this is the reward field
        x = curr_field[0]
        y = curr_field[1]
        fieldnumber = x + y* grid_size
        # test if this has already been activated!
        if loc_matrix[fieldnumber, i] == 0:
            # if so, then don't overwrite it.
            loc_matrix[fieldnumber, i] = 1
        else:   
            loc_matrix[fieldnumber, :] = 0
            loc_matrix[fieldnumber, i] = 1
    loc_per_sec = np.repeat(loc_matrix, repeats = step_time, axis=1)    
    return loc_matrix, loc_per_sec


    
#
#
# PART 2: PLOTTING
# create functions to plot the matrices

def plot_without_legends(any_matrix, prediction = None,  hrf = None, grid_size = None, step_time = None, reward_no = None, perms = None):
    # import pdb; pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(any_matrix, aspect = 'auto') 
    ax.xlabel = 'neural activity over some timescale'
    ax.ylabel = 'neurons'
    if 'hrf' in locals():
        hrf_set = '_hrf=' + str(hrf)
    if 'grid_size' in locals():
        grid_set = '_gridsize_' + str(grid_size) + 'x' + str(grid_size)
    if 'step_time' in locals():
        time_set = '_' + str(step_time) + 'ms_per_step_'
    if 'reward_no' in locals():
        rew_set = str(reward_no) + '_rewards_' 
    if 'perms' in locals():
        perm_set = '_' + str(perms) + '_perms'
        plt.title('settings:_' + prediction + hrf_set + grid_set + time_set + rew_set + perm_set)
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
    


# loop function to create an average prediction.
def many_configs_loop(loop_no, which_matrix):
    # import pdb; pdb.set_trace()
    if which_matrix != 'clocks' and which_matrix != 'location':
        raise TypeError("Please enter 'location' or 'clocks' to create the correct matrix")   
    for loop in range(loop_no):
        reward_coords = mc.simulation.grid.create_grid()
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords)
        if which_matrix == 'location':
            temp_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0) 
        elif which_matrix == 'clocks':
            temp_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)
        if loop < 1:
            sum_matrix = temp_matrix[:]
        else:
            sum_matrix = np.nansum(np.dstack((sum_matrix[:],temp_matrix[:])),2)
    average_matrix = sum_matrix[:]/loop_no
    return average_matrix





