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

def set_clocks(walked_path, step_number, phases, peak_activity = 1, neighbour_activation = 0.5):
    # import pdb; pdb.set_trace()
    n_states = len(step_number)
    n_columns = phases*n_states
    # and number of rows is locations*phase*neurons per clock
    # every field (9 fields) -> can be the anchor of 3 phase clocks
    # -> of 12 neurons each. 9 x 3 x 12 
    # to make it easy, the number of neurons = number of one complete loop (12)
    n_rows = 9*phases*(phases*n_states)    
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
                anchor_field = x + y*3
                anchor_phase_start = (anchor_field * n_columns * 3) + (phase * n_columns)
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
    return clocks_matrix, total_steps  


# input is: reshaped_visited_fields and all_stepnums from mc.simulation.grid.walk_paths(reward_coords)
def set_clocks_bytime(walked_path, step_number, phases, step_time):
    # for simplicity, I will just use the same number of neurons as before.
    # this is a bit annoying though because now I can't propagate the signal
    # by 1 neuron per step, because the step lengths will be different.
    # so. I need some sort of interpolation over the clock neurons. At the same
    # time, the interpolation for phases will be obsolete.
    # first, set up the same matrix structure as before.
    # import pdb; pdb.set_trace()
    n_states = len(step_number)
    # the number of columns will be in secs. Each step takes 2secs for now.
    n_columns = len(walked_path) * step_time
    # and number of rows is locations*phase*neurons per clock
    # every field (9 fields) -> can be the anchor of 3 phase clocks
    # -> of 12 neurons each. 9 x 3 x 12 
    # as before, the number of neurons = number of one complete loop (12)
    n_rows = 9*phases*(phases*n_states)
    clocks_matrix = np.empty([n_rows,n_columns]) # fields times phases.
    clocks_matrix[:] = np.nan # 324 x stepnum (e.g. 7)
    

    # NOT SURE IF I WILL NEED THE FOLLOWING STEPS
    # MAYBE DELETE LATER
    phase_loop = list(range(0,phases))
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1]
    all_phases = list(range(n_columns))  
    
    
    # set up neurons of a clock.
    # here I now need to interpolate.
    # propagation of neurons is 12/ steplength
    # !! this function only works for steplength smaller than 24
    # otherwise it will round down to 0
    # I cant really have non-integers, thus: use rounding, and make the last 
    # neuron longer if round up, and activate two neurons if round down.
    
    clock_neurons = np.zeros([phases*n_states, n_columns])
    no_activated_neurons = round((phases*n_states) / len(walked_path))

    neurons_activated = 0
    for i in range(0,len(clock_neurons)):  
        # import pdb; pdb.set_trace()
        neurons_activated = i*no_activated_neurons
        if (neurons_activated >= len(clock_neurons)) or (i >= (len(clock_neurons[0]))):
            break
        # check if rounded up or down in the end
        clock_neurons[neurons_activated:(i+1)*no_activated_neurons, i]= 1  
    if n_columns == 8: 
        clock_neurons[-2:None, -2:None] = 1 # this isnt ideal bc the last neuron is now super long (x3)
    if (n_columns == 5) or (n_columns == 10):
        clock_neurons[-2:None, -1:None] = 1 # also not ideal, the last 4 steps don't propagate
    if (n_columns == 9):
        clock_neurons[-3:None, -1:None] = 1 # also not ideal, the last 4 steps don't propagate
    if (n_columns == 10):
        clock_neurons[-3:None, -1:None] = 1 # also not ideal, the last 4 steps don't propagate




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
def set_location_matrix(walked_path, step_number, phases, neighbour_activation = 0):
    #import pdb; pdb.set_trace()
    n_states = len(step_number)
    n_columns = phases*n_states
    location_matrix = np.zeros([9,n_columns]) # fields times phases.
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
                fieldnumber = x + y*3
                location_matrix[fieldnumber, ((phases*count_paths)+phase)] = 1 # currstep = phases
    return location_matrix, total_steps  
 
# create functions to plot the matrices

def plotclocks(clocks_matrix):
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    plt.imshow(clocks_matrix, aspect = 'auto') 
    ax.set_xticks([2,5,8,11])
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back @ r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,36,72,108,144,180,216,252,288])
    ax.set_yticklabels(['anchor 1', 'anchor 2','anchor 3', 'anchor 4', 'anchor 5', 'anchor 6', 'anchor 7', 'anchor 8', 'anchor 9'])
    #return fig

def plot_one_clock(one_clock_matrix):
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    plt.imshow(one_clock_matrix, aspect = 'auto')
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back @ r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_yticklabels(['neuron 1', 'neuron2','neuron 3', 'neuron 4', 'neuron 5', 'neuron 6', 'neuron 7', 'neuron 8', 'neuron 9', 'neuron 10', 'neuron 11', 'neuron 12'])


def plot_one_anchor_all_clocks(one_anchor_matrix):
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    plt.imshow(one_anchor_matrix, aspect = 'auto')
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back @ r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,12,24])
    ax.set_yticklabels(['early_phase', 'mid_phase','late_phase'])
    
    
def plotlocation(location_matrix):
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    plt.imshow(location_matrix, aspect = 'auto')
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back @ r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,1,2,3,4,5,6,7,8])
    ax.set_yticklabels(['field 1', 'field 2','field 3', 'field 4', 'field 5', 'field 6', 'field 7', 'field 8', 'feld 9'])
    #plt.plot([0, total_steps+1],[] )
 

# create polar plots that visualize neuron firing per phase
def plot_neurons(data):  
    #import pdb; pdb.set_trace()
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





