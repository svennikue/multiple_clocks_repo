#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:12:25 2022

@author: svenja

"""
# %reset -f

import random
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from numpy import pi
from matplotlib.gridspec import GridSpec


## Section 1.
## Create the task
##
# first create the 3 x 3 grid and plot.
coord = [list(p) for p in product(range(3), range(3))]
cmap = cm.get_cmap('tab20b')
plt.scatter([x[0] for x in coord], [x[1] for x in coord], c=cmap(6), s=250)

# create 4 reward locations
points = random.sample(coord, 4)
# note that points[0:4] are my states: 
    # points[1] = A - dark red
    # points[2] = B - red
    # points[3] = C - medium red
    # points[4] = D - bright red
for i, x in enumerate(points):
    plt.scatter(x[0], x[1], c=cmap(i+11), s=250)

plt.yticks([0,1,2])
plt.xticks([0,1,2])
plt.grid(True)


# now create the path connecting the points.

# I define a shortest path as: 
    # first, compare the x coordinates - start(x) minus stop(x)
    # go this distance on the x-axis
    # stepsxdir = stop[0]-start[0]
    # second, compare the y coordinates - start(y) minus stop(y)
    # go this distance on the y-axis
    # stepsydir = stop[1]-start[1]
    # this of course can be made more elaborate, to track all possible paths
    # buuuut that's for later.
    
def walksteps(startcoords, stopcoords):
    #import pdb; pdb.set_trace()
    stepsxdir = stopcoords[0]-startcoords[0]
    stepsydir = stopcoords[1]-startcoords[1]
    num_steps = abs(stepsxdir) + abs(stepsydir) 
    currcoord = list(startcoords)
    path = list()
    path.append(startcoords)
    for i in range(abs(stepsxdir)):
        if stepsxdir < 0: # if smaller than 0, go left
            currcoord[0]=currcoord[0]-1
            path.append([x for x in currcoord])
        elif stepsxdir > 0: # if smaller than 0, go right
            currcoord[0]=currcoord[0]+1
            path.append([x for x in currcoord])                
    for i in range(abs(stepsydir)):
        if stepsydir < 0: # if smaller than 0, go up
            currcoord[1]=currcoord[1]-1
            path.append([x for x in currcoord])
        elif stepsydir > 0: # if bigger than 0, go down
            currcoord[1]=currcoord[1]+1
            path.append([x for x in currcoord])        
    return path, num_steps


# now make use of the walksteps functions and loop through all points that 
# we generated before in points, and plot: 
for i, (x) in enumerate(points):
    plt.scatter(x[0], x[1], c=cmap(i+12), s=250)

# prepare looping through pairs of points to mimic the walks
all_paths = []  
all_stepnums = []
visited_fields = [[points[0]]]

# now loop through pairs of points using the walksteps function 
for i, (x) in enumerate(points):
    start = points[i]
    if i == (len(points)-1):
        stop = points[0]
    else:
        stop = points[i+1]
    path, num_steps = walksteps(start,stop)  
    all_paths.append([x for x in path])    
    visited_fields.append([x for x in path[1:]])
    all_stepnums.append(num_steps)
    # jitter to make the order of paths visible 
    # (thats a bit ugly now, but serves the purpose of visibilty...)
    plotpath = np.array(path) + 0.1*np.random.randn(len(path), 2)
    # plot the path
    for currstep, nextstep in zip(plotpath[:-1,:], plotpath[1:,:]):
        plt.plot([currstep[0], nextstep[0]], [currstep[1], nextstep[1]], c=cmap(i+12))

# reshape the visited_fields variable to a not-nested list
reshaped_visited_fields=[]    
for path in visited_fields:
    for coord in path:
        reshaped_visited_fields.append(coord)

# cut the first field: I am not starting at the reward.
# reshaped_visited_fields= reshaped_visited_fields[1:None]
# all_stepnums[0] = all_stepnums[0] - 1
           
##############  
        
   
## Section 2. 
## Setting the Clocks and Location Matrix. 
##
# I now have 4 location, 4 states, and paths between the locations/states,
# including going back to A after D.
# 4 locations: coordinates in points[0:4]
# 4 states: points[0:4]
# 4 paths between locations/states: all_paths[0:4]
# A-B, B-C, C-D, D-A
# (number of steps for each path: all_stepnums)


# for setting clocks
# there will be a matrix of 9*3*12 (field-anchors * phases * neurons) x 12 (3phase*4rewards)
# activate: phase1-field-clock for an early field
# activate: phase2-field-clock for a -1reward field (just before).
# activate: phase3-field-clock for a reward-field.
# advance every activated clock for progressing 'one phase'


# input will later be: reshaped_visited_fields and all_stepnums
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
    

# next, set location matrix.
# this will be a matrix which is 9 (fields) x  12 (phases).
# every field visit will activate the respective field. 
# since there always have to be 3 phases between 2 reward fields, I need to interpolate.
# my current solution for this:
# 1 step = 2 fields → both are early, late and reward (reward old and reward new)
# 2 steps = 3 fields → leave current field as is; 2nd is early and late; 3rd is reward
# 3 steps = 4 fields → leave current field as is, 2nd is early, 3rd is late, fourth is reward
# 4 steps = 5 fields → leave current field as is, 2nd is early, 3rd is early, 4th is late, 5th is reward

# input will later be: reshaped_visited_fields and all_stepnums
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
        print('Entered loop which goes through every subpath, currently at', count_paths)
        phasecount = len(phase_loop) #this needs to be reset for every subpath.
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
        print('Now I defined the current walked path:', curr_path)
        # if pathlength < phases -> 
        # it can be either pathlength == 1 or == 2. In both cases,
        # dublicate the field until it matches length phases
        # if pathlength > phases
        # dublicate the first phase so it matches length of path
        # so that, if finally, pathlength = phases
        # zip both lists and loop through them together.
        if pathlength < phasecount: 
            finished = False
            print('Entered a loop for paths shorter than 3 phases')
            while not finished:
                curr_path.insert(0, curr_path[0]) # dublicate first field 
                pathlength = len(curr_path)
                finished = pathlength == phasecount
        elif pathlength > phasecount:
            finished = False
            print('Entered a loop for paths longer than 3 phases')
            while not finished:
                phase_loop.insert(0,phase_loop[0]) #make more early phases
                phasecount = len(phase_loop)
                finished = pathlength == phasecount
        if pathlength == phasecount:
            print('Now finally entered a loop where paths = phases')
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
    return fig
    
def plotlocation(clocks_matrix, walked_path, step_number):
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    plt.imshow(clocks_matrix, aspect = 'auto')
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(['early', 'mid','reward 2','early', 'mid', 'reward 3','early','mid', 'reward 4', 'early','mid', 'back @ r1'])
    plt.xticks(rotation = 45)
    ax.set_yticks([0,1,2,3,4,5,6,7,8])
    ax.set_yticklabels(['field 1', 'field 2','field 3', 'field 4', 'field 5', 'field 6', 'field 7', 'field 8', 'feld 9'])
    #plt.plot([0, total_steps+1],[] )
 
    
# now create the two matrices, print them and plot them. 
first_clocks_matrix, total_steps  = set_clocks(reshaped_visited_fields, all_stepnums, 3)           
location_matrix = set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0) 
 
print(first_clocks_matrix)
print(location_matrix)
 
plotlocation(location_matrix, reshaped_visited_fields, all_stepnums)
plotclocks(first_clocks_matrix)


#########################
# Section 3. Create 'neuron plots'

# check if the neurons fulfill the same predictions as they do in Mohammadys case.
# write a function that plots the firing patterns of a certain neuron in the clock
# as a polar plot.
# this is from the matplotlib example.

# first step is to plot the firing pattern in a simple histogram
# this will be phases times firing, across different patterns.
# thus, recreate and store 100 matrices. 
# then, plot the 'firing' of every phase-neuron in the clock. 
# (or maybe not EVERY neuron...)

# 12.01.: CONTINUE BY LOOPING THROUGH THE PATH GENERATOR + CREATING SEVERAL MATRICES.
# for each neuron (= each row), I need to sum the values per phase ('timepoint')
# this will then be plotted in the polarplots.


def plot_neurons(data):  
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
    ax2.set_rgrids([10, 20, 30])
    
  
   

data = pd.DataFrame({'value': [0, 20, 30, 20, 0, 10, 15, 10, 0, 0, 0, 0], #these will be averages
                     'bearing': range(0, 360, 30),
                     'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})      

plot_neurons(data)



#########################
# Section 4. Create RDMs.

###########################
# Section 5. Correlate RDMs.

# later: create a loop and try out many combinations.  
# only save those combinations with a correlation lower than a certain value 
# or: first create a distribuion of corr values
# then define cut-off 


# x is defined by number of neurons and number of steps

# create distribution with scipy.stats.norm.pdf(x, mean, sigma)
    
        

            
    

################################
###### NOTES SECTION ###########
    
    
    # to index x,y coords in 1dim i = x + y * max_x   
    # x = i % x_max #remainder after division > modulo
    # y = int(i / x_max) # round down when converting to integer
    
# Legend.
# clocks_m[0:3,0] relates to early-med-late field 1
# clocks_m[3:6,0] relates to early-med-late field 2
# clocks_m[6:9,0] relates to early-med-late field 3
# clocks_m[9:12,0] relates to early-med-late field 4
# clocks_m[12:15,0] relates to early-med-late field 5
# clocks_m[15:18,0] relates to early-med-late field 6
# clocks_m[18:21,0] relates to early-med-late field 7
# clocks_m[21:24,0] relates to early-med-late field 8
# clocks_m[24:28,0] relates to early-med-late field 9
# coord[n-1] are the coordinates of field n = 1-9 



# # so now, I am going step by step, beginning at 'start' -> first element of points.
# # this will be the first column. 
# # to find out which row I have to fill in, I need to compare the current coordinates
# # with the column-coordinates.
# for x in points:
#     if points[x] == (0,2):
#         print('go to row zero')
#         clockrow = 0
#         elif points[x] == (1,2):
#             print('go to the third row')
#             clockrow = 3
#             elif points[x] == (2,2):
#                 print('go to the sixth row')
#                 clockrow = 6
#                 elif points[x] == (0,1):
#                     print('go to the ninth row')
#                     clockrow = 9
#                     elif points[x] == (1,1):
#                         print('go to the twelth row')
#                         clockrow = 12
#                         elif points[x] == (1,2):
#                             print('go to the fifteenth row')
#                             clockrow = 15
#                             elif points[x] == (2,0):
#                                 print('go to the eighteenth row')
#                                 clockrow = 18
#                                 elif points[x] == (2,1):
#                                     print('go to the twentyfirst row')
#                                     clockrow = 21
#                                     elif points[x] == (2,2):
#                                         print('go to the twentyfourth row')
#                                         clockrow = 24
                                        
#     # now check how the length of the path maps to the phases.
#     # if looking at the paths separetly, then its always 
#     colums_to_fill = all_stepnums[x]
    
#     # do 360° for a full cycle
#     # divide by total number of steps
#     # initiate clock (0°) at early reward phase
    
    
    
    
#     # I am not using this for now. 
#     # define gaussian distribution of clock-tuning:
#     def early(x):
#         return scipy.stats.norm.pdf(x, 1/6, 1/12)

#     def mid(x):
#         return scipy.stats.norm.pdf(x, 3/6, 1/12)

#     def late(x):
#         return scipy.stats.norm.pdf(x, 5/6, 1/12)
#     # define x exemplarily as x = np.linspace(-1,2,100)
#     # to plot, do plt.figure(); plt.plot(x, early); plt.plot(x, mid);plt.plot(x, late);  
    






# # for plotting an animation when doing the walk
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# # Creating the Animation object
# ani = animation.FuncAnimation(
#     fig, walksteps, num_steps, fargs=(walks, lines), interval=100)

# plt.show()


# Jacobs fancy plotting function 
#     for i, (currstep, nextstep) in enumerate(zip(path[:-1], path[1:])):
#        plt.plot([currstep[0], nextstep[0]], [currstep[1], nextstep[1]], c=cmap(i/len(path)))
        

# [tuple[index] for tuple in list]

