#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:51:56 2023

@author: xpsy1114
"""

# CONTINUOUS MODELS

# I am not using this for now. 
# define gaussian distribution of clock-tuning:
from matplotlib import pyplot as plt
import scipy
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import mc



plt.close('all')
#
#
# IN THIS SCRIPT I TRY AROUND TO FIND OUT HOW THE FUCK I CAN CODE UP A CONTINUOUS CLOCK MODEL
#
#

# start small.
#
# first quest: CONTINUOUS LOCATION MODEL
# update: done :)
#
#

# try how a 2d guassian would look like.
x,y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x,y))
rv = multivariate_normal([0, 0], cov = 0.5)
rv_two = multivariate_normal([0.5, 0.5], cov = 0.1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x,y,rv_two.pdf(pos), 1000)
# not sure if I can plot several together

plt.figure(); plt.imshow(rv.pdf(pos))
plt.figure(); plt.imshow(rv_two.pdf(pos))
plt.figure(); plt.imshow(rv.pdf(pos) + rv_two.pdf(pos))
# anyways, the plotting might be a bit more tricky. buuuut I can do the combo of 'neurons
# eg for a 3 x 3 grid

x,y = np.mgrid[-1:3:.01, -1:3:.01]
pos = np.dstack((x,y))
field_zero = multivariate_normal([0, 0], cov = 0.25)
field_one = multivariate_normal([1, 0], cov = 0.25)
field_two = multivariate_normal([2, 0], cov = 0.25)
field_three = multivariate_normal([0, 1], cov = 0.25)
field_four = multivariate_normal([1, 1], cov = 0.25)
field_five = multivariate_normal([2, 1], cov = 0.25)
field_six = multivariate_normal([0, 2], cov = 0.25)
field_seven = multivariate_normal([1, 2], cov = 0.25)
field_eight = multivariate_normal([2, 2], cov = 0.25)


field_zero = multivariate_normal([0, 0], cov = 0.5)
field_one = multivariate_normal([1, 0], cov = 0.5)
field_two = multivariate_normal([2, 0], cov = 0.5)
field_three = multivariate_normal([0, 1], cov = 0.5)
field_four = multivariate_normal([1, 1], cov = 0.5)
field_five = multivariate_normal([2, 1], cov = 0.5)
field_six = multivariate_normal([0, 2], cov = 0.5)
field_seven = multivariate_normal([1, 2], cov = 0.5)
field_eight = multivariate_normal([2, 2], cov = 0.5)


plt.figure(); plt.imshow(field_zero.pdf(pos))
plt.figure(); plt.imshow(field_one.pdf(pos))
plt.figure(); plt.imshow(field_two.pdf(pos))

plt.figure(); plt.imshow(field_two.pdf(pos) + field_one.pdf(pos) + field_zero.pdf(pos)+
                         field_three.pdf(pos) + field_four.pdf(pos) + field_five.pdf(pos)+
                         field_six.pdf(pos) + field_seven.pdf(pos) + field_eight.pdf(pos))
cbar = plt.colorbar()
plt.clim(0, 1)
for interval in [100, 200, 300]:
    plt.axvline(interval, color = 'white')
    plt.axhline(interval, color = 'white')
    
plt.figure(); plt.imshow(field_three.pdf(pos) + field_five.pdf(pos))    
cbar = plt.colorbar()
for interval in [100, 200, 300]:
    plt.axvline(interval, color = 'white')
    plt.axhline(interval, color = 'white')
    
# or
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x,y,field_two.pdf(pos) + field_one.pdf(pos) + field_zero.pdf(pos), 5)

# make a simulation to play around with this new model.
reward_coords = mc.simulation.grid.create_grid(plot = False)
walked_path, step_number = mc.simulation.grid.walk_paths(reward_coords, plotting = False)


#now, in order to make a model, loop through timepoints and sample neuron activity accordingly
# for simplicity, assume one coordinate is one timpoint.
# first, make a matrix that I am going to fill.
grid_size = 3 
loc_matrix = np.empty([grid_size*grid_size,len(walked_path)])
loc_matrix[:] = np.nan
# then fill the matrix with the neural readout of each of the 2d normalised place-neurons
for timestep, coords in enumerate(walked_path):
    loc_matrix[0,timestep] = field_zero.pdf(coords)
    loc_matrix[1,timestep] = field_one.pdf(coords)
    loc_matrix[2,timestep] = field_two.pdf(coords)
    loc_matrix[3,timestep] = field_three.pdf(coords)
    loc_matrix[4,timestep] = field_four.pdf(coords)
    loc_matrix[5,timestep] = field_five.pdf(coords)
    loc_matrix[6,timestep] = field_six.pdf(coords)
    loc_matrix[7,timestep] = field_seven.pdf(coords)
    loc_matrix[8,timestep] = field_eight.pdf(coords)


 
# just the field number function from predictions
def field_to_number(currentfield, size_of_grid):
    x = currentfield[0]
    y = currentfield[1]
    fieldnumber = x + y * size_of_grid
    return fieldnumber


# this is just the location function, maye I can use something from it    
step_time = 10 # I pretend that this is one second.
grid_size = 3
field_no_given = None
phases = 3

# first thing I do is to make an artificial timecourse
locations = [None] * len(walked_path)
for index, currfield in enumerate(walked_path):
    locations[index] = field_to_number(currfield, size_of_grid=3)

reward_fields = [None] * len(reward_coords)
for index, currreward in enumerate(reward_coords):
    reward_fields[index] = field_to_number(currreward, size_of_grid=3)


locations = np.repeat(locations, repeats = step_time)
timings = [step*step_time for step in step_number]
timings = np.cumsum(timings)

# then cut the path in subpaths.
for reward_no, curr_time in enumerate(timings):
    if reward_no == 0:
        subpath = locations[0:curr_time]
    elif reward_no > 0:
        subpath = locations[timings[reward_no - 1] : curr_time]
    # then for every subpath read out the phases
    # not really sure how lol
    
# ok. I now coded up a continuous version of the locations.
# THIS IS THE FINAL CONTINUUOUS LOCATION MODEL, YEY!
location_matrix = mc.simulation.predictions.set_location_contin(walked_path, step_time = 1)
plt.figure; plt.imshow(loc_matrix, aspect = 'auto')
#
#
#
# Quest one: success!


#
#
# Quest two: coding up a continuous PHASE model
#
#
# this is just a continuous version of phase 
# I am coding up phase in three neurons that approximate the phase-space
# and where 2 have a lot of overlap and the first one doesnt as much (as Mohamady found)


#
# Tim said something about having an actual location x phase representation.
# one can probably multiply the neurons from the location matrix with the phase-neurons.
# this uis how I define the phase-coding for now.
x = np.linspace(0, 1 ,1000)
early = norm.pdf(x, loc = 0.125, scale = 0.05)
mid = 2*norm.pdf(x, loc = 0.5, scale = 0.1)
late = 1.5*norm.pdf(x, loc = 0.75, scale = 0.075)
plt.figure();
plt.plot(x,early)
plt.plot(x,mid)
plt.plot(x,late)
plt.show()

# the readout of the neural activity would be as follows:
# phases = x
phases = [0.33, 0.66, 1]
for x in phases:
    early = norm.pdf(x, loc = 0.125, scale = 0.05)
    mid = 2*norm.pdf(x, loc = 0.5, scale = 0.1)
    late = 1.5*norm.pdf(x, loc = 0.75, scale = 0.075)
    print(early, mid, late)
    
    
phase_matrix = mc.simulation.predictions.set_phase_contin(walked_path, step_number, step_time = 3)
plt.figure(); 
plt.imshow(phase_matrix, aspect='auto')
# QUEST 2: DONE!!


# quest 3. continuous midnight model

# now that I have a continous phase and a continous location model, I want to build a 
# phase * location conjunction model.
# I believe this should be reeeeally similar to the phase model - but also considering locations.
# idea:
    # I will first compute the location matrix, and make 3 location neurons instead of one.
    # then I will compute the phase matrix
    # then I will multiply the respective location with the phase neurons (low will be nearly off, high will be on )

midnight_matrix = mc.simulation.predictions.set_midnight_contin(walked_path, step_number, step_time = 3)
plt.figure(); 
plt.imshow(midnight_matrix, aspect='auto')
# QUEST 3: DONE!!


# quest 4: continuous clocks model.

#now, to this continous midnight model, I need to add neurons that keep tracking phase and state.
# this comes with a few decisions.
# 1. do I activate a clock for a certain threshold or does the activity in which the 
    # rest of the neurons fire depend on how strong the midnight neuron was activated?
    # option 1 would be more descrete
    # option 2 would be a bit like a shooting star, where the clocks are the afterglow
    # of the initial activation.
# 2. I have to make the phase model neurons state-dependent. 
    # this should be easy. I can probably just create something similar to the locations function
    # and then multiply the subpaths with this matrix. 
# 3. put it all together. I can even take the midnight model, and squeeze the state-dependent neurons
    # in the middle. this can again be 12 - make it dependent on no_rewards and no_phases. 
location_matrix, phase_matrix, state_matrix, midnight_matrix, clocks_matrix, phase_state_matrix = mc.simulation.predictions.set_continous_models(walked_path, step_number, step_time = 1)
plt.figure(); 
plt.imshow(location_matrix, aspect='auto')
plt.figure(); 
plt.imshow(phase_matrix, aspect='auto')
plt.figure();  
plt.imshow(state_matrix, aspect='auto')
plt.figure()
plt.imshow(midnight_matrix, aspect='auto')
plt.figure(); 
plt.imshow(phase_state_matrix, aspect='auto')
plt.figure(); 
plt.imshow(clocks_matrix, aspect='auto')
for row in range(0, len(clocks_matrix), len(phase_state_matrix)):
    plt.axhline(row, color='white', ls='dashed')


# I BELIEVE IT WORKS!! :)
# NEXT STEP:
    # 1. check how RDMs look like.
    # 2. write the code for the ephys stuff.

# check and plot RDMs.
mc.simulation.RDMs.within_task_RDM(clocks_matrix, plotting=True, titlestring='continuous clocks')
mc.simulation.RDMs.within_task_RDM(phase_state_matrix, plotting=True, titlestring='continuous phase state')
mc.simulation.RDMs.within_task_RDM(midnight_matrix, plotting=True, titlestring='continuous midnight')
mc.simulation.RDMs.within_task_RDM(location_matrix, plotting=True, titlestring='continuous location')
mc.simulation.RDMs.within_task_RDM(phase_matrix, plotting=True, titlestring='continuous phase')
mc.simulation.RDMs.within_task_RDM(state_matrix, plotting=True, titlestring='continuous state')

# now write the functions for ephys and check how they work!!






