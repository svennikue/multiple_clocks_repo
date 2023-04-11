#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:58:33 2023

@author: Svenja Küchenhoff
This script validates my models with Mohamadys ephys data, making use of an RSA.
This is the plan:
 1.  Based on a location file (do you have something like that?),  model how the mouse was running: where was it for how long
 2.  Create a model RDM (time x time) based on my model using the coordinates from step (1)
 3.  Run an RSA where I compute data a RDM from all of your neuron time-series (all neurons across time, then correlate the time x time axes for the RDM)
 4.  Run a regression between my model RDM and your data RDM
 
 
 'Data is here: https://drive.google.com/drive/folders/1vJw8AVZmHQrUnvqkASUwAd4t549uKN6b'
"""


# first bit is Mohamadys notebook

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os, sys, pickle, time, re, csv
import mc
import pandas as pd
from datetime import datetime 


Data_folder='/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_050423/' 

# start with a single mouse/ recording session first.
mouse_recday='me11_05122021_06122021'
session=0
location = np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')


# 1.  Based on a location file (do you have something like that?),  model how the mouse was running: where was it for how long
# Mohammady codes both for bridges and nodes. I am only considering nodes. 
# Thus, first try what happend if I approximate the location to nodes.

 
# from the reward coordinates, I can infer how many steps between each reward were taken
# goal is to create 'steps_per_walk' which is a list of e.g. 4 numbers that tells me how many steps between 2 rewards are taken.

# then, write the numbers of the fields as coordinates. 
# I can probably use some sort of reversed function as:
    # first step is to do -1 (bc I have numbers 1 til 9 but I need 0 til 8)
    # divide the number by 3 (grid size) into 2 whole numbers
    # the result is x, the remainder is y.
    # x = floor(number/grid_size)
    # y = number - x
#    y = step[1]
#    x = step[0]
#    anchor_field = x + y*size_grid


# the bridges are all numbers between 10-21
# maybe I want to replace all bridges with the neighbouring place fields.

grid_size = 3

# firstly, exchange all bridges for the fields the mouse was on before
# secondly, create a 'walk' list made of x and y coordinates 
for task_no in range(len(location)):
    # import pdb; pdb.set_trace() 
    curr_task = location[task_no]
    curr_coords = []
    for i, field in enumerate(curr_task):
        if field > 9: 
            curr_task[i] = curr_task[i-1]
        field_x_corr = curr_task[i]//grid_size
        field_y_corr = curr_task[i] - field_x_corr*grid_size
        curr_coords.append([field_x_corr, field_y_corr])
    if task_no == 0:
        curr_coords_np = np.array(curr_coords)
        walk_coords = np.expand_dims(curr_coords_np, axis = 0)
    if task_no > 0:
        curr_coords_np = np.array(curr_coords)
        walk_coords = np.concatenate([walk_coords, curr_coords_np.reshape([1, curr_coords_np.shape[0],curr_coords_np.shape[1]])], axis = 0)


# thirdly, reverse the rewards_configs file into coordinates
     

# fourth, count how many steps the mice made between each reward



#####################
## NOTEBOOK #########
#####################

'Data is here: https://drive.google.com/drive/folders/1vJw8AVZmHQrUnvqkASUwAd4t549uKN6b'

'''
####################
###Data structure###
####################


##Neuron npy files:
These contain the normalized firing rates of each neuron - spikes per frame (frame rate is 60 Hz)

npy files for each session
matrix with dimensions [neuron,trial,bin]

360 bins per trial, every state is 90 bins

##Location npy files:
These contain the location of the animal in each bin (should correspond exactly to the neuron bins)
locations1-9 are the 9 nodes

Then the remaining locations 10-21 are the bridges, coded in the "Edge_grid" array. just subtract 10 and that
gives you the index of the Edge_grid array which tells you which bridges are being referenced
e.g. an entry of 10 means index 0 which is array([1, 2]) (i.e. animal is at the the bridge between nodes 1 and 2)

##task_data
this is the task sequence used for that day (which nodes are rewarded in what order)

'''






 # 1.  Based on a location file (do you have something like that?),  model how the mouse was running: where was it for how long
 # Mohammady codes both for bridges and nodes. I am only considering nodes. 
 # Thus, first try what happend if I approximate the location to nodes.
 
 
# the location file per mouse and recording day is 
##Example Location array


mouse_recday='me11_05122021_06122021'
session=0
location_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')

print(np.shape(location_))
location_[0]

# the bridges are all numbers between 10-21
# maybe I want to replace all bridges with the neighbouring place fields.

curr_task_location = location_[0]
for i, place in enumerate(curr_task_location):
    if place > 9:
        curr_task_location[i] = curr_task_location[i-1]
 

# to know the reward locations, look here:
##example task array
mouse_recday='me11_05122021_06122021'
Tasks=np.load(Data_folder+'Task_data_'+mouse_recday+'.npy')

Tasks

# from the reward coordinates, I can infer how many steps between each reward were taken
# goal is to create 'steps_per_walk' which is a list of e.g. 4 numbers that tells me how many steps between 2 rewards are taken.

# then, write the numbers of the fields as coordinates. 
# I can probably use some sort of reversed function as:
    # first step is to do -1 (bc I have numbers 1 til 9 but I need 0 til 8)
    # divide the number by 3 (grid size) into 2 whole numbers
    # the result is x, the remainder is y.
    # x = floor(number/grid_size)
    # y = number - x
#    y = step[1]
#    x = step[0]
#    anchor_field = x + y*size_grid

for i in range(len(location_)):
    curr_task = location_[i]
    for field in curr_task:
        print('yey')
        

# locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, time_per_step, grid)
# single_clock, midnight_matrix, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, time_per_step, grid)
    
 
#############################################


# THIS IS FROM MOHAMMADYS NOTEBOOK
##Recording days used
'''These are pairs of days which were spike sorted together to give a total of 6 tasks: animals do 3 tasks a day'''

Recording_days=np.load(Data_folder+'Recording_days_combined.npy')
Recording_days


##example task array
mouse_recday='me11_05122021_06122021'
Tasks=np.load(Data_folder+'Task_data_'+mouse_recday+'.npy')

Tasks


#Edge grid
'''use this to make sense of the location arrays (see data structure above)'''
Edge_grid=np.load(Data_folder+'Edge_grid.npy')
Edge_grid



##Example Location array

mouse_recday='me11_05122021_06122021'
session=0
location_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')

print(np.shape(location_))
location_[0]


##Example Neuron activity
mouse_recday='me11_05122021_06122021'
session=0
neuron=0

data_neurons=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
data_neuron0=data_neurons[neuron]

plt.matshow(data_neuron0)
for angle in np.arange(4)*90:
    plt.axvline(angle,color='red',ls='dashed')
plt.show()

print(np.shape(data_neuron0))

data_neuron0








