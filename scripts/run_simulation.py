#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:05:51 2023

@author: Svenja Kuechenhoff

this script calls on several simulation functions to eventually create RDMs
and check if predictions of location neuron activations are sufficiently 
distinct from phase clock neuron activation patterns.

"""

# %reset -f

import mc
import pandas as pd

## Section 1.
## Create the task
##
reward_coords = mc.simulation.grid.create_grid()
reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords)

############## 

 
## Section 2. 
## Setting the Clocks and Location Matrix. 
##

# now create the two matrices, print them and plot them. 
first_clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
location_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0) 
 
print(first_clocks_matrix)
print(location_matrix)
 
mc.simulation.predictions.plotlocation(location_matrix)
mc.simulation.predictions.plotclocks(first_clocks_matrix)

#########################


## Section 3. 
## Create 'neuron plots'
##


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

# the simple solution will in the end be filling each row in a pd.DataFrame the way
# it is done below. But right now I first need to solve why the function doesnt work
# for multiple clocks, since this is the crucial one!! It just creates an empty matrix.

    
loops = 50
matrixtype = 'location'
first_average_loc = mc.simulation.predictions.manf_configs_loop(loops, matrixtype)
matrixtypetwo = 'clocks'
first_average_clock = mc.simulation.predictions.manf_configs_loop(loops, matrixtypetwo)
 

mc.simulation.predictions.plotlocation(first_average_loc)
mc.simulation.predictions.plotclocks(first_average_clock)

 
# then fill in pd.DataFrame
# then plot selected neurons > adjust the plot neuron function for which row/ neuron to plot.



# data = pd.DataFrame({'value': [0, 20, 30, 20, 0, 10, 15, 10, 0, 0, 0, 0], #these will be averages
#                      'bearing': range(0, 360, 30),
#                      'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})      

# mc.simulation.predictions.plot_neurons(data)


#



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
    
        
