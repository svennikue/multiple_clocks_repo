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
 
mc.simulation.predictions.plotlocation(location_matrix, reshaped_visited_fields, all_stepnums)
mc.simulation.predictions.plotclocks(first_clocks_matrix)

#########################


## Section 3. 
## Create 'neuron plots'
##