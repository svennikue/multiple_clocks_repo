#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:18:05 2023

@author: Svenja Küchenhoff

This script aims at identifying task configurations that have a low correlation
between clocks and location predicitons.

"""

import mc
import pandas as pd
from datetime import datetime 
import numpy as np

phases = ['first_early', 'first_late', 'first_reward', 'scnd_early', 'scnd_late', 'scnd_reward', 'third_early', 'third_late', 'third_reward', 'fourth_early', 'fourth_late', 'fourth_reward']
pathlengths = []
all_pathlengths = []
indices_test = []


### Between Task RSMs.

loc_RSM_matrix, clock_RSM_matrix, df_clock, df_loc, df_task_configs = mc.simulation.RDMs.between_task_RDM(10, phases, plotting = True)
similarity_between = mc.simulation.RDMs.corr_matrices(loc_RSM_matrix, clock_RSM_matrix) 



## Optimize between and within task RSM
# SOMETHING DOESNT WORK HERE! Continue in 'RDMS.py', find_best_tasks.
# should be nearly there but something is going wrong with the temp.variables and 
# the looping.
# df_clock_opt, df_loc_opt, task_configs_opt, similarity_opt = mc.simulation.RDMs.find_best_tasks(20, 12, phases)



# start the loop.
cut_off = 0.45
count = 0
for i in range(0, 1000):
    ## Section 1.
    ## Create the task and paths
    ##
    reward_coords = mc.simulation.grid.create_grid()
    reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords)
    
    ## Section 2. 
    ## Setting the Clocks and Location Matrix. 
    ##
    
    # now create the two matrices 
    first_clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
    location_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0) 
     
    # Section 3. Create RDMs.
    
    loc_RSM = mc.simulation.RDMs.within_task_RDM(location_matrix, phases)
    clock_RSM = mc.simulation.RDMs.within_task_RDM(first_clocks_matrix, phases)
    similarity = mc.simulation.RDMs.corr_matrices(loc_RSM, clock_RSM)    
    all_pathlengths= np.append(all_pathlengths, len(reshaped_visited_fields))
    
    if similarity[1,0] < cut_off:
        # save task configuration and coefficient
        pathlengths= np.append(pathlengths, len(reshaped_visited_fields))
        indices_test = np.append(indices_test, i)
        path = pd.DataFrame(reshaped_visited_fields)
        coef = pd.DataFrame(similarity[0])
        rewards = pd.DataFrame(reward_coords)
        if count == 0:
            save_task = pd.concat([coef, path, rewards], axis = 1)
        else: 
            save_task = pd.concat([save_task, coef, path, rewards], axis = 1)
        count += 1 
        

# folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/results/good_configs'
# time = datetime.now().strftime("%Y%m%d-%H%M%S")
# file = '.csv'
# filename = folder + time + file
# if count > 0:
#     save_task.to_csv(filename)