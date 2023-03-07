#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:04:41 2023

@author: Svenja Küchenhoff

This script defines functions for creating RDMs and plotting them.
"""

import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
import numpy as np
import mc


def within_task_RDM(activation_matrix, column_names, ax=None, plotting = False):
    # import pdb; pdb.set_trace()
    dataframe = pd.DataFrame(activation_matrix)
    dataframe.fillna(0)
    dataframe.columns = column_names
    corr_matrix = dataframe.corr() # pairwise pearson corr of columns, excluding NA/nulls
    if plotting == True:       
        if ax is None:
            plt.figure()
            ax = plt.axes()   
        print(corr_matrix)
        sn.heatmap(corr_matrix, annot = False)
    RSM = corr_matrix.to_numpy()
    return RSM

def between_task_RDM(no_tasks, column_names, ax=None, plotting = False):
    #import pdb; pdb.set_trace()
    for i in range(0, no_tasks):
        ## Create the task and paths
        reward_coords = mc.simulation.grid.create_grid()
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords) 
        ## Setting the Clocks and Location Matrix. 
        clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
        loc_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0)
        df_rewards = pd.DataFrame(reward_coords)
        if i == 0:
            df_clocks = pd.DataFrame(clocks_matrix)
            df_locs = pd.DataFrame(loc_matrix)
            df_clocks.columns = column_names
            df_locs.columns = column_names
            df_task_configs = pd.DataFrame(reshaped_visited_fields)
            df_task_configs = pd.concat([df_rewards, df_task_configs], axis = 1)
        else:
            temp_clocks = pd.DataFrame(clocks_matrix)
            temp_locs = pd.DataFrame(loc_matrix)
            temp_clocks.columns = column_names
            temp_locs.columns = column_names
            temp_path = pd.DataFrame(reshaped_visited_fields)
            df_clocks = pd.concat([df_clocks, temp_clocks], axis = 1)               
            df_locs = pd.concat([df_locs, temp_locs], axis = 1)
            df_task_configs = pd.concat([df_task_configs, df_rewards, temp_path], axis = 1)
    df_clocks.fillna(0, inplace = True)  
    df_locs.fillna(0, inplace = True)
    corr_clocks = df_clocks.corr()
    corr_locs = df_locs.corr()
    clocks_RSM = corr_clocks.to_numpy()
    locs_RSM = corr_locs.to_numpy()
    if plotting == True:       
        if ax is None:
            plt.figure()
            fig, ax =plt.subplots(1,2)  
        sn.heatmap(corr_clocks, annot = False, ax=ax[0])
        ax[0].set_title('Clocks')
        print(corr_clocks)
        sn.heatmap(corr_locs, annot = False, ax=ax[1])
        ax[1].set_title('Location')
        print(corr_locs)  
    return clocks_RSM, locs_RSM, df_clocks, df_locs, df_task_configs
            

def find_best_tasks(loop_no, no_columns, column_names): 
#    import pdb; pdb.set_trace()
#     # this needs to be something like:
#         # 1. create 10 random tasks and the between-task corr maps.
#         # 2. compute similarity between those 2 big matrices (this needs to be exclude_diag = False!! bc thats the within task one)
#         # 3. stepwise go through each task configuration and check if replacing it with 
#         #      a new one reduces the similarity value
#         # do this a number of loops
#         # always store the current configurations/ toss the one I am replacing
    # first, create one 10 tasks x 10 tasks matrix for clocks and locations
    task_config_no = 10
    clock_RSM_matrix, loc_RSM_matrix, df_clock, df_loc, task_configs = mc.simulation.RDMs.between_task_RDM(task_config_no, column_names, plotting = False)
    # and get the similarity between those. 
    similarity_between = mc.simulation.RDMs.corr_matrices(loc_RSM_matrix, clock_RSM_matrix)
    # based on this, try to optimize the correlation coefficient (similarity_between)
    for i in range(0, loop_no):        
        # then first take the first 10 columns of df_clock and df_loc and replace it with a new config
        # create new configuration
        reward_coords = mc.simulation.grid.create_grid()
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords) 
        df_rewards = pd.DataFrame(reward_coords)
        df_task_configs = pd.DataFrame(reshaped_visited_fields)
        df_temp_task_configs = pd.concat([df_rewards, df_task_configs], axis = 1)
        # create new neural predictions for this task config
        clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
        loc_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0)
        # turn those into dataframe
        temp_clocks = pd.DataFrame(clocks_matrix)
        temp_locs = pd.DataFrame(loc_matrix)
        temp_clocks.columns = column_names
        temp_locs.columns = column_names
        # prepare loop here       
        temp_similarity = np.ones((2,2))
        
        count = 1 # change back to 1 once debugging is done
        # then, replace each of the 10 tasks with the new config and text if similarity is now less (= better)
        # step out of the loop either way once looped through all columns, or when temp_similarity is lower
        while (temp_similarity[0,1] > similarity_between[0,1]) and (count < task_config_no):
            # have a counter for all columns   
            temp_df_loc = df_loc.copy()
            # replace the first (count) 12 columns with the new configuration
            temp_df_loc.iloc[:, ((count-1)*no_columns):(count*no_columns)] = temp_locs
            # temp_df_loc.iloc[:, (count*no_columns):((count*no_columns)+no_columns)] = temp_locs
            temp_df_loc.fillna(0, inplace = True)
            temp_df_clock = df_clock.copy()
            temp_df_clock.iloc[:, ((count-1)*no_columns):(count*no_columns)] = temp_clocks
            temp_df_clock.fillna(0, inplace = True)
            # create new correlation matrices for the new clocks and location matrix
            temp_corr_clocks = temp_df_clock.corr()
            temp_corr_locs = temp_df_loc.corr()
            temp_clocks_RSM = temp_corr_clocks.to_numpy()
            temp_locs_RSM = temp_corr_locs.to_numpy() 
            # test the new similarity between the new RSMs
            temp_similarity = mc.simulation.RDMs.corr_matrices(temp_locs_RSM, temp_clocks_RSM)
            if temp_similarity[0,1] < similarity_between[0,1]:
                # task_configs is structured a little different, its always x,y of paths and then x,y of rewards.
                # -> 4 columns per task config              
                task_configs.iloc[:,((count-1)*4):(count*4)] = df_temp_task_configs
                # if the new RSMs correlate less, replace the current configuration and RSM with the new
                # and continue to optimize further.
                df_clock = temp_df_clock.copy()
                df_loc = temp_df_loc.copy()
                similarity_between = temp_similarity.copy()
            count += 1 
            del temp_df_loc
            del temp_df_clock
            del temp_corr_clocks
            del temp_corr_locs
            del temp_similarity
            temp_similarity = np.ones((2,2))
        
    return df_clock, df_loc, task_configs, similarity_between





# def optimise_0angle_clocks(permutations):
    
#     return optimal_settings, optimal_tasks, max_dissimilarity

    
def corr_matrices(matrix_one, matrix_two, exclude_diag = True):
    # import pdb; pdb.set_trace()
    dimension = len(matrix_one) 
    if exclude_diag == True:
        diag_array_one = list(matrix_one[np.tril_indices(dimension, -1)])
        diag_array_two = list(matrix_two[np.tril_indices(dimension, -1)])
    else: # this is diagonal plus upper triangle
        diag_array_one = list(matrix_one[np.triu_indices(dimension)])
        diag_array_two = list(matrix_two[np.triu_indices(dimension)])
    coef = np.corrcoef(diag_array_one, diag_array_two) #pearson's R change this to kandalls tau!!
    return coef

lon








