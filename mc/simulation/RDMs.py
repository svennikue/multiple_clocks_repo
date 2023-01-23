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

def within_task_RDM(activation_matrix, column_names,ax=None, plotting = False):
    # import pdb; pdb.set_trace()
    dataframe = pd.DataFrame(activation_matrix)
    dataframe.fillna(0)
    dataframe.columns = column_names
    corr_matrix = dataframe.corr()
    if plotting == True:       
        if ax is None:
            plt.figure()
            ax = plt.axes()   
        print(corr_matrix)
        sn.heatmap(corr_matrix, annot = False)
    RSM = corr_matrix.to_numpy()
    return RSM

def between_task_RDM(no_tasks, column_names, ax=None, plotting = False):
    pathlengths = []
    for i in range(0, no_tasks):
        ## Create the task and paths
        reward_coords = mc.simulation.grid.create_grid()
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords) 
        ## Setting the Clocks and Location Matrix. 
        clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
        loc_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0)
        if i == 0:
            df_clocks = pd.DataFrame(clocks_matrix)
            df_locs = pd.DataFrame(loc_matrix)
        else:
            temp_clocks = pd.DataFrame(clocks_matrix)
            temp_locs = pd.DataFrame(loc_matrix)
            df_clocks = pd.concat([df_clocks, temp_clocks], axis = 1)               
            df_locs = pd.concat([df_locs, temp_locs], axis = 1)
    df_clocks.fillna(0)  
    df_locs.fillna(0)
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
    return clocks_RSM, locs_RSM
            

# def find_best_tasks(loop_no):
#     # this needs to be something like:
#         # 1. create 10 random tasks and the between-task corr maps.
#         # 2. compute similarity between those 2 big matrices (this needs to be exclude_diag = False!! bc thats the within task one)
#         # 3. stepwise go through each task configuration and check if replacing it with 
#         #      a new one reduces the similarity value
#         # do this a number of loops
#         # always store the current configurations/ toss the one I am replacing
        
   
#     return best_tasks 

    
    

    
def corr_matrices(matrix_one, matrix_two, exclude_diag = True):
    # import pdb; pdb.set_trace()
    dimension = len(matrix_one) 
    if exclude_diag == True:
        diag_array_one = list(matrix_one[np.tril_indices(dimension)])
        diag_array_two = list(matrix_two[np.tril_indices(dimension)])
    else:
        diag_array_one = list(matrix_one[np.triu_indices(dimension)])
        diag_array_two = list(matrix_two[np.triu_indices(dimension)])
    coef = np.corrcoef(diag_array_one, diag_array_two)
    return coef