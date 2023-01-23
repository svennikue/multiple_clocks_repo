#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:42:09 2023

@author: Svenja Kuechenhoff

This module creates the task space for the multiple clock task.
It creates a grid, rewards on the grid, paths that connect the rewards,
and plots the whole thing.
"""

from itertools import product
from matplotlib import cm
import matplotlib.pyplot as plt
import random
import mc
import numpy as np


def create_grid(size_grid = 3, num_rewards = 4, ax = None, plot = False):
    # first create the 3 x 3 grid and plot.
    coord = [list(p) for p in product(range(size_grid), range(size_grid))]
    # create 4 reward locations
    reward_coords = random.sample(coord, num_rewards)
    if plot == True:
        if ax is None:
            plt.figure()
            ax = plt.axes()
        cmap = cm.get_cmap('tab20b')
        plt.scatter([x[0] for x in coord], [x[1] for x in coord], color =cmap(6), s=250)
        for i, x in enumerate(reward_coords):
            plt.scatter(x[0], x[1], color=cmap(i+11), s=250)
            # note that points[0:4] are my states: 
                # reward_coords[1] = A - dark red
                # reward_coords[2] = B - red
                # reward_coords[3] = C - medium red
                # reward_coords[4] = D - bright red           
        plt.yticks(list(range(size_grid)))
        plt.xticks(list(range(size_grid)))
        plt.grid(True)
    return reward_coords

    
def find_paths(startcoords, stopcoords):
    # import pdb; pdb.set_trace()
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

# create a function that identifies all possible paths
# if the path length is 4, then there will be 6 possible paths 
# if path length is 3, then there will be 3 possible paths
# if path length is 2, there will be 2 paths if one reward is at position 5
# if path length is 2, there will be 1 path if no reward is at position 5
# if path lenght is 1, there will be only 1 path
# def find_all_paths(startcoords, stopcoords):
#     # firstly, get absolut distance in both directions.
#     stepsxdir = stopcoords[0]-startcoords[0]
#     stepsydir = stopcoords[1]-startcoords[1]
#     # secondly, create 'step lists'
#     x_step_list = [1]* abs(stepsxdir)
#     y_step_list = [1]* abs(stepsydir)
#     if stepsxdir < 0:
#         x_step_list = [ -x for x in x_step_list]
#     if stepsydir < 0: 
#         y_step_list = [ -y for y in y_step_list]  
    
#     # this needs to be a bit adjusted. I want all possible combinations of steps,
#     # but
#     [[x_, y_] for x_ in x_step_list for y_ in y_step_list]

# #  alternatively, to loop through all possible combinations:    
#     for x_ in x:
#         for y_ in y:
#             print(x_,y_)
    
#     num_steps = abs(stepsxdir) + abs(stepsydir) 
#     currcoord = list(startcoords)
#     path = list()
#     path.append(startcoords)
    
    



def walk_paths(points, size_grid = 3, ax = None, plotting = False):
    # import pdb; pdb.set_trace()
    # loop through pairs of points to mimic the walks, using the find_paths 
    # function from above. Also plot paths connecting the rewards generated
    # by the create_grid function.
    coord = [list(p) for p in product(range(size_grid), range(size_grid))]
    # now set up the paths.
    all_stepnums = []
    visited_fields = [[points[0]]]
    if plotting == True:
        # first plot the grid (this is actually the same as in create_grid)
        if ax is None:
            plt.figure()
            ax = plt.axes()       
        cmap = cm.get_cmap('tab20b')
        plt.scatter([x[0] for x in coord], [x[1] for x in coord], color=cmap(6), s=250)
        for i, x in enumerate(points):
            plt.scatter(x[0], x[1], color =cmap(i+11), s=250)
        plt.yticks(list(range(size_grid)))
        plt.xticks(list(range(size_grid)))
        plt.grid(True)
        for i, (x) in enumerate(points):
            plt.scatter(x[0], x[1], color =cmap(i+12), s=250)
        # plot the path
    for i, (x) in enumerate(points):
        start = points[i]
        if i == (len(points)-1):
            stop = points[0]
        else:
            stop = points[i+1]
        path, num_steps = mc.simulation.grid.find_paths(start,stop)  
        # all_paths.append([x for x in path])    
        visited_fields.append([x for x in path[1:]])
        all_stepnums.append(num_steps)
        if plotting == True:
            # jitter to make the order of paths visible 
            # (thats a bit ugly now, but serves the purpose of visibilty...)
            plotpath = np.array(path) + 0.1*np.random.randn(len(path), 2)
            for currstep, nextstep in zip(plotpath[:-1,:], plotpath[1:,:]):
                plt.plot([currstep[0], nextstep[0]], [currstep[1], nextstep[1]], color=cmap(i+12))
    # reshape the visited_fields variable to a not-nested list        
    reshaped_visited_fields=[]    
    for path in visited_fields:
        for coord in path:
            reshaped_visited_fields.append(coord)
    
    return reshaped_visited_fields, all_stepnums
