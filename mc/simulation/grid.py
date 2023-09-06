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


def create_grid(size_grid = 3, num_rewards = 4, ax = None, plot = False, old_rewards = None, step_longer_one = False):
    #import pdb; pdb.set_trace()
    # first create the 3 x 3 grid and plot.
    coord = [list(p) for p in product(range(size_grid), range(size_grid))]

    # If reward coordinates are not provided: set them here, then go on as usual
    if old_rewards is None:
        # create 4 reward locations, except if those are given
        reward_coords = [np.array(random.sample(coord, num_rewards))]
        if step_longer_one == True:
            # for i in range(0, len(reward_coords[0])):
            #     distance = sum(abs(reward_coords[0][i -1]- reward_coords[0][i]))
            #     while abs(distance) <= 1:
            #         replacement = random.sample(coord, 1)[0]
            #         reward_coords[0][i] = replacement 
            #         # now check if distance between replacement and any except the to-replaced
            #         # item is bigger than 1:
            #         for j in range(0, len(reward_coords[0])):
            #             if j != i:
            #                 the_same = sum(abs(replacement - reward_coords[0][j]))
            #             distance = sum(abs(reward_coords[0][j -1]- reward_coords[0][j]))
            #             # check if its the same as any sample
            #             if (abs(the_same) == 0) or abs(distance) <= 1:
            #                 replacement = random.sample(coord, 1)[0]
            #                 break
                    
                    
            for i in range(0, len(reward_coords[0])):
                for j in range(0, len(reward_coords[0])):
                    the_same = sum(abs(reward_coords[0][i] - reward_coords[0][j]))
                    if j == i:
                        the_same = 4
                    distance = sum(abs(reward_coords[0][i -1]- reward_coords[0][i]))
                    while (abs(distance) <= 1) or (the_same == 0):
                        replacement = random.sample(coord, 1)[0]
                        reward_coords[0][i] = replacement
                        the_same = sum(abs(reward_coords[0][i] - reward_coords[0][j]))
                        if j == i:
                            the_same = 4
                        distance = sum(abs(reward_coords[0][i -1]- reward_coords[0][i]))
                                
                            
                            
                            
                        # distance = sum(abs(reward_coords[0][i -1]- reward_coords[0][i]))
                        # while abs(distance) <= 1:
                        #     replacement = random.sample(coord, 1)[0]
                        #     reward_coords[0][i] = replacement 
                        #     # now check if distance between replacement and any except the to-replaced
                        #     # item is bigger than 1:
                        #     for j in range(0, len(reward_coords[0])):
                        #         if j != i:
                        #             the_same = sum(abs(replacement - reward_coords[0][j]))
                        #         distance = sum(abs(reward_coords[0][j -1]- reward_coords[0][j]))
                        #         # check if its the same as any sample
                        #         if (abs(the_same) == 0) or abs(distance) <= 1:
                        #             replacement = random.sample(coord, 1)[0]
                        #             break
                         
                    
                #distance = sum(abs(reward_coords[0][i -1] - reward_coords[0][i]))
                
                
    
                
                # # first check if every entry is unique.
                # for j in range(0, len(reward_coords[0])):
                #     the_same = sum(abs(reward_coords[0][i]-reward_coords[0][j]))
                    
                    
                    
                # # then check if they are more than 1 step apart.
                # distance = sum(abs(reward_coords[0][i -1]- reward_coords[0][i]))
                # while abs(distance) <= 1:
                #     replacement = random.sample(coord, 1)[0]
                #     reward_coords[0][i] = replacement
                #     distance = sum(abs(reward_coords[0][i -1] - reward_coords[0][i]))
                
                    
                    
                    
                    
                #     already_exists = 0
                #     while already_exists == 0:
                #         for j in range(0, len(reward_coords[0])):
                #             already_exists = sum(abs(replacement-reward_coords[0][j]))
                #             if already_exists == 0:
                #                 replacement = random.sample(coord, 1)[0]
                #                 #already_exists = sum(abs(replacement-reward_coords[0][j]))
                #                 already_exists = 0
                #                 break 
                #     reward_coords[0][i] = replacement
                #     distance = sum(abs(reward_coords[0][i -1] - reward_coords[0][i]))

                #print(f"distance is now {sum(reward_coords[0][i -1]-reward_coords[0][i])}")
    else:
        # read the reward coordinations and the path
        # careful! Here, every 4th column are the reward locations.
        # this is because I did a bad job saving these configuration lists...
        reward_coords = old_rewards.to_numpy()
        reward_coords = [reward_coords[:num_rewards,((i+1)*5-2):((i+1)*5)] 
                         for i in range(int(reward_coords.shape[1]/5))]  
    if plot == True:        
        for curr_coords in reward_coords:    
            if ax is None:
                plt.figure()
                plt.axes()
            cmap = cm.get_cmap('tab20b')
            plt.scatter([x[0] for x in coord], [x[1] for x in coord], color =cmap(6), s=250)
                # note that points[0:4] are my states: 
                    # reward_coords[1] = A - dark red
                    # reward_coords[2] = B - red
                    # reward_coords[3] = C - medium red
                    # reward_coords[4] = D - bright red           
            for i, x in enumerate(curr_coords):
                plt.scatter(x[0], x[1], color=cmap(i+11), s=250)
                plt.yticks(list(range(size_grid)))
                plt.xticks(list(range(size_grid)))
                plt.grid(True)
            
    reward_coords = reward_coords[0].tolist()
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



# def plot_paths(points, size_grid = 3, ax = None, plotting = False)
def plot_paths(rewards, path, size_grid = 3, ax = None, plotting = True):
    # import pdb; pdb.set_trace()
    # loop through pairs of points to mimic the walks 
    # Also plot paths connecting the rewards generated by the 
    # create_grid function.
    
    # first set up the grid.
    coord = [list(p) for p in product(range(size_grid), range(size_grid))]

    if plotting == True:
        # first plot the grid (this is actually the same as in create_grid)
        if ax is None:
            plt.figure()
            ax = plt.axes()       
        cmap = cm.get_cmap('tab20b')
        plt.scatter([x[0] for x in coord], [x[1] for x in coord], color=cmap(6), s=250)
        plt.yticks(list(range(size_grid)))
        plt.xticks(list(range(size_grid)))
        plt.grid(True)
        # now plot the rewards
        for i, x in enumerate(rewards):
            plt.scatter(x[0], x[1], color =cmap(i+11), s=250)
        for i, (x) in enumerate(rewards):
            plt.scatter(x[0], x[1], color =cmap(i+12), s=250)
        # finally, plot the path

    # now set up the paths.
    all_stepnums = []
    visited_fields = [[path[0]]]

    # plot the path
    for i, (x) in enumerate(path):
        start = path[i]
        if i == (len(path)-1):
            stop = path[0]
        else:
            stop = path[i+1]
        step, num_steps = mc.simulation.grid.find_paths(start,stop)  
        # all_paths.append([x for x in path])    
        visited_fields.append([x for x in step[1:]])
        all_stepnums.append(num_steps)
        if plotting == True:
            # jitter to make the order of paths visible 
            # (thats a bit ugly now, but serves the purpose of visibilty...)
            plotpath = np.array(step) + 0.1*np.random.randn(len(step), 2)
            for currstep, nextstep in zip(plotpath[:-1,:], plotpath[1:,:]):
                plt.plot([currstep[0], nextstep[0]], [currstep[1], nextstep[1]], color=cmap(i+12))
    # reshape the visited_fields variable to a not-nested list        
    reshaped_visited_fields=[]    
    for step in visited_fields:
        for coord in step:
            reshaped_visited_fields.append(coord)
    
    return reshaped_visited_fields, all_stepnums






