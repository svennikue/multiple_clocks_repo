#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:14:44 2023
This file is to open and clean my behavioural variables.

@author: xpsy1114
"""


import pandas as pd
import numpy as np
import mc
import matplotlib.pyplot as plt
import scipy.special as sps  


def print_stuff(string_input):
    print(string_input)
    
    

def jitter(expected_step_no):
    # first randomly sample from a gamma distribution
    shape = 5.75 # this is what the mean subpath is supposed to be
    draw = np.random.standard_gamma(shape)
    
    # then make an array for each step + reward I expect to take
    step_size_maker = np.random.randint(1, expected_step_no + 4, size= expected_step_no + 1)
    
    # make the last one, the reward, twice as long as the average step
    ave_step = np.mean(step_size_maker)
    step_size_maker[-1] = ave_step*2
    
    # then multiply the fraction of all step sizes with the actual subpath length
    stepsizes = np.empty(expected_step_no + 1)
    for i in range(expected_step_no+ 1):
        stepsizes[i] = (step_size_maker[i]/ (sum(step_size_maker))) * draw
        
    # stepsizes [-1] will be reward length. if more steps than stepsizes[0:-2], randomly sample.
    
    return(stepsizes)



    # plotting how I draw the randomly jittered steps
    
    # # first randomly sample from a gamma distribution
    # # or from an exponantial
    # # then sample no of optimal steps random numbers 
    # # e.g. if 3 random numbers
    # # a/(a+b+c) * randomly sampled goal
    # # then 
    repeats = 10000
    shape, scale = 5.75, 1. # mean and width
    s = np.empty(repeats)
    for i in range(repeats):
        draw = np.random.standard_gamma(shape)
        while (draw < 3) or (draw > 15):
            draw = np.random.standard_gamma(shape)
        s[i] = draw
    
    step_no = 3
    step_size_maker = np.random.randint(1, step_no + 4, size= step_no + 1)
    ave_step = np.mean(step_size_maker)
    step_size_maker[-1] = ave_step*2
    
    # Find the index of the maximum value in the array
    # max_index = np.argmax(step_size_maker) 
    # Swap the maximum value with the last element
    # step_size_maker[max_index], step_size_maker[-1] = step_size_maker[-1], step_size_maker[max_index]

    stepsizes = np.empty(step_no + 1)
    for i in range(step_no+ 1):
        stepsizes[i] = (step_size_maker[i]/ (sum(step_size_maker))) * draw
        
    print (f'Step 1 = {stepsizes[0]} Step 2 = {stepsizes[1]} Step 3 = {stepsizes[1]}, rew = {stepsizes[-1]}, sum = {sum(stepsizes)}')
    
    plt.figure()
    count, bins, ignored = plt.hist(s, 50, density=True)
    y = bins**(shape-1) * ((np.exp(-bins/scale))/(sps.gamma(shape) * scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')  
    plt.show()
    
    

# code snippet to create a regressor
def create_EV(onset, duration, magnitude, name, folder, TR_at_sec):
    if len(onset) > len(duration):
        onset = onset[:len(duration)]
        magnitude = magnitude[:len(duration)]
    elif len(duration) > len(onset):
        duration = onset[:len(onset)]
        magnitude = magnitude[:len(onset)]
    regressor_matrix = np.ones((len(magnitude),3))
    regressor_matrix[:,0] = [(time - TR_at_sec) for time in onset]
    regressor_matrix[:,1] = duration
    regressor_matrix[:,2] = magnitude
    # import pdb; pdb.set_trace()
    np.savetxt(str(folder) + 'ev_' + str(name) + '.txt', regressor_matrix, delimiter="    ", fmt='%f')
    return regressor_matrix


# to transform the locations
def transform_coord(coord, is_x = False, is_y = False):
    if is_x:
        if coord == -0.21:
            return 0
        elif coord == 0:
            return 1
        elif coord == 0.21:
            return 2
    if is_y:
        if coord == -0.29:
            return 0
        elif coord == 0:
            return 1
        elif coord == 0.29:
            return 2
    # Add more conditions if needed
    else:
        return None



# use to check if the EV making went wrong
def check_for_nan(array):
    if np.isnan(array).any():
        print(f"Careful! There are Nans in {array}. Pausing script")
        import pdb; pdb.set_trace()



def make_loc_EV(dataframe, x_coord, y_coord):
    # import pdb; pdb.set_trace()

    skip_next = False
    loc_dur = []
    loc_on = []
    loc_df = dataframe[(dataframe['curr_loc_x'] == x_coord) & (dataframe['curr_loc_y'] == y_coord)]
    #loc_one_on = loc_one['t_step_press_global'].to_list()
    for index in loc_df.index:
        if index > 0:  # Check if the index is valid (not the first row)
            if skip_next:
                skip_next = False
                continue
            # next, check if it's a row with a reward that has started within a task
            if not np.isnan(dataframe.at[index,'rew_loc_x']): # if this is a reward field.
                if index+2 < len(dataframe):
                    if dataframe.at[index, 'task_config'] == dataframe.at[index+1, 'task_config']:
                        start = dataframe.at[index, 't_reward_start'] 
                        # if its within the same task config, compute duration like so
                        duration = dataframe.at[index + 1, 't_step_press_global'] - start
                    elif dataframe.at[index, 'task_config'] != dataframe.at[index+1, 'task_config']:
                        start = dataframe.at[index, 't_reward_start'] 
                        # if its not within same task config, take reward afterwait to compute duration
                        duration = dataframe.at[index, 't_reward_afterwait'] - start
                else:
                    # if this is the last row, then ignore- this is where the task stopped.
                    continue
                
            
            
            if np.isnan(dataframe.at[index, 't_step_press_global']):
                # first check if this is where the task stopped, or if the next reward is where the task stopped.
                # ignore these as they are not complete.
                if index+2 < len(dataframe):
                    if dataframe.at[index, 'task_config'] == dataframe.at[index+1, 'task_config']:
                        start = dataframe.at[index, 't_reward_start'] 
                        # if its within the same task config, compute duration like so
                        duration = dataframe.at[index + 1, 't_step_press_global'] - start
                    elif dataframe.at[index, 'task_config'] != dataframe.at[index+1, 'task_config']:
                        start = dataframe.at[index, 't_reward_start'] 
                        # if its not within same task config, take reward afterwait to compute duration
                        duration = dataframe.at[index, 't_reward_afterwait'] - start
                else:
                    # if this is the last row, then ignore- this is where the task stopped.
                    continue
              
                loc_on.append(start)
                loc_dur.append(duration)
                # and then skip the next row
                skip_next = True
                    
            else: 
                # in case a new task config just started
                if dataframe.at[index, 'task_config'] != dataframe.at[index-1, 'task_config']:
                    start = dataframe.at[index, 'start_ABCD_screen']
                else:
                    start = dataframe.at[index - 1, 't_step_press_global']  # Extract 'goal' value from index-1
                duration = dataframe.at[index, 't_step_press_global'] - start
                loc_on.append(start)
                loc_dur.append(duration)            
    loc_mag = np.ones(len(loc_on))
    
    return(loc_on,loc_dur, loc_mag)









