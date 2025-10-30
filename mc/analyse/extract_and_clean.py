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





def define_futsteps_x_locs_regressors(beh_df):
    # import pdb; pdb.set_trace()
    unique_tasks = beh_df['task_config_ex'].unique()
    beh = beh_df.copy()
    
    # defining whenever a new reward is found within a task.
    beh['rew_no'] = beh_df.groupby('task_config_ex')['curr_rew'].apply(lambda s: s.ne(s.shift()).cumsum())
    
    # 2) One row per run + future runs (curr, +1, +2, +3)
    r = (beh.drop_duplicates(['task_config_ex','rew_no'])[['task_config_ex','rew_no','curr_rew']].rename(columns={'curr_rew':'curr'}))
    r['one_fut']   = r.groupby('task_config_ex')['curr'].shift(-1)
    r['two_fut']   = r.groupby('task_config_ex')['curr'].shift(-2)
    r['three_fut'] = r.groupby('task_config_ex')['curr'].shift(-3)

    
    # 3) Broadcast back to all rows in that run
    beh = beh.merge(r, on=['task_config_ex','rew_no'], how='left')
    
    # 4) Make the 36 one-hot columns (locations 1..9 × {curr, +1, +2, +3})
    for step in ['curr','one_fut','two_fut','three_fut']:
        for loc in range(1, 10):
            beh[f'loc_{loc}_{step}'] = (beh[step] == loc).astype('int8')

    return beh


def print_stuff(string_input):
    print(string_input)
   
    
   
def flatten_nested_dict(nested_dict):
    flattened_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):  # If the value is a dictionary, extend the flat dictionary with its items
            flattened_dict.update(value)
        else:
            flattened_dict[key] = value
    return flattened_dict
 

   
def order_task_according_to_rewards(reward_per_task_per_taskhalf_dict):  
    # import pdb; pdb.set_trace() 
    rewards_experiment = mc.analyse.extract_and_clean.flatten_nested_dict(reward_per_task_per_taskhalf_dict)
    ordered_config_names = {half: [] for half in reward_per_task_per_taskhalf_dict}  

    no_duplicates_list = []    
    for i, task_reference in enumerate(sorted(rewards_experiment.keys())):
        if task_reference not in no_duplicates_list:
            for task_comp in rewards_experiment:
                if task_comp not in no_duplicates_list:
                    if not task_reference == task_comp:
                        if rewards_experiment[task_reference] == rewards_experiment[task_comp]:
                            ordered_config_names['1'].append(task_reference)
                            ordered_config_names['2'].append(task_comp)
                            no_duplicates_list.append(task_reference)
                            no_duplicates_list.append(task_comp)
                            
    return ordered_config_names



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


    
    
    #   plotting how I draw the randomly jittered steps
    
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