#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:01:01 2024

First run LFP_ripple_dection.py
This is to visualise and run stats on detected ripple events.

@author: Svenja Kuchenhoff
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os     


# import pdb; pdb.set_trace() 
names_blks_short = ['EMU-117_subj-YEU_task-ABCD_run-01_blk-01_NSP-1','EMU-118_subj-YEU_task-ABCD_run-01_blk-02_NSP-1']
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"
ROI = 'HPC' # HPC mPFC
analysis_type = 'grid_wise' # grid_wise, exploration_trials 



subjects = ['s12', 's25']
#subjects = ['s13', 's12', 's25']
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"
for sub in subjects :
    ns3_files = glob.glob(os.path.join(f"{LFP_dir}/{sub}/", '*.ns3'))
    names_blks_short = [os.path.splitext(os.path.basename(file))[0] for file in ns3_files]

    
    # load behaviour that defines my snippets.
    behaviour = np.genfromtxt(f"{LFP_dir}/{sub}/exploration_trials_times_and_ncorrect.csv", delimiter=',')
    behaviour_all = np.genfromtxt(f"{LFP_dir}/{sub}/all_trials_times.csv", delimiter=',')
    feedback = np.genfromtxt(f"{LFP_dir}/{sub}/feedback.csv", delimiter=',')
    
    
    # with open(f"{result_dir}/{sub}_{ROI}_{analysis_type}_ripple_events_dir.pkl", 'rb') as file:
    #     events_dict = pickle.load(file)
        
    with open(f"{result_dir}/{sub}_{ROI}_{analysis_type}_ripple_by_seconds.pkl", 'rb') as file:
        ripples = pickle.load(file)
    
    
    # maybe first normalise the amount of ripples by the ripple count of this task??
    # otherwise some tasks are completely over represented...
    
    
    # Plot the following across tasks:
        # a: timebin in bins defined as subpaths. Plot ripple amount per bin.
        # d: timebin by state, collapsed across repeats.
        # c: timebin in bins defined as correct repeat. Plot ripple amount per bin.
        # d: divide into 3 bits: pre finding D, pre doing one correct, post doing first correct. Plot 3 bars.
        # e: Sort ripples into: locked to neg feedback (feedback + 1000 samples); locked to pos feedb; not locked to feedb
        
        
        
    # a. timebin in bins defined as subpaths of correct solves. Plot ripple amount per bin.
    # there will be 4*10 bins.
    ripples_by_state_per_repeat = {}
    for repeat in range(len(behaviour_all)):
        curr_task = behaviour_all[repeat, -1]
        if curr_task not in ripples:
            continue
        curr_correct_solve = behaviour_all[repeat, 0]
        ripples_per_state_repeat_curr_task = []
        for state in [1,2,3,4]:
            bin_to_fill = (curr_correct_solve)*4 + state
            if state == 1:
                lower = behaviour_all[repeat, 10]
            else:
                lower = behaviour_all[repeat, state-1]
            upper = behaviour_all[repeat, state]
            #then check for ripples in this bin
            for channel in ripples[curr_task]:
                for rip in ripples[curr_task][channel]:
                    if rip > lower and rip < upper:
                        if bin_to_fill not in ripples_by_state_per_repeat:
                            ripples_by_state_per_repeat[bin_to_fill] = 1
                        else:
                            ripples_by_state_per_repeat[bin_to_fill] = ripples_by_state_per_repeat[bin_to_fill] + 1
    
    # Initialize the heights list for 40 bars
    heights = []
    
    # Loop from 1 to 40 and get the corresponding value from the dictionary
    for i in range(1, 41):
        if i in ripples_by_state_per_repeat:
            heights.append(ripples_by_state_per_repeat[i])  # Use the value from the dictionary
        else:
            heights.append(0)  # If no key exists for this index, append 0
    
    # Generate labels for the x-axis (numbers 1 to 40)
    labels = [str(i) for i in range(1, 41)]
    
    plt.figure();
    # Create the bar plot
    plt.bar(labels, heights)
    
    # Add title and labels
    plt.title(f"Ripples per state and repeat for subj {sub}")
    plt.xlabel('State A-D correct solve 0 to 10')
    plt.ylabel('Sum of ripples across tasks, per state and correct solve')
    
    # Show the plot
    plt.show()
    
                        
    
    
    # b per state, collapsed across repeats.
    ripples_by_state = {}
    for repeat in range(len(behaviour_all)):
        curr_task = behaviour_all[repeat, -1]
        if curr_task not in ripples:
            continue
        for state in [1,2,3,4]:
            if state == 1:
                lower = behaviour_all[repeat, 10]
            else:
                lower = behaviour_all[repeat, state-1]
            upper = behaviour_all[repeat, state]
            
            #then check for ripples in this bin
            for channel in ripples[curr_task]:
                for rip in ripples[curr_task][channel]:
                    if rip > lower and rip < upper:
                        if state not in ripples_by_state:
                            ripples_by_state[state] = 1
                        ripples_by_state[state] = ripples_by_state[state] + 1
    
    
    plt.figure();
    categories = list(ripples_by_state.keys())
    heights = list(ripples_by_state.values())
    
    # Create a bar plot
    plt.bar(categories, heights)
    plt.ylabel('Sum of ripples across tasks and repeats')
    plt.title(f"Ripples per state for subj {sub}")
    plt.xticks([1,2,3,4], ['Going to A', 'Going to B', 'Going to C', 'Going to D'])
    
    
    
    # c: timebin in bins defined as correct repeat. Plot ripple amount per bin.
    ripples_by_repeat = {}
    for repeat in range(len(behaviour_all)):
        curr_task = behaviour_all[repeat, -1]
        if curr_task not in ripples:
            continue
        curr_repeat = behaviour_all[repeat, 0]
        lower = behaviour_all[repeat, 10]
        upper = behaviour_all[repeat, 4]
        #then check for ripples in this bin
        for channel in ripples[curr_task]:
            for rip in ripples[curr_task][channel]:
                if rip > lower and rip < upper:
                    if curr_repeat not in ripples_by_repeat:
                        ripples_by_repeat[curr_repeat] = 1
                    ripples_by_repeat[curr_repeat] = ripples_by_repeat[curr_repeat] + 1
        
    
    plt.figure();
    categories = list(ripples_by_repeat.keys())
    heights = list(ripples_by_repeat.values())
    
    # Create a bar plot
    plt.bar(categories, heights)
    plt.ylabel('Sum of ripples per repeat across grids')
    plt.title(f"Ripples per repeat for subj {sub}")
    plt.xlabel('Repeats 0-10')
    
    
    # d: divide into 3 bits: pre finding D, pre doing one correct, post doing first correct. Plot 3 bars.
    found_first_D = []
    
    ripples_by_exploration = {}
    # first get indices for new tasks.
    next_task_index = [0]
    for repeat in range(1, len(behaviour_all)):
        if behaviour_all[repeat, -1] > behaviour_all[repeat-1, -1]:
            next_task_index.append(repeat)
        
    
    for curr_task, line in enumerate(next_task_index):
        if curr_task+1 not in ripples:
            continue
        lowest = behaviour_all[line, 10] # task onset
        middle_low = behaviour_all[line, 4] # found first 
        if curr_task+1 < len(next_task_index):
            for repeats in range(next_task_index[curr_task+1]-next_task_index[curr_task]):
                if behaviour_all[line+repeats, 0] == 1 and behaviour_all[line+repeats-1,0] < 1:
                    middle_high = behaviour_all[line+repeats, 4] # completed first task
                if behaviour_all[line+repeats,0] == 9:
                    highest = behaviour_all[line+repeats, 4] # finished task
        # import pdb; pdb.set_trace() 
        for channel in ripples[curr_task+1]:
            for rip in ripples[curr_task+1][channel]:
                if rip > lowest and rip < middle_low:
                    if 'exploration' not in ripples_by_exploration:
                        ripples_by_exploration['exploration'] = 1
                    else:
                        ripples_by_exploration['exploration'] = ripples_by_exploration['exploration'] + 1
                elif rip > middle_low and rip < middle_high:
                    if 'first correct' not in ripples_by_exploration:
                        ripples_by_exploration['first correct'] = 1
                    else: 
                        ripples_by_exploration['first correct'] = ripples_by_exploration['first correct'] + 1
                elif rip > middle_high and rip < highest:
                    if 'repeats' not in ripples_by_exploration:
                        ripples_by_exploration['repeats'] = 1
                    else: 
                        ripples_by_exploration['repeats'] = ripples_by_exploration['repeats'] + 1
            
    
    plt.figure();
    categories = list(ripples_by_exploration.keys())
    heights = list(ripples_by_exploration.values())
    
    # Create a bar plot
    plt.bar(categories, heights)
    plt.ylabel('Sum of ripples per execution phase across grids')
    plt.title(f"Ripples per execution phase for subj {sub}")
    plt.xlabel('Exploration - solving first grid correctly - the other 9 repeats')
    
    
    
    
    
    
    
    
    
    
    
