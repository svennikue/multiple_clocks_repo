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
import mc
import pandas as pd


# import pdb; pdb.set_trace() 
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"
ROI = 'HPC' # HPC mPFC
analysis_type = 'grid_wise' # grid_wise, exploration_trials 

preproc_type = 'referenced' # channel_wise

distribution = False

# import pdb; pdb.set_trace() 

# Plot the following across tasks:
    # a: timebin in bins defined as subpaths. Plot ripple amount per bin.
    # d: timebin by state, collapsed across repeats.
    # c: timebin in bins defined as correct repeat. Plot ripple amount per bin.
    # d: divide into 3 bits: pre finding D, pre doing one correct, post doing first correct. Plot 3 bars.
    # e: Sort ripples into: locked to neg feedback (feedback + 1000 samples); locked to pos feedb; not locked to feedb


# ok right. So I'm not really seeing more ripples/time (ripplerate) for 
                    
            
sessions = ['s5', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's25']
session_per_subj = {
    'sub-1': ['s5'],
    'sub-2': ['s7', 's8', 's9'],
    'sub-3': ['s10', 's11'],
    'sub-4': ['s12', 's13', 's14'],
    'sub-5': ['s25']
    }

# sessions = ['s7', 's8', 's9']
# session_per_subj = {
#     'sub-2': ['s7', 's8', 's9'],
#     }



# note that these subjects are the same person, respectively:
# - sub1 = s5 YEJ
# - sub2 = s7 + s8 + s9 YEK
# - sub3 = s10, s11 YEL
# - sub4 = s12, 13, 14 YEN
# - sub5 = s25 YEU

# make sure to consider all sessions per subject.
# import pdb; pdb.set_trace() 


# check what I want to show in the lab meeting
# start with stats across subjects
# 



#subjects = ['s13', 's12', 's25']
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"
for sub in session_per_subj:
    ripples_all_repeats_all_sessions = {}                
    ripple_rate_across_session = {}
    ripple_count_correct_solve_cross_sessions = {}
          
    for sesh in session_per_subj[sub]:
        # at some point do all data computations in here, and then collect them all for all sessions one subject had.
        
        # load behaviour that defines my snippets.
        behaviour_all = np.genfromtxt(f"{LFP_dir}/{sesh}/all_trials_times.csv", delimiter=',')
        # load behavioural stuff you need for plotting all sorts of stuff
        seconds_lower, seconds_upper, task_config, task_index, task_onset, new_grid_onset, found_first_D = mc.analyse.ripple_helpers.prep_behaviour(behaviour_all)

        with open(f"{result_dir}/{sub}_{ROI}_{analysis_type}_{preproc_type}_ripple_by_seconds.pkl", 'rb') as file:
            ripples = pickle.load(file)
        
        with open(f"{result_dir}/{sesh}_{ROI}_{analysis_type}_feedback.pkl", 'rb') as file:
            feedback_dict = pickle.load(file)
            
        # add a counter which trial per grid you are currently in (independent of correct)
        index_repeat = np.empty((len(behaviour_all),1))
        repeat_counter = 0
        for trial in range(0, len(behaviour_all)):
            if trial > 0:
                if behaviour_all[trial, 0] < behaviour_all[trial-1, 0]:
                    repeat_counter = 0
            index_repeat[trial] = repeat_counter
            repeat_counter = repeat_counter + 1
            
        # plot ripple distribution: all ripples.
        # for task_to_check in ripples:
        #     feedback_correct_curr_task = feedback_dict[f"{int(task_to_check)}_correct"]
        #     feedback_error_curr_task = feedback_dict[f"{int(task_to_check)}_error"]
        #     index_lower = np.where(np.array(task_index)== task_to_check)[0][0]
        #     index_upper = np.where(np.array(task_index)== task_to_check)[0][-1]
        #     ripples_across_channels = []
        #     for channel in ripples[task_to_check]:
        #         ripples_across_channels.extend(ripples[task_to_check][channel])
        #     mc.analyse.ripple_helpers.plot_ripple_distribution(ripples, task_to_check, feedback_error_curr_task, feedback_correct_curr_task, ripples_across_channels, found_first_D, seconds_upper, index_upper, index_lower, seconds_lower, sub)
    
        # only consider ripples from the inner-most channels, not the entire wire. 
        # These should include '07', '08' or '09'.
        channels_of_interest = ['Ha08','HaEa08','Ha09','HaEa09','Hb08','HbE08','Hb09','HbE09']
        ripple_hpc = {}
        for task_to_check in ripples:
            ripples_per_task = {}
            for channel in ripples[task_to_check]:
                if any(coi in channel for coi in channels_of_interest):
                    ripples_per_task[channel] = ripples[task_to_check][channel]
            ripple_hpc[task_to_check] = ripples_per_task
            
        for task_to_check in ripple_hpc:
            feedback_correct_curr_task = feedback_dict[f"{int(task_to_check)}_correct"]
            feedback_error_curr_task = feedback_dict[f"{int(task_to_check)}_error"]
            index_lower = np.where(np.array(task_index)== task_to_check)[0][0]
            index_upper = np.where(np.array(task_index)== task_to_check)[0][-1]
            ripples_across_hpc_channels = []
            for channel in ripple_hpc[task_to_check]:
                ripples_across_hpc_channels.extend(ripple_hpc[task_to_check][channel])
            if distribution == True:
                mc.analyse.plotting_ripples.plot_ripple_distribution(ripple_hpc, task_to_check, feedback_error_curr_task, feedback_correct_curr_task, ripples_across_hpc_channels, found_first_D, seconds_upper, index_upper, index_lower, seconds_lower, sub)
    
    
 
        # a. timebin in bins defined as subpaths of correct solves. Plot ripple amount per bin.
        # there will be 4*10 bins.
        ripples_by_state_per_repeat = {}
        ripple_rate_by_state_per_repeat = {}
        ripple_rate_by_state_per_try = {}
        for repeat in range(len(behaviour_all)):
            curr_task = behaviour_all[repeat, -1]
            if curr_task not in ripple_hpc:
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
                for channel in ripple_hpc[curr_task]:
                    for rip in ripple_hpc[curr_task][channel]:
                        if rip > lower and rip < upper:
                            if bin_to_fill not in ripples_by_state_per_repeat:
                                ripples_by_state_per_repeat[bin_to_fill] = 1
                            else:
                                ripples_by_state_per_repeat[bin_to_fill] = ripples_by_state_per_repeat[bin_to_fill] + 1
                duration_bin = upper-lower
                if bin_to_fill in ripples_by_state_per_repeat:
                    ripple_rate_by_state_per_repeat[bin_to_fill] = ripples_by_state_per_repeat[bin_to_fill]/duration_bin
                    
        
        # before plotting, first add all ripples across sessions that one subject completed!!
        
        title = f"Ripples per state and repeat for {sub}"
        x_title = 'State A-D correct solve 0 to 10'
        y_title = 'Sum of ripples across tasks, per state and correct solve'
        bars = 40
        mc.analyse.plotting_ripples.ripple_count_barchart(ripple_rate_by_state_per_repeat, bars, title, x_title, y_title)
        
        title = f"Ripple Rate per state and repeat for {sub}"
        x_title = 'State A-D correct solve 0 to 10'
        y_title = '[Sum of ripples/ duration] across tasks, per state and correct solve'
        mc.analyse.plotting_ripples.ripple_count_barchart(ripple_rate_by_state_per_repeat, bars, title, x_title, y_title)



        
        # now do the same thing, but count how many ripples per grid as a single datapoint
        # and plot as scatter.
        # a.2 timebin in bins defined as subpaths of repeats, irrespective of correct/incorrect. 
        # Plot ripple rate (sum ripples/ duration) per bin, across grids.
        # there will be at least 4*10 bins, potentially more if there are more repeats.
        
        
        # first loop: go through tasks.
        # inner loop: go through repeats.
        # most inner loop: go through states 1,2,3,4.
        
        # save ripple rate per state-repeat-task, and collect all in a list per task.
        
        # then plot.
        
        behaviour_df = pd.DataFrame(behaviour_all)
        
        # I don't think this is ideal since it compare successful with unsuccessfull repeats.
        # maybe better to take the entire 'correct' after all.
        ripples_all_repeat_dict = {}
        max_repeats = 0
        for task in ripple_hpc:
            ripples_for_grid = {}
            beh_subset_task = behaviour_df[behaviour_df[12] == task]
            for repeat in range(0, len(beh_subset_task)):
                if len(beh_subset_task) > max_repeats:
                    max_repeats = len(beh_subset_task)
                for state in [1,2,3,4]:
                    # define where to add the ripples to
                    bin_to_fill = (repeat)*4 + state
                    # then define at what time to look
                    if state == 1:
                        lower = beh_subset_task.iloc[repeat][10]
                    else:
                        lower = beh_subset_task.iloc[repeat][state-1]
                    upper = beh_subset_task.iloc[repeat][state]
                    duration = upper-lower 
                    # then check for ripples in within this time
                    for channel in ripple_hpc[task]:
                        for rip in ripple_hpc[task][channel]:
                            if rip > lower and rip < upper:
                                if bin_to_fill not in ripples_for_grid:
                                    ripples_for_grid[bin_to_fill] = 1
                                else:
                                    ripples_for_grid[bin_to_fill] = ripples_for_grid[bin_to_fill] + 1   
                    if bin_to_fill not in ripples_for_grid:
                        ripples_for_grid[bin_to_fill] = np.nan
                    if bin_to_fill in ripples_for_grid:
                        # print(f"Found {ripples_for_grid[bin_to_fill]} ripples for state {state} and repeat {repeat}")
                        ripples_for_grid[bin_to_fill] = ripples_for_grid[bin_to_fill]/duration
            # 
            ripples_all_repeat_dict[task] = ripples_for_grid
        ripples_all_repeats_all_sessions[sesh] = ripples_all_repeat_dict
        
        
        # first loop: go through tasks.
        # inner loop: go through repeats.
        # most inner loop: go through states 1,2,3,4.
        
        # save ripple rate per state for every correct solve
        # then plot.
        
        behaviour_df = pd.DataFrame(behaviour_all)
        
        # only for 'correct' solves. -> sum up incorrect repeats!
        rate_result_dict = {}
        amount_result_dict = {}
        for task in ripple_hpc:
            ripples_for_grid = {}
            ripple_rate_per_grid = {}
            beh_subset_task = behaviour_df[behaviour_df[12] == task]
            for correct_solve in range(0, 10):
                beh_subset_repeats = beh_subset_task[beh_subset_task[0] == correct_solve]
                cum_duration = np.zeros((4))
                for repeat in range(len(beh_subset_repeats)):
                    for state in [1,2,3,4]:
                        # if task == 3:
                        #      import pdb; pdb.set_trace() 
                        # define where to add the ripples to
                        bin_to_fill = (correct_solve)*4 + state
                        # then define at what time to look
                        if state == 1:
                            lower = beh_subset_repeats.iloc[repeat][10]
                        else:
                            lower = beh_subset_repeats.iloc[repeat][state-1]
                        upper = beh_subset_repeats.iloc[repeat][state]
                        # print(f"duration for state {state} and attempt {repeat} in repeat {correct_solve} of task {task} is {upper-lower}")
                        cum_duration[state-1] = cum_duration[state-1] + upper-lower 
                        # then check for ripples in within this time
                        for channel in ripple_hpc[task]:
                            for rip in ripple_hpc[task][channel]:
                                if rip > lower and rip < upper:
                                    if bin_to_fill not in ripples_for_grid:
                                        ripples_for_grid[bin_to_fill] = 1
                                    else:
                                        ripples_for_grid[bin_to_fill] = ripples_for_grid[bin_to_fill] + 1  
                        if repeat == (len(beh_subset_repeats)-1) and bin_to_fill in ripples_for_grid:
                            #print(f"Found {ripples_for_grid[bin_to_fill]} ripples for state {state} and in repeat {correct_solve} of task {task}, duration overall {cum_duration[state-1]}")
                            #print(f"in bin {bin_to_fill} the rate will be {ripples_for_grid[bin_to_fill]/cum_duration[state-1]}")
                            ripple_rate_per_grid[bin_to_fill] = ripples_for_grid[bin_to_fill]/cum_duration[state-1]
                            
                        if repeat == (len(beh_subset_repeats)-1) and bin_to_fill not in ripples_for_grid:
                            ripples_for_grid[bin_to_fill] = np.nan
                            ripple_rate_per_grid[bin_to_fill] = np.nan
                            
                amount_result_dict[task] = ripples_for_grid
                rate_result_dict[task] = ripple_rate_per_grid
                
        ripple_rate_across_session[sesh] = rate_result_dict
        ripple_count_correct_solve_cross_sessions[sesh] = amount_result_dict
        
        
        # # Loop from 1 to 40 and get the corresponding value from the dictionary
        # heights = []
        # for i in range(1, 41):
        #     if i in ripple_rate_by_state_per_repeat:
        #         heights.append(ripple_rate_by_state_per_repeat[i])  # Use the value from the dictionary
        #     else:
        #         heights.append(0)  # If no key exists for this index, append 0
        
        # # Generate labels for the x-axis (numbers 1 to 40)
        # labels = [str(i) for i in range(1, 41)]
        
        # plt.figure();
        # # Create the bar plot
        # plt.bar(labels, heights)
        
        # # Add title and labels
        # plt.title(f"Ripple Rate per state and repeat for {sub}")
        # plt.xlabel('State A-D correct solve 0 to 10')
        # plt.ylabel('[Sum of ripples/ duration] across tasks, per state and correct solve')
        
        # # Show the plot
        # plt.show()
             
        
        
        # # b per state, collapsed across repeats.
        # ripples_by_state = {}
        # for repeat in range(len(behaviour_all)):
        #     curr_task = behaviour_all[repeat, -1]
        #     if curr_task not in ripple_hpc:
        #         continue
        #     for state in [1,2,3,4]:
        #         if state == 1:
        #             lower = behaviour_all[repeat, 10]
        #         else:
        #             lower = behaviour_all[repeat, state-1]
        #         upper = behaviour_all[repeat, state]
                
        #         #then check for ripples in this bin
        #         for channel in ripple_hpc[curr_task]:
        #             for rip in ripple_hpc[curr_task][channel]:
        #                 if rip > lower and rip < upper:
        #                     if state not in ripples_by_state:
        #                         ripples_by_state[state] = 1
        #                     ripples_by_state[state] = ripples_by_state[state] + 1
        
        
        # plt.figure();
        # categories = list(ripples_by_state.keys())
        # heights = list(ripples_by_state.values())
        
        # # Create a bar plot
        # plt.bar(categories, heights)
        # plt.ylabel('Sum of ripples across tasks and repeats')
        # plt.title(f"Ripples per state for subj {sub}")
        # plt.xticks([1,2,3,4], ['Going to A', 'Going to B', 'Going to C', 'Going to D'])
        
        
        
        # # c: timebin in bins defined as correct repeat. Plot ripple amount per bin.
        # ripples_by_repeat = {}
        # for repeat in range(len(behaviour_all)):
        #     curr_task = behaviour_all[repeat, -1]
        #     if curr_task not in ripple_hpc:
        #         continue
        #     curr_repeat = behaviour_all[repeat, 0]
        #     lower = behaviour_all[repeat, 10]
        #     upper = behaviour_all[repeat, 4]
        #     #then check for ripples in this bin
        #     for channel in ripple_hpc[curr_task]:
        #         for rip in ripple_hpc[curr_task][channel]:
        #             if rip > lower and rip < upper:
        #                 if curr_repeat not in ripples_by_repeat:
        #                     ripples_by_repeat[curr_repeat] = 1
        #                 ripples_by_repeat[curr_repeat] = ripples_by_repeat[curr_repeat] + 1
            
        
        # plt.figure();
        # categories = list(ripples_by_repeat.keys())
        # heights = list(ripples_by_repeat.values())
        
        # # Create a bar plot
        # plt.bar(categories, heights)
        # plt.ylabel('Sum of ripples per repeat across grids')
        # plt.title(f"Ripples per repeat for {sub}")
        # plt.xlabel('Repeats 0-10')
        
        
        # d: divide into 3 bits: pre finding D, pre doing one correct, post doing first correct. Plot 3 bars.
        found_first_D = []
        
        ripples_by_exploration = {}
        # first get indices for new tasks.
        next_task_index = [0]
        for repeat in range(1, len(behaviour_all)):
            if behaviour_all[repeat, -1] > behaviour_all[repeat-1, -1]:
                next_task_index.append(repeat)
            
        
        for curr_task, line in enumerate(next_task_index):
            if curr_task+1 not in ripple_hpc:
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
            for channel in ripple_hpc[curr_task+1]:
                for rip in ripple_hpc[curr_task+1][channel]:
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
        plt.title(f"Ripples per execution phase for {sub}")
        plt.xlabel('Exploration - solving first grid correctly - the other 9 repeats')
    
        
    
    # per subject, but across sessions.
    plotting_dict = {}
    for session in ripples_all_repeats_all_sessions.keys():
        for task in ripples_all_repeats_all_sessions[session].keys():
            for boxplot_idx in ripples_all_repeats_all_sessions[session][task].keys():
                if boxplot_idx not in plotting_dict:
                    plotting_dict[boxplot_idx] = []
                plotting_dict[boxplot_idx].append(ripples_all_repeats_all_sessions[session][task][boxplot_idx])
            # Collect data for this boxplot (task and boxplot_idx specific)

    # first collect all sessions a subject completed!!!
    title = f"Ripplerate per repeat across grids for subject {sub}"
    x_title = 'A-D = one repeat. Repeats depend on errors made.'
    y_title = 'Ripple Rate'
    mc.analyse.plotting_ripples.ripple_amount_violin_scatter(plotting_dict, title, x_title, y_title)
    
    # plot rate 
    plotting_dict = {}
    for session in ripple_rate_across_session.keys():
        for task in ripple_rate_across_session[session].keys():
            for boxplot_idx in ripple_rate_across_session[session][task].keys():
                if boxplot_idx not in plotting_dict:
                    plotting_dict[boxplot_idx] = []
                plotting_dict[boxplot_idx].append(ripple_rate_across_session[session][task][boxplot_idx])
                # Collect data for this boxplot (task and boxplot_idx specific)

    title = f"Ripplerate per correct solve across grids for {sub}"
    x_title = 'A-D = one correct solve. Repeats depend on errors made.'
    y_title = 'Ripple Rate'
    mc.analyse.plotting_ripples.ripple_amount_violin_scatter(plotting_dict, title, x_title, y_title)
    
    # plot amount 
    plotting_dict = {}
    for session in ripple_count_correct_solve_cross_sessions.keys():
        for task in ripple_count_correct_solve_cross_sessions[session].keys():
            for boxplot_idx in ripple_count_correct_solve_cross_sessions[session][task].keys():
                if boxplot_idx not in plotting_dict:
                    plotting_dict[boxplot_idx] = []
                plotting_dict[boxplot_idx].append(ripple_count_correct_solve_cross_sessions[session][task][boxplot_idx])
            # Collect data for this boxplot (task and boxplot_idx specific)

    title = f"Ripple Count per correct solve across grids for {sub}"
    x_title = 'A-D = one correct solve. Repeats depend on errors made.'
    y_title = 'Ripple Count'
    mc.analyse.plotting_ripples.ripple_amount_violin_scatter(plotting_dict, title, x_title, y_title)
    

    
    
    
    
